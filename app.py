"""
Fraud Detection API - FastAPI Backend
Serves predictions from pre-trained ML models using full preprocessing pipeline.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, Literal
import joblib
import numpy as np
from pathlib import Path
import logging
import pandas as pd
import shap
from preprocessing import preprocess_input
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API", version="2.0.0")

try:
    from llm_explainer import generate_llm_explanation
except ImportError as e:
    logging.warning(f"LLM Explainer module failed to import: {e}")
    generate_llm_explanation = None

# --- CONFIG ---
MODELS_DIR = Path("models")
APP_VERSION = "1.5.1"
THRESHOLD_AUTO_FLAG = 0.53

# Model registry
MODELS = {}

# SHAP Configuration
BACKGROUND_DATA_PATH = MODELS_DIR / "shap_background.npy"
FEATURE_NAMES_PATH = MODELS_DIR / "shap_feature_names.joblib"
METADATA_PATH = MODELS_DIR / "feature_metadata.joblib"

SHAP_EXPLAINERS = {}
SHAP_INIT_ERRORS = {}
BACKGROUND_DATA = None
SHAP_FEATURE_NAMES = None
FEATURE_METADATA = None
EXPLANATION_SOURCE_MODEL = "ExtraTrees_uncalibrated"

# Feature Name Mapping (Technical -> User)
# We keep this for the final display Mapping
FEATURE_MAP = {
    "total_claim_amount": "Claim Value",
    "injury_share": "Injury Cost Portion",
    "property_share": "Property Damage Portion",
    "incident_hour_of_the_day": "Incident Time",
    "months_as_customer": "Policy Tenure",
    "policy_annual_premium": "Annual Premium",
    "vehicle_age": "Vehicle Age",
    "age": "Insured Age",
    "capital-gains": "Capital Gains",
    "capital-loss": "Capital Losses",
    "umbrella_limit": "Umbrella Limit",
    "bodily_injuries": "Bodily Injuries",
    "number_of_vehicles_involved": "Vehicles Involved",
    "incident_severity_Major Damage": "Major Damage Severity",
    "incident_severity_Total Loss": "Total Loss Severity",
    "collision_type_Rear Collision": "Rear Collision Type",
    "authorities_contacted_Police": "Police Contacted"
}

class ClaimInput(BaseModel):
    """Input schema accepting Raw + New Categorical Features"""
    # Numeric
    policy_annual_premium: float = Field(..., description="Annual policy premium")
    total_claim_amount: float = Field(..., description="Total claim amount")
    vehicle_age: int = Field(..., description="Age of vehicle in years")
    days_since_bind: int = Field(..., description="Days since policy binding")
    months_as_customer: int = Field(..., description="Months as customer")
    capital_gains: float = Field(0.0, alias="capital-gains")
    capital_loss: float = Field(0.0, alias="capital-loss")
    injury_share: float = Field(..., description="Share of injury damage")
    property_share: float = Field(..., description="Share of property damage")
    age: int = Field(38, description="Insured Age")
    umbrella_limit: int = Field(..., description="Umbrella policy limit")
    incident_hour_of_the_day: int = Field(..., ge=0, le=23)
    
    # New Categorical Fields
    collision_type: Optional[str] = Field(None, description="Front Collision, Side Collision, Rear Collision, or ?")
    incident_severity: Optional[str] = Field(None, description="Major Damage, Minor Damage, Total Loss, Trivial Damage")
    authorities_contacted: Optional[str] = Field(None, description="Police, Fire, Ambulance, Other, None")
    number_of_vehicles_involved: Optional[int] = Field(1, description="Number of vehicles")
    bodily_injuries: Optional[int] = Field(0, description="Number of injuries")
    police_report_available: Optional[str] = Field(None, description="YES, NO, ?")
    
    class Config:
        populate_by_name = True

class ExplanationItem(BaseModel):
    feature: str
    direction: str # "UP" or "DOWN"
    text: str
    importance: float

class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    model: str
    calibrated: bool
    probability: float
    threshold_flag: Optional[str] = None
    scenario: str
    explanation: Optional[list[ExplanationItem]] = None
    llm_explanation: Optional[dict] = None
    app_version: str = "1.0.0"

def load_models():
    """Load all available models on startup"""
    model_types = ["RandomForest", "ExtraTrees", "XGBoost", "VotingEnsemble"]
    calibration_types = ["calibrated", "uncalibrated"]
    
    for model_type in model_types:
        for cal_type in calibration_types:
            filename = f"best_tree_models_{cal_type}.joblib"
            filepath = MODELS_DIR / filename
            
            if filepath.exists():
                try:
                    models_dict = joblib.load(filepath)
                    if 'Trees' in models_dict and model_type in models_dict['Trees']:
                        key = f"{model_type}_{cal_type}"
                        MODELS[key] = models_dict['Trees'][model_type]
                        logger.info(f"Loaded model: {key}")
                except Exception as e:
                    logger.error(f"Error loading {filepath}: {e}")
    
    logger.info(f"Total models loaded: {len(MODELS)}")

def get_pipeline_components(model):
    """Extract (preprocessor, estimator) from Pipeline"""
    if hasattr(model, 'steps'):
        # Usually steps=[('prep', ColumnTransformer), ('clf', Estimator)]
        return model.steps[0][1], model.steps[-1][1]
    return None, model

def load_shap_resources():
    """Load pre-calculated SHAP artifacts and initialize explainers"""
    global BACKGROUND_DATA, SHAP_FEATURE_NAMES, FEATURE_METADATA
    
    # 1. Load Pre-Processed Background Data
    if BACKGROUND_DATA_PATH.exists():
        BACKGROUND_DATA = np.load(BACKGROUND_DATA_PATH)
        logger.info(f"Loaded processed SHAP background: {BACKGROUND_DATA.shape}")
    else:
        logger.warning("SHAP background (npy) not found.")

    # 2. Load Feature Names
    if FEATURE_NAMES_PATH.exists():
        SHAP_FEATURE_NAMES = joblib.load(FEATURE_NAMES_PATH)
        logger.info(f"Loaded {len(SHAP_FEATURE_NAMES)} feature names.")
    
    # 3. Load Metadata
    if METADATA_PATH.exists():
        try:
            FEATURE_METADATA = joblib.load(METADATA_PATH)
            logger.info("Loaded feature metadata.")
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")

    # 4. Initialize TreeExplainers (ONLY for the Source Model)
    # We explicitly skip XGBoost to avoid version crashes, and Voting.
    # We only really need the EXPLANATION_SOURCE_MODEL.
    
    target_models = [EXPLANATION_SOURCE_MODEL]
    
    for key in target_models:
        if key not in MODELS:
            logger.warning(f"Explanation source {key} not loaded in MODELS.")
            continue
            
        if BACKGROUND_DATA is None:
            SHAP_INIT_ERRORS[key] = "Background data missing"
            continue

        try:
             model = MODELS[key]
             _, estimator = get_pipeline_components(model)
             
             # Direct initialization
             explainer = shap.TreeExplainer(estimator, BACKGROUND_DATA)
             SHAP_EXPLAINERS[key] = explainer
             logger.info(f"Initialized SHAP for {key} (Canonical Explanation Source)")
             
        except Exception as e:
            logger.error(f"Failed to init SHAP for {key}: {e}")
            SHAP_INIT_ERRORS[key] = str(e)

@app.on_event("startup")
async def startup_event():
    load_models()
    load_shap_resources()

def get_readable_explanation(feature_name, shap_val, metadata=None):
    """
    Generate explanation based on SHAP direction and Feature Meaning.
    Does NOT use raw values or thresholds, only direction and presence.
    """
    direction = "Increased risk" if shap_val > 0 else "Reduced risk"
    
    # Resolve human name
    # 1. Try Metadata Origin
    raw_feat = feature_name
    meta = metadata.get(feature_name) if metadata else None
    
    if meta:
        raw_feat = meta.get("raw_feature", feature_name)
    
    # 2. Map raw feature to user label
    user_label = FEATURE_MAP.get(raw_feat, raw_feat.replace("_", " ").title())
    
    # 3. Generate Reason
    reason = f"{user_label} factor"
    
    # categorical logic (if it looks like OneHot)
    # If feature is "collision_type_Rear Collision", logic:
    # "Rear Collision Type detected" (if positive shap)
    # We rely on direction for now.
    
    if shap_val > 0:
        reason = f"{user_label} contributes to higher risk"
    else:
        reason = f"{user_label} reduces risk estimate"
        
    return direction, reason

@app.get("/")
async def root():
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": len(MODELS)}

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    claim_data: ClaimInput,
    model: Literal["rf", "et", "xgb", "voting"] = Query("rf"),
    calibrated: bool = Query(True),
    scenario: Literal["auto_flagger", "dashboard"] = Query("dashboard"),
    explain: bool = Query(True),
    llm_explain: bool = Query(False)
):
    model_map = {"rf": "RandomForest", "et": "ExtraTrees", "xgb": "XGBoost", "voting": "VotingEnsemble"}
    model_name = model_map[model]
    
    cal_type = "calibrated" if calibrated else "uncalibrated"
    if scenario == "auto_flagger": cal_type = "uncalibrated"
    elif scenario == "dashboard": cal_type = "calibrated"
    
    model_key = f"{model_name}_{cal_type}"
    
    if model_key not in MODELS:
        if cal_type == 'calibrated': model_key = f"{model_name}_uncalibrated"
        if model_key not in MODELS:
             raise HTTPException(status_code=404, detail=f"Model {model_key} not found")
    
    loaded_model = MODELS[model_key]
    
    try:
        input_dict = claim_data.dict(by_alias=True)
        final_df = preprocess_input(input_dict)
        
        # Predict
        if hasattr(loaded_model, "predict_proba"):
             proba = loaded_model.predict_proba(final_df)[0, 1]
        else:
             start_pred = loaded_model.predict(final_df)
             proba = float(start_pred[0])
        
        # SHAP EXPLANATION (Canonical Source)
        explanation_items = []
        if explain:
             # ALWAYS use the canonical source for explanations
             source_key = EXPLANATION_SOURCE_MODEL
             explainer = SHAP_EXPLAINERS.get(source_key)
             
             if explainer and source_key in MODELS:
                 try:
                     # Use the PREPROCESSOR form the SOURCE model to ensure alignment
                     source_model = MODELS[source_key]
                     prep, _ = get_pipeline_components(source_model)
                     
                     if prep:
                         # Transform Query to match Explanation Space
                         X_query = prep.transform(final_df)
                         if hasattr(X_query, 'toarray'): X_query = X_query.toarray()
                         
                         # Safety Check: Contract Alignment
                         if X_query.shape[1] != BACKGROUND_DATA.shape[1]:
                             raise ValueError(f"Shape Mismatch: Query {X_query.shape[1]} != BG {BACKGROUND_DATA.shape[1]}")
                         
                         # Calculate SHAP
                         shap_values = explainer.shap_values(X_query)
                         
                         # Handle output shape
                         if isinstance(shap_values, list):
                             vals = shap_values[1][0]
                         elif len(shap_values.shape) == 3:
                             # Shape (1, features, 2) -> We want Sample 0, All features, Class 1
                             vals = shap_values[0, :, 1]
                         else:
                             # Shape (1, features) -> regression or binary XGBoost
                             vals = shap_values[0]
                         
                         # Map to Names
                         feature_names = SHAP_FEATURE_NAMES if SHAP_FEATURE_NAMES is not None else []
                         
                         items_temp = []
                         if isinstance(vals, (float, int)): vals = [vals]
                         
                         for i, sh_val in enumerate(vals):
                             if abs(sh_val) < 1e-4: continue
                             fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                             
                             items_temp.append({
                                 'feature': fname,
                                 'shap': sh_val
                             })
                         
                         # 5. Filter and Sort
                         items_temp.sort(key=lambda x: abs(x['shap']), reverse=True)
                         
                         # Whitelist of features the user can actually control/see
                         VISIBLE_ROOTS = {
                             "total_claim_amount", "injury_share", "property_share", "incident_hour_of_the_day",
                             "months_as_customer", "policy_annual_premium", "vehicle_age", "age",
                             "capital-gains", "capital-loss", "umbrella_limit", "bodily_injuries",
                             "number_of_vehicles_involved", "incident_severity", "collision_type",
                             "authorities_contacted", "police_report_available"
                         }
                         
                         count = 0
                         
                         # Fix: Get actual input vector to check feature presence
                         input_vector = X_query[0] if len(X_query.shape) == 2 else X_query

                         for idx_item, item in enumerate(items_temp):
                             if count >= 5: break
                             
                             feat_name = item['feature']
                             
                             # Filter: If one-hot feature (has underscore) and value is 0, SKIP it
                             # This prevents "Minor Damage" showing up when "Major Damage" is active.
                             # We find the index of this feature in feature_names to get its value
                             try:
                                 f_index = list(feature_names).index(feat_name)
                                 f_val = input_vector[f_index]
                                 # If it looks like a one-hot category and is NOT present (0), skip
                                 if "_" in feat_name and abs(f_val) < 1e-4:
                                     continue
                             except ValueError:
                                 pass # Feature not found in columns, safe to proceed or skip? Proceed.

                             # Resolve Root Feature Name
                             meta = FEATURE_METADATA.get(feat_name) if FEATURE_METADATA else None
                             root_feat = meta.get("raw_feature", feat_name) if meta else feat_name
                             
                             # Clean up potential "onehot__" prefix if metadata missing
                             if "onehot__" in root_feat: root_feat = root_feat.split("__")[1].split("_")[0] 
                             
                             # Heuristic for roots
                             for v in VISIBLE_ROOTS:
                                 if feat_name.startswith(v):
                                     root_feat = v
                                     break

                             if root_feat not in VISIBLE_ROOTS and root_feat not in FEATURE_MAP: 
                                 continue
                                 
                             direction, text = get_readable_explanation(feat_name, item['shap'], FEATURE_METADATA)
                             
                             # Clean display name: Remove underscores, Title Case
                             # e.g. "incident_severity_Major Damage" -> "Major Damage Severity"
                             clean_name = feat_name
                             if "_" in clean_name:
                                 parts = clean_name.split('_')
                                 # Heuristic: if last part is the category ("Major Damage"), put it first?
                                 # Or just replace underscores.
                                 # User wanted "Major Damage Severity"
                                 if len(parts) >= 2:
                                      # Generic clean: "Incident Severity Minor Damage"
                                      clean_name = clean_name.replace("_", " ").title()
                             
                             explanation_items.append(ExplanationItem(
                                 feature=FEATURE_MAP.get(feat_name, clean_name), # Use cleaned name as fallback
                                 direction="UP" if item['shap'] > 0 else "DOWN",
                                 text=text,
                                 importance=float(abs(item['shap']))
                             ))
                             count += 1
                     else:
                          explanation_items.append(ExplanationItem(feature="System Error", direction="DOWN", text="Explanation preprocessor missing", importance=0))
                 except Exception as e:
                     logger.error(f"Explanation generation failed: {e}")
                     explanation_items.append(ExplanationItem(feature="System Error", direction="DOWN", text="Explanation error", importance=0))
             else:
                 # Check for init error on the source
                 err = SHAP_INIT_ERRORS.get(source_key, "Use of unsupported model for explanation")
                 explanation_items.append(ExplanationItem(
                     feature="Init Failed", direction="DOWN", text=f"Explainer Error: {err}", importance=0
                 ))
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}"
        logger.error(f"Prediction error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
    
    threshold_flag = None
    if scenario == "auto_flagger":
        threshold_flag = "AUTO_FLAG" if proba >= THRESHOLD_AUTO_FLAG else "AUTO_APPROVE"
    
    # LLM Explanation Logic
    llm_result = None
    if llm_explain:
        if not explain:
             llm_result = {"error": "Explanation (SHAP) processing was disabled."}
        elif not explanation_items:
             llm_result = {"error": "No risk drivers found to explain. (SHAP returned empty)"}
        elif not generate_llm_explanation:
             llm_result = {"error": "LLM Module failed to load on server startup."}
        else:
            try:
                # Determine readable model name for prompt
                model_nice_name = model
                if model == "xgb": model_nice_name = "XGBoost"
                elif model == "voting": model_nice_name = "Voting Ensemble"
                elif model == "rf": model_nice_name = "Random Forest"
                elif model == "et": model_nice_name = "Extra Trees"

                llm_result = generate_llm_explanation(
                    selected_model_name=model_nice_name,
                    reference_model_name="ExtraTrees (Reference)", # From EXPLANATION_SOURCE_MODEL
                    risk_score=float(proba),
                    explanation_items=explanation_items
                )
            except Exception as e:
                logger.error(f"LLM generation failed in endpoint: {e}")
                llm_result = {"error": f"Endpoint Error: {str(e)}"}

    return PredictionResponse(
        model=model_name,
        calibrated=("calibrated" in model_key),
        probability=float(proba),
        threshold_flag=threshold_flag,
        scenario=scenario,
        explanation=explanation_items,
        llm_explanation=llm_result,
        app_version=APP_VERSION
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
