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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API", version="2.0.0")

# Model configuration
MODELS_DIR = Path("models")
THRESHOLD_AUTO_FLAG = 0.53

# Model registry
MODELS = {}

# SHAP Configuration
BACKGROUND_DATA_PATH = MODELS_DIR / "shap_background.npy"
FEATURE_NAMES_PATH = MODELS_DIR / "shap_feature_names.joblib"
METADATA_PATH = MODELS_DIR / "feature_metadata.joblib"

SHAP_EXPLAINERS = {}
BACKGROUND_DATA = None
SHAP_FEATURE_NAMES = None
FEATURE_METADATA = None

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

    # 2. Load Feature Names corresponding to the matrix columns
    if FEATURE_NAMES_PATH.exists():
        SHAP_FEATURE_NAMES = joblib.load(FEATURE_NAMES_PATH)
        logger.info(f"Loaded {len(SHAP_FEATURE_NAMES)} feature names.")
    
    # 3. Load Feature Metadata (Origin)
    if METADATA_PATH.exists():
        try:
            FEATURE_METADATA = joblib.load(METADATA_PATH)
            logger.info("Loaded feature metadata.")
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")

    # 4. Initialize TreeExplainers only (Sane Architecture)
    for key, model in MODELS.items():
        if "Voting" in key: continue
        
        # Only init on uncalibrated models to get clean trees
        if "uncalibrated" in key:
            try:
                 if BACKGROUND_DATA is None:
                     raise ValueError("Background data not loaded")
                     
                 _, estimator = get_pipeline_components(model)
                 
                 # Direct initialization: Estimator + Pre-Processed Data
                 # No transforms here. The contract is: estimator takes what BACKGROUND_DATA is.
                 explainer = shap.TreeExplainer(estimator, BACKGROUND_DATA)
                 SHAP_EXPLAINERS[key] = explainer
                 
                 # Map calibrated version to this explainer
                 cal_key = key.replace("uncalibrated", "calibrated")
                 SHAP_EXPLAINERS[cal_key] = explainer
                 logger.info(f"Initialized SHAP for {key}")
                 
            except Exception as e:
                logger.warning(f"Failed to init SHAP for {key}: {e}")
                SHAP_INIT_ERRORS[key] = str(e)
                # Map error to calibrated too so we see it
                cal_key = key.replace("uncalibrated", "calibrated")
                SHAP_INIT_ERRORS[cal_key] = str(e)

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
    explain: bool = Query(True)
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
        
        # SHAP (Sane Architecture)
        explanation_items = []
        if explain and "Voting" not in model_name and BACKGROUND_DATA is not None:
             try:
                 # Check Explainer
                 explainer = SHAP_EXPLAINERS.get(model_key)
                 if not explainer and "calibrated" in model_key:
                     # Try fallback to uncalibrated key
                     explainer = SHAP_EXPLAINERS.get(model_key.replace("calibrated", "uncalibrated"))
                 
                 if explainer:
                     # 1. Transform Query
                     # We need the PREPROCESSOR specific to the model being explained.
                     # If we are explaining User's Model, we use User's Model Prep.
                     prep, _ = get_pipeline_components(loaded_model)
                     if not prep and "calibrated" in model_key:
                          uncal_key = model_key.replace("calibrated", "uncalibrated")
                          if uncal_key in MODELS:
                              prep, _ = get_pipeline_components(MODELS[uncal_key])
                     
                     if prep:
                         # Transform to match Training Space
                         X_query = prep.transform(final_df)
                         if hasattr(X_query, 'toarray'): X_query = X_query.toarray()
                         
                         # 2. Calculate SHAP
                         shap_values = explainer.shap_values(X_query)
                         
                         # Handle output shape
                         if isinstance(shap_values, list):
                             vals = shap_values[1][0]
                         elif len(shap_values.shape) > 1 and shap_values.shape[1] > 1:
                             vals = shap_values[0][1] # Should not happen for TreeExplainer on binary usually?
                         else:
                             vals = shap_values[0] # XGBoost output is raw margin or log-odds
                         
                         # 3. Map to Names
                         feature_names = SHAP_FEATURE_NAMES if SHAP_FEATURE_NAMES is not None else []
                         
                         items_temp = []
                         # Ensure vals is iterable
                         if isinstance(vals, (float, int)): vals = [vals]
                         
                         for i, sh_val in enumerate(vals):
                             if abs(sh_val) < 1e-4: continue
                             
                             fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                             
                             items_temp.append({
                                 'feature': fname,
                                 'shap': sh_val
                             })
                         
                         # 4. Sort and Extract Top
                         items_temp.sort(key=lambda x: abs(x['shap']), reverse=True)
                         top_5 = items_temp[:5]
                         
                         for item in top_5:
                             direction, text = get_readable_explanation(item['feature'], item['shap'], FEATURE_METADATA)
                             explanation_items.append(ExplanationItem(
                                 feature=FEATURE_MAP.get(item['feature'], item['feature']), # Fallback to tech name if no mapping
                                 direction="UP" if item['shap'] > 0 else "DOWN",
                                 text=text,
                                 importance=float(abs(item['shap']))
                             ))
                     else:
                         # No preprocessor found
                         pass
                 else:
                     # Check for specific init error
                     init_error = SHAP_INIT_ERRORS.get(model_key, "Unknown initialization failure")
                     
                     explanation_items.append(ExplanationItem(
                         feature="Init Failed",
                         direction="DOWN", 
                         text=f"Error: {init_error}",
                         importance=0.0
                     ))
             except Exception as e:
                 logger.warning(f"SHAP Error: {e}")
                 explanation_items.append(ExplanationItem(feature="Error", direction="DOWN", text=str(e), importance=0))
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}"
        logger.error(f"Prediction error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
    
    threshold_flag = None
    if scenario == "auto_flagger":
        threshold_flag = "AUTO_FLAG" if proba >= THRESHOLD_AUTO_FLAG else "AUTO_APPROVE"
    
    return PredictionResponse(
        model=model_name,
        calibrated=("calibrated" in model_key),
        probability=float(proba),
        threshold_flag=threshold_flag,
        scenario=scenario,
        explanation=explanation_items
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
