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
import json
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
APP_VERSION = "1.6.1"
THRESHOLD_AUTO_FLAG = 0.53

# Model registry
MODELS = {}
LOADING_ERRORS = {}

# SHAP Configuration
BACKGROUND_DATA_PATH = MODELS_DIR / "shap_background.npy"
FEATURE_NAMES_PATH = MODELS_DIR / "shap_feature_names.joblib"
METADATA_PATH = MODELS_DIR / "feature_metadata.joblib"
TREND_REGISTRY_PATH = MODELS_DIR / "trend_registry.json"

SHAP_EXPLAINERS = {}
SHAP_INIT_ERRORS = {}
BACKGROUND_DATA = None
SHAP_FEATURE_NAMES = None
FEATURE_METADATA = None
TREND_REGISTRY = None
EXPLANATION_SOURCE_MODEL = "ExtraTrees_uncalibrated"

# Feature Name Mapping (Technical -> User)
# We keep this for the final display Mapping
# Feature Name Mapping (Technical -> User)
from mappings import DISPLAY_LABEL_MAP as FEATURE_MAP

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
    injury_share: Optional[float] = Field(None, description="Share of injury damage")
    property_share: Optional[float] = Field(None, description="Share of property damage")
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
    explanation_source: Optional[str] = None
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
                    LOADING_ERRORS[key] = str(e)
    
    logger.info(f"Total models loaded: {len(MODELS)}")
    if LOADING_ERRORS:
        logger.warning(f"Models failed to load: {list(LOADING_ERRORS.keys())}")

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
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")

    # 4. Load Trend Registry
    global TREND_REGISTRY
    if TREND_REGISTRY_PATH.exists():
        try:
            with open(TREND_REGISTRY_PATH, "r") as f:
                TREND_REGISTRY = json.load(f)
            logger.info("Loaded Trend Registry.")
        except Exception as e:
            logger.warning(f"Failed to load Trend Registry: {e}")

    # 5. Initialize TreeExplainers (ONLY for the Source Model)
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
    
    # categorical logic
    # If feature looks like "authorities_contacted_Police", we want "Police Contacted" or "Contacting Police"
    
    if "_" in feature_name:
        parts = feature_name.split("_")
        # Heuristic: The last part is likely the category if it's capitalized or distinct
        # E.g. incident_severity_Major Damage -> Major Damage
        # E.g. authorities_contacted_Police -> Police
        
        category = parts[-1]
        root = " ".join(parts[:-1]).title()
        
        # Override user label for categories to be more specific
        if "Authorities" in root:
             if category == "None": user_label = "No Authorities Contacted"
             else: user_label = f"Contacting {category}"
        elif "Severity" in root:
             user_label = f"{category} Severity"
        elif "Collision" in root:
             user_label = f"{category} Type"
        else:
             # Fallback: "Major Damage (incident_severity)"
             # Actually, just appending category is usually good
             user_label = f"{category} ({FEATURE_MAP.get(raw_feat, root)})"

    if shap_val > 0:
        reason = f"{user_label} contributes to higher risk"
    else:
        reason = f"{user_label} reduces risk estimate"
        
    return direction, reason

def get_nuanced_explanation(feature_name, shap_val, feature_val, metadata=None, original_name=None):
    """
    Generate explanation with relative-to-typical context and value descriptors.
    """
    baseline_direction = "UP" if shap_val > 0 else "DOWN"
    
    # 1. Resolve Name
    # Use original_name (specific) if provided, else feature_name (root)
    name_to_resolve = original_name if original_name else feature_name
    
    raw_feat = name_to_resolve
    if metadata and name_to_resolve in metadata:
        raw_feat = metadata[name_to_resolve].get("raw_feature", name_to_resolve)
    user_label = FEATURE_MAP.get(raw_feat, raw_feat.replace("_", " ").title())
    
    # Precise Categorical Handling for Nuanced Text
    # We prefer the original name (e.g. authorities_contacted_Police) for category extraction
    # The feature_name passed is usually length 1 (root), original is specific.
    
    target_for_parsing = original_name if original_name else feature_name
    
    if "_" in target_for_parsing and target_for_parsing not in FEATURE_MAP:
         parts = target_for_parsing.split("_")
         category = parts[-1]
         root = " ".join(parts[:-1]).title() 
         
         # Sanity check: if root became empty or weird, ignore
         if root:
             # Normalize category for checks
             cat_lower = category.lower()
             
             if "Authorities" in root:
                  if cat_lower == "none": user_label = "No Authorities Contacted"
                  elif cat_lower in ["missing", "unknown", "?", "nan"]: user_label = "Authority Contact Info Missing"
                  else: user_label = f"Contacting {category}"
             elif "Severity" in root: 
                  if cat_lower in ["missing", "unknown", "?", "nan"]: user_label = "Unknown Severity"
                  else: user_label = f"{category} Severity"
             elif "Collision" in root: 
                  if cat_lower in ["missing", "unknown", "?", "nan"]: user_label = "Unknown Collision Type"
                  else: user_label = f"{category} Type"
             elif "Report" in root:
                  # Police Report Available
                  if cat_lower == "yes": user_label = "Police Report Available"
                  elif cat_lower == "no": user_label = "No Police Report"
                  elif cat_lower in ["missing", "unknown", "?", "nan"]: user_label = "Police Report Status Unknown"
                  else: user_label = f"Police Report: {category}"
             else: 
                 # Generic Fallback
                 if cat_lower in ["missing", "unknown", "?", "nan"]: user_label = f"Unknown {FEATURE_MAP.get(raw_feat, root)}"
                 else: user_label = f"{category} ({FEATURE_MAP.get(raw_feat, root)})"

    # 2. Trend Analysis
    trend_text = ""
    value_desc = ""
    
    if TREND_REGISTRY and feature_name in TREND_REGISTRY:
        try:
            entry = TREND_REGISTRY[feature_name]
            bins = entry["bins"]
            shaps = entry["shap_values"]
            ref_idx = entry.get("ref_idx", len(bins)//2)
            min_val, max_val = entry.get("min_val", bins[0]), entry.get("max_val", bins[-1])
            
            # Find current bin
            curr_idx = (np.abs(np.array(bins) - feature_val)).argmin()
            
            curr_shap_med = shaps[curr_idx]
            ref_shap_med = shaps[ref_idx]
            typical_delta = curr_shap_med - ref_shap_med
            
            # Value Descriptor (Low/High/Typical)
            # Simple percentile check
            rng = max_val - min_val
            if rng > 0:
                rel_pos = (feature_val - min_val) / rng
                if rel_pos < 0.33: value_desc = "Low "
                elif rel_pos > 0.66: value_desc = "High "
                else: value_desc = "Typical "
            
            # Threshold for "significant" relative difference
            if typical_delta > 0.0005:
                # Riskier than typical
                trend_text = "associated with higher risk than average"
            elif typical_delta < -0.0005:
                # Safer than typical
                trend_text = "associated with lower risk than average"
                    
        except Exception:
            pass

    # 3. Construct Final Sentence
    # Case A: Trend info available and significant
    if trend_text:
        # "Low Injury Cost Portion is associated with higher risk than average."
        if value_desc == "Typical ": value_desc = "" # Omit "Typical" prefix usually
        full_text = f"{value_desc}{user_label} is {trend_text}"
        
        # Add baseline context if it contradicts?
        # If baseline is DOWN but trend is RISKIER -> "Reduces risk overall, but Low X is associated with higher risk than average"
        # User implies they just want the "causes it higher" part.
        # "Low Injury Cost Portion is associated with higher risk than average" is very clear.
        pass
    else:
        # Case B: Standard Baseline Fallback
        if shap_val > 0:
            full_text = f"{user_label} contributes to risk"
        else:
            full_text = f"{user_label} reduces risk estimate"

    return baseline_direction, full_text

@app.get("/")
async def root():
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(MODELS),
        "registry_loaded": TREND_REGISTRY is not None,
        "loading_errors": LOADING_ERRORS
    }

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
                                 
                             # Retrieve value for nuanced explanation
                             # We already extracted f_val, but need to ensure it corresponds to root_feat name if we check registry
                             # The registry keys match raw continuous features (e.g. "injury_share")
                             # feat_name might be "injury_share" or "numerical__injury_share..."
                             # Our registry builder used "injury_share".
                             
                             # Clean Name for display
                             clean_name = feat_name
                             if "_" in clean_name:
                                 parts = clean_name.split('_')
                                 if len(parts) >= 2:
                                      clean_name = clean_name.replace("_", " ").title()
                             
                             # Determine Value to pass
                             # If categorical (one-hot), value is 1/0. 
                             # If continuous, value is f_val.
                             # Registry lookup key is root_feat (e.g. injury_share)
                             
                             # We use root_feat for registry lookup
                             try:
                                f_index_reg = list(feature_names).index(feat_name)
                                val_for_trend = input_vector[f_index_reg]
                                direction, text = get_nuanced_explanation(root_feat, item['shap'], val_for_trend, FEATURE_METADATA, original_name=feat_name)
                             except:
                                direction, text = get_readable_explanation(feat_name, item['shap'], FEATURE_METADATA)
                             
                             
                             explanation_items.append(ExplanationItem(
                                 feature=FEATURE_MAP.get(root_feat, clean_name), # Map root feat to Label
                                 direction=direction,
                                 text=text,
                                 importance=float(abs(item['shap']))
                             ))
                             count += 1
                             if count >= 5: break
                             
                     # Safety Net: If filtering removed all features (rare), add generic backup
                     if not explanation_items:
                         explanation_items.append(ExplanationItem(
                             feature="No Key Drivers", 
                             direction="DOWN", 
                             text="No single feature exceeded importance threshold.", 
                             importance=0.0
                         ))
                         
                     # Fallback for empty list (should be covered above, but just in case)
                     # This 'pass' was part of an extraneous 'else' block.
                     pass
                          
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
        explanation_source=EXPLANATION_SOURCE_MODEL if explain else None,
        llm_explanation=llm_result,
        app_version=APP_VERSION
    )

# --- ASYNC BATCH JOB MANAGEMENT ---
from fastapi import BackgroundTasks
import uuid
import datetime

# In-memory Job Store
# Structure: { job_id: { "status": "pending"|"running"|"completed"|"failed", "progress": int, "result": dict, "error": str, "created_at": str } }
JOBS = {}

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: str
    api_version: str = "v1"

async def run_batch_job(job_id: str, file_content: bytes, model_name: str, scenario: str, explain: bool, team_size: int, review_time: int, operating_mode: str, review_window_days: int, current_backlog_cases: int):
    """
    Background Task Wrapper for Batch Processing
    """
    try:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["progress"] = 0
        
        # Prepare Context (moved from submit_batch)
        model_context = MODELS.copy()
        model_context["shap_explainer"] = SHAP_EXPLAINERS.get(EXPLANATION_SOURCE_MODEL)
        model_context["feature_names"] = SHAP_FEATURE_NAMES
        source_model = MODELS.get(EXPLANATION_SOURCE_MODEL)
        prep_step, _ = get_pipeline_components(source_model)
        model_context["preprocessor"] = prep_step
        
        async def update_progress(p):
            JOBS[job_id]["progress"] = p
            
        # Run the heavy lifting
        result = await batch_ingest.process_batch_file(
            file_content=file_content,
            model_context=model_context,
            model_key=model_name, # Use model_name from signature
            scenario=scenario,
            explain=explain,
            progress_callback=update_progress,
            team_size=team_size,
            review_time=review_time,
            operating_mode=operating_mode,
            review_window_days=review_window_days,
            current_backlog_cases=current_backlog_cases
        )
        
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["result"] = result
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)

# --- BATCH PREDICTION ENDPOINT (V1) ---
from fastapi import UploadFile, File
import batch_ingest

@app.post("/batch_predict")
async def batch_predict(
    file: UploadFile = File(...),
    model: str = "xgb",
    scenario: str = "auto_flagger",
    explain: bool = False
):
    """
    Legacy Synchronous Endpoint (Keep for backward compatibility)
    """
    if not file.filename.endswith('.csv'):
        return {"summary": {"status": "error", "message": "File must be CSV"}, "rows": []}
    
    contents = await file.read()
    
    # Context Setup (Same as before)
    model_context = MODELS.copy()
    model_context["shap_explainer"] = SHAP_EXPLAINERS.get(EXPLANATION_SOURCE_MODEL)
    model_context["feature_names"] = SHAP_FEATURE_NAMES
    source_model = MODELS.get(EXPLANATION_SOURCE_MODEL)
    prep_step, _ = get_pipeline_components(source_model)
    model_context["preprocessor"] = prep_step
    
    result = await batch_ingest.process_batch_file(
        file_content=contents,
        model_context=model_context,
        model_key=model,
        scenario=scenario,
        explain=explain
    )
    return result

# TODO: Refactor Risk Context to POST body (JSON) instead of query params (Issue 5)
# Example: { "risk_context": { "team_size": 5, ... } }
@app.post("/submit_batch")
async def submit_batch(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Query("xgb", pattern="^(xgb|rf|et|voting)$"),
    scenario: str = Query("dashboard", pattern="^(dashboard|auto_flagger|underwriter)$"),
    explain: bool = True,
    team_size: int = Query(5, ge=1, description="Number of investigators"),
    review_time: int = Query(20, ge=1, description="Minutes per review"),
    operating_mode: str = Query("daily_ops", description="Operational Context"),
    review_window_days: int = Query(1, ge=1, description="Days to review"),
    current_backlog_cases: int = Query(0, ge=0, description="Existing Backlog")
):
    """
    Async Batch Submission. Returns Job ID.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be CSV")

    contents = await file.read()
    job_id = str(uuid.uuid4())

    # Initialize Job
    JOBS[job_id] = {
        "status": "pending",
        "progress": 0,
        "created_at": datetime.datetime.now().isoformat()
    }

    # Prepare Context
    model_context = MODELS.copy()
    model_context["shap_explainer"] = SHAP_EXPLAINERS.get(EXPLANATION_SOURCE_MODEL)
    model_context["feature_names"] = SHAP_FEATURE_NAMES
    source_model = MODELS.get(EXPLANATION_SOURCE_MODEL)
    prep_step, _ = get_pipeline_components(source_model)
    model_context["preprocessor"] = prep_step
    
    # Launch Background Task
    # Launch Background Task
    background_tasks.add_task(
        run_batch_job, 
        job_id, 
        contents, 
        model, 
        scenario, 
        explain,
        team_size,
        review_time,
        operating_mode,
        review_window_days,
        current_backlog_cases
    )
    
    return {"job_id": job_id, "status": "pending"}

@app.get("/job_status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = JOBS[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0),
        result=job.get("result"),
        error=job.get("error"),
        created_at=job.get("created_at"),
        api_version="v1"
    )

@app.get("/outputs/{filename}")
async def download_output(filename: str):
    file_path = Path("outputs") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type='text/csv', filename=filename)

@app.get("/download_template")
async def download_template():
    file_path = Path("static") / "claim_template.csv"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Template not found")
    return FileResponse(file_path, media_type='text/csv', filename="claim_template.csv")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
