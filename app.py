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
BACKGROUND_DATA_PATH = MODELS_DIR / "shap_background.csv"
SHAP_EXPLAINERS = {}
BACKGROUND_DATA = None

# Feature Name Mapping (Technical -> User)
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
    """Load background data and initialize explainers"""
    global BACKGROUND_DATA
    if BACKGROUND_DATA_PATH.exists():
        # Load and ensure it matches input schema (drop target if present)
        bg = pd.read_csv(BACKGROUND_DATA_PATH)
        if 'target' in bg.columns:
            bg = bg.drop(columns=['target'])
        BACKGROUND_DATA = bg
        logger.info(f"Loaded SHAP background data: {len(BACKGROUND_DATA)} rows")
    else:
        logger.warning(f"SHAP background data not found at {BACKGROUND_DATA_PATH}")

    # Pre-compute explainers
    for key, model in MODELS.items():
        if "Voting" in key: continue
        
        if "uncalibrated" in key:
            try:
                 prep, estimator = get_pipeline_components(model)
                 if BACKGROUND_DATA is not None:
                     # TRANSFORM background data
                     if prep:
                         # sklearn pipeline preprocessor needs dataframe with correct columns
                         # We assume BACKGROUND_DATA has superset of columns needed.
                         X_bg = prep.transform(BACKGROUND_DATA)
                     else:
                         X_bg = BACKGROUND_DATA
                     
                     explainer = shap.TreeExplainer(estimator, X_bg)
                     SHAP_EXPLAINERS[key] = explainer
                     
                     # Map calibrated
                     cal_key = key.replace("uncalibrated", "calibrated")
                     SHAP_EXPLAINERS[cal_key] = explainer
                     # We also need to cache the preprocessor for the calibrated key lookup later?
                     # No, we can get it from the model at predict time.
            except Exception as e:
                logger.warning(f"Could not init SHAP for {key}: {e}")

# ... (startup_event default)

# ... (get_readable_explanation default)

# ...

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    claim_data: ClaimInput,
    model: Literal["rf", "et", "xgb", "voting"] = Query("rf"),
    calibrated: bool = Query(True),
    scenario: Literal["auto_flagger", "dashboard"] = Query("dashboard"),
    explain: bool = Query(True)
):
    # ... (model selection logic same)
    model_map = {"rf": "RandomForest", "et": "ExtraTrees", "xgb": "XGBoost", "voting": "VotingEnsemble"}
    model_name = model_map[model]
    
    cal_type = "calibrated" if calibrated else "uncalibrated"
    if scenario == "auto_flagger": cal_type = "uncalibrated"
    elif scenario == "dashboard": cal_type = "calibrated"
    
    model_key = f"{model_name}_{cal_type}"
    
    # ... (lookup fallback logic same)
    if model_key not in MODELS:
        # Fallback to uncalibrated if calibrated not found (common dev issue)
        if cal_type == 'calibrated':
             model_key = f"{model_name}_uncalibrated"
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
        
        # SHAP
        explanation_items = []
        if explain and "Voting" not in model_name and BACKGROUND_DATA is not None:
             try:
                 explainer = SHAP_EXPLAINERS.get(model_key)
                 if explainer:
                     # 1. Transform Input
                     prep, _ = get_pipeline_components(loaded_model)
                     # If loaded_model is CalibratedClassifierCV, it doesn't have steps directly usually.
                     # But we mapped SHAP info based on the *Uncalibrated* key structure.
                     # However, 'loaded_model' here might be the Calibrated one.
                     # We need the preprocessor.
                     # If Calibrated, it wraps the pipeline? No, usually CalibratedClassifierCV(base_estimator=Pipeline).
                     # So loaded_model.estimator would be the pipeline?
                     # Let's rely on the fact that if we use the uncalibrated key mapping, we should use the uncalibrated model's preprocessor too.
                     # Simpler: Load the uncalibrated model to get the preprocessor if needed.
                     
                     if "calibrated" in model_key:
                         # Get the uncalibrated sibling for structural access
                         uncal_key = model_key.replace("calibrated", "uncalibrated")
                         sibling = MODELS.get(uncal_key)
                         if sibling: prep, _ = get_pipeline_components(sibling)
                     
                     if prep:
                         X_query = prep.transform(final_df)
                         # Get Feature Names
                         # If prep is ColumnTransformer
                         if hasattr(prep, 'get_feature_names_out'):
                             feature_names = prep.get_feature_names_out()
                         else:
                             # Fallback, maybe X_query has columns if df? no, transform returns array usually
                             feature_names = [f"feature_{i}" for i in range(X_query.shape[1])]
                     else:
                         X_query = final_df
                         feature_names = final_df.columns
                     
                     shap_values = explainer.shap_values(X_query)
                     
                     if isinstance(shap_values, list):
                         vals = shap_values[1][0]
                     else:
                         vals = shap_values[0]
                     
                     # Create DF
                     # We need to map feature_names to vals
                     # Remove "onehot__" or "remainder__" prefix from get_feature_names_out results typically
                     clean_feats = [f.split('__')[-1] if '__' in f else f for f in feature_names]
                     
                     # We need the original values for text generation?
                     # The humanizer logic uses `val_raw`. We can look up in final_df?
                     # But final_df has raw columns. 
                     # What if OneHot? "collision_type_Rear Collision".
                     # We can try to match.
                     
                     # Let's store importance and raw feature name
                     # X_query contains transformed values.
                     
                     # Construct items
                     # We need to iterate top features.
                     # We have `clean_feats` (e.g. "collision_type_Rear Collision") and `vals` (SHAP).
                     # We need `val` (User input).
                     # For "collision_type_Rear Collision", the user input is in `final_df['collision_type']`.
                     
                     # Simplify: Pass 0 as value to get_readable_explanation and rely on SHAP sign?
                     # get_readable_explanation uses `val_raw` for nuances ("Long tenure" vs "Low tenure").
                     # Use final_df for value lookup.
                     
                     items_temp = []
                     for i, feat_name in enumerate(clean_feats):
                         sh_val = vals[i]
                         if abs(sh_val) < 1e-4: continue
                         
                         # Find raw value
                         # If feat_name is like "collision_type_Rear Collision", raw col is "collision_type"
                         # Heuristic matching
                         raw_val = 0
                         # Try exact match in final_df
                         if feat_name in final_df.columns:
                             raw_val = final_df.iloc[0][feat_name]
                         else:
                             # Try prefix match logic or just pass dummy
                             # For OneHot: "Rear Collision" implies value is 1 (if feature is active).
                             # If SHAP is high, it likely is active.
                             pass 
                             
                         items_temp.append({
                             'feature': feat_name,
                             'shap': sh_val,
                             'val': raw_val
                         })
                     
                     # Sort
                     items_temp.sort(key=lambda x: abs(x['shap']), reverse=True)
                     top_5 = items_temp[:5]
                     
                     for item in top_5:
                         direction, text = get_readable_explanation(item['feature'], item['val'], item['shap'], 0)
                         explanation_items.append(ExplanationItem(
                             feature=FEATURE_MAP.get(item['feature'], item['feature']),
                             direction="UP" if item['shap'] > 0 else "DOWN",
                             text=text,
                             importance=float(abs(item['shap']))
                         ))
                         
             except Exception as e:
                 logger.warning(f"SHAP gen failed: {e}")

@app.on_event("startup")
async def startup_event():
    load_models()
    load_shap_resources()

def get_readable_explanation(feature, val_raw, shap_val, mean_val):
    """Generate human-friendly text based on feature value and SHAP direction"""
    direction = "Increased risk" if shap_val > 0 else "Reduced risk"
    fname = FEATURE_MAP.get(feature, feature.replace("_", " ").title())
    
    # Generic logic
    reason = "factor"
    if shap_val > 0:
        if val_raw > mean_val: reason = f"Higher {fname} than typical"
        else: reason = f"Specific {fname} configuration"
    else:
        if val_raw > mean_val and "tenure" in feature: reason = "Long-standing customer history"
        elif val_raw < mean_val: reason = f"Lower {fname} than typical"
        else: reason = f"Favorable {fname} profile"

    # Specific Overrides
    if feature == "total_claim_amount":
        if shap_val > 0: reason = "Larger-than-usual claim size"
        else: reason = "Smaller-than-usual claim size"
    elif feature == "incident_hour_of_the_day":
        if shap_val > 0: reason = "Off-hours incident timing"
        else: reason = "Daytime incident timing"
    elif feature == "injury_share":
        if shap_val > 0: reason = "High proportion of injury costs"
        else: reason = "Low proportion of injury costs"
    elif feature == "age":
        if shap_val > 0: reason = "Insured age group associated with higher risk"
        else: reason = "Insured age group associated with lower risk"
    
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
        # Fallback to uncalibrated if calibrated not found (common dev issue)
        if cal_type == 'calibrated':
             model_key = f"{model_name}_uncalibrated"
        if model_key not in MODELS:
             raise HTTPException(status_code=404, detail=f"Model {model_key} not found")
    
    loaded_model = MODELS[model_key]
    
    try:
        # Convert Pydantic to Dict
        input_dict = claim_data.dict(by_alias=True)
        
        # FULL PREPROCESSING
        final_df = preprocess_input(input_dict)
        
        # Predict
        # Check if pipeline or raw model
        if hasattr(loaded_model, "predict_proba"):
             proba = loaded_model.predict_proba(final_df)[0, 1]
        else:
             # Basic fallback
             start_pred = loaded_model.predict(final_df)
             proba = float(start_pred[0])
        
        # SHAP EXPLANATION
        explanation_items = []
        if explain and "Voting" not in model_name and BACKGROUND_DATA is not None:
             # Only simple tree models for now
             try:
                 explainer = SHAP_EXPLAINERS.get(model_key)
                 if not explainer:
                     # Lazy init
                     explainer = shap.TreeExplainer(loaded_model, BACKGROUND_DATA)
                     SHAP_EXPLAINERS[model_key] = explainer
                 
                 shap_values = explainer.shap_values(final_df)
                 # Handle list output (for classification) -> take index 1 (positive class) or 0 if regression
                 if isinstance(shap_values, list):
                     vals = shap_values[1][0]
                 else:
                     vals = shap_values[0] # assuming single row, SHAP returns array
                 
                 # Create DF
                 shap_df = pd.DataFrame(list(zip(final_df.columns, vals, final_df.iloc[0])), columns=['feature', 'shap', 'val'])
                 shap_df['abs_shap'] = shap_df['shap'].abs()
                 top_5 = shap_df.sort_values('abs_shap', ascending=False).head(5)
                 
                 for _, row in top_5.iterrows():
                     direction, text = get_readable_explanation(row['feature'], row['val'], row['shap'], 0) # 0 is dummy mean for now
                     explanation_items.append(ExplanationItem(
                         feature=FEATURE_MAP.get(row['feature'], row['feature']),
                         direction="UP" if row['shap'] > 0 else "DOWN",
                         text=text,
                         importance=float(row['abs_shap'])
                     ))
                     
             except Exception as e:
                 logger.warning(f"SHAP gen failed: {e}")
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
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
