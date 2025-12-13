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

def get_tree_estimator(model):
    """Extract the actual tree estimator from Pipelines"""
    # Check for sklearn Pipeline (without importing sklearn if possible, or catch)
    if hasattr(model, 'steps'):
        # Assume the last step is the estimator
        return model.steps[-1][1]
    return model

def load_shap_resources():
    """Load background data and initialize explainers"""
    global BACKGROUND_DATA
    if BACKGROUND_DATA_PATH.exists():
        BACKGROUND_DATA = pd.read_csv(BACKGROUND_DATA_PATH)
        logger.info(f"Loaded SHAP background data: {len(BACKGROUND_DATA)} rows")
    else:
        logger.warning(f"SHAP background data not found at {BACKGROUND_DATA_PATH}")

    # Pre-compute explainers for loaded models where possible
    # STRATEGY: Initialize on _uncalibrated (raw trees) and map _calibrated to them.
    for key, model in MODELS.items():
        if "Voting" in key: continue # SHAP for voting is complex
        
        # We only init based on the Uncalibrated (Pipeline) version to get the raw tree
        if "uncalibrated" in key:
            try:
                 estimator = get_tree_estimator(model)
                 if BACKGROUND_DATA is not None:
                     explainer = shap.TreeExplainer(estimator, BACKGROUND_DATA)
                     SHAP_EXPLAINERS[key] = explainer
                     
                     # Also allow the calibrated version to use this explainer
                     # (Calibration is monotonic, so risk drivers are generally preserved)
                     cal_key = key.replace("uncalibrated", "calibrated")
                     SHAP_EXPLAINERS[cal_key] = explainer
                     logger.info(f"Initialized SHAP for {key} (and mapped {cal_key})")
            except Exception as e:
                logger.warning(f"Could not init SHAP for {key}: {e}")

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
