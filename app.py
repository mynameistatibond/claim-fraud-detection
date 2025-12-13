"""
Fraud Detection API - FastAPI Backend
Serves predictions from pre-trained ML models using full preprocessing pipeline.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, Literal
import shap
import pandas as pd

# ... (Previous imports kept implicitly by replace_tool context if not ensuring full file view. 
# Waiting, I should replace blocks. I will do a big replace to ensure imports are there.)

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

def load_shap_resources():
    """Load background data and initialize explainers"""
    global BACKGROUND_DATA
    if BACKGROUND_DATA_PATH.exists():
        BACKGROUND_DATA = pd.read_csv(BACKGROUND_DATA_PATH)
        # Ensure dummy columns match model expectation if needed, but preprocessed_for_trees should be good.
        logger.info(f"Loaded SHAP background data: {len(BACKGROUND_DATA)} rows")
    else:
        logger.warning(f"SHAP background data not found at {BACKGROUND_DATA_PATH}")

    # Pre-compute explainers for loaded models where possible
    # Note: TreeExplainer is fast, but better to cache.
    for key, model in MODELS.items():
        if "Voting" in key: continue # SHAP for voting is complex, we might skip or approx
        try:
             # Just cache the explainer if we have data
             if BACKGROUND_DATA is not None:
                 # Check if model has direct estimator or via pipeline steps? 
                 # Assuming loaded models are pipelines or naked estimators.
                 # The 'best_tree_models' are usually estimators.
                 SHAP_EXPLAINERS[key] = shap.TreeExplainer(model, BACKGROUND_DATA)
        except Exception as e:
            logger.warning(f"Could not init SHAP for {key}: {e}")

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
    uvicorn.run(app, host="0.0.0.0", port=7860)
