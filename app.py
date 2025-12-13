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
                         try:
                             X_bg = prep.transform(BACKGROUND_DATA)
                             if hasattr(X_bg, 'toarray'): X_bg = X_bg.toarray()
                         except Exception as e:
                             logger.warning(f"Background transform failed for {key}: {e}")
                             X_bg = BACKGROUND_DATA 
                     else:
                         X_bg = BACKGROUND_DATA
                     
                     # Attempt 1: TreeExplainer (Fast, Precise)
                     try:
                         explainer = shap.TreeExplainer(estimator, X_bg)
                         logger.info(f"Initialized TreeExplainer for {key}")
                     except Exception as e_tree:
                         logger.warning(f"TreeExplainer failed for {key} ({e_tree}). Falling back to KernelExplainer.")
                         # Attempt 2: KernelExplainer (Slow, Model-Agnostic)
                         # Use k-means summary to speed it up (10 centroids)
                         try:
                             if hasattr(estimator, 'predict_proba'):
                                 pred_fn = estimator.predict_proba
                             else:
                                 pred_fn = estimator.predict
                             
                             # Summary background for Speed
                             try:
                                 X_bg_summary = shap.kmeans(X_bg, 10)
                             except:
                                 X_bg_summary = X_bg[0:10] # Fallback subsample
                                 
                             explainer = shap.KernelExplainer(pred_fn, X_bg_summary)
                             logger.info(f"Initialized KernelExplainer for {key}")
                         except Exception as e_kernel:
                             logger.error(f"All SHAP inits failed for {key}: {e_kernel}")
                             explainer = None

                     if explainer:
                         SHAP_EXPLAINERS[key] = explainer
                         cal_key = key.replace("uncalibrated", "calibrated")
                         SHAP_EXPLAINERS[cal_key] = explainer

            except Exception as e:
                logger.warning(f"Could not init SHAP for {key}: {e}")

# ... (startup)

# ...

# ... inside predict ...
        # SHAP
        explanation_items = []
        if explain and "Voting" not in model_name and BACKGROUND_DATA is not None:
             try:
                 explainer = SHAP_EXPLAINERS.get(model_key)
                 if not explainer:
                     # Fallback check for sibling if calibrated
                     if "calibrated" in model_key:
                         uncal_key = model_key.replace("calibrated", "uncalibrated")
                         explainer = SHAP_EXPLAINERS.get(uncal_key)

                 if explainer:
                     # 1. Transform Input
                     prep, _ = get_pipeline_components(loaded_model)
                     
                     if not prep and "calibrated" in model_key:
                         uncal_key = model_key.replace("calibrated", "uncalibrated")
                         sibling = MODELS.get(uncal_key)
                         if sibling: prep, _ = get_pipeline_components(sibling)
                     
                     if prep:
                         X_query = prep.transform(final_df)
                         if hasattr(X_query, 'toarray'): X_query = X_query.toarray()
                         
                         if hasattr(prep, 'get_feature_names_out'):
                             feature_names = prep.get_feature_names_out()
                         else:
                             feature_names = [f"feature_{i}" for i in range(X_query.shape[1])]
                     else:
                         X_query = final_df
                         feature_names = final_df.columns
                     
                     shap_values = explainer.shap_values(X_query)
                     
                     if isinstance(shap_values, list):
                         vals = shap_values[1][0] # Positive class for Classifier
                     elif len(shap_values.shape) > 1 and shap_values.shape[1] > 1:
                         vals = shap_values[0][1] # KernelExplainer (nsamples, nclasses) -> (1, 2)
                     else:
                         vals = shap_values[0] # Regression or flat array
                         # Handle KernelExplainer single sample output shape quirks
                         if vals.shape == (2,): vals = vals[1]
                         elif len(vals.shape) == 0: vals = float(vals) # scalar
                     
                     # Map Items
                     clean_feats = [f.split('__')[-1] if '__' in f else f for f in feature_names]
                     
                     items_temp = []
                     # Handle if vals is not iterable
                     if isinstance(vals, (float, int)): vals = [vals]
                     
                     for i, feat_name in enumerate(clean_feats):
                         if i >= len(vals): break
                         sh_val = vals[i]
                         if abs(sh_val) < 1e-4: continue
                         
                         raw_val = 0
                         if feat_name in final_df.columns:
                             raw_val = final_df.iloc[0][feat_name]
                             
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
                 else:
                     # Explicitly report initialization failure via UI
                     explanation_items.append(ExplanationItem(
                         feature="Initialization Failed",
                         direction="DOWN", 
                         text="SHAP explainer could not be loaded on startup. Check logs.",
                         importance=0.0
                     ))
                         
             except Exception as e:
                 logger.warning(f"SHAP gen failed: {e}")
                 # Debugging: Return error as explanation
                 explanation_items.append(ExplanationItem(
                     feature="System Error",
                     direction="DOWN", 
                     text=f"SHAP Error: {str(e)}",
                     importance=0.0
                 ))
        
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
