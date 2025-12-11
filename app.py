"""
Fraud Detection API - FastAPI Backend
Serves predictions from pre-trained ML models (RandomForest, ExtraTrees, XGBoost)
Supports both calibrated and uncalibrated versions with two deployment scenarios.
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API", version="1.0.0")

# Model configuration
MODELS_DIR = Path("models")
THRESHOLD_AUTO_FLAG = 0.53  # Placeholder - adjust based on your F2 optimization

# Model registry
MODELS = {}

class ClaimInput(BaseModel):
    """Input schema for claim predictions"""
    policy_annual_premium: float = Field(..., description="Annual policy premium")
    total_claim_amount: float = Field(..., description="Total claim amount")
    vehicle_age: int = Field(..., description="Age of vehicle in years")
    days_since_bind: int = Field(..., description="Days since policy binding")
    months_as_customer: int = Field(..., description="Months as customer")
    capital_gains: float = Field(0.0, alias="capital-gains")
    capital_loss: float = Field(0.0, alias="capital-loss")
    injury_share: float = Field(..., description="Share of injury damage")
    property_share: float = Field(..., description="Share of property damage")
    umbrella_limit: int = Field(..., description="Umbrella policy limit")
    incident_hour_of_the_day: int = Field(..., ge=0, le=23)
    hour_sin: Optional[float] = None
    hour_cos: Optional[float] = None
    
    class Config:
        populate_by_name = True

class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    model: str
    calibrated: bool
    probability: float
    threshold_flag: Optional[str] = None
    scenario: str

def load_models():
    """Load all available models on startup"""
    model_types = ["RandomForest", "ExtraTrees", "XGBoost"]
    calibration_types = ["calibrated", "uncalibrated"]
    
    for model_type in model_types:
        for cal_type in calibration_types:
            # Expected filename format: best_tree_models_calibrated.joblib or best_tree_models_uncalibrated.joblib
            filename = f"best_tree_models_{cal_type}.joblib"
            filepath = MODELS_DIR / filename
            
            if filepath.exists():
                try:
                    models_dict = joblib.load(filepath)
                    # Models are stored in dict structure: {'Trees': {'RandomForest': model, 'XGBoost': model, ...}}
                    if 'Trees' in models_dict and model_type in models_dict['Trees']:
                        key = f"{model_type}_{cal_type}"
                        MODELS[key] = models_dict['Trees'][model_type]
                        logger.info(f"Loaded model: {key}")
                except Exception as e:
                    logger.error(f"Error loading {filepath}: {e}")
    
    logger.info(f"Total models loaded: {len(MODELS)}")
    if not MODELS:
        logger.warning("No models loaded! Check models directory.")

@app.on_event("startup")
async def startup_event():
    """Load models on application startup"""
    load_models()

@app.get("/")
async def root():
    """Serve the frontend HTML"""
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(MODELS),
        "available_models": list(MODELS.keys())
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    claim_data: ClaimInput,
    model: Literal["rf", "et", "xgb"] = Query("rf", description="Model type: rf=RandomForest, et=ExtraTrees, xgb=XGBoost"),
    calibrated: bool = Query(True, description="Use calibrated model"),
    scenario: Literal["auto_flagger", "dashboard"] = Query("dashboard", description="Prediction scenario")
):
    """
    Predict fraud probability for an insurance claim.
    
    - **Scenario A (auto_flagger)**: Uses uncalibrated model + threshold for auto-flagging
    - **Scenario B (dashboard)**: Uses calibrated model for ranking/prioritization
    """
    
    # Map shorthand to full model names
    model_map = {"rf": "RandomForest", "et": "ExtraTrees", "xgb": "XGBoost"}
    model_name = model_map[model]
    
    # Determine calibration type
    cal_type = "calibrated" if calibrated else "uncalibrated"
    model_key = f"{model_name}_{cal_type}"
    
    # Override calibration based on scenario
    if scenario == "auto_flagger":
        cal_type = "uncalibrated"
        model_key = f"{model_name}_uncalibrated"
    elif scenario == "dashboard":
        cal_type = "calibrated"
        model_key = f"{model_name}_calibrated"
    
    # Get model
    if model_key not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_key} not found. Available: {list(MODELS.keys())}"
        )
    
    loaded_model = MODELS[model_key]
    
    # Prepare input data
    # Calculate hour_sin and hour_cos if not provided
    if claim_data.hour_sin is None or claim_data.hour_cos is None:
        hour_rad = (claim_data.incident_hour_of_the_day / 24) * 2 * np.pi
        claim_data.hour_sin = np.sin(hour_rad)
        claim_data.hour_cos = np.cos(hour_rad)
    
    # Convert to dict and create feature array
    # Note: The model expects the preprocessor to handle feature engineering
    # We'll pass raw features as a dict
    features_dict = claim_data.dict(by_alias=True)
    
    # For deployment, you would typically have a preprocessor that was saved with the model
    # Here we assume the model is already wrapped in a pipeline that handles preprocessing
    try:
        # Create input array - order must match training
        # The pipeline should handle the transformation
        input_data = {
            'policy_annual_premium': features_dict['policy_annual_premium'],
            'total_claim_amount': features_dict['total_claim_amount'],
            'vehicle_age': features_dict['vehicle_age'],
            'days_since_bind': features_dict['days_since_bind'],
            'months_as_customer': features_dict['months_as_customer'],
            'capital-gains': features_dict['capital-gains'],
            'capital-loss': features_dict['capital-loss'],
            'injury_share': features_dict['injury_share'],
            'property_share': features_dict['property_share'],
            'umbrella_limit': features_dict['umbrella_limit'],
            'incident_hour_of_the_day': features_dict['incident_hour_of_the_day'],
            'hour_sin': features_dict['hour_sin'],
            'hour_cos': features_dict['hour_cos']
        }
        
        # If model is a pipeline, it expects a DataFrame
        import pandas as pd
        input_df = pd.DataFrame([input_data])
        
        # Get prediction probability
        proba = loaded_model.predict_proba(input_df)[0, 1]  # Probability of fraud (class 1)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    # Determine threshold flag for auto_flagger scenario
    threshold_flag = None
    if scenario == "auto_flagger":
        threshold_flag = "AUTO_FLAG" if proba >= THRESHOLD_AUTO_FLAG else "AUTO_APPROVE"
    
    return PredictionResponse(
        model=model_name,
        calibrated=(cal_type == "calibrated"),
        probability=float(proba),
        threshold_flag=threshold_flag,
        scenario=scenario
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
