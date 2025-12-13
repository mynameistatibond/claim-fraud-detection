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

class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    model: str
    calibrated: bool
    probability: float
    threshold_flag: Optional[str] = None
    scenario: str

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

@app.on_event("startup")
async def startup_event():
    load_models()

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
    scenario: Literal["auto_flagger", "dashboard"] = Query("dashboard")
):
    model_map = {"rf": "RandomForest", "et": "ExtraTrees", "xgb": "XGBoost", "voting": "VotingEnsemble"}
    model_name = model_map[model]
    
    cal_type = "calibrated" if calibrated else "uncalibrated"
    if scenario == "auto_flagger": cal_type = "uncalibrated"
    elif scenario == "dashboard": cal_type = "calibrated"
    
    model_key = f"{model_name}_{cal_type}"
    
    if model_key not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model {model_key} not found")
    
    loaded_model = MODELS[model_key]
    
    try:
        # Convert Pydantic to Dict
        input_dict = claim_data.dict(by_alias=True)
        
        # FULL PREPROCESSING
        final_df = preprocess_input(input_dict)
        
        # Predict
        proba = loaded_model.predict_proba(final_df)[0, 1]
        
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
        calibrated=(cal_type == "calibrated"),
        probability=float(proba),
        threshold_flag=threshold_flag,
        scenario=scenario
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
