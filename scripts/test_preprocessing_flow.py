
import requests
import json
import sys

# We can run this LOCALLY against the "app" logic if we import it, 
# But easier to just unit test `preprocessing.py` and model loading.

import sys
from pathlib import Path
# Add parent dir to path to import preprocessing
sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing import preprocess_input, EXPECTED_COLS
import joblib
import pandas as pd

# Mock Input mimicking the new frontend
mock_input = {
    "policy_annual_premium": 1200,
    "total_claim_amount": 52000,
    "vehicle_age": 12,
    "days_since_bind": 300,
    "months_as_customer": 48,
    "capital-gains": 0,
    "capital-loss": 0,
    "injury_share": 0.1,
    "property_share": 0.1,
    "umbrella_limit": 0,
    "incident_hour_of_the_day": 14,
    
    # Cats
    "incident_severity": "Major Damage",
    "collision_type": "Rear Collision",
    "police_report_available": "YES",
    "authorities_contacted": "Police",
    "number_of_vehicles_involved": 2,
    "bodily_injuries": 1
}

print("Running Preprocessing...")
df = preprocess_input(mock_input)
print("Resulting DataFrame shape:", df.shape)
print("Columns:", list(df.columns))

# Verify specific engineered cols
print(f"Hour Bin: {df['hour_bin_4'].values[0]}")
print(f"Collision Missing Flag: {df['collision_type_missing'].values[0]}")

# Load Model
print("\nLoading Model...")
models = joblib.load(str(Path(__file__).resolve().parent.parent / "models/best_tree_models_uncalibrated.joblib"))
model = list(models['Trees'].values())[0]

print("Predicting...")
try:
    pred = model.predict_proba(df)
    print(f"Success! Prediction: {pred[0]}")
except Exception as e:
    print(f"FAILED: {e}")
    # Inspect difference
    if hasattr(model, 'feature_names_in_'):
        miss = set(model.feature_names_in_) - set(df.columns)
        print(f"Missing cols: {miss}")
