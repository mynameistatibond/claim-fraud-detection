
import joblib
import numpy as np
import shap
import pandas as pd
import json
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_tree_models_uncalibrated.joblib"
# FEATURE_NAMES_PATH might be missing based on previous steps, we extract from pipeline
BACKGROUND_DATA_PATH = MODELS_DIR / "shap_background.npy"
OUTPUT_PATH = MODELS_DIR / "trend_registry.json"

# Features to analyze (Numeric features from VISIBLE_ROOTS)
# Based on app.py and metadata
CONTINUOUS_FEATURES = [
    "total_claim_amount", 
    "injury_share", 
    "property_share", 
    "incident_hour_of_the_day",
    "months_as_customer", 
    "policy_annual_premium", 
    "vehicle_age", 
    "age",
    "capital-gains", 
    "capital-loss", 
    "umbrella_limit", 
    "bodily_injuries",
    "number_of_vehicles_involved"
]

def build_registry():
    print("Loading resources...")
    
    # 1. Load Model
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        return

    models_dict = joblib.load(MODEL_PATH)
    if 'Trees' in models_dict and 'ExtraTrees' in models_dict['Trees']:
        model = models_dict['Trees']['ExtraTrees']
        print("Loaded ExtraTrees model.")
    else:
        print("Could not find ExtraTrees in model file.")
        return

    # 2. Get Estimator & Preprocessor
    if hasattr(model, 'steps'):
        preprocessor = model.steps[0][1]
        estimator = model.steps[-1][1]
    else:
        print("Model is not a pipeline!")
        return
        
    # 3. Extract Feature Names
    try:
        feature_names = list(preprocessor.get_feature_names_out())
        print(f"Extracted {len(feature_names)} feature names.")
    except Exception as e:
        print(f"Failed to extract feature names: {e}")
        return

    # 4. Load Background Data
    if BACKGROUND_DATA_PATH.exists():
        background = np.load(BACKGROUND_DATA_PATH)
        print(f"Loaded background data: {background.shape}")
        # Create a sturdy baseline (median)
        baseline_sample = np.median(background, axis=0).reshape(1, -1)
    else:
        print("Background data not found.")
        return

    # 5. Initialize Explainer
    print("Initializing TreeExplainer...")
    explainer = shap.TreeExplainer(estimator, background)
    
    registry = {}

    # 6. Iterate Features
    for feat_raw in CONTINUOUS_FEATURES:
        print(f"Analyzing {feat_raw}...")
        
        # Find index in transformed matrix
        # Note: transformed names might differ (e.g. passthrough for numeric)
        # We look for exact match or contains
        matches = [f for f in feature_names if feat_raw == f]
        if not matches:
            # Try finding it as part of name (e.g. "numerical__injury_share")
            matches = [f for f in feature_names if feat_raw in f]
            
        if not matches:
            print(f"  [WARN] Feature {feat_raw} not found in model input features. Skipping.")
            continue
            
        # Use first match (assuming 1-to-1 for continuous)
        feat_col = matches[0]
        idx = feature_names.index(feat_col)
        
        # Define Scan Range
        # Ideally we'd scan the valid domain.
        # For columns like 'age', 'vehicle_age', etc.
        # Check if we have domain info? No.
        # Let's define smart ranges or use background min/max?
        # Background might be scaled/encoded? 
        # Wait, PREPROCESSOR output is what goes into SHAP.
        # The background IS transformed data.
        # So we should use background distribution to define appropriate scan range!
        
        col_values = background[:, idx]
        min_val = np.min(col_values)
        max_val = np.max(col_values)
        
        # If constant, skip
        if min_val == max_val:
            print(f"  [SKIP] Constant feature dynamics in background.")
            continue
            
        # Create a grid of N points
        N = 10
        # Use quantiles or linspace? Linspace is safer for "trend" across full range.
        # But quantiles capture density. User suggested "quantiles ... or a fixed grid"
        # Let's use linspace to cover extremes (riskier!)
        grid_values = np.linspace(min_val, max_val, N)
        
        # Create synthetic batch
        X_synth = np.repeat(baseline_sample, N, axis=0)
        X_synth[:, idx] = grid_values
        
        # Compute SHAP
        # shap_values shape: (N, n_features, n_classes) or (N, n_features)
        shap_out = explainer.shap_values(X_synth)
        
        if isinstance(shap_out, list):
            # Class 1 (Fraud)
            shap_vec = shap_out[1]
        elif len(shap_out.shape) == 3:
            shap_vec = shap_out[:, :, 1]
        else:
            shap_vec = shap_out
            
        # Extract mean SHAP for THIS feature for each bin
        # SHAP represents contribution. 
        # For a single point changing, we just take the SHAP value of that point.
        # But we might want to be robust? No, just raw curve.
        shap_feature_curve = shap_vec[:, idx]
        
        # Store in registry
        # We store bins and corresponding SHAP medians/means
        # "ref_bin" -> Typically the middle bin? Or the median value bin?
        # User said "compare to mid-range values".
        # Let's say Ref Bin is index N//2 (middle of range)
        registry[feat_raw] = {
            "bins": grid_values.tolist(),
            "shap_values": shap_feature_curve.tolist(),
            "ref_idx": int(N // 2), 
            "min_val": float(min_val),
            "max_val": float(max_val)
        }
        
    # 7. Save Artifact
    print(f"Saving registry to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    build_registry()
