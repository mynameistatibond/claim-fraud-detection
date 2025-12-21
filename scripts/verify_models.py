
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_models")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS = {}

def load_models():
    """Load all available models using the same logic as app.py"""
    model_types = ["RandomForest", "ExtraTrees", "XGBoost"]
    calibration_types = ["calibrated", "uncalibrated"]
    
    for model_type in model_types:
        for cal_type in calibration_types:
            filename = f"best_tree_models_{cal_type}.joblib"
            filepath = MODELS_DIR / filename
            
            print(f"Checking for {filepath}...")
            if filepath.exists():
                try:
                    models_dict = joblib.load(filepath)
                    if 'Trees' in models_dict and model_type in models_dict['Trees']:
                        key = f"{model_type}_{cal_type}"
                        MODELS[key] = models_dict['Trees'][model_type]
                        print(f"SUCCESS: Loaded model: {key}")
                except Exception as e:
                    print(f"ERROR: Error loading {filepath}: {e}")
            else:
                print(f"WARNING: File not found: {filepath}")
    
    print(f"Total models loaded: {len(MODELS)}")
    if len(MODELS) == 6: # 3 models * 2 types
        print("VERIFICATION SUCCESS: All 6 expected models loaded.")
    else:
        print(f"VERIFICATION FAILED: Expected 6 models, loaded {len(MODELS)}.")

if __name__ == "__main__":
    load_models()
