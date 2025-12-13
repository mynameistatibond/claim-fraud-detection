
import joblib
import os
import sys
from sklearn.base import is_classifier

models_dir = "models"
files = [
    "best_tree_models_calibrated.joblib",
    "best_tree_models_uncalibrated.joblib"
]

print(f"Checking models in {models_dir}...")

for f in files:
    path = os.path.join(models_dir, f)
    try:
        models = joblib.load(path)
        
        if isinstance(models, dict) and 'Trees' in models:
            xgb_model = models['Trees'].get('XGBoost')
            if xgb_model:
                print(f"\n--- {f} XGBoost Analysis ---")
                print(f"Type: {type(xgb_model)}")
                
                # Check sklearn version
                if hasattr(xgb_model, '_sklearn_version'):
                    print(f"Sklearn Ver: {xgb_model._sklearn_version}")
                
                # Check classifier status
                print(f"is_classifier(model): {is_classifier(xgb_model)}")
                
                # Check Pipeline internals
                if hasattr(xgb_model, 'steps'):
                    print("Pipeline Steps:")
                    for name, step in xgb_model.steps:
                        print(f"  - {name}: {type(step).__name__}")
                        if hasattr(step, '_sklearn_version'):
                             print(f"    Step Sklearn Ver: {step._sklearn_version}")
                        if hasattr(step, '__class__'):
                             print(f"    is_classifier(step): {is_classifier(step)}")

    except Exception as e:
        print(f"Error loading {f}: {e}")
