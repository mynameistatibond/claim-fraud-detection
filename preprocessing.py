
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# Load metadata
BASE_DIR = Path(__file__).resolve().parent
META_PATH = BASE_DIR / "preprocessing_metadata.json"

with open(META_PATH, "r") as f:
    META = json.load(f)

EXPECTED_COLS = META["expected_columns"]
DEFAULTS = META["defaults"]

def get_hour_bin(h):
    """Categorize hour into 4 bins"""
    if 0 <= h < 6: return "Night (0-5)"
    if 6 <= h < 12: return "Morning (6-11)"
    if 12 <= h < 18: return "Afternoon (12-17)"
    return "Evening (18-23)"

def preprocess_input(input_data: dict) -> pd.DataFrame:
    """
    Transform raw user input (dict) into the exact DataFrame structure
    expected by the model pipeline (42 columns).
    """
    
    # 1. Start with Defaults
    # We copy the defaults so we have a full dictionary of 42 fields
    final_data = DEFAULTS.copy()
    
    # 2. Update with User Input
    # Only update keys that exist in the input and are not None
    for k, v in input_data.items():
        if v is not None:
            final_data[k] = v
            
    # 3. Handle Feature Engineering
    
    # Hour Logic
    if 'incident_hour_of_the_day' in input_data:
        h = input_data['incident_hour_of_the_day']
        
        # Sine/Cosine
        hour_rad = (h / 24) * 2 * np.pi
        final_data['hour_sin'] = np.sin(hour_rad)
        final_data['hour_cos'] = np.cos(hour_rad)
        
        # Binning
        final_data['hour_bin_4'] = get_hour_bin(h)
    
    # Missing Flags Logic
    # collision_type
    col_type = input_data.get('collision_type')
    if col_type == '?' or col_type is None:
        final_data['collision_type_missing'] = "YES" # Assuming boolean or string "YES"/"NO"??
        # Let's check the defaults type.
        # If defaults['collision_type_missing'] is 0 or 1, use that.
        # But 'preprocesed_for_trees.csv' usually has numeric 0/1 for booleans?
        # Let's assume 1/0 based on standard engineering.
        # Wait, notebook output showed "collision_type_missing" column exists.
        # I'll stick to 1/0 if default is number.
        if isinstance(DEFAULTS.get('collision_type_missing'), (int, float)):
             final_data['collision_type_missing'] = 1
        else:
             final_data['collision_type_missing'] = "YES"
    else:
        if isinstance(DEFAULTS.get('collision_type_missing'), (int, float)):
             final_data['collision_type_missing'] = 0
        else:
             final_data['collision_type_missing'] = "NO"

    # police_report_available_missing
    police = input_data.get('police_report_available')
    if police == '?' or police is None:
         # Same logic
         if isinstance(DEFAULTS.get('police_report_available_missing'), (int, float)):
             final_data['police_report_available_missing'] = 1
         else:
             final_data['police_report_available_missing'] = "YES"
         
         # Usually if it is missing, we might impute the original column too? 
         # The Pipeline likely handles '?' as a category if it was trained on it.
    else:
         if isinstance(DEFAULTS.get('police_report_available_missing'), (int, float)):
             final_data['police_report_available_missing'] = 0
    
    # authorities_contacted_missing
    auth = input_data.get('authorities_contacted')
    if auth is None or auth == "None": # Check string "None" too
         if isinstance(DEFAULTS.get('authorities_contacted_missing'), (int, float)):
             final_data['authorities_contacted_missing'] = 1
    else:
         if isinstance(DEFAULTS.get('authorities_contacted_missing'), (int, float)):
             final_data['authorities_contacted_missing'] = 0

    # 4. Create DataFrame
    # Ensure strict column order
    df = pd.DataFrame([final_data])
    
    # Select only expected columns in correct order
    df = df[EXPECTED_COLS]
    
    return df
