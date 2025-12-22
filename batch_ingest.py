import pandas as pd
import numpy as np
import io
import json
import asyncio
import time
from pathlib import Path
from preprocessing import preprocess_input, DEFAULTS
from triage_agent import FraudTriageAgent
from risk_appetite_agent import RiskAppetiteAgent

# --- CONFIGURATION (Spec 3, 2.2) ---
FRIENDLY_TO_INTERNAL = {
    "Claim Value": "total_claim_amount",
    "Injury Cost Portion": "injury_share",
    "Property Damage Portion": "property_share",
    "Incident Time": "incident_hour_of_the_day",
    "Policy Tenure": "months_as_customer",
    "Annual Premium": "policy_annual_premium",
    "Vehicle Age": "vehicle_age",
    "Insured Age": "age",
    "Capital Gains": "capital-gains",
    "Capital Losses": "capital-loss",
    "Umbrella Limit": "umbrella_limit",
    "Bodily Injuries": "bodily_injuries",
    "Vehicles Involved": "number_of_vehicles_involved",
    "Days Since Policy Start": "days_since_bind",
    "Police Report Available": "police_report_available",
    "Incident Severity": "incident_severity",
    "Collision Type": "collision_type",
    "Authorities Contacted": "authorities_contacted"
}

# --- TYPE SAFETY HELPERS ---
def safe_cast(key, val):
    """
    Cast user input string to the correct type found in DEFAULTS.
    Useful because CSV reads as strings/mixed, but model expects specific types.
    """
    default_val = DEFAULTS.get(key)
    if default_val is None: 
        return val # No default, pass through (likely categorical string)
    
    target_type = type(default_val)
    
    try:
        if target_type == int:
            return int(float(val)) # Handles "5.0" -> 5
        elif target_type == float:
            return float(val)
        elif target_type == str:
            return str(val)
    except (ValueError, TypeError):
        # If casting fails, return original. 
        # Downstream validaton (or pipeline failure) handles it.
        return val
        
    return val

def map_row_to_internal(row_dict):
    """
    Maps a single user-facing row (dict) to internal keys.
    Handles Friendly Names -> Internal Keys
    Handles Internal Keys (snake_case) directly
    Handles Flag Resolution (YES/NO) but prefers direct values
    Handles Type Casting
    """
    internal_dict = {}
    
    # 1. Map Direct Fields (Friendly OR Internal)
    for friendly, internal in FRIENDLY_TO_INTERNAL.items():
        val = None
        # Try Friendly Name
        if friendly in row_dict:
            val = row_dict[friendly]
        # Try Internal Name (Fallback)
        elif internal in row_dict:
            val = row_dict[internal]
            
        if val is not None and not pd.isna(val) and str(val).strip() != "":
            internal_dict[internal] = safe_cast(internal, val)

    # 2. Categoricals & Flags
    # A. Incident Severity
    # Check if already mapped (from Friendly Name) or exists in raw dict (snake_case)
    if not internal_dict.get("incident_severity"):
        if "incident_severity" in row_dict and not pd.isna(row_dict["incident_severity"]):
            internal_dict["incident_severity"] = row_dict["incident_severity"]
        else:
            # Legacy Flag Resolution
            major = str(row_dict.get("Major Damage Severity", "")).upper() == "YES"
            total = str(row_dict.get("Total Loss Severity", "")).upper() == "YES"
            if major and total:
                 internal_dict["incident_severity"] = "Total Loss"
            elif major:
                internal_dict["incident_severity"] = "Major Damage"
            elif total:
                internal_dict["incident_severity"] = "Total Loss"
            # Else: Default (Minor Damage) applies in preprocessing

    # B. Collision Type
    if not internal_dict.get("collision_type"):
        if "collision_type" in row_dict and not pd.isna(row_dict["collision_type"]):
            internal_dict["collision_type"] = row_dict["collision_type"]
        elif str(row_dict.get("Rear Collision Type", "")).upper() == "YES":
            internal_dict["collision_type"] = "Rear Collision"

    # C. Authorities Contacted
    if not internal_dict.get("authorities_contacted"):
        if "authorities_contacted" in row_dict and not pd.isna(row_dict["authorities_contacted"]):
            internal_dict["authorities_contacted"] = row_dict["authorities_contacted"]
        elif str(row_dict.get("Police Contacted", "")).upper() == "YES":
            internal_dict["authorities_contacted"] = "Police"
        
    # Claim ID (Pass through or Generate)
    claim_id = row_dict.get("Claim ID")
    if not claim_id:
        claim_id = row_dict.get("claim_id") # Try snake_case
    
    return internal_dict, claim_id

async def process_batch_file(
    file_content: bytes, 
    model_context: dict,
    model_key: str = "xgb", # Default model
    scenario: str = "auto_flagger",
    explain: bool = False,
    progress_callback: callable = None,
    team_size: int = 5,
    review_time: int = 20,
    operating_mode: str = "daily_ops",
    review_window_days: int = 1,
    current_backlog_cases: int = 0
) -> dict:
    """
    Main entry point for batch processing.
    
    Args:
        file_content: Raw bytes of the CSV file
        model_context: Dict containing loaded models (passed from app.py)
        model_key: Key of the model to use (e.g. 'xgb', 'rf')
        scenario: 'auto_flagger' or 'dashboard'
        explain: If true, compute drivers (expensive)
        
    Returns:
        JSON summary and per-row results
    """
    
    # 1. Parse CSV
    try:
        # Use simple read_csv. Treat all as object first to assume nothing.
        df_raw = pd.read_csv(io.BytesIO(file_content), dtype=object)
    except Exception as e:
        return {
            "summary": {"status": "error", "message": f"CSV Parsing Failed: {str(e)}"},
            "rows": []
        }
        
    results = []
    valid_rows = [] # (index, processed_df) tuples
    
    # 2. Iterate & Map
    # 2. Iterate & Map
    total_raw = len(df_raw)
    for idx, row in df_raw.iterrows():
        # Yield control every 50 rows preventing freeze
        if idx % 50 == 0: 
            await asyncio.sleep(0)
            if progress_callback:
                pct = int((idx / total_raw) * 5)
                await progress_callback(pct)

        # Generate row ID if missing
        row_id = row.get("Claim ID")
        if pd.isna(row_id) or str(row_id).strip() == "":
            row_id = f"row_{idx+1:04d}"
            
        try:
            # A. Map & Validate
            internal_dict, _ = map_row_to_internal(row)
            
            # B. Preprocess (Fills Defaults + Configures Columns)
            # preprocess_input returns a 1-row DataFrame
            processed_df = preprocess_input(internal_dict)
            
            # Store for Batch Prediction
            valid_rows.append((idx, processed_df))
            
            # Initialize success entry (will fill score later)
            results.append({
                "claim_id": str(row_id),
                "status": "pending_prediction", 
                "errors": [],
                "original_index": idx
            })
            
        except Exception as e:
            # Row Failure
            results.append({
                "claim_id": str(row_id),
                "status": "failed",
                "errors": [str(e)],
                "original_index": idx
            })

    # 3. Batch Predict
    if not valid_rows:
        return {
            "summary": {
                "total_rows": len(df_raw),
                "processed_rows": 0,
                "failed_rows": len(df_raw)
            },
            "rows": results
        }
        
    # Re-assemble valid batch
    # valid_rows is list of (idx, df)
    # We need to map predictions back to results list by index
    
    # Concat all 1-row DFs
    batch_X = pd.concat([x[1] for x in valid_rows], ignore_index=True)
    
    # Update Progress: Parsing Done
    if progress_callback: await progress_callback(5)
    
    # Select Model
    MODEL_NAME_MAP = {
        'xgb': 'XGBoost',
        'rf': 'RandomForest',
        'et': 'ExtraTrees',
        'voting': 'VotingEnsemble'
    }
    long_name = MODEL_NAME_MAP.get(model_key, 'XGBoost')
    calibrated_suffix = "_calibrated" if scenario == 'dashboard' else "_uncalibrated"
    full_model_name = f"{long_name}{calibrated_suffix}"
    
    model = model_context.get(full_model_name)
    if not model:
        raise ValueError(f"Model {full_model_name} not found in context.")

    # Predict
    try:
        probs = model.predict_proba(batch_X)[:, 1] # Probability of Class 1
    except Exception as e:
        raise RuntimeError(f"Model Batch Prediction Failed: {str(e)}")

    # 4. Helpers for SHAP
    explainer = None
    if explain:
        explainer = model_context.get("shap_explainer")
    
    valid_indices = [x[0] for x in valid_rows] # original indices
    total_valid = len(valid_indices)
    
    for i, original_idx in enumerate(valid_indices):
        if i % 10 == 0: await asyncio.sleep(0) # Yield
        
        # Update Progress (Throttled)
        if progress_callback and (i % 10 == 0 or i == total_valid - 1):
            pct = 10 + int((i / total_valid) * 90)
            await progress_callback(pct)
        
        prob = float(probs[i])

        prob = float(probs[i])
        
        # Find the result entry
        res_entry = next(r for r in results if r["original_index"] == original_idx)
        
        res_entry["status"] = "success"
        res_entry["probability"] = prob
        
    # ... (inside loop)
        res_entry["status"] = "success"
        res_entry["probability"] = prob
        
        if explain and explainer:
            try:
                # Use batch_X row
                row_df = batch_X.iloc[[i]]
                
                # Use batch_X row
                row_df = batch_X.iloc[[i]]
                
                # TRANSFORM Data for SHAP (Must match Training Space)
                preprocessor = model_context.get("preprocessor")
                if preprocessor:
                    row_array = preprocessor.transform(row_df)
                    if hasattr(row_array, 'toarray'):
                        row_array = row_array.toarray()
                else:
                    row_array = row_df # Fallback, likely fail if strings exist
                
                # Calculate SHAP Values
                # Determine method: .shap_values() or call()
                # app.py uses explainer.shap_values(X_query)
                shap_raw = explainer.shap_values(row_array)
                
                # Handle output shape (List[Array] for classifier, Array for regressor)
                if isinstance(shap_raw, list):
                    # For binary classifier, we usually want Class 1 (index 1)
                    # Shape: [ (N, M), (N, M) ]
                    # We have N=1
                    vals = shap_raw[1][0]
                elif len(shap_raw.shape) == 3:
                    # Shape (N, M, Classes)
                    vals = shap_raw[0, :, 1]
                else:
                    # Shape (N, M) - XGBoost or binary
                    vals = shap_raw[0]

                # Get Feature Names
                feature_names = model_context.get("feature_names")
                if feature_names is None or len(feature_names) == 0:
                    # Fallback to column names (likely wrong for transformed data but prevents crash)
                    # Actually, if transformed, we have more columns. 
                    # app.py relies on SHAP_FEATURE_NAMES being correct for transformed data.
                   feature_names = [f"Feature {k}" for k in range(len(vals))]
                
                # Consolidate SHAP values by Feature Group
                # This solves "Major Damage ↓" appearing when the actual value is "Trivial"
                
                contributions_map = {}
                
                # Mappings for aggregation
                # key = prefix, value = (raw_col_name, display_label)
                GROUP_CONFIG = {
                    "incident_severity": ("incident_severity", "Severity"),
                    "collision_type": ("collision_type", "Collision"),
                    "authorities_contacted": ("authorities_contacted", "Authorities"),
                    # Add others if needed
                }

                # 1. Aggregate
                for f_idx, val in enumerate(vals):
                     fname = feature_names[f_idx] if f_idx < len(feature_names) else f"Feature {f_idx}"
                     
                     matched_group = None
                     for prefix, (raw_col, label) in GROUP_CONFIG.items():
                         # Strict prefix match
                         # Also ensure we don't accidentally match "collision_type_missing" into "collision_type"
                         if fname.startswith(prefix):
                             is_missing_flag = fname.lower().endswith("missing")
                             if not is_missing_flag:
                                 matched_group = prefix
                                 break
                     
                     if matched_group:
                         if matched_group not in contributions_map:
                             contributions_map[matched_group] = 0.0
                         contributions_map[matched_group] += val
                     else:
                         contributions_map[fname] = val
                
                # 2. Format Output
                final_contributions = []
                
                # Get access to raw values for labels
                raw_row = row_df.iloc[0]
                
                for key, val in contributions_map.items():
                    # Check if it's a group key
                    if key in GROUP_CONFIG:
                        raw_col, label = GROUP_CONFIG[key]
                        raw_val = raw_row.get(raw_col, "Unknown")
                        display_name = f"{label}: {raw_val}"
                    else:
                        # Standard Feature
                        # Clean up "collision_type_missing" -> "Collision Type Missing"
                        display_name = key.replace("_", " ").title()
                        
                        # Handle specific messy names if needed
                        if "Collision Type Missing" in display_name:
                             # If value is 1 (True), it increases risk usually (if bias is neg).
                             # But let's just show the name.
                             pass

                    final_contributions.append((display_name, val))

                # Sort by MAGNITUDE (absolute value) descending
                final_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                
                # Take Top 5 (index.html uses 3)
                top_drivers = []
                for name, val in final_contributions[:5]:
                    effect = "↑ Increases Risk" if val > 0 else "↓ Decreases Risk"
                    top_drivers.append({
                        "name": name,
                        "effect": effect,
                        "value": float(val)
                    })
                    
                res_entry["drivers"] = top_drivers

            except Exception as e:
                res_entry["errors"].append(f"Explanation Error: {str(e)}")

    # 5. Agentic Triage (Spec 3, 2.1)
    # Convert results to DataFrame for Agent analysis
    rows_list = [{k:v for k,v in r.items() if k != "original_index"} for r in results]
    scored_df = pd.DataFrame(rows_list)
    
    # Initialize Agents
    triage_agent = FraudTriageAgent()
    risk_agent = RiskAppetiteAgent()
    
    # 5a. Compute Risk Appetite (Product-Grade)
    probs = scored_df['probability'].astype(float).values
    
    # Enforce Stats Contract (Default safe values)
    batch_stats = {
        "p95": 0.0,
        "share_ge_0_3": 0.0,
        "count": 0
    }
    
    if len(probs) > 0:
        batch_stats.update({
            "p95": float(np.percentile(probs, 95)),
            "share_ge_0_3": float((probs >= 0.3).mean()),
            "count": len(probs)
        })
        
    # Decide Appetite
    risk_decision = risk_agent.decide_appetite(
        team_size=team_size,
        review_time_mins=review_time,
        batch_size=len(scored_df), 
        batch_stats=batch_stats,
        operating_mode=operating_mode,
        review_window_days=review_window_days,
        current_backlog=current_backlog_cases
    )
    
    appetite_str = risk_decision['risk_appetite']

    # 5b. Run Triage with Derived Appetite
    triage_result = triage_agent.triage_batch(
        scored_df, 
        batch_size=len(scored_df),
        team_size=team_size,
        review_time_mins=review_time,
        risk_appetite=appetite_str,
        review_window_days=review_window_days,
        current_backlog_cases=current_backlog_cases
    )
    
    # Attach Risk Analysis to Result
    triage_result['risk_analysis'] = risk_decision
    
    # Save Full Results CSV
    output_filename = f"batch_results_{int(time.time())}.csv"
    output_path = Path("outputs") / output_filename
    output_path.parent.mkdir(exist_ok=True)
    
    if "full_df" in triage_result:
        # Clean up for export (remove complex objects if any)
        export_df = triage_result["full_df"].copy()
        # Ensure drivers are stringified if needed, or drop complex columns
        if 'drivers' in export_df.columns:
            export_df['drivers'] = export_df['drivers'].apply(lambda x: str(x) if isinstance(x, list) else x)
            
        # Write with Metadata Headers
        with open(output_path, 'w') as f:
            f.write(f"# Risk Appetite: {risk_decision.get('risk_appetite', 'Unknown')}\n")
            f.write(f"# Confidence: {risk_decision.get('confidence', 'N/A')}\n")
            f.write(f"# Operating Mode: {operating_mode}\n")
            f.write(f"# Review Window: {review_window_days} days\n")
            f.write(f"# Generated: {time.ctime()}\n")
            export_df.to_csv(f, index=False)
            
        triage_result["csv_file"] = output_filename
        del triage_result["full_df"] # Remove from memory return
        
    # Merge Processing Stats into Agent Summary
    processed_count = len(valid_rows)
    failed_count = len(df_raw) - processed_count
    
    triage_result["summary"].update({
        "processed_rows": processed_count,
        "failed_rows": failed_count
    })
    
    return triage_result
