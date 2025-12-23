"""
Centralized Field Mappings for Fraud Detector.
Acts as the Single Source of Truth for:
1. CSV Ingestion (Friendly -> Internal)
2. UI Display (Internal -> Friendly)
"""

# Core Definitions: { Internal_Key: Friendly_Label }
_CORE_FIELD_DEFINITIONS = {
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
    "days_since_bind": "Days Since Policy Start",
    "police_report_available": "Police Report Available",
    
    # Categorical Roots (Useful for loose matching or raw ingest)
    "incident_severity": "Incident Severity",
    "collision_type": "Collision Type",
    "authorities_contacted": "Authorities Contacted"
}

# 1. CSV Input Map: Friendly -> Internal
# Used by batch_ingest.py to map user columns to internal keys
CSV_INPUT_MAP = {v: k for k, v in _CORE_FIELD_DEFINITIONS.items()}

# 2. Display Label Map: Internal -> Friendly
# Used by app.py for SHAP explanations and UI labels
# We extend the core definitions with model-specific One-Hot keys if needed
_MODEL_DISPLAY_OVERRIDES = {
    "incident_severity_Major Damage": "Major Damage Severity",
    "incident_severity_Total Loss": "Total Loss Severity",
    "collision_type_Rear Collision": "Rear Collision Type",
    "authorities_contacted_Police": "Police Contacted"
}

DISPLAY_LABEL_MAP = _CORE_FIELD_DEFINITIONS.copy()
DISPLAY_LABEL_MAP.update(_MODEL_DISPLAY_OVERRIDES)
