import os
import requests
import json
import logging
import time
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

# Constants
# Constants
HF_API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.3"
# NOTE: We use the router endpoint with /hf-inference/ path as required by the new infrastructure.

def build_driver_lines(explanation_items: list, max_items: int = 5) -> str:
    """
    Convert ExplanationItem list into newline string lines:
    - {feature} | {direction} | {text}
    """
    lines = []
    # Sort by importance just in case, though usually already sorted
    # Items might be dicts or objects depending on where they come from. 
    # In app.py they are ExplanationItem objects (tuples/dataclasses) or dicts.
    # The prompt implies they are structured items.
    
    # Assuming standard sorting is done by caller, but we limit to max_items
    for item in explanation_items[:max_items]:
        # Handle if item is dict or object
        if isinstance(item, dict):
            feat = item.get('feature', 'Unknown')
            direction = item.get('direction', 'N/A')
            text = item.get('text', '')
        else:
            feat = getattr(item, 'feature', 'Unknown')
            direction = getattr(item, 'direction', 'N/A')
            text = getattr(item, 'text', '')
            
        lines.append(f"- {feat} | {direction} | {text}")
    
    return "\n".join(lines)

@lru_cache(maxsize=256)
def _cached_llm_request(selected_model: str, ref_model: str, risk_str: str, drivers_tuple: tuple) -> dict | None:
    """
    Internal cached function. Arguments must be hashable.
    risk_str: "45.2" (formatted risk score)
    drivers_tuple: tuple of strings (lines)
    """
    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        logger.warning("HF_TOKEN missing. Skipping LLM explanation.")
        return {"error": "HF_TOKEN environment variable is missing."}

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    drivers_text = "\n".join(drivers_tuple)
    
    # Prompt Template
    prompt = f"""You generate user-facing explanations for an insurance claim risk score.

Rules:
- Use ONLY the provided drivers. Do not invent facts.
- Do NOT say “fraud” or imply certainty. This is a risk signal, not proof.
- Speak plainly. No ML jargon. No mention of SHAP.
- Explain each driver in terms of “tends to be associated with higher/lower risk patterns”.
- If a driver looks like a one-hot category (contains underscores or category names), explain it as “This specific case includes X”.
- Return valid JSON only, matching the schema exactly.

Context:
- Prediction model selected by user: {selected_model}
- Risk score: {risk_str}%
- Explanation source model (reference): {ref_model}
- Drivers (top {len(drivers_tuple)}):
{drivers_text}

Return JSON:
{{
  "summary": "...",
  "bullets": ["...", "...", "..."],
  "disclaimer": "..."
}}
"""

    payload = {
        "inputs": prompt, # Plain text (Instruct format removed as per diagnostics)
        "parameters": {
            "temperature": 0.01, # Almost deterministic
            "max_new_tokens": 350,
            "return_full_text": False
        }
    }

    # Retry logic for model loading (503)
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            # DEBUG: Log the exact URL being hit
            logger.error(f"HF REQUEST URL: {HF_API_URL}")

            response = requests.post(
                HF_API_URL, 
                headers=headers, 
                json=payload, 
                timeout=(5.0, 20.0) # Increased timeout (connect, read)
            )
            
            if response.status_code == 503:
                if attempt < max_retries:
                    time.sleep(5) # Increased wait for cold starts
                    continue
                else:
                    return {"error": "Model is loading (503). Try again in a moment."}
            
            if response.status_code == 429:
                return {"error": "Rate limit reached (429)."}
            
            if response.status_code == 401:
                return {"error": "Invalid HF_TOKEN (401). Check permissions."}

            if response.status_code != 200:
                logger.error(f"HF API Error: {response.text}")
                return {"error": f"HF API Error {response.status_code}: {response.text[:50]}"}
                
            # Parse Response
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
            elif isinstance(result, dict):
                generated_text = result.get('generated_text', '')
            else:
                generated_text = ''
            
            # RAW LOGGING for Debugging
            logger.info(f"RAW LLM RESPONSE: {generated_text[:500]}")

            # Robust JSON Extraction (Primary Method, not fallback)
            clean_text = generated_text.strip()
            
            # 1. Attempt to find JSON object bounds
            start = clean_text.find('{')
            end = clean_text.rfind('}')
            
            if start != -1 and end != -1:
                json_str = clean_text[start:end+1]
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    return {"error": "Failed to parse JSON (syntax error)."}
            else:
                return {"error": "No JSON object found in response."}
            
            # Relaxed Schema Validation
            if "summary" not in data:
                return {"error": "LLM response missing 'summary'."}
            
            # Defaults
            data.setdefault("bullets", [])
            data.setdefault("disclaimer", "This explanation reflects statistical patterns, not proof.")
            
            # Ensure bullets is a list
            if not isinstance(data["bullets"], list):
                data["bullets"] = [str(data["bullets"])] # Force list if single string
                
            return data

        except requests.exceptions.Timeout:
            return {"error": "HF API Request Timed Out (Cold model?)."}
        except Exception as e:
            logger.error(f"LLM Explainer Exception: {e}")
            return {"error": f"Internal Error: {str(e)}"}
            
    return {"error": "Unknown error."}

def generate_llm_explanation(
    selected_model_name: str,
    reference_model_name: str,
    risk_score: float,
    explanation_items: list,
    timeout_s: tuple = (3, 10)
) -> dict | None:
    """
    Public facade for generating LLM explanations.
    """
    try:
        if not explanation_items:
            return None
            
        # Format inputs for cache key
        risk_str = f"{risk_score * 100:.1f}"
        driver_str = build_driver_lines(explanation_items, max_items=5)
        drivers_tuple = tuple(driver_str.split('\n'))
        
        return _cached_llm_request(
            selected_model_name, 
            reference_model_name, 
            risk_str, 
            drivers_tuple
        )
    except Exception as e:
        logger.error(f"Error in generate_llm_explanation wrapper: {e}")
        return {"error": f"Wrapper Error: {str(e)}"}
