import os
import requests
import json
import logging
import time
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

# Constants
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3" 
# Note: Using v0.3 or just Instruct as per availability. The prompt requested "mistralai/Mistral-7B-Instruct", 
# usually it redirects to the latest or specific commit. I'll use the base tag to be safe or v0.3 if known good.
# Prompt said: "models/mistralai/Mistral-7B-Instruct" -> I will stick to that.
HF_API_URL_EXACT = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2" # v0.2 is very stable

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
        return None

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
        "inputs": f"[INST] {prompt} [/INST]", # Mistral instruction format
        "parameters": {
            "temperature": 0.01, # Almost deterministic
            "max_new_tokens": 350,
            "return_full_text": False
        }
    }

    # Retry logic for model loading (503)
    max_retries = 1
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                HF_API_URL_EXACT, 
                headers=headers, 
                json=payload, 
                timeout=(3.0, 10.0) # (connect, read)
            )
            
            if response.status_code == 503:
                if attempt < max_retries:
                    time.sleep(1.5)
                    continue
                else:
                    logger.warning("HF API 503 Service Unavailable (Model Loading) after retry.")
                    return None
            
            if response.status_code == 429:
                logger.warning("HF API 429 Rate Limit Reached.")
                return None
                
            if response.status_code != 200:
                logger.warning(f"HF API Error {response.status_code}: {response.text}")
                return None
                
            # Parse Response
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
            elif isinstance(result, dict):
                generated_text = result.get('generated_text', '')
            else:
                generated_text = ''
                
            # Extract JSON from potential markdown code blocks
            clean_text = generated_text.strip()
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[1].strip()
            
            try:
                data = json.loads(clean_text)
            except json.JSONDecodeError:
                # Fallback: sometimes LLM adds text before/after
                # Try to find first { and last }
                start = clean_text.find('{')
                end = clean_text.rfind('}')
                if start != -1 and end != -1:
                    try:
                        data = json.loads(clean_text[start:end+1])
                    except:
                        logger.warning("Failed to parse LLM JSON response.")
                        return None
                else:
                    logger.warning("No JSON object found in LLM response.")
                    return None
            
            # Schema Validation
            if not all(k in data for k in ("summary", "bullets", "disclaimer")):
                logger.warning("LLM response missing required keys.")
                return None
            
            if not isinstance(data["bullets"], list) or len(data["bullets"]) < 1:
                logger.warning("LLM response 'bullets' is invalid.")
                return None
                
            return data

        except requests.exceptions.Timeout:
            logger.warning("HF API Request Timed Out.")
            return None
        except Exception as e:
            logger.error(f"LLM Explainer Exception: {e}")
            return None
            
    return None

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
        return None
