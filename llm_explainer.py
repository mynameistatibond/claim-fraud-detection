import os
import requests
import json
import logging
import time
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

# Constants
# Candidate Models (Tried in order)
HF_CANDIDATE_URLS = [
    "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
    "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
    "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
]

@lru_cache(maxsize=256)
def _cached_llm_request(selected_model: str, ref_model: str, risk_str: str, drivers_tuple: tuple) -> dict | None:
    """
    Internal cached function. Arguments must be hashable.
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
    
    # Prompt Template (Generic enough for all instruction models)
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
        "inputs": prompt,
        "parameters": {
            "temperature": 0.01,
            "max_new_tokens": 350,
            "return_full_text": False
        }
    }

    last_error = "No models attempted."

    # Iterate through candidates
    for model_url in HF_CANDIDATE_URLS:
        logger.info(f"Attempting LLM request to: {model_url}")
        
        # Retry logic per model (for 503 loading states)
        max_retries = 2
        model_success = False
        
        for attempt in range(max_retries + 1):
            try:
                # DEBUG: Log exact URL
                logger.info(f"HF REQUEST URL: {model_url} (Attempt {attempt+1})")

                response = requests.post(
                    model_url, 
                    headers=headers, 
                    json=payload, 
                    timeout=(5.0, 20.0)
                )
                
                # Handling status codes
                if response.status_code == 503:
                    if attempt < max_retries:
                        time.sleep(3) # Wait for load
                        continue
                    else:
                        last_error = f"{model_url} timeout (503)"
                        break # Try next model
                
                if response.status_code == 429:
                    last_error = "Rate limit (429)"
                    break # Try next model immediately
                
                if response.status_code == 404 or response.status_code == 410:
                    last_error = f"{model_url} Not Found/Gone ({response.status_code})"
                    break # Definitely try next model
                
                if response.status_code != 200:
                    last_error = f"HF Error {response.status_code}: {response.text[:50]}"
                    break # Try next model
                    
                # Success parsing
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                elif isinstance(result, dict):
                    generated_text = result.get('generated_text', '')
                else:
                    generated_text = ''
                
                if not generated_text:
                    last_error = "Empty response from model"
                    break # Try next model

                # --- JSON Parsing Logic ---
                clean_text = generated_text.strip()
                start = clean_text.find('{')
                end = clean_text.rfind('}')
                
                if start != -1 and end != -1:
                    json_str = clean_text[start:end+1]
                    try:
                        data = json.loads(json_str)
                        # Minimal validation
                        if "summary" in data:
                            data.setdefault("bullets", [])
                            data.setdefault("disclaimer", "Automated explanation.")
                            if not isinstance(data["bullets"], list):
                                data["bullets"] = [str(data["bullets"])]
                            return data # <--- SUCCESS RETURN
                    except json.JSONDecodeError:
                        last_error = "JSON parse failed"
                        # Don't break immediately, maybe retry? No, deterministic failure.
                        break
                else:
                    last_error = "No JSON found"
                    break
                    
            except Exception as e:
                logger.error(f"Error calling {model_url}: {e}")
                last_error = str(e)
                # Try next attempt or model
    
    # If we get here, all models failed
    return {"error": f"All models failed. Last error: {last_error}"}

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
