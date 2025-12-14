
import os
import requests
import json
import logging
import time
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# LLM API CONFIGURATION
# ============================================================================
def get_llm_config():
    """
    Check which LLM API is available and return configuration.
    Priority: GROQ > Together AI > OpenRouter > HuggingFace
    """
    
    # Option 1: GROQ (BEST - Free, fast, reliable)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return {
            "provider": "groq",
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "key": groq_key,
            "model": "llama-3.3-70b-versatile",
            "format": "openai"
        }
    
    # Option 2: Together AI
    together_key = os.getenv("TOGETHER_API_KEY")
    if together_key:
        return {
            "provider": "together",
            "url": "https://api.together.xyz/v1/chat/completions",
            "key": together_key,
            "model": "mistralai/Mistral-7B-Instruct-v0.1",
            "format": "openai"
        }
    
    # Option 3: OpenRouter
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        return {
            "provider": "openrouter",
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "key": openrouter_key,
            "model": "mistralai/mistral-7b-instruct:free",
            "format": "openai"
        }
    
    # Option 4: HuggingFace (Backup - Router + Zephyr)
    hf_key = os.getenv("HF_TOKEN")
    if hf_key:
        return {
            "provider": "huggingface",
            "url": "https://router.huggingface.co/hf-inference/models/HuggingFaceH4/zephyr-7b-beta",
            "key": hf_key,
            "model": "HuggingFaceH4/zephyr-7b-beta",
            "format": "huggingface"
        }
    
    return None

def build_driver_lines(explanation_items: list, max_items: int = 5) -> str:
    """
    Convert ExplanationItem list into newline string lines:
    - {feature} | {direction} | {text}
    """
    lines = []
    # Handle both dicts and objects
    for item in explanation_items[:max_items]:
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

def call_openai_format_api(config: dict, prompt: str) -> str:
    """Call OpenAI-compatible APIs (GROQ, Together, OpenRouter)"""
    headers = {
        "Authorization": f"Bearer {config['key']}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": config["model"],
        "messages": [
            {
                "role": "system",
                "content": "You are an expert at explaining fraud detection model predictions in simple, clear language."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.3
    }
    
    response = requests.post(
        config["url"],
        headers=headers,
        json=payload,
        timeout=10
    )
    
    if response.status_code != 200:
        raise Exception(f"API Error {config['provider'].upper()} {response.status_code}: {response.text[:200]}")
    
    result = response.json()
    return result["choices"][0]["message"]["content"]

def call_huggingface_api(config: dict, prompt: str) -> str:
    """Call HuggingFace Inference API"""
    headers = {
        "Authorization": f"Bearer {config['key']}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 2048,
            "temperature": 0.3,
            "return_full_text": False
        }
    }
    
    response = requests.post(
        config["url"],
        headers=headers,
        json=payload,
        timeout=10
    )
    
    if response.status_code != 200:
        raise Exception(f"API Error {config['provider'].upper()} {response.status_code}: {response.text[:200]}")
    
    result = response.json()
    if isinstance(result, list) and len(result) > 0:
        return result[0].get("generated_text", "")
    return result.get("generated_text", "")

@lru_cache(maxsize=256)
def _cached_llm_request(selected_model: str, ref_model: str, risk_str: str, drivers_tuple: tuple) -> dict | None:
    """
    Internal cached function. Arguments must be hashable.
    """
    config = get_llm_config()
    
    if not config:
        logger.warning("No LLM API key found (GROQ/TOGETHER/OPENROUTER/HF).")
        return {"error": "No LLM credentials configured."}

    drivers_text = "\n".join(drivers_tuple)
    
    # Build prompt
    prompt = f"""You explain insurance claim risk scores to busy humans.


    Audience:

A claims reviewer or operations specialist skimming many cases.

They want quick orientation and context, not theory or instructions.

Tone:

Clear, calm, and human.

Slightly playful understatement is welcome (Douglas Adams–style).

Avoid bureaucratic, legalistic, or brochure-like language.

Rules:

Use ONLY the provided drivers. Do not invent facts.

List ALL provided drivers.

For each driver, clearly indicate whether it pushes risk up or pulls it down.

Do NOT say "fraud" or imply certainty.

Do NOT claim causality. Describe statistical associations only.

Do NOT mention models, SHAP, ML, or methodology.

Avoid generic phrases like "this assessment is based on" or "the model has identified".

Avoid repeating feature names mechanically; paraphrase naturally.

Style guidance:

Start by explaining what the score means in human terms.

Describe drivers as forces acting on the score (pushing it up or pulling it down).

For each driver, provide a short but meaningful explanation (2–3 sentences),
explaining what the signal represents and how it typically relates to risk patterns.

Be concrete, but not technical.

Prefer clarity and readability over extreme brevity.

Synthesis guidance (CRITICAL):
- The "summary" must be a cohesive narrative that WEAVES specific drivers together.
- Do NOT just say "there are mixed signals". Explain WHICH signals conflict.
- ADDRESS CONTRADICTIONS DIRECTLY:
    - Example: If "Authorities Contacted: Police" is Safe (↓), but "No Police Report" is Risky (↑), explain the REAL-WORLD implication. 
    - Say: "Use of police is typically safe, but the lack of a report suggests the claimant *asserted* police contact without official verification, or police attended but declined to file a report. This discrepancy is a key risk driver."
    - Do not just say "it's a gap". Explain *why* it's suspicious.
- Connect the dots for the user.

Optional guidance (Dynamic based on Risk Score):
- Look at the "Risk score" in the Context.
- If Risk < 50%: Suggest standard/routine checking.
- If Risk 50-75%: Suggest "Heightened Scrutiny" or "Closer Review".
- If Risk > 75%: Suggest "Immediate Escalation" or "Priority Investigation".
- MATCH THE URGENCY TO THE SCORE. Do not understate a high risk (e.g. >70%) as "moderate".

Context:
- Prediction model: {selected_model}
- Risk score: {risk_str}%
- Reference model: {ref_model}
- Top drivers:
{drivers_text}

Output:
Return ONLY valid JSON (no markdown, no extra text), matching this schema exactly:

{{
"summary": "A 2-3 sentence narrative that connects the strongest drivers into a cohesive story. Explicitly mention how key factors interact (e.g., 'X offsets Y').",
"drivers": [
{{
"name": "Readable driver name",
"effect": "up or down",
"explanation": "2–3 short sentences explaining what this factor represents and how it tends to influence risk patterns."
}}
],
"guidance": "One short sentence describing what this usually means for attention or handling.",
"disclaimer": "One short sentence noting this reflects statistical patterns, not proof."
}}
}}


"""

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Using LLM: {config['provider']} ({config['model']})")
            
            # Call appropriate API
            if config["format"] == "openai":
                generated_text = call_openai_format_api(config, prompt)
            else:
                generated_text = call_huggingface_api(config, prompt)
            
            logger.info(f"RAW LLM RESPONSE: {generated_text[:500]}")
            
            # Extract JSON
            clean_text = generated_text.strip()
            
            # Remove markdown code blocks if present
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0]
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[1].split("```")[0]
            
            # Find JSON object
            start = clean_text.find('{')
            end = clean_text.rfind('}')
            
            if start != -1 and end != -1:
                json_str = clean_text[start:end+1]
                data = json.loads(json_str)
                
                # Check for error style return if it's already a dict from fallback? No, text parsing.
                
                # Validate schema
                if "summary" not in data:
                    logger.warning("Missing 'summary' in response")
                    return {"error": "LLM response schema invalid (missing summary)"}
                
                # Set defaults
                data.setdefault("bullets", [])
                data.setdefault("disclaimer", "This explanation reflects statistical patterns, not proof.")
                
                # Ensure bullets is a list
                if not isinstance(data["bullets"], list):
                    data["bullets"] = [str(data["bullets"])]
                
                logger.info(f"✅ Successfully generated LLM explanation using {config['provider']}")
                return data
            else:
                logger.warning("No JSON found in response")
                return {"error": "LLM response contained no JSON"}
                
        except requests.exceptions.Timeout:
            logger.warning(f"API timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries:
                time.sleep(2)
                continue
            return {"error": f"LLM API Timeout ({config['provider']})"}
            
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            if attempt < max_retries:
                time.sleep(2)
                continue
            return {"error": f"LLM API Error: {str(e)}"}
    
    return {"error": "LLM retries exhausted."}

def generate_llm_explanation(
    selected_model_name: str,
    reference_model_name: str,
    risk_score: float,
    explanation_items: list,
    timeout_s: tuple = (3, 10)
) -> dict | None:
    """
    Public API for generating LLM explanations.
    FALLBACK IS DISABLED per user request. Returns Error dict on failure.
    """
    try:
        if not explanation_items:
            return None
        
        # Format inputs
        risk_str = f"{risk_score * 100:.1f}"
        
        # This function was missing before, leading to NameError. It is now defined above.
        driver_str = build_driver_lines(explanation_items, max_items=5) 
        drivers_tuple = tuple(driver_str.split('\n'))
        
        # Try LLM
        result = _cached_llm_request(
            selected_model_name,
            reference_model_name,
            risk_str,
            drivers_tuple
        )
        
        if result and "error" not in result:
             return result
             
        # If result has error, return it directly
        if result:
             return result
        
        return {"error": "Unknown Code Path in Helper"}
        
    except Exception as e:
        logger.error(f"Error in generate_llm_explanation: {e}")
        return {"error": f"LLM Module Error: {str(e)}"}
