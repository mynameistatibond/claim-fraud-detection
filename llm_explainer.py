
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
# GROQ is RECOMMENDED: Free, fast (14,400 requests/day)
# Sign up at: https://console.groq.com/
# Add GROQ_API_KEY to your HuggingFace Space secrets

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
            "model": "mixtral-8x7b-32768",  # Fast and smart
            "format": "openai"  # OpenAI-compatible API
        }
    
    # Option 2: Together AI (Good - 1M tokens/month free)
    together_key = os.getenv("TOGETHER_API_KEY")
    if together_key:
        return {
            "provider": "together",
            "url": "https://api.together.xyz/v1/chat/completions",
            "key": together_key,
            "model": "mistralai/Mistral-7B-Instruct-v0.1",
            "format": "openai"
        }
    
    # Option 3: OpenRouter (OK - Free tier available)
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        return {
            "provider": "openrouter",
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "key": openrouter_key,
            "model": "mistralai/mistral-7b-instruct:free",
            "format": "openai"
        }
    
    # Option 4: HuggingFace (Unreliable - not recommended)
    hf_key = os.getenv("HF_TOKEN")
    if hf_key:
        return {
            "provider": "huggingface",
            "url": "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
            "key": hf_key,
            "model": "mistralai/Mistral-7B-Instruct-v0.1",
            "format": "huggingface"
        }
    
    # No API key found
    return None

def build_driver_lines(explanation_items: list, max_items: int = 5) -> str:
    """
    Convert ExplanationItem list into newline string lines:
    - {feature} | {direction} | {text}
    """
    lines = []
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
        "max_tokens": 400,
        "temperature": 0.3
    }
    
    response = requests.post(
        config["url"],
        headers=headers,
        json=payload,
        timeout=30
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
            "max_new_tokens": 400,
            "temperature": 0.3,
            "return_full_text": False
        }
    }
    
    response = requests.post(
        config["url"],
        headers=headers,
        json=payload,
        timeout=30
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
        logger.warning("No LLM API key found. Please set GROQ_API_KEY, TOGETHER_API_KEY, or OPENROUTER_API_KEY")
        # Don't return None yet, let caller use fallback
        return None

    drivers_text = "\n".join(drivers_tuple)
    
    # Build prompt
    prompt = f"""You generate user-facing explanations for an insurance claim risk score.

Rules:
- Use ONLY the provided drivers. Do not invent facts.
- Do NOT say "fraud" or imply certainty. This is a risk signal, not proof.
- Speak plainly. No ML jargon. No mention of SHAP.
- Explain each driver in terms of "tends to be associated with higher/lower risk patterns".
- Return valid JSON only, matching the schema exactly.

Context:
- Prediction model: {selected_model}
- Risk score: {risk_str}%
- Reference model: {ref_model}
- Top drivers:
{drivers_text}

Return ONLY valid JSON (no markdown, no extra text):
{{
  "summary": "2-3 sentence summary of the risk assessment",
  "bullets": ["bullet 1 explaining a driver", "bullet 2", "bullet 3"],
  "disclaimer": "Standard disclaimer about statistical patterns"
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
                
                # Validate schema
                if "summary" not in data:
                    logger.warning("Missing 'summary' in response")
                    return None
                
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
                return None
                
        except requests.exceptions.Timeout:
            logger.warning(f"API timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries:
                time.sleep(3)
                continue
            return None
            
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            if attempt < max_retries:
                time.sleep(2)
                continue
            return None
    
    return None

def generate_smart_fallback(selected_model: str, risk_score: float, explanation_items: list) -> dict:
    """
    High-quality fallback when LLM APIs are unavailable.
    """
    risk_pct = risk_score * 100
    
    if risk_pct >= 75:
        risk_level = "High"
        summary = f"Risk Level: {risk_level} ({risk_pct:.1f}%). This claim presents significant concern based on analyzed patterns and warrants immediate review."
    elif risk_pct >= 50:
        risk_level = "Elevated"
        summary = f"Risk Level: {risk_level} ({risk_pct:.1f}%). This claim shows moderate concern and should be reviewed carefully."
    elif risk_pct >= 25:
        risk_level = "Moderate"
        summary = f"Risk Level: {risk_level} ({risk_pct:.1f}%). This claim presents some attention and may benefit from additional review."
    else:
        risk_level = "Low"
        summary = f"Risk Level: {risk_level} ({risk_pct:.1f}%). This claim appears within normal parameters with minimal concern."
    
    bullets = []
    for item in explanation_items[:5]:
        if isinstance(item, dict):
            feat = item.get('feature', 'Unknown')
            text = item.get('text', '')
            direction = item.get('direction', '')
        else:
            feat = getattr(item, 'feature', 'Unknown')
            text = getattr(item, 'text', '')
            direction = getattr(item, 'direction', '')
        
        display_feat = feat.replace('_', ' ').title()
        
        if 'increase' in direction.lower():
            bullet = f"**{display_feat}**: {text} - This factor elevates risk as it aligns with patterns historically associated with problematic claims."
        elif 'decrease' in direction.lower():
            bullet = f"**{display_feat}**: {text} - This factor reduces risk as it matches characteristics of typical legitimate claims."
        else:
            bullet = f"**{display_feat}**: {text}"
        
        bullets.append(bullet)
    
    return {
        "summary": summary,
        "bullets": bullets,
        "disclaimer": "This explanation is based on statistical patterns. It represents risk signals, not definitive proof."
    }

def generate_llm_explanation(
    selected_model_name: str,
    reference_model_name: str,
    risk_score: float,
    explanation_items: list,
    timeout_s: tuple = (3, 10)
) -> dict | None:
    """
    Public API for generating LLM explanations.
    Falls back to smart rule-based explanation if LLM unavailable.
    """
    try:
        if not explanation_items:
            return None
        
        # Format inputs
        risk_str = f"{risk_score * 100:.1f}"
        driver_str = build_driver_lines(explanation_items, max_items=5)
        drivers_tuple = tuple(driver_str.split('\n'))
        
        # Try LLM first - if config missing or call fails, it returns None
        result = _cached_llm_request(
            selected_model_name,
            reference_model_name,
            risk_str,
            drivers_tuple
        )
        
        if result is not None:
            return result
        
        # Fallback to smart explanation
        logger.info("Using fallback explanation (LLM unavailable)")
        return generate_smart_fallback(
            selected_model_name,
            risk_score,
            explanation_items
        )
        
    except Exception as e:
        logger.error(f"Error in generate_llm_explanation: {e}")
        return generate_smart_fallback(
            selected_model_name,
            risk_score,
            explanation_items
        )
