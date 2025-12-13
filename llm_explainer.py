
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
    
    # Option 4: HuggingFace (Backup - using Router + Zephyr)
    hf_key = os.getenv("HF_TOKEN")
    if hf_key:
        return {
            "provider": "huggingface",
            "url": "https://router.huggingface.co/hf-inference/models/HuggingFaceH4/zephyr-7b-beta",
            "key": hf_key,
            "model": "HuggingFaceH4/zephyr-7b-beta",
            "format": "huggingface"
        }
    
    # No API key found
    return None

# ... (rest of file)

def generate_llm_explanation(
    selected_model_name: str,
    reference_model_name: str,
    risk_score: float,
    explanation_items: list,
    timeout_s: tuple = (3, 10)
) -> dict | None:
    """
    Public API for generating LLM explanations.
    """
    try:
        if not explanation_items:
            return None
        
        # Format inputs
        risk_str = f"{risk_score * 100:.1f}"
        driver_str = build_driver_lines(explanation_items, max_items=5)
        drivers_tuple = tuple(driver_str.split('\n'))
        
        # Try LLM
        result = _cached_llm_request(
            selected_model_name,
            reference_model_name,
            risk_str,
            drivers_tuple
        )
        
        if result is not None:
            return result
        
        # Fallback Disabled per user request
        logger.error("LLM Request returned None (Error or Empty). Returning Error state.")
        return {"error": "LLM generation failed (Check logs for details)."}
        
    except Exception as e:
        logger.error(f"Error in generate_llm_explanation: {e}")
        return {"error": f"LLM Error: {str(e)}"}
