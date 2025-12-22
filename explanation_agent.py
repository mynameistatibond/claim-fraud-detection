# explanation_agent.py
import json
import logging
import requests
from llm_explainer import get_llm_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExplanationAgent:
    def __init__(self):
        self.llm_config = get_llm_config()

    def generate_ops_briefing(self, fact_sheet: dict) -> dict:
        """
        Generates a human-centric operational explanation based strictly on the provided Fact Sheet.
        """
        if not self.llm_config:
            return self._generate_fallback(fact_sheet)

        system_prompt = (
            "You are a claims operations lead. "
            "Your goal is to explain the system's review mode and workload commitments to a human operator. "
            "Use ONLY the provided JSON fact sheet. "
            "Do NOT invent numbers, thresholds, or UI elements. "
            "Do NOT use the word 'audit'. Use 'technical assumptions' instead. "
            "Be decisive (one recommendation). "
            "Output strictly valid JSON matching the schema."
        )

        user_prompt = f"""
        Generate an operational explanation based on this fact sheet.

        INPUT FACT SHEET:
        {json.dumps(fact_sheet, indent=2)}

        OUTPUT SCHEMA (Strict JSON):
        {{
            "headline": "Short 1-line headline (e.g. 'Broad review mode — spare capacity')",
            "impact": "1-2 sentences. What does this change for work today? (Visibility/Volume)",
            "workload_commitment": "What work is committed? Explicitly mention review window and P0 count. Frame P0 as primary.",
            "why_this_is_safe": "Why is this reasonable? Capacity vs Workload fit. No jargon.",
            "recommended_next_step": "One decisive instruction. Start with P0. Conditionally mention P1 if spare capacity.",
            "technical_assumptions": [
                "Bullet 1 (e.g. Thresholds)",
                "Bullet 2 (e.g. Capacity)",
                "Bullet 3 (e.g. Window)"
            ]
        }}
        """

        try:
            headers = {
                "Authorization": f"Bearer {self.llm_config['key']}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>",
                "parameters": {
                    "max_new_tokens": 512,
                    "temperature": 0.3, # Low temp for factual consistency
                    "return_full_text": False
                }
            }

            response = requests.post(self.llm_config['url'], headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and 'generated_text' in result[0]:
                    text = result[0]['generated_text'].strip()
                    # Clean markdown code blocks if present
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0].strip()
                    elif "```" in text:
                        text = text.split("```")[1].split("```")[0].strip()
                    
                    return json.loads(text)
                else:
                    logger.error(f"Unexpected LLM response format: {result}")
                    return self._generate_fallback(fact_sheet)
            else:
                logger.error(f"LLM Error {response.status_code}: {response.text}")
                return self._generate_fallback(fact_sheet)

        except Exception as e:
            logger.error(f"Explanation Agent Exception: {e}")
            return self._generate_fallback(fact_sheet)

    def _generate_fallback(self, fact_sheet: dict) -> dict:
        """
        Deterministic fallback matching the new schema.
        """
        mode = fact_sheet.get("mode_label", "Standard Review")
        status = fact_sheet.get("capacity_status", "balanced")
        p0 = fact_sheet.get("workload", {}).get("p0_cases", "?")
        window = fact_sheet.get("review_window_days", 1)
        cap = fact_sheet.get("capacity", {}).get("daily_capacity_cases", "?")
        
        return {
            "headline": f"{mode} — {status.replace('_', ' ')}",
            "impact": "The system has set thresholds to prioritize claims based on your available capacity.",
            "workload_commitment": f"You are committed to reviewing {p0} high-priority cases over the next {window} day(s).",
            "why_this_is_safe": "The workload is calibrated to fit within your team's estimated operational limits.",
            "recommended_next_step": "Start by reviewing the identified P0 cases immediately.",
            "technical_assumptions": [
                f"Team Capacity: ~{cap}/day",
                f"Review Window: {window} day(s)",
                "Thresholds applied based on appetite"
            ]
        }
