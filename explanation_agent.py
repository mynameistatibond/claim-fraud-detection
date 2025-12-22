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
        Generates a human-centric operational briefing based strictly on the provided Fact Sheet.
        """
        if not self.llm_config:
            return self._generate_fallback(fact_sheet)

        system_prompt = (
            "You are a claims operations lead. "
            "Use only the provided JSON fact sheet. "
            "Do not invent numbers or thresholds. "
            "Do not mention specific stats like p95 or share_ge unless in the audit details. "
            "Output JSON only in the specified schema."
        )

        user_prompt = f"""
        Generate an operational briefing based on this fact sheet.

        FACT SHEET:
        {json.dumps(fact_sheet, indent=2)}

        OUTPUT SCHEMA:
        {{
            "title": "Short 1-line headline (e.g. 'Broad review mode — capacity available')",
            "summary": "2-3 sentences explaining the situation clearly.",
            "what_you_get": ["Bullet 1", "Bullet 2"],
            "why_this_mode": ["Reason 1", "Reason 2", "Reason 3 (optional)"],
            "recommended_next_step": {{
                "action": "One concrete recommended action",
                "reason": "Why this action constitutes a good trade-off"
            }},
            "audit_details": {{
                "thresholds": "e.g. P0 >= 0.XX",
                "capacity": "e.g. ~120/day",
                "risk_shape": "e.g. P95=0.77"
            }}
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
                    "temperature": 0.3, # Low temperature for factual consistency
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
        Deterministic fallback if LLM falls.
        """
        mode = fact_sheet.get("mode_label", "Standard Review")
        status = fact_sheet.get("workload", {}).get("status", "Balanced")
        
        return {
            "title": f"{mode} — {status.replace('_', ' ').capitalize()}",
            "summary": "The system has analyzed the batch and set thresholds based on your team's capacity.",
            "what_you_get": [
                f"Prioritized list of {fact_sheet.get('counts', {}).get('p0', '?')} high-risk claims",
                "Full CSV export for detailed analysis"
            ],
            "why_this_mode": [
                "Based on current capacity constraints",
                "Optimized for available review time"
            ],
            "recommended_next_step": {
                "action": "Review P0 cases first",
                "reason": "Ensures highest risk claims are covered"
            },
            "audit_details": {
                "thresholds": "Check advanced panel",
                "capacity": "Check settings",
                "risk_shape": "N/A"
            }
        }
