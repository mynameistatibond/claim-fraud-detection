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

class ExplanationAgent:
    def __init__(self):
        self.llm_config = get_llm_config()

    def generate_ops_briefing(self, decision_contract: dict) -> dict:
        """
        Generates the official operational explanation based on the Decision Contract.
        """
        if not self.llm_config:
            return self._generate_fallback(decision_contract)

        system_prompt = (
            "You are a senior claims operations lead writing the official explanation of a system decision. "
            "The decision has already been made. "
            "Your job is to explain it clearly, calmly, and authoritatively. "
            "Rules: "
            "- Use ONLY the provided JSON. "
            "- Do NOT reference UI elements, panels, or settings. "
            "- Do NOT use the word 'audit'. "
            "- Do NOT hedge or offer multiple alternatives. "
            "- Speak as the system, not about the system. "
            "- Explain decisions, not deliberations. "
            "- Output valid JSON matching the schema."
        )

        user_prompt = f"""
        Using the decision contract below, generate an explanation for the user.

        DECISION CONTRACT:
        {json.dumps(decision_contract, indent=2)}

        OUTPUT SCHEMA (Strict JSON):
        {{
            "headline": "Strict review mode — spare capacity",
            "impact": "How this decision was made (Paragraph)",
            "workload_commitment": "Workload commitment (Paragraph, must mention P0 and Window)",
            "why_this_is_safe": "Why this is safe (Paragraph)",
            "recommended_next_step": "What to do next (Paragraph, decisive)",
            "technical_assumptions": [
                "Assumption 1",
                "Assumption 2",
                "Assumption 3"
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
                    "temperature": 0.1, # Lowest temp for authority and facts
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
                    return self._generate_fallback(decision_contract)
            else:
                logger.error(f"LLM Error {response.status_code}: {response.text}")
                return self._generate_fallback(decision_contract)

        except Exception as e:
            logger.error(f"Explanation Agent Exception: {e}")
            return self._generate_fallback(decision_contract)

    def _generate_fallback(self, decision_contract: dict) -> dict:
        """
        Deterministic fallback matching the new schema.
        """
        mode = decision_contract.get("review_mode", {}).get("label", "Standard Review")
        status = decision_contract.get("review_mode", {}).get("capacity_status", "balanced")
        workload = decision_contract.get("workload_commitment", {})
        
        return {
            "headline": f"{mode} — {status}",
            "impact": "The system has calibrated thresholds to match your team's specific capacity constraints today.",
            "workload_commitment": f"You are committed to reviewing {workload.get('p0_cases', '?')} high-priority cases over the next {workload.get('review_window_days', 1)} day(s).",
            "why_this_is_safe": "This workload volume sits within your team's safe operational limits, preventing backlog accumulation.",
            "recommended_next_step": "Review P0 cases immediately, as they represent the highest calibrated fraud risk.",
            "technical_assumptions": [
                f"Review Window: {workload.get('review_window_days', 1)} day(s)",
                "Model: Calibrated Fraud Detection",
                "Capacity: Optimized"
            ]
        }
