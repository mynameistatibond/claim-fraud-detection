import os
import json
import logging
import re
import requests
from llm_explainer import get_llm_config

logger = logging.getLogger(__name__)

class DecisionContractBuilder:
    """
    Deterministic builder for the Decision Contract.
    Enforces all arithmetic and logic rules from Tech Spec 3.3.
    """
    @staticmethod
    def build(triage_result, scored_df, review_window_days, team_size, review_time_mins, appetite_str):
        # 1. Deterministic Computation Rules
        team_capacity_cases_per_day = int((team_size * 480) / review_time_mins)
        
        p0_count = triage_result["summary"]["p0_count"]
        p1_count = triage_result["workload_summary"]["assigned_work"]["p1"]
        p2_count = len(scored_df) - p0_count - p1_count
        
        # Capacity Equivalent Demand: p0 + 0.6 * p1
        demand = p0_count + (0.6 * p1_count)
        total_capacity = team_capacity_cases_per_day * review_window_days
        
        if total_capacity > 0:
            cap_ratio = demand / total_capacity
        else:
            cap_ratio = 999.0
        
        if cap_ratio <= 0.9:
            capacity_status = "spare capacity"
        elif cap_ratio <= 1.1:
            capacity_status = "at capacity"
        else:
            capacity_status = "backlog risk"

        # Window Label
        if review_window_days == 1:
            window_label = "today"
        # elif review_window_days == 7:
        #     window_label = "next 7 days"
        else:
            window_label = f"next {review_window_days} days"

        # Mode Label
        mode_label = f"{'Strict' if appetite_str == 'Conservative' else ('Broad' if appetite_str == 'Aggressive' else 'Standard')} review mode"

        # 2. Construct Contract
        contract = {
            "contract_version": "1.0",
            "review_mode": {
                "mode_label": mode_label,
                "capacity_status": capacity_status
            },
            "decision_basis": {
                "model_statement": "Fraud scores are produced by a calibrated fraud detection model.",
                "inputs_used": {
                    "review_window_days": review_window_days,
                    "team_size": team_size,
                    "review_time_mins_per_case": review_time_mins,
                    "team_capacity_cases_per_day": team_capacity_cases_per_day
                },
                "how_thresholds_were_set": "Thresholds were set to prioritize high-confidence fraud cases while staying within your operational capacity."
            },
            "workload_commitment": {
                "window_label": window_label,
                "p0_cases": p0_count,
                "p1_cases": p1_count,
                "p2_cases": p2_count,
                "ordering_guarantee": "Cases are ordered from most to least likely fraud."
            },
            "policy": {
                "thresholds": triage_result["decision_policy"]["thresholds"],
                "p0_strictness_statement": "P0 thresholds are intentionally strict to ensure the most credible fraud cases are reviewed first.",
                "p1_flex_statement": "P1/P2 are more flexible and expand or shrink based on available capacity."
            },
            "safety_statement": {
                "why_safe": "Lower-priority claims have weaker fraud signals and can be reviewed later if capacity allows without significantly increasing risk."
            }
        }
        return contract

class ExplanationAgent:
    def __init__(self):
        self.llm_config = get_llm_config()

    # Added import for specific helper if not already there, but keeping it clean
    from llm_explainer import call_openai_format_api

    def generate_explanation(self, decision_contract: dict) -> dict:
        """
        Generates the explanation text, validates it, and parses it for the UI.
        Returns a dict matching the UI schema.
        """
        if not self.llm_config:
            return self._generate_fallback(decision_contract)

        system_prompt = (
            "You are a senior claims operations lead writing the official explanation of a decision that has already been made.\n\n"
            "Rules:\n"
            "- Use ONLY the provided DecisionContract JSON.\n"
            "- Do NOT reference UI elements, panels, settings, buttons, links, or 'advanced views'.\n"
            "- Do NOT use the word 'audit'.\n"
            "- Do NOT invent numbers, thresholds, metrics, or missing fields.\n"
            "- Be explicit, human, and accountable.\n"
            "- P0 must be described as strict and the first priority. P1/P2 are optional depending on capacity.\n"
            "- Provide exactly one recommended next step.\n"
            "- Output the explanation in the specified section structure only, using plain text."
        )

        user_prompt = f"""
        Write the user-facing explanation using the DecisionContract below.

        Use exactly this structure:
        1) Headline
        2) How this decision was made
        3) Workload commitment
        4) Why this is safe
        5) What to do next
        6) System assumptions (bullet list)

        DecisionContract:
        {json.dumps(decision_contract, indent=2)}
        """

        try:
            # Check format from config (added in llm_explainer)
            api_format = self.llm_config.get("format", "huggingface")

            if api_format == "openai":
                # GROQ / OPENROUTER PATH
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                # call_openai_format_api expects (config, prompt)
                # But looking at llm_explainer.py, it constructs messages from prompt.
                # Let's double check llm_explainer.py signature in next step if needed, 
                # but based on reading it earlier:
                # def call_openai_format_api(config, prompt): matches.
                # It puts prompt in user message.
                
                text = call_openai_format_api(self.llm_config, full_prompt)
                
            else:
                # HUGGING FACE PATH
                headers = {
                    "Authorization": f"Bearer {self.llm_config['key']}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "inputs": f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>",
                    "parameters": {
                        "max_new_tokens": 600,
                        "temperature": 0.1, 
                        "return_full_text": False
                    }
                }
                
                response = requests.post(self.llm_config['url'], headers=headers, json=payload, timeout=15)
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and 'generated_text' in result[0]:
                        text = result[0]['generated_text'].strip()
                    else:
                        logger.error(f"Unexpected LLM response format: {result}")
                        return self._generate_fallback(decision_contract)
                else:
                    logger.error(f"LLM Error {response.status_code}: {response.text}")
                    return self._generate_fallback(decision_contract)

            # Validate and Parse (Shared)
            if text and self._validate_text(text, decision_contract):
                return self._parse_to_ui_schema(text)
            else:
                logger.warning(f"LLM output failed validation. Text: {text[:100]}...")
                return self._generate_fallback(decision_contract)

        except Exception as e:
            logger.error(f"Explanation Agent Exception: {e}")
            return self._generate_fallback(decision_contract)

    def _validate_text(self, text: str, contract: dict) -> bool:
        """
        Deterministic Explanation Validator (Spec 6.2).
        """
        text_lower = text.lower()
        
        # 1. Check sections (Loose match to handle minor formatting vars)
        required_sections = [
            "how this decision was made",
            "workload commitment",
            "why this is safe",
            "what to do next",
            "system assumptions"
        ]
        for section in required_sections:
            if section not in text_lower:
                logger.warning(f"Validation Error: Missing section '{section}'")
                return False

        # 2. Forbidden terms
        forbidden = ["audit", "panel", "settings", "click", "link", "advanced view", "ui", "optimization"]
        contract_str = json.dumps(contract).lower()
        
        for term in forbidden:
            if term in text_lower:
                # Exception for "optimization" if strictly needed, but spec says forbid unless verbatim.
                if term == "optimization" and "optimization" in contract_str:
                    continue
                logger.warning(f"Validation Error: Forbidden term '{term}' found.")
                return False

        return True

    def _parse_to_ui_schema(self, text: str) -> dict:
        """
        Parses strictly formatted text back to the JSON schema expected by valid index.html
        Schema: headline, impact, workload_commitment, why_this_is_safe, recommended_next_step, technical_assumptions
        """
        sections = {
            "headline": "",
            "impact": "",
            "workload_commitment": "",
            "why_this_is_safe": "",
            "recommended_next_step": "",
            "technical_assumptions": []
        }
        
        lines = text.split('\n')
        current_section = "headline"
        buffer = []

        header_map = {
            "headline": "headline", # Should be first line usually
            "how this decision was made": "impact",
            "workload commitment": "workload_commitment",
            "why this is safe": "why_this_is_safe",
            "what to do next": "recommended_next_step",
            "system assumptions": "technical_assumptions"
        }
        
        # First line is Headline (usually)
        if lines:
            first_line = lines[0].strip()
            # If first line looks like a header (e.g. "Headline"), skip it, but spec says output structure: 1) Headline...
            # User Prompt: "1) Headline" (header), then content? Or "Headline: content"?
            # Prompt: "Use exactly this structure: 1) Headline...". 
            # Usually LLM outputs "Headline\nContent".
            # We assume the first non-empty line IS the headline content if it doesn't match a header keyword string
            pass

        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Normalize line to check for header
            # Remove numbering and punctuation
            lower_line = re.sub(r'^\d+[\).]\s*', '', line.lower().replace(":", "")).strip()
            
            if lower_line in header_map:
                # Flush buffer to *previous* section
                valid_content = [b for b in buffer if b]
                if valid_content:
                    if current_section == "technical_assumptions":
                        sections[current_section] = valid_content
                    elif current_section == "headline": # Special case if we captured lines before first explicit header
                         sections[current_section] = " ".join(valid_content).strip()
                    else:
                        sections[current_section] = " ".join(valid_content).strip()
                
                # Switch section
                current_section = header_map[lower_line]
                buffer = []
            else:
                # Add to buffer
                if current_section == "technical_assumptions":
                    # Only add bullet points
                    if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                        buffer.append(line.lstrip("-•* ").strip())
                elif current_section == "headline":
                    # If we are in headline mode and encounter a line that isn't a header, assume it's the headline text
                    buffer.append(line)
                else:
                    buffer.append(line)
        
        # Flush last buffer
        valid_content = [b for b in buffer if b]
        if valid_content:
            if current_section == "technical_assumptions":
                sections[current_section] = valid_content
            else:
                sections[current_section] = " ".join(valid_content).strip()

        return sections

    def _generate_fallback(self, contract: dict) -> dict:
        """
        Deterministic fallback template (Spec 6.3).
        """
        c = contract["workload_commitment"]
        p = contract["policy"]
        
        return {
            "headline": f"{contract['review_mode']['mode_label']} — {contract['review_mode']['capacity_status']}",
            "impact": f"{contract['decision_basis']['model_statement']} {contract['decision_basis']['how_thresholds_were_set']}",
            "workload_commitment": f"You are committed to reviewing {c['p0_cases']} strict P0 cases over the {c['window_label']}. {c['ordering_guarantee']}",
            "why_this_is_safe": contract['safety_statement']['why_safe'],
            "recommended_next_step": "Review P0 cases immediately; they are your verified priority.",
            "technical_assumptions": [
                f"Review Window: {c['window_label']}",
                f"Team Capacity: {contract['decision_basis']['inputs_used']['team_capacity_cases_per_day']}/day",
                f"Thresholds: P0 > {p['thresholds']['p0']}, P1 > {p['thresholds']['p1']}"
            ]
        }
