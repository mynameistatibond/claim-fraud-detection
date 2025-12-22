import os
import json
import logging
import re
import requests
from llm_explainer import get_llm_config, call_openai_format_api, robust_api_call

logger = logging.getLogger(__name__)

class DecisionContractBuilder:
    """
    Deterministic builder for the Decision Contract.
    Enforces all arithmetic and logic rules from Tech Spec 3.3.
    """
    @staticmethod
    @staticmethod
    def build(triage_result, scored_df, review_window_days, team_size, review_time_mins, appetite_str, current_backlog_cases=0):
        # 1. Deterministic Computation Rules
        team_capacity_cases_per_day = int((team_size * 480) / review_time_mins)
        
        p0_count = triage_result["summary"]["p0_count"]
        # Handle new keys safely (or assume exists from new TriageAgent)
        p1_count = triage_result["summary"].get("p1_count", triage_result["workload_summary"]["assigned_work"]["p1"])
        p2_count = triage_result["summary"].get("p2_count", 0)
        
        # Capacity Equivalent Demand: p0 (1.0) + p1 (0.6) + p2 (0.2)
        demand = p0_count + (0.6 * p1_count) + (0.2 * p2_count)
        
        # Effective Capacity (deducting backlog)
        total_window_capacity = (team_capacity_cases_per_day * review_window_days) - current_backlog_cases
        total_window_capacity = max(0, total_window_capacity)
        
        if total_window_capacity > 0:
            cap_ratio = demand / total_window_capacity
        else:
            cap_ratio = 999.0
        
        if cap_ratio <= 0.9:
            capacity_status = "spare capacity"
        elif cap_ratio <= 1.1:
            capacity_status = "at capacity"
        else:
            capacity_status = "backlog risk"

        # Detailed Workload Math (Per Person)
        # Cases assigned vs Capacity (Count P0 + P1 fully + 50% P2 for headcount planning?)
        # Or Just P0 + P1? User says P2 is "Maybe".
        # Let's count them all but rely on recommended next step to prioritize.
        total_assigned = p0_count + p1_count + p2_count
        cases_per_person = round(total_assigned / team_size, 1)
        hours_per_person_needed = round((cases_per_person * review_time_mins) / 60, 1)
        
        # Available hours per person (assuming 8h day)
        available_hours_per_person = review_window_days * 8
        utilization_pct = round((hours_per_person_needed / available_hours_per_person) * 100, 1) if available_hours_per_person > 0 else 100

        # P0 Specifics for Overload Context
        p0_cases_per_person = round(p0_count / team_size, 1)
        p0_hours_per_person = round((p0_cases_per_person * review_time_mins) / 60, 1)
        hours_remaining_after_p0 = round(available_hours_per_person - p0_hours_per_person, 1)

        # Non-P0 Demand
        p1_p2_cases = p1_count + p2_count
        p1_p2_hours_needed = round(((p1_p2_cases / team_size) * review_time_mins) / 60, 1)

        workload_analysis = {
            "cases_per_person": cases_per_person,
            "hours_needed_per_person": hours_per_person_needed,
            "available_hours_per_person": available_hours_per_person,
            "utilization_pct": utilization_pct,
            "p0_hours_per_person": p0_hours_per_person,
            "hours_remaining_after_p0": max(0, hours_remaining_after_p0),
            "p1_p2_hours_needed": p1_p2_hours_needed,
            "is_overloaded": utilization_pct > 100,
            "recommendation": "Free time available" if utilization_pct < 85 else ("Balanced workload" if utilization_pct < 105 else "Overload risk")
        }

        # Window Label
        if review_window_days == 1:
            window_label = "today"
        else:
            window_label = f"next {review_window_days} days"

        # Mode Label
        mode_label = f"{'Strict' if appetite_str == 'Conservative' else ('Broad' if appetite_str == 'Aggressive' else 'Standard')} review mode"

        # 2. Construct Contract
        contract = {
            "contract_version": "1.1",
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
                    "team_capacity_cases_per_day": team_capacity_cases_per_day,
                    "existing_backlog": current_backlog_cases
                },
                "how_thresholds_were_set": "Agentic AI set 3 dynamic thresholds (P0, P1, P2) to optimize fraud capture within operational capacity."
            },
            "workload_commitment": {
                "window_label": window_label,
                "p0_cases": p0_count,
                "p1_cases": p1_count,
                "p2_cases": p2_count,
                "ordering_guarantee": "Strict ordering: P0 -> P1 -> P2. Review P2 only if time remains."
            },
            "workload_analysis": workload_analysis,
            "policy": {
                "thresholds": triage_result["decision_policy"]["thresholds"],
                "p0_strictness_statement": "P0 thresholds are strict to ensure the most credible fraud cases are covered.",
                "p1_flex_statement": "P1/P2 expand or shrink based on available capacity and fraud score distribution."
            },
            "safety_statement": {
                "why_safe": "Claims below P2 Threshold (or < 0.25) are explicitly ignored as noise."
            }
        }
        return contract

class ExplanationAgent:
    def __init__(self):
        self.llm_config = get_llm_config()
        if self.llm_config:
            logger.info(f"ExplanationAgent initialized with provider: {self.llm_config.get('provider')}")
        else:
            logger.warning("ExplanationAgent initialized with NO LLM config.")

    def generate_explanation(self, decision_contract: dict) -> dict:
        """
        Generates the explanation text, validates it, and parses it for the UI.
        Returns a dict matching the UI schema with metadata.
        """
        if not self.llm_config:
            return self._generate_fallback(decision_contract, reason="No LLM Config")

        system_prompt = (
            "You are a senior claims operations lead writing the official explanation of a decision that has already been made.\n\n"
            "Rules:\n"
            "- Use ONLY the provided DecisionContract JSON.\n"
            "- Do NOT reference UI elements, panels, settings, buttons, links, or 'advanced views'.\n"
            "- Do NOT use the word 'audit'.\n"
            "- Use the provided Workload Analysis metrics. Do not invent new ones.\n"
            "- Be explicit, human, and accountable.\n"
            "- **Distinction:** Clearly state that the 'ML Model' provided the scores, but the 'Agentic AI' decided the thresholds and capacity allocation.\n"
            "- P0 must be described as strict and the first priority. P1/P2 are optional depending on capacity.\n"
            "- **IF IS_OVERLOADED (utilization > 100%):** You MUST explicitly state that P0 cases alone will take X hours per person. Recommend finishing P0 first. State clearly how many hours (if any) remain for P1.\n"
            "- **IF NOT OVERLOADED:** Compare 'Hours Remaining after P0' vs 'Hours Required for P1/P2' to confirm fit.\n"
            "- Provide exactly one recommended next step.\n"
            "- Output the explanation in the specified section structure only, using plain text."
        )

        user_prompt = f"""
        Write the user-facing explanation using the DecisionContract below.

        Use exactly this structure:
        1) Headline
        2) How this decision was made
        3) Workload commitment (include ordering guarantee)
        4) Team Workload Analysis (breakdown per person and utilization recommendation)
        5) Why this is safe
        6) What to do next
        7) System assumptions (bullet list)

        DecisionContract:
        {json.dumps(decision_contract, indent=2)}
        """

        try:
            # Check format from config (added in llm_explainer)
            api_format = self.llm_config.get("format", "huggingface")
            text = ""

            if api_format == "openai":
                # GROQ / OPENROUTER PATH - Using Robust Retry Helper
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                try:
                    text = robust_api_call(self.llm_config, messages, temperature=0.1)
                except Exception as e:
                    logger.error(f"LLM Robust Call Failed: {e}")
                    return self._generate_fallback(decision_contract, reason=str(e))
                
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
                        return self._generate_fallback(decision_contract, reason="Invalid HF Response")
                else:
                    logger.error(f"LLM Error {response.status_code}: {response.text}")
                    return self._generate_fallback(decision_contract, reason=f"API Error {response.status_code}")

            # Validate and Parse (Shared)
            if text and self._validate_text(text, decision_contract):
                parsed = self._parse_to_ui_schema(text)
                parsed["meta"] = {
                    "source": "LLM",
                    "provider": self.llm_config.get('provider', 'unknown'),
                    "model": self.llm_config.get('model', 'unknown')
                }
                return parsed
            else:
                logger.warning(f"LLM output failed validation. Text: {text[:100]}...")
                return self._generate_fallback(decision_contract, reason="Validation Failed")

        except Exception as e:
            logger.error(f"Explanation Agent Exception: {e}")
            return self._generate_fallback(decision_contract, reason=f"Exception: {str(e)}")

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
        forbidden = ["audit", "panel", "settings", "click", "link", "advanced view", "optimization"]
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
            "headline": "headline",
            "how this decision was made": "impact",
            "workload commitment": "workload_commitment",
            "team workload analysis": "workload_commitment", # MERGE into workload commitment
            "why this is safe": "why_this_is_safe",
            "what to do next": "recommended_next_step",
            "system assumptions": "technical_assumptions"
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            lower_line = re.sub(r'^\d+[\).]\s*', '', line.lower().replace(":", "")).strip()
            
            if lower_line in header_map:
                valid_content = [b for b in buffer if b]
                if valid_content:
                    if current_section == "technical_assumptions":
                        sections[current_section] = valid_content
                    elif current_section == "workload_commitment" and header_map[lower_line] == "workload_commitment":
                         # We are APPENDING to the same section (Analysis merged into Commitment)
                         # So flush buffer to existing content with a newline
                         existing = sections[current_section]
                         new_text = " ".join(valid_content).strip()
                         sections[current_section] = f"{existing}\n\n{new_text}".strip()
                    else:
                        sections[current_section] = " ".join(valid_content).strip()
                
                # Switch section
                current_section = header_map[lower_line]
                buffer = []
            else:
                if current_section == "technical_assumptions":
                    if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                        buffer.append(line.lstrip("-•* ").strip())
                else:
                    buffer.append(line)
        
        # Flush last buffer
        valid_content = [b for b in buffer if b]
        if valid_content:
            if current_section == "technical_assumptions":
                sections[current_section] = valid_content
            elif current_section == "workload_commitment":
                # Special flush for merge scenario
                 existing = sections[current_section]
                 new_text = " ".join(valid_content).strip()
                 if existing:
                     sections[current_section] = f"{existing}\n\n{new_text}".strip()
                 else:
                     sections[current_section] = new_text
            else:
                sections[current_section] = " ".join(valid_content).strip()

        return sections

    def _generate_fallback(self, contract: dict, reason: str = "Unknown") -> dict:
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
            ],
            "meta": {
                "source": "Fallback",
                "reason": reason,
                "provider": "Internal Rule Engine"
            }
        }
