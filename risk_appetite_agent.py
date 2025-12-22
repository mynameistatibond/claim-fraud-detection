
import logging
import json
import requests
import numpy as np
import math
from typing import Dict, Any, Optional, List
from llm_explainer import get_llm_config

logger = logging.getLogger(__name__)

class RiskAppetiteAgent:
    """
    Deterministically derives review posture (Conservative/Balanced/Aggressive)
    from capacity, batch stats, and operational intent.
    """
    
    APPETITE_LEVELS = ["Conservative", "Balanced", "Aggressive"]
    
    # 0 = Conservative, 1 = Balanced, 2 = Aggressive
    APPETITE_MAP = {k: i for i, k in enumerate(APPETITE_LEVELS)}
    IDX_MAP = {i: k for i, k in enumerate(APPETITE_LEVELS)}

    def __init__(self):
        self.llm_config = get_llm_config()

    def decide_appetite(self, 
                        team_size: int,
                        review_time_mins: int,
                        batch_size: int,
                        batch_stats: Dict[str, Any],
                        operating_mode: str = "daily_ops",
                        review_window_days: int = 1,
                        current_backlog: Optional[int] = None) -> Dict[str, Any]:
        """
        Main entry point. Returns appetite decision dict.
        """
        # Validate Inputs
        if operating_mode not in ["daily_ops", "incident_sweep", "regulatory_audit"]:
            operating_mode = "daily_ops"
        
        # 1. Compute Capacity Signals
        signals = self._compute_signals(
            team_size, review_time_mins, batch_size, 
            review_window_days, current_backlog
        )
        
        # Inject Risk signals for Explainability
        if batch_stats:
            signals.update({
                "p95": batch_stats.get("p95"),
                "share_ge_0_3": batch_stats.get("share_ge_0_3")
            })
        
        # 2. Compute Deterministic Baseline
        baseline_appetite, baseline_reason = self._determine_baseline(
            signals, batch_stats, operating_mode
        )
        
        # 3. LLM Refinement (Optional)
        final_appetite = baseline_appetite
        final_rationale = [baseline_reason]
        used_fallback = False
        
        llm_result = self._call_llm_refinement(
            baseline_appetite, baseline_reason, 
            signals, batch_stats, operating_mode
        )
        
        if llm_result:
            # Apply Change Limiter (Max 1 step)
            final_appetite = self._apply_limiter(baseline_appetite, llm_result['risk_appetite'])
            final_rationale = llm_result.get('rationale', [baseline_reason])
        else:
            used_fallback = True
            
        return {
            "risk_appetite": final_appetite,
            "risk_appetite": final_appetite,
            "confidence": 0.8 if not used_fallback else 0.6,
            "confidence_type": "operational", # Not statistical
            "rationale": final_rationale,
            "signals": signals,
            "guardrails": {
                "used_fallback": used_fallback,
                "baseline_appetite": baseline_appetite,
                "previous_appetite": None # Stateless V1
            }
        }

    def _compute_signals(self, team_size, review_time, batch_size, window, backlog):
        daily_capacity = math.floor((team_size * 480) / max(1, review_time))
        effective_capacity = daily_capacity * window
        if backlog is not None:
            effective_capacity -= backlog
        
        effective_capacity = max(effective_capacity, 1) # Prevent div/0
        capacity_ratio = round(batch_size / effective_capacity, 2)
        
        return {
            "daily_capacity": daily_capacity,
            "effective_capacity": effective_capacity,
            "capacity_ratio": capacity_ratio,
            "batch_size": batch_size
        }

    def _determine_baseline(self, signals, stats, mode):
        ratio = signals['capacity_ratio']
        
        # Outlier Detection (Deterministic)
        # Assuming stats keys match what typical batch_ingest produces
        # We need flexible key access
        p95 = stats.get('p95', 0)
        share_high = stats.get('share_ge_0_3', 0) # Fallback key
        
        outliers_exist = (share_high >= 0.08 or p95 >= 0.45)
        
        # Core Matrix
        if ratio >= 4.0:
            appetite = "Conservative"
        elif ratio >= 0.9:
            appetite = "Balanced"
        else:
            # Low ratio (< 0.9)
            if outliers_exist:
                appetite = "Aggressive"
            else:
                appetite = "Balanced"
                
        # Operating Mode Bias
        idx = self.APPETITE_MAP[appetite]
        reason = f"Baseline based on Capacity Ratio {ratio}"
        
        if mode == "incident_sweep":
            if idx < 2: 
                idx += 1
                reason += " + Incident Sweep bias (Aggressive)"
        elif mode == "regulatory_audit":
            if idx > 0: 
                idx -= 1
                reason += " + Audit bias (Conservative)"
                
        return self.IDX_MAP[idx], reason

    def _call_llm_refinement(self, baseline, reason, signals, stats, mode):
        if not self.llm_config:
            return None
            
        system_prompt = (
            "You are an operational risk policy assistant. "
            "Your goal is to set the review posture (Conservative, Balanced, Aggressive). "
            "Output valid JSON only."
        )
        
        user_prompt = f"""
        Determine Risk Appetite.
        
        Baseline: {baseline} (Reason: {reason})
        Context:
        - Mode: {mode}
        - Capacity Ratio: {signals['capacity_ratio']} (Load vs Capacity)
        - Batch Stats: {json.dumps(stats, indent=2)}
        
        Rules:
        1. You may shift the baseline by MAX 1 step.
        2. Conservative: Prioritize precision, ignore noise.
        3. Aggressive: Prioritize recall, fill capacity.
        
        Output JSON:
        {{
            "risk_appetite": "...",
            "rationale": ["bullet 1", "bullet 2"]
        }}
        """
        
        try:
            # Simplified LLM call using requests similar to triage_agent
            headers = {
                "Authorization": f"Bearer {self.llm_config['key']}",
                "Content-Type": "application/json"
            }
            if self.llm_config['provider'] == "openrouter":
                headers["HTTP-Referer"] = "https://fraud-detector.internal"
                
            payload = {
                "model": self.llm_config['model'],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"} if self.llm_config['format'] == "openai" else None
            }
            
            response = requests.post(self.llm_config['url'], headers=headers, json=payload, timeout=8)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                if "```" in content:
                    content = content.split("```json")[1].split("```")[0].strip() if "```json" in content else content.split("```")[1].split("```")[0]
                
                result = json.loads(content)
                
                # Robustness check for rationale
                if "rationale" in result:
                     if not isinstance(result["rationale"], list):
                         result["rationale"] = [str(result["rationale"])]
                         
                return result
        except Exception as e:
            logger.warning(f"Risk Appetite LLM Failed: {e}")
            return None
            
        return None

    def _apply_limiter(self, baseline, proposed):
        # Enforce max 1 step change from baseline
        b_idx = self.APPETITE_MAP[baseline]
        if proposed not in self.APPETITE_MAP:
            return baseline
            
        p_idx = self.APPETITE_MAP[proposed]
        
        if abs(p_idx - b_idx) > 1:
            # Clamp
            if p_idx > b_idx: return self.IDX_MAP[b_idx + 1]
            else: return self.IDX_MAP[b_idx - 1]
            
        return proposed
