
import pandas as pd
import numpy as np
import json
import logging
import requests
import time
from llm_explainer import get_llm_config

logger = logging.getLogger(__name__)

class FraudTriageAgent:
    """
    Agentic decision layer for allocating human attention efficiently.
    Uses LLM reasoning to determine triage thresholds based on batch distribution and capacity.
    """
    
    # Constants
    UI_FULL_DISPLAY_THRESHOLD = 50
    UI_PARTIAL_DISPLAY_THRESHOLD = 500
    
    # Explicit Decision Policy (Audit Trail)
    THRESHOLDS_BY_APPETITE = {
        "Conservative": {"p0": 0.60, "p1": 0.40},
        "Balanced":     {"p0": 0.40, "p1": 0.20},
        "Aggressive":   {"p0": 0.20, "p1": 0.10}
    }
    
    def __init__(self):
        self.llm_config = get_llm_config()

    def triage_batch(self, scored_df: pd.DataFrame, batch_size: int, 
                     team_size: int = 5, review_time_mins: int = 20,
                     risk_appetite: str = "balanced") -> dict:
        """
        Main entry point for triaging a scored batch.
        """
        # 1. Determine Strategy
        strategy = self._determine_strategy(batch_size)
        
        # 2. Compute Capacity
        # Daily minutes / review time * team size
        # Assuming 8 hour work day (480 mins)
        daily_capacity = int((team_size * 480) / max(1, review_time_mins))
        
        # 3. Prepare Data
        df = scored_df.copy()
        df['probability'] = df['probability'].astype(float)
        df = df.sort_values(by='probability', ascending=False).reset_index(drop=True)
        df['priority_rank'] = df.index + 1
        
        # 4. Agentic Decision (LLM)
        # We try to get thresholds from LLM. If fail, fall back to rule-based.
        llm_decision = self._call_llm_triage(df, batch_size, daily_capacity, risk_appetite)
        
        if llm_decision:
            # Apply LLM Thresholds
            df = self._apply_thresholds(df, llm_decision['p0_threshold'], llm_decision['p1_threshold'])
            rationale = llm_decision['rationale']
        else:
            # Fallback Rule-Based (Appetite-Aware)
            logger.warning("LLM Triage failed, reverting to rule-based.")
            df = self._allocate_tiers_fallback(df, strategy, daily_capacity, risk_appetite)
            p0_count = len(df[df['triage_decision'] == 'P0_IMMEDIATE'])
            rationale = self._generate_fallback_summary(batch_size, strategy, p0_count)

        # 5. Determine UI Rows
        ui_rows_df = self._select_ui_rows(df, strategy)
        
        # Format output
        def fmt_row(r):
            return r.to_dict()
            
        ui_rows = [fmt_row(r) for _, r in ui_rows_df.iterrows()]
        
        p0_count = len(df[df['triage_decision'] == 'P0_IMMEDIATE'])
        
        # Active Thresholds (for Explainability)
        active_thresholds = self.THRESHOLDS_BY_APPETITE.get(risk_appetite, self.THRESHOLDS_BY_APPETITE["Balanced"])
        
        return {
            "rows": ui_rows,
            "summary": {
                "total_rows": batch_size,
                "ui_displayed_count": len(ui_rows),
                "p0_count": p0_count,
                "rationale": rationale,
                "strategy": strategy,
                "capacity_used": daily_capacity
            },
            "decision_policy": {
                "risk_appetite": risk_appetite,
                "thresholds": active_thresholds,
                "method": "LLM_REFINED" if llm_decision else "RULE_BASED_FALLBACK"
            },
            "full_dataset_available": True,
            "full_df": df # Return full dataframe for CSV export
        }

    def _determine_strategy(self, batch_size: int) -> str:
        if batch_size <= self.UI_FULL_DISPLAY_THRESHOLD:
            return "FULL"
        elif batch_size <= self.UI_PARTIAL_DISPLAY_THRESHOLD:
            return "PARTIAL"
        else:
            return "TRIAGE_DASHBOARD"

    def _call_llm_triage(self, df: pd.DataFrame, batch_size: int, capacity: int, risk_appetite: str):
        if not self.llm_config:
            return None
            
        # Compute Stats for Prompt
        probs = df['probability'].values
        if len(probs) == 0: return None
        
        stats = {
            "count": len(probs),
            "p95": round(float(np.percentile(probs, 95)), 3),
            "p75": round(float(np.percentile(probs, 75)), 3),
            "p50": round(float(np.percentile(probs, 50)), 3),
            "mean": round(float(np.mean(probs)), 3),
            "max": round(float(np.max(probs)), 3)
        }
        
        # Construct Prompt
        system_prompt = (
            "You are an expert Fraud Operations Manager. Your goal is to triage a batch of insurance claims "
            "to maximize fraud detection within limited human review capacity.\n"
            "Output must be valid JSON only."
        )
        
        user_prompt = f"""
        Analyze this batch of insurance claims and set score thresholds for review.
        
        Context:
        - Total Claims: {batch_size}
        - Team Capacity: ~{capacity} reviews per day
        - Risk Appetite: {risk_appetite}
        
        Batch Statistics (Risk Scores 0.0 - 1.0):
        {json.dumps(stats, indent=2)}
        
        Task:
        1. Determine the 'p0_threshold' (Score minimum for Immediate Review). 
           - Ideally, the count of P0 claims should be close to or less than Team Capacity.
           - But do not set the threshold too low (<0.4) just to fill capacity; suppress noise.
        2. Determine 'p1_threshold' (Queue).
        3. Provide a 'rationale' (1-2 sentences) explaining your decision based on the stats and capacity.
        
        Response Format (JSON):
        {{
            "p0_threshold": 0.XX,
            "p1_threshold": 0.XX,
            "rationale": "..."
        }}
        """
        
        try:
            headers = {
                "Authorization": f"Bearer {self.llm_config['key']}",
                "Content-Type": "application/json"
            }
            
            # Provider specific adjustments
            if self.llm_config['provider'] == "openrouter":
                headers["HTTP-Referer"] = "https://fraud-detector.internal"
            
            payload = {
                "model": self.llm_config['model'],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"} if self.llm_config['format'] == "openai" else None
            }
            
            response = requests.post(self.llm_config['url'], headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                res_json = response.json()
                content = res_json['choices'][0]['message']['content']
                # Clean markdown if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                    
                return json.loads(content)
            else:
                logger.error(f"LLM Error: {response.text}")
                return None
        except Exception as e:
            logger.error(f"LLM Exception: {e}")
            return None

    def _apply_thresholds(self, df: pd.DataFrame, p0_thresh: float, p1_thresh: float) -> pd.DataFrame:
        decisions = []
        for i, row in df.iterrows():
            score = row['probability']
            if score >= p0_thresh:
                decisions.append("P0_IMMEDIATE")
            elif score >= p1_thresh:
                decisions.append("P1_REVIEW_IF_CAPACITY")
            else:
                decisions.append("P2_MONITOR")
        df['triage_decision'] = decisions
        return df

    def _allocate_tiers_fallback(self, df: pd.DataFrame, strategy: str, capacity: int, risk_appetite: str) -> pd.DataFrame:
        # Appetite-Aware Logic
        thresholds = self.THRESHOLDS_BY_APPETITE.get(risk_appetite, self.THRESHOLDS_BY_APPETITE["Balanced"])
        t_p0 = thresholds["p0"]
        t_p1 = thresholds["p1"]
        
        scores = df['probability'].values
        decisions = []
        for i, score in enumerate(scores):
            rank = i + 1
            if rank <= capacity and score >= t_p0:
                decisions.append("P0_IMMEDIATE")
            elif rank <= capacity * 2 and score >= t_p1:
                decisions.append("P1_REVIEW_IF_CAPACITY")
            else:
                decisions.append("P2_MONITOR")
        df['triage_decision'] = decisions
        return df

    def _generate_fallback_summary(self, batch_size: int, strategy: str, p0: int) -> str:
        if strategy == "FULL":
            return f"Showing all {batch_size} claims. {p0} identified for immediate review based on capacity."
        else:
            return f"Note: Out of {batch_size} claims, {p0} were prioritized for review. Displaying top cases only."

    def _select_ui_rows(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        if strategy == "FULL":
            return df
        elif strategy == "PARTIAL":
            limit = min(100, int(len(df) * 0.2))
            limit = max(limit, 10) 
            return df.head(limit)
        else:
            return df.head(50)
