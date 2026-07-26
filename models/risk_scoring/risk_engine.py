# =============================================================================
# Cybersecurity Anomaly Detection - UEBA System
# Module: Enterprise Risk Scoring Engine
# =============================================================================
"""
Enterprise Risk Scoring Engine (Enhancement 1).

Combines the raw anomaly score from the Isolation Forest with contextual
business and threat signals to produce a normalized Risk Score (0-100),
a Risk Level, and a breakdown of contributing factors.

Usage:
    from models.risk_scoring.risk_engine import RiskScoringEngine
    risk_engine = RiskScoringEngine(config_path="config/config.yaml")
    scored_df = risk_engine.run(raw_events, anomaly_scores, attack_predictions)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)


class RiskScoringEngine:
    """Combines multiple signals into a final unified Risk Score (0-100)."""

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.config = self._load_config(config_path)
        
        # Load weights and normalize them so they sum to 1.0 internally
        raw_weights = self.config.get("risk", {}).get("weights", {})
        if not raw_weights:
            logger.warning("No risk weights found in config. Using defaults.")
            raw_weights = {
                "anomaly_confidence": 35,
                "resource_sensitivity": 20,
                "failed_attempts": 10,
                "new_device": 10,
                "new_location": 5,
                "geo_velocity": 5,
                "off_hours": 5,
                "privilege_level": 10,
            }
            
        total_weight = sum(raw_weights.values())
        self.weights = {k: v / total_weight for k, v in raw_weights.items()}
        self.max_score = 100.0
        
        # Risk thresholds
        levels = self.config.get("risk", {}).get("levels", {})
        self.low_max = levels.get("low_max", 39)
        self.medium_max = levels.get("medium_max", 69)
        self.high_max = levels.get("high_max", 89)

        logger.info(
            "RiskScoringEngine initialized. Total weight normalized from %d. "
            "Thresholds: Low<=%d, Med<=%d, High<=%d",
            total_weight, self.low_max, self.medium_max, self.high_max
        )

    @staticmethod
    def _load_config(config_path: str) -> dict:
        resolved = Path(_PROJECT_ROOT) / config_path
        if not resolved.exists():
            resolved = Path(config_path)
        with open(resolved, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _get_risk_level(self, score: float) -> str:
        """Map numerical score to string risk level."""
        if score <= self.low_max:
            return "Low"
        elif score <= self.medium_max:
            return "Medium"
        elif score <= self.high_max:
            return "High"
        else:
            return "Critical"

    def run(
        self,
        events: pd.DataFrame,
        anomaly_scores: np.ndarray,
        attack_predictions: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Calculate unified risk score and contributing factors for each event.
        
        Args:
            events: Raw or enriched events containing context columns.
            anomaly_scores: 1D array of base anomaly scores [0, 100].
            attack_predictions: Optional 1D array of attack type strings.
            
        Returns:
            DataFrame containing 'risk_score', 'risk_level', and 'risk_contributors'.
        """
        logger.info("=" * 70)
        logger.info("  RISK SCORING ENGINE - START")
        logger.info("=" * 70)
        logger.info("Scoring %d events...", len(events))
        
        results = []
        
        for i in range(len(events)):
            row = events.iloc[i]
            base_anomaly_score = anomaly_scores[i]
            
            total_score = 0.0
            contributors: List[str] = []
            
            # 1. Anomaly Confidence (Isolation Forest)
            # anomaly_scores are already 0-100.
            w_anomaly = self.weights.get("anomaly_confidence", 0) * self.max_score
            pts_anomaly = (base_anomaly_score / 100.0) * w_anomaly
            if pts_anomaly > 0:
                total_score += pts_anomaly
                if pts_anomaly >= (w_anomaly * 0.5):
                    contributors.append(f"+{int(pts_anomaly)} High anomaly confidence")
                elif pts_anomaly > 0:
                    contributors.append(f"+{int(pts_anomaly)} Moderate anomaly confidence")

            # 2. Resource Sensitivity
            sens = str(row.get("resource_sensitivity", "Low")).lower()
            w_res = self.weights.get("resource_sensitivity", 0) * self.max_score
            pts_res = 0.0
            if sens == "critical":
                pts_res = w_res
                contributors.append(f"+{int(pts_res)} Critical resource")
            elif sens == "high":
                pts_res = w_res * 0.75
                contributors.append(f"+{int(pts_res)} High sensitivity resource")
            elif sens == "medium":
                pts_res = w_res * 0.25
                # Usually don't flag medium in contributors unless it's the only thing, but we add it to score
            total_score += pts_res

            # 3. Privilege Level
            priv = str(row.get("privilege_level", "User")).lower()
            w_priv = self.weights.get("privilege_level", 0) * self.max_score
            pts_priv = 0.0
            if priv == "admin":
                pts_priv = w_priv
                contributors.append(f"+{int(pts_priv)} Admin privilege")
            elif priv == "poweruser":
                pts_priv = w_priv * 0.7
                contributors.append(f"+{int(pts_priv)} PowerUser privilege")
            elif priv == "service":
                pts_priv = w_priv * 0.5
            total_score += pts_priv

            # 4. Off-Hours Activity
            is_off = row.get("is_off_hours", False)
            if str(is_off).lower() == "true" or is_off is True or is_off == 1:
                w_off = self.weights.get("off_hours", 0) * self.max_score
                total_score += w_off
                contributors.append(f"+{int(w_off)} Off-hours login")

            # 5. New Device
            new_dev = row.get("new_device", False)
            if str(new_dev).lower() == "true" or new_dev is True or new_dev == 1:
                w_dev = self.weights.get("new_device", 0) * self.max_score
                total_score += w_dev
                contributors.append(f"+{int(w_dev)} New device")

            # 6. New Location
            new_loc = row.get("new_location", False)
            if str(new_loc).lower() == "true" or new_loc is True or new_loc == 1:
                w_loc = self.weights.get("new_location", 0) * self.max_score
                total_score += w_loc
                contributors.append(f"+{int(w_loc)} New location")

            # 7. Failed Attempts (scaled up to max weight if >= 5)
            fails = float(row.get("failed_attempts", 0))
            if fails > 0:
                w_fail = self.weights.get("failed_attempts", 0) * self.max_score
                pts_fail = min(fails / 5.0, 1.0) * w_fail
                total_score += pts_fail
                if fails >= 3:
                    contributors.append(f"+{int(pts_fail)} Multiple failed attempts ({int(fails)})")

            # 8. Geo Velocity (Impossible Travel)
            velocity = float(row.get("geo_velocity_kmph", 0))
            if velocity > 500:
                w_vel = self.weights.get("geo_velocity", 0) * self.max_score
                total_score += w_vel
                contributors.append(f"+{int(w_vel)} Impossible travel ({int(velocity)} km/h)")
                
            # If Attack classification is provided and it's not normal
            if attack_predictions is not None:
                atk = str(attack_predictions[i])
                if atk != "Normal":
                    # We can add a flat bump for confirmed attack patterns
                    # Note: We didn't explicitly add this to the weight divisor, 
                    # so this can push the score up closer to 100 if it was lagging.
                    attack_bump = 15.0
                    total_score = min(total_score + attack_bump, 100.0)
                    contributors.append(f"+{int(attack_bump)} Suspected {atk} attack pattern")

            # Cap at 100
            final_score = min(max(total_score, 0.0), 100.0)
            
            results.append({
                "risk_score": final_score,
                "risk_level": self._get_risk_level(final_score),
                "risk_contributors": " | ".join(contributors)
            })

        df_results = pd.DataFrame(results)
        
        logger.info(
            "Risk Scoring complete. Levels: %s", 
            df_results["risk_level"].value_counts().to_dict()
        )
        logger.info("=" * 70)
        
        return df_results

# ---------------------------------------------------------------------------
# Simple module test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from feature_engineering.feature_engineer import FeatureEngineer
    # Dummy run to verify imports
    eng = RiskScoringEngine()
    print("RiskScoringEngine initialized successfully.")
