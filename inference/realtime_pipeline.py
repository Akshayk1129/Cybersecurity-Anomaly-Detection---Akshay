# =============================================================================
# Cybersecurity Anomaly Detection - UEBA System
# Module: Real-Time Inference Pipeline
# =============================================================================
"""
API-ready class for real-time inference of single cybersecurity events.

Loads all pre-trained artifacts (profiles, encoders, Isolation Forest, LightGBM)
and streams a single raw event dictionary through the exact same processing
pipeline used in batch mode, returning a scored JSON response.

Usage:
    from inference.realtime_pipeline import UEBARealTimePipeline
    
    pipeline = UEBARealTimePipeline("config/config.yaml")
    result = pipeline.predict({
        "timestamp": "2024-03-20 14:00:00",
        "entity_id": "usr_999",
        "department": "Engineering",
        "login_status": "Success",
        # ... other raw fields
    })
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from feature_engineering.feature_engineer import FeatureEngineer
from models.baseline.baseline_profiler import BaselineProfiler
from preprocessing.data_preprocessor import DataPreprocessor
from models.risk_scoring.risk_engine import RiskScoringEngine
from explainability.explainability_engine import ExplainabilityEngine
from incident_response.response_engine import IncidentResponseEngine
from utils.logger import get_logger

logger = get_logger(__name__)


class UEBARealTimePipeline:
    """End-to-End Real-Time Inference Pipeline for UEBA."""

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        logger.info("Initializing UEBARealTimePipeline...")
        
        # 1. Initialize Pipeline Components
        self.feature_engineer = FeatureEngineer(config_path)
        self.profiler = BaselineProfiler(config_path)
        self.preprocessor = DataPreprocessor(config_path)
        self.risk_engine = RiskScoringEngine(config_path)
        self.explainability_engine = ExplainabilityEngine(config_path)
        self.response_engine = IncidentResponseEngine(config_path)
        
        # 2. Load Artifacts
        self._load_artifacts()
        logger.info("UEBARealTimePipeline initialized successfully.")

    def _load_config(self, config_path: str) -> dict:
        resolved = Path(_PROJECT_ROOT) / config_path
        if not resolved.exists():
            resolved = Path(config_path)
        with open(resolved, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_artifacts(self) -> None:
        """Load all pre-computed profiles and models from disk."""
        models_dir = Path(_PROJECT_ROOT) / self.config["paths"]["saved_models_dir"]
        
        # Load Profiles
        entity_prof_path = models_dir / "entity_profiles.joblib"
        dept_prof_path = models_dir / "department_profiles.joblib"
        if entity_prof_path.exists():
            self.profiler.profiles = joblib.load(str(entity_prof_path))
        if dept_prof_path.exists():
            self.profiler.department_profiles = joblib.load(str(dept_prof_path))
            
        # Load Preprocessor Transformers
        self.preprocessor.load_transformers()
        
        # Load Models
        iforest_path = models_dir / "isolation_forest.joblib"
        if iforest_path.exists():
            self.iforest = joblib.load(str(iforest_path))
        else:
            raise FileNotFoundError(f"Missing Isolation Forest at {iforest_path}")
            
        classifier_path = models_dir / "attack_classifier.joblib"
        if classifier_path.exists():
            self.attack_classifier = joblib.load(str(classifier_path))
        else:
            raise FileNotFoundError(f"Missing Attack Classifier at {classifier_path}")

        # The ExplainabilityEngine loads its own tree explainers upon init.

    def predict(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process and score a single raw event.
        
        Args:
            event: A dictionary representing a single raw access log.
            
        Returns:
            Dictionary containing Risk Score, Level, Classifications, and Explanation.
        """
        # 1. Convert to DataFrame
        df_raw = pd.DataFrame([event])
        
        if "timestamp" in df_raw.columns:
            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
        
        # Ensure event_id exists (required by feature engineer for indexing sometimes)
        if "event_id" not in df_raw.columns:
            df_raw["event_id"] = "rt_" + str(pd.Timestamp.now().timestamp())
            
        # 2. Feature Engineering (Temporal, rolling approximations)
        # Note: Rolling metrics on a single row without a state cache will approximate to 1.
        df_fe = self.feature_engineer.run(df_raw)
        
        # 3. Baseline Deviations (Apply loaded profiles)
        df_profiled = self.profiler.compute_profile_deviations(df_fe)
        
        # 4. Preprocessing (Encode & Scale)
        df_features = self.preprocessor.transform(df_profiled)
        
        # Clean numeric alignment
        X = df_features.select_dtypes(include=[np.number])
        
        # Ensure feature columns align EXACTLY with model expected features
        if hasattr(self.iforest, 'feature_names_in_'):
            expected_cols = self.iforest.feature_names_in_
            for col in expected_cols:
                if col not in X.columns:
                    X[col] = 0
            X = X[expected_cols]
        
        # 5. Anomaly Detection
        pred_if = self.iforest.predict(X)[0]
        raw_score = self.iforest.decision_function(X)[0]
        
        # Normalise anomaly score 0-100 (approximated for single event if we don't have min/max bounds)
        # Ideally, we should save min_s/max_s from training. Here we approximate bound logic.
        # Isolation Forest decision function usually falls between -0.5 and 0.5.
        min_s, max_s = -0.3, 0.2  # Approx bounds from previous logs
        risk_score_base = 100 * (1 - (raw_score - min_s) / (max_s - min_s))
        risk_score_base = max(0, min(100, risk_score_base))
        
        # 6. Attack Classification
        attack_class = "Normal"
        if pred_if == -1:
            attack_class = self.attack_classifier.predict(X)[0]
            
        # 7. Risk Engine
        # Requires base 0-100 anomaly score and string attack prediction
        risk_df = self.risk_engine.run(
            df_profiled, 
            np.array([risk_score_base]), 
            np.array([attack_class])
        )
        final_risk_score = float(risk_df["risk_score"].iloc[0])
        final_risk_level = risk_df["risk_level"].iloc[0]
        contributors = risk_df["risk_contributors"].iloc[0]
        
        # 8. Explainability
        explanation = "Normal behavior"
        if pred_if == -1:
            try:
                # We need to tell explainability engine which model to explain.
                # It handles it internally based on predicted_attack_type.
                explanation_str = self.explainability_engine.explain_event(
                    X, top_k=3, predicted_attack_type=attack_class if attack_class != "Normal" else None
                )
                explanation = explanation_str
            except Exception as e:
                explanation = f"Could not generate explanation: {str(e)}"
                
        # 9. Format Response
        response = {
            "event_id": df_raw["event_id"].iloc[0],
            "entity_id": df_raw.get("entity_id", pd.Series(["Unknown"])).iloc[0],
            "is_anomaly": bool(pred_if == -1),
            "attack_classification": attack_class,
            "risk_score": round(final_risk_score, 2),
            "risk_level": final_risk_level,
            "risk_contributors": contributors,
            "explanation": explanation,
            "recommended_actions": self.response_engine.get_actions(attack_class, final_risk_level)
        }
        
        return response

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    
    # Simple Mock Event test
    pipeline = UEBARealTimePipeline()
    
    mock_event = {
        "timestamp": "2024-03-20 03:00:00",
        "entity_id": "usr_test123",
        "entity_type": "User",
        "department": "Engineering",
        "login_status": "Success",
        "login_hour": 3,
        "is_off_hours": True,
        "geo_location": "US-West",
        "new_location": True,
        "new_device": True,
        "resource_accessed": "AdminConsole",
        "resource_sensitivity": "Critical",
        "session_duration_min": 10.0,
        "bytes_uploaded": 5000000.0,
        "bytes_downloaded": 1000.0,
        "failed_attempts": 0,
        "geo_velocity_kmph": 0.0,
        "device_fingerprint": "win_chrome_99",
        "protocol": "HTTPS",
        "auth_method": "MFA"
    }
    
    print("Scoring event...")
    result = pipeline.predict(mock_event)
    print(json.dumps(result, indent=2))
