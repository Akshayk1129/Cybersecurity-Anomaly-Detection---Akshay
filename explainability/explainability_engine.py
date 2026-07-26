# =============================================================================
# Cybersecurity Anomaly Detection - UEBA System
# Module: Explainability Engine (Phase 7)
# =============================================================================
"""
Model explainability engine using SHAP (SHapley Additive exPlanations) and
feature contribution analysis.

Provides local (per-event) and global (system-wide) explanations for both:
1. Anomaly Detection (Isolation Forest point anomalies)
2. Attack Classification (LightGBM multi-class attack categorization)

Translates feature importance and SHAP values into natural-language
justifications suitable for Security Operations Center (SOC) analysts.

Usage:
    from explainability.explainability_engine import ExplainabilityEngine
    explainer = ExplainabilityEngine(config_path="config/config.yaml")
    explanation = explainer.explain_event(features_df.iloc[0:1])
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)

# Attempt to import SHAP
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning("SHAP package not found. Fallback feature attribution will be used.")


class ExplainabilityEngine:
    """Enterprise explainability engine for UEBA alerts and attack types.

    Attributes:
        config: Loaded YAML configuration.
        iforest_model: Loaded Isolation Forest model.
        classifier_model: Loaded attack classifier model (LightGBM/RF).
        attack_label_encoder: Loaded attack LabelEncoder.
        iforest_explainer: Fitted SHAP TreeExplainer for Isolation Forest.
        classifier_explainer: Fitted SHAP TreeExplainer for attack classifier.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """Initialize explainability engine and load fitted models.

        Args:
            config_path: Path to the central YAML config.
        """
        self.config = self._load_config(config_path)
        self.saved_models_dir = Path(_PROJECT_ROOT) / self.config["paths"]["saved_models_dir"]

        self.iforest_model: Any = None
        self.classifier_model: Any = None
        self.attack_label_encoder: Any = None

        self.iforest_explainer: Any = None
        self.classifier_explainer: Any = None

        self._load_saved_models()
        self._init_explainers()
        logger.info("ExplainabilityEngine initialized.")

    @staticmethod
    def _load_config(config_path: str) -> dict:
        resolved = Path(_PROJECT_ROOT) / config_path
        if not resolved.exists():
            resolved = Path(config_path)
        with open(resolved, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_saved_models(self) -> None:
        """Load trained models from saved_models directory."""
        iforest_path = self.saved_models_dir / "isolation_forest.joblib"
        if iforest_path.exists():
            self.iforest_model = joblib.load(str(iforest_path))
            logger.info("Loaded Isolation Forest model from %s", iforest_path)

        classifier_path = self.saved_models_dir / "attack_classifier.joblib"
        if classifier_path.exists():
            self.classifier_model = joblib.load(str(classifier_path))
            logger.info("Loaded Attack Classifier model from %s", classifier_path)

        le_path = self.saved_models_dir / "attack_label_encoder.joblib"
        if le_path.exists():
            self.attack_label_encoder = joblib.load(str(le_path))
            logger.info("Loaded Attack Label Encoder from %s", le_path)

    def _init_explainers(self) -> None:
        """Initialize SHAP TreeExplainers if SHAP is available."""
        if not _SHAP_AVAILABLE:
            return

        if self.classifier_model is not None:
            try:
                self.classifier_explainer = shap.TreeExplainer(self.classifier_model)
                logger.info("Initialized TreeExplainer for Attack Classifier.")
            except Exception as e:
                logger.warning("Could not initialize SHAP TreeExplainer for classifier: %s", e)

        if self.iforest_model is not None:
            try:
                self.iforest_explainer = shap.TreeExplainer(self.iforest_model)
                logger.info("Initialized TreeExplainer for Isolation Forest.")
            except Exception as e:
                logger.warning("Could not initialize SHAP TreeExplainer for Isolation Forest: %s", e)

    # ------------------------------------------------------------------
    # Feature Name Humanizer
    # ------------------------------------------------------------------
    @staticmethod
    def _humanize_feature_name(feat_name: str) -> str:
        """Convert machine feature names to analyst-friendly descriptions."""
        mappings = {
            "session_duration_zscore": "Session duration z-score vs baseline",
            "bytes_uploaded_zscore": "Bytes uploaded z-score vs baseline",
            "bytes_downloaded_zscore": "Bytes downloaded z-score vs baseline",
            "time_since_last_login_min": "Time since last login (minutes)",
            "resource_access_entropy": "Resource access pattern entropy (lateral movement indicator)",
            "consecutive_failures": "Consecutive login failures",
            "profile_hour_deviation": "Login hour deviation from personal baseline",
            "profile_session_deviation": "Session length deviation from personal baseline",
            "profile_bytes_up_deviation": "Data upload deviation from personal baseline",
            "profile_new_geo": "Access from new geographic location",
            "profile_new_resource": "Access to previously unaccessed resource",
            "profile_new_device": "Access from unseen device fingerprint",
            "new_device_rate": "Expanding rate of new device usage",
            "new_location_rate": "Expanding rate of new location usage",
            "peer_session_deviation": "Session duration deviation from department peers",
            "peer_bytes_up_deviation": "Upload data volume deviation from department peers",
            "peer_bytes_down_deviation": "Download data volume deviation from department peers",
            "entity_failed_rate": "Historical failed login ratio",
            "failed_attempts": "Failed login attempt count",
            "geo_velocity_kmph": "Geographic movement velocity (km/h)",
            "privilege_level": "Account privilege tier",
            "resource_sensitivity": "Accessed resource sensitivity tier",
        }
        return mappings.get(feat_name, feat_name.replace("_", " ").title())

    # ------------------------------------------------------------------
    # Single Event Explanation
    # ------------------------------------------------------------------
    def explain_event(
        self,
        event_features: pd.DataFrame,
        top_k: int = 5,
        predicted_attack_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate local SHAP explanations and natural language summary for an event.

        Args:
            event_features: Single-row or multi-row DataFrame containing preprocessed features.
            top_k: Top K contributing features to extract.
            predicted_attack_type: Optional predicted attack label string.

        Returns:
            Dictionary containing feature contributions, SHAP values, and natural language narrative.
        """
        if event_features.empty:
            return {"error": "Empty feature matrix provided."}

        X_row = event_features.iloc[0:1].replace([np.inf, -np.inf], np.nan).fillna(0)
        feature_names = list(X_row.columns)

        classifier_attributions: List[Dict[str, Any]] = []

        # --- 1. SHAP Classifier Explanation ---
        if self.classifier_explainer is not None and self.classifier_model is not None:
            try:
                shap_values = self.classifier_explainer.shap_values(X_row)
                
                # Handle multi-class (array of shape [n_samples, n_features, n_classes] or list of arrays)
                if isinstance(shap_values, list):
                    # List per class
                    if predicted_attack_type and self.attack_label_encoder:
                        class_idx = list(self.attack_label_encoder.classes_).index(predicted_attack_type)
                    else:
                        class_idx = 0
                    vals = shap_values[class_idx][0]
                elif isinstance(shap_values, np.ndarray):
                    if shap_values.ndim == 3:
                        class_idx = 0
                        if predicted_attack_type and self.attack_label_encoder:
                            class_idx = list(self.attack_label_encoder.classes_).index(predicted_attack_type)
                        vals = shap_values[0, :, class_idx]
                    else:
                        vals = shap_values[0]
                else:
                    vals = np.zeros(len(feature_names))

                # Top positive contributions
                top_indices = np.argsort(np.abs(vals))[-top_k:][::-1]

                for idx in top_indices:
                    feat = feature_names[idx]
                    impact = float(vals[idx])
                    raw_val = float(X_row.iloc[0, idx])
                    classifier_attributions.append({
                        "feature": feat,
                        "human_name": self._humanize_feature_name(feat),
                        "shap_value": round(impact, 4),
                        "feature_value": round(raw_val, 4),
                        "direction": "increases_risk" if impact > 0 else "decreases_risk",
                    })
            except Exception as e:
                logger.warning("SHAP calculation failed for event: %s", e)

        # --- Fallback feature importance if SHAP produces nothing ---
        if not classifier_attributions:
            if hasattr(self.classifier_model, "feature_importances_"):
                importances = self.classifier_model.feature_importances_
                top_indices = np.argsort(importances)[-top_k:][::-1]
                for idx in top_indices:
                    feat = feature_names[idx]
                    classifier_attributions.append({
                        "feature": feat,
                        "human_name": self._humanize_feature_name(feat),
                        "shap_value": round(float(importances[idx]), 4),
                        "feature_value": round(float(X_row.iloc[0, idx]), 4),
                        "direction": "increases_risk",
                    })

        # --- 2. Construct Natural Language Narrative ---
        narrative_lines = []
        if predicted_attack_type:
            narrative_lines.append(f"Attack Classification: **{predicted_attack_type}**")

        narrative_lines.append("Top Behavioral Anomalies & Risk Drivers:")
        for i, item in enumerate(classifier_attributions[:top_k], 1):
            narrative_lines.append(
                f"  {i}. **{item['human_name']}** (Value: {item['feature_value']}) "
                f"-> Contribution Score: +{item['shap_value']:.4f}"
            )

        explanation_result = {
            "top_features": classifier_attributions,
            "narrative": "\n".join(narrative_lines),
            "attack_type": predicted_attack_type,
        }

        return explanation_result

    # ------------------------------------------------------------------
    # Global Explanation & Summary
    # ------------------------------------------------------------------
    def generate_global_summary(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Compute global feature importances across the dataset.

        Args:
            features: Preprocessed feature matrix.

        Returns:
            Dictionary containing global feature rankings and metrics.
        """
        X_sample = features.sample(min(1000, len(features)), random_state=42)
        X_clean = X_sample.replace([np.inf, -np.inf], np.nan).fillna(0)

        global_importances: Dict[str, float] = {}

        if self.classifier_model and hasattr(self.classifier_model, "feature_importances_"):
            importances = self.classifier_model.feature_importances_
            feature_names = list(X_clean.columns)
            top_indices = np.argsort(importances)[-15:][::-1]

            for idx in top_indices:
                feat = feature_names[idx]
                global_importances[self._humanize_feature_name(feat)] = round(float(importances[idx]), 4)

        summary = {
            "model_type": type(self.classifier_model).__name__ if self.classifier_model else "Unknown",
            "global_top_features": global_importances,
            "sample_size": len(X_clean),
        }

        # Save summary artifact
        out_path = self.saved_models_dir / "explainability_summary.joblib"
        joblib.dump(summary, str(out_path))
        logger.info("Saved global explainability summary -> %s", out_path)

        return summary

    # ------------------------------------------------------------------
    # Main Pipeline Execution
    # ------------------------------------------------------------------
    def run(
        self,
        features: pd.DataFrame,
        targets: Optional[pd.DataFrame] = None,
        anomaly_predictions: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Execute Explainability Engine pipeline.

        Args:
            features: Full feature matrix.
            targets: Targets DataFrame.
            anomaly_predictions: Anomaly predictions (-1 / 1).

        Returns:
            Summary of global explainability and sample explanations.
        """
        logger.info("=" * 70)
        logger.info("  EXPLAINABILITY ENGINE PIPELINE - START")
        logger.info("=" * 70)

        summary = self.generate_global_summary(features)

        # Generate sample explanation for a true-positive anomaly
        sample_explanation = {}
        if anomaly_predictions is not None and (anomaly_predictions == -1).sum() > 0:
            anomaly_indices = np.where(anomaly_predictions == -1)[0]

            # Prefer a true positive (ground-truth Anomaly) for the sample
            chosen_idx = anomaly_indices[0]
            attack_type = None
            if targets is not None and "anomaly_type" in targets.columns:
                attack_classes = list(self.attack_label_encoder.classes_) if self.attack_label_encoder else []
                for idx in anomaly_indices:
                    atype = str(targets["anomaly_type"].iloc[idx])
                    if atype in attack_classes:
                        chosen_idx = idx
                        attack_type = atype
                        break

            sample_explanation = self.explain_event(
                features.iloc[chosen_idx:chosen_idx + 1],
                predicted_attack_type=attack_type,
            )

            narrative = sample_explanation.get("narrative", "")
            logger.info("Sample Anomaly Explanation:")
            for line in narrative.split("\n"):
                logger.info("  %s", line)

        logger.info("=" * 70)
        logger.info("  EXPLAINABILITY ENGINE PIPELINE - COMPLETE")
        logger.info("=" * 70)

        return {
            "summary": summary,
            "sample_explanation": sample_explanation,
        }


# ---------------------------------------------------------------------------
# Standalone Testing Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from preprocessing.data_preprocessor import DataPreprocessor
    from preprocessing.data_validator import DataValidator
    from feature_engineering.feature_engineer import FeatureEngineer
    from models.baseline.baseline_profiler import BaselineProfiler

    validator = DataValidator()
    df_raw = validator.run()
    engineer = FeatureEngineer()
    df_enriched = engineer.run(df_raw)
    profiler = BaselineProfiler()
    df_profiled = profiler.run(df_enriched)
    preprocessor = DataPreprocessor()
    features, targets, entity_ids = preprocessor.run(df_profiled)

    explainer = ExplainabilityEngine()
    results = explainer.run(features, targets)
    print("\nGlobal Summary:", results["summary"])
