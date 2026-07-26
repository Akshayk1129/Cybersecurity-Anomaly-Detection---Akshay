# =============================================================================
# Cybersecurity Anomaly Detection - UEBA System
# Module: Attack Classifier
# =============================================================================
"""
Supervised multi-class classifier for attack type categorization.

This model is trained ONLY on events flagged as anomalous (either by the
Isolation Forest or by ground truth), then classifies them into one of
seven attack categories:

    - Brute Force
    - Credential Stuffing
    - Impossible Travel
    - Lateral Movement
    - Device Spoofing
    - Low-and-Slow Exfiltration
    - Insider Drift

Design decisions:
    - Three models are compared: Random Forest, XGBoost, and LightGBM.
    - The best model is selected based on macro F1-score (since attack
      types are imbalanced, macro-average treats all classes equally).
    - Class weights / sample weights handle the class imbalance within
      the anomaly subset.

Usage:
    from models.classification.attack_classifier import AttackClassifier
    classifier = AttackClassifier(config_path="config/config.yaml")
    results = classifier.run(features, targets, anomaly_mask)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)


class AttackClassifier:
    """Multi-class attack type classifier.

    Compares Random Forest, XGBoost, and LightGBM on the anomalous
    event subset and selects the best performer.

    Attributes:
        config: Full YAML configuration.
        best_model: The fitted model with the highest macro F1.
        best_model_name: Name of the best model.
        label_encoder: LabelEncoder for anomaly_type classes.
        model_comparison: Dict of model_name -> metrics for all models.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.config = self._load_config(config_path)
        self.best_model: Any = None
        self.best_model_name: str = ""
        self.label_encoder = LabelEncoder()
        self.model_comparison: Dict[str, Dict[str, Any]] = {}
        logger.info("AttackClassifier initialized.")

    @staticmethod
    def _load_config(config_path: str) -> dict:
        resolved = Path(_PROJECT_ROOT) / config_path
        if not resolved.exists():
            resolved = Path(config_path)
        with open(resolved, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Model Factory
    # ------------------------------------------------------------------
    def _get_candidate_models(self) -> Dict[str, Any]:
        """Create candidate classifiers for comparison.

        Returns:
            Dictionary of model_name -> unfitted model instance.
        """
        models: Dict[str, Any] = {}
        cls_cfg = self.config.get("models", {}).get("attack_classification", {})

        # Random Forest — strong baseline, handles imbalance natively
        if cls_cfg.get("use_random_forest", True):
            rf_cfg = cls_cfg.get("random_forest", {})
            models["RandomForest"] = RandomForestClassifier(
                n_estimators=rf_cfg.get("n_estimators", 200),
                max_depth=rf_cfg.get("max_depth", 15),
                class_weight=rf_cfg.get("class_weight", "balanced"),
                random_state=rf_cfg.get("random_state", 42),
                n_jobs=rf_cfg.get("n_jobs", -1),
            )

        # XGBoost — gradient boosting, often best on tabular data
        if cls_cfg.get("use_xgboost", True):
            try:
                from xgboost import XGBClassifier
                xgb_cfg = cls_cfg.get("xgboost", {})
                models["XGBoost"] = XGBClassifier(
                    n_estimators=xgb_cfg.get("n_estimators", 200),
                    max_depth=xgb_cfg.get("max_depth", 8),
                    learning_rate=xgb_cfg.get("learning_rate", 0.1),
                    subsample=xgb_cfg.get("subsample", 0.8),
                    colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
                    random_state=xgb_cfg.get("random_state", 42),
                    eval_metric=xgb_cfg.get("eval_metric", "mlogloss"),
                    n_jobs=xgb_cfg.get("n_jobs", -1),
                    verbosity=xgb_cfg.get("verbosity", 0),
                )
            except ImportError:
                logger.warning("XGBoost not installed. Skipping.")

        # LightGBM — fast gradient boosting
        if cls_cfg.get("use_lightgbm", True):
            try:
                from lightgbm import LGBMClassifier
                lgb_cfg = cls_cfg.get("lightgbm", {})
                models["LightGBM"] = LGBMClassifier(
                    n_estimators=lgb_cfg.get("n_estimators", 200),
                    max_depth=lgb_cfg.get("max_depth", 10),
                    learning_rate=lgb_cfg.get("learning_rate", 0.1),
                    subsample=lgb_cfg.get("subsample", 0.8),
                    colsample_bytree=lgb_cfg.get("colsample_bytree", 0.8),
                    class_weight=lgb_cfg.get("class_weight", "balanced"),
                    random_state=lgb_cfg.get("random_state", 42),
                    n_jobs=lgb_cfg.get("n_jobs", -1),
                    verbose=lgb_cfg.get("verbose", -1),
                )
            except ImportError:
                logger.warning("LightGBM not installed. Skipping.")

        return models

    # ------------------------------------------------------------------
    # Training & Comparison
    # ------------------------------------------------------------------
    def compare_models(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        cv_folds: int = 5,
    ) -> str:
        """Train and compare all candidate models via cross-validation.

        Args:
            X: Feature matrix (anomalous events only).
            y: Encoded target labels.
            cv_folds: Number of stratified CV folds.

        Returns:
            Name of the best-performing model.
        """
        models = self._get_candidate_models()
        best_score = -1.0
        best_name = ""

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for name, model in models.items():
            logger.info("Training %s...", name)

            try:
                scores = cross_val_score(
                    model,
                    X,
                    y,
                    cv=skf,
                    scoring="f1_macro",
                    n_jobs=-1,
                )
                mean_f1 = scores.mean()
                std_f1 = scores.std()

                self.model_comparison[name] = {
                    "cv_f1_mean": round(mean_f1, 4),
                    "cv_f1_std": round(std_f1, 4),
                    "cv_scores": [round(s, 4) for s in scores],
                }

                logger.info(
                    "  %s: F1 macro = %.4f (+/- %.4f)", name, mean_f1, std_f1
                )

                if mean_f1 > best_score:
                    best_score = mean_f1
                    best_name = name

            except Exception as e:
                logger.error("  %s failed: %s", name, str(e))
                self.model_comparison[name] = {"error": str(e)}

        logger.info("Best model: %s (F1=%.4f)", best_name, best_score)
        return best_name

    def train_best_model(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model_name: str,
    ) -> None:
        """Train the selected best model on the full anomaly dataset.

        Args:
            X: Feature matrix.
            y: Encoded labels.
            model_name: Name of the model to train.
        """
        models = self._get_candidate_models()
        self.best_model = models[model_name]
        self.best_model_name = model_name

        self.best_model.fit(X, y)
        logger.info("Trained final %s on %d samples.", model_name, len(y))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Generate detailed evaluation metrics on training data.

        Args:
            X: Feature matrix.
            y: True encoded labels.

        Returns:
            Dictionary of evaluation results.
        """
        y_pred = self.best_model.predict(X)

        # Decode labels for readable report
        classes = self.label_encoder.classes_

        report = classification_report(
            y,
            y_pred,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y, y_pred)

        # Feature importance (if available)
        importances = {}
        if hasattr(self.best_model, "feature_importances_"):
            imp = self.best_model.feature_importances_
            top_indices = np.argsort(imp)[-15:][::-1]
            for idx in top_indices:
                if idx < len(X.columns):
                    importances[X.columns[idx]] = round(float(imp[idx]), 4)

        results = {
            "model": self.best_model_name,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "classes": list(classes),
            "top_features": importances,
            "model_comparison": self.model_comparison,
        }

        # Log the report
        logger.info("-" * 50)
        logger.info("  CLASSIFICATION REPORT (%s)", self.best_model_name)
        logger.info("-" * 50)
        report_str = classification_report(
            y, y_pred, target_names=classes, zero_division=0
        )
        for line in report_str.split("\n"):
            logger.info("  %s", line)
        logger.info("-" * 50)

        if importances:
            logger.info("  Top 15 Features:")
            for feat, imp_val in importances.items():
                logger.info("    %-35s: %.4f", feat, imp_val)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_model(self) -> str:
        """Save the best classifier and label encoder."""
        save_dir = Path(_PROJECT_ROOT) / self.config["paths"]["saved_models_dir"]
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "attack_classifier.joblib"
        joblib.dump(self.best_model, str(model_path))
        logger.info("Saved %s -> %s", self.best_model_name, model_path)

        le_path = save_dir / "attack_label_encoder.joblib"
        joblib.dump(self.label_encoder, str(le_path))
        logger.info("Saved attack label encoder -> %s", le_path)

        # Save comparison report
        comp_path = save_dir / "model_comparison.joblib"
        joblib.dump(self.model_comparison, str(comp_path))

        return str(model_path)

    # ------------------------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------------------------
    def run(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        anomaly_predictions: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Execute the attack classification pipeline.

        If ``anomaly_predictions`` is provided, the classifier is trained
        only on events predicted as anomalous. Otherwise, ground-truth
        labels are used to filter anomalies.

        Args:
            features: Full preprocessed feature matrix.
            targets: Full target DataFrame with 'label' and 'anomaly_type'.
            anomaly_predictions: Optional array of -1/1 from the anomaly
                                 detector (used for filtering).

        Returns:
            Dictionary with evaluation results and model comparison.
        """
        logger.info("=" * 70)
        logger.info("  ATTACK CLASSIFICATION PIPELINE - START")
        logger.info("=" * 70)

        # Filter to anomalous events only (use ground truth for training)
        if "label" in targets.columns and "anomaly_type" in targets.columns:
            anomaly_mask = targets["label"] == "Anomaly"
            X_anomaly = features.loc[anomaly_mask].reset_index(drop=True)
            y_raw = targets.loc[anomaly_mask, "anomaly_type"].reset_index(drop=True)

            # Remove 'Normal' from anomaly_type (shouldn't be there, but safety)
            valid_mask = y_raw != "Normal"
            X_anomaly = X_anomaly.loc[valid_mask].reset_index(drop=True)
            y_raw = y_raw.loc[valid_mask].reset_index(drop=True)
        else:
            logger.error("Cannot train classifier: missing label/anomaly_type.")
            return {}

        logger.info(
            "Training on %d anomalous events across %d attack types.",
            len(X_anomaly),
            y_raw.nunique(),
        )

        # Handle NaN/Inf
        X_anomaly = X_anomaly.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Encode attack types
        y_encoded = self.label_encoder.fit_transform(y_raw)
        logger.info("Attack classes: %s", list(self.label_encoder.classes_))

        # Compare models
        best_name = self.compare_models(X_anomaly, y_encoded)

        # Train best model on full anomaly set
        self.train_best_model(X_anomaly, y_encoded, best_name)

        # Evaluate
        results = self.evaluate(X_anomaly, y_encoded)

        # Save
        self.save_model()

        logger.info("=" * 70)
        logger.info("  ATTACK CLASSIFICATION COMPLETE")
        logger.info("=" * 70)

        return results


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(str(Path(_PROJECT_ROOT) / "data" / "processed_dataset.csv"))
    entity_ids = df["entity_id"]
    targets = df[["label", "anomaly_type"]]
    features = df.drop(columns=["entity_id", "label", "anomaly_type"], errors="ignore")

    classifier = AttackClassifier()
    results = classifier.run(features, targets)
