# =============================================================================
# Cybersecurity Anomaly Detection - UEBA System
# Module: Anomaly Detector
# =============================================================================
"""
Unsupervised anomaly detection using Isolation Forest.

Design decisions:
    - **Isolation Forest** is chosen as the primary model because:
      1. It is CPU-friendly and scales linearly with data size.
      2. It excels at detecting point anomalies in high-dimensional spaces.
      3. It requires no assumption about data distribution.
      4. It naturally handles the 2% anomaly contamination rate.
    - **Contamination** is set to match the known anomaly rate (0.02) for
      calibrated thresholds. In a real deployment, this would be tuned.
    - **Anomaly scores** are continuous (higher = more anomalous), which
      feeds into the risk scoring engine in Phase 7.

Usage:
    from models.anomaly_detection.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector(config_path="config/config.yaml")
    predictions, scores = detector.run(features, targets)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)


class AnomalyDetector:
    """Isolation Forest based anomaly detector for UEBA.

    The model learns the structure of normal data and assigns anomaly
    scores. Events with high anomaly scores (low decision function
    values) are flagged for investigation.

    Attributes:
        config: Full YAML configuration.
        model: Fitted IsolationForest instance.
        contamination: Expected fraction of anomalies.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        contamination: float = 0.02,
        n_estimators: int = 200,
        max_features: float = 0.8,
        random_state: int = 42,
    ) -> None:
        """Initialize the anomaly detector.

        Args:
            config_path: Path to YAML config.
            contamination: Expected anomaly fraction (default fallback).
            n_estimators: Number of trees in the forest (default fallback).
            max_features: Fraction of features per tree (default fallback).
            random_state: Seed for reproducibility.
        """
        self.config = self._load_config(config_path)
        
        # Load from config if available, else fallback to kwargs
        model_cfg = self.config.get("models", {}).get("anomaly_detection", {}).get("isolation_forest", {})
        
        self.contamination = model_cfg.get("contamination", contamination)
        _n_estimators = model_cfg.get("n_estimators", n_estimators)
        _max_features = model_cfg.get("max_features", max_features)
        _random_state = model_cfg.get("random_state", random_state)
        
        self.model = IsolationForest(
            n_estimators=_n_estimators,
            contamination=self.contamination,
            max_features=_max_features,
            random_state=_random_state,
            n_jobs=-1,
            verbose=0,
        )
        logger.info(
            "AnomalyDetector initialized (contamination=%.3f, "
            "n_estimators=%d, max_features=%.1f).",
            self.contamination,
            _n_estimators,
            _max_features,
        )

    @staticmethod
    def _load_config(config_path: str) -> dict:
        resolved = Path(_PROJECT_ROOT) / config_path
        if not resolved.exists():
            resolved = Path(config_path)
        with open(resolved, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame) -> None:
        """Fit the Isolation Forest on the feature matrix.

        Args:
            X: Numeric feature matrix (preprocessed, scaled).
        """
        # Handle any remaining NaN/Inf
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        logger.info(
            "Fitting Isolation Forest on %d samples x %d features...",
            X_clean.shape[0],
            X_clean.shape[1],
        )
        self.model.fit(X_clean)
        logger.info("Isolation Forest training complete.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self, X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate anomaly predictions and continuous scores.

        Args:
            X: Feature matrix.

        Returns:
            Tuple of (predictions, anomaly_scores):
                - predictions: 1 = normal, -1 = anomaly (sklearn convention)
                - anomaly_scores: Continuous scores where lower (more
                  negative) = more anomalous. Rescaled to [0, 100] range
                  for the risk engine.
        """
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        predictions = self.model.predict(X_clean)
        raw_scores = self.model.decision_function(X_clean)

        # Rescale: raw_scores are centered around 0 (negative = anomalous)
        # Convert to risk score: 0 = normal, 100 = most anomalous
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        score_range = max_score - min_score
        if score_range == 0:
            anomaly_scores = np.zeros_like(raw_scores)
        else:
            # Invert: lower raw = higher risk
            anomaly_scores = ((max_score - raw_scores) / score_range) * 100

        logger.info(
            "Predictions: %d anomalies detected out of %d events.",
            (predictions == -1).sum(),
            len(predictions),
        )
        return predictions, anomaly_scores

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        predictions: np.ndarray,
        anomaly_scores: np.ndarray,
        y_true: pd.Series,
    ) -> dict:
        """Evaluate anomaly detection performance against ground truth.

        Args:
            predictions: Isolation Forest predictions (-1/1).
            anomaly_scores: Continuous risk scores (0-100).
            y_true: Ground truth labels ('Normal'/'Anomaly').

        Returns:
            Dictionary of evaluation metrics.
        """
        # Convert to binary: Anomaly=1, Normal=0
        y_binary = (y_true == "Anomaly").astype(int)
        pred_binary = (predictions == -1).astype(int)

        precision = precision_score(y_binary, pred_binary, zero_division=0)
        recall = recall_score(y_binary, pred_binary, zero_division=0)
        f1 = f1_score(y_binary, pred_binary, zero_division=0)

        # ROC AUC using continuous scores
        try:
            roc_auc = roc_auc_score(y_binary, anomaly_scores)
        except ValueError:
            roc_auc = 0.0

        cm = confusion_matrix(y_binary, pred_binary)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Top-1% alert budget: what fraction of true anomalies are in
        # the top 1% highest scores?
        top_1_pct = int(len(anomaly_scores) * 0.01)
        top_indices = np.argsort(anomaly_scores)[-top_1_pct:]
        top_1_recall = y_binary.iloc[top_indices].sum() / y_binary.sum() if y_binary.sum() > 0 else 0.0

        metrics = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "false_positive_rate": round(fpr, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "top_1pct_recall": round(float(top_1_recall), 4),
        }

        logger.info("-" * 50)
        logger.info("  ANOMALY DETECTION METRICS")
        logger.info("-" * 50)
        for k, v in metrics.items():
            logger.info("  %-25s: %s", k, v)
        logger.info("-" * 50)

        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_model(self) -> str:
        """Save the fitted model to disk."""
        save_dir = Path(_PROJECT_ROOT) / self.config["paths"]["saved_models_dir"]
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / "isolation_forest.joblib"
        joblib.dump(self.model, str(path))
        logger.info("Saved Isolation Forest -> %s", path)
        return str(path)

    # ------------------------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------------------------
    def run(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Execute the full anomaly detection pipeline.

        Args:
            features: Preprocessed numeric feature matrix.
            targets: Target labels DataFrame (must contain 'label').

        Returns:
            Tuple of (predictions, anomaly_scores, metrics).
        """
        logger.info("=" * 70)
        logger.info("  ANOMALY DETECTION PIPELINE - START")
        logger.info("=" * 70)

        self.fit(features)
        predictions, anomaly_scores = self.predict(features)

        # Evaluate if ground truth available
        metrics = {}
        if "label" in targets.columns:
            metrics = self.evaluate(predictions, anomaly_scores, targets["label"])

        self.save_model()

        logger.info("=" * 70)
        logger.info("  ANOMALY DETECTION COMPLETE")
        logger.info("=" * 70)

        return predictions, anomaly_scores, metrics


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_features = pd.read_csv(
        str(Path(_PROJECT_ROOT) / "data" / "processed_dataset.csv")
    )
    # Separate entity_id and targets
    entity_ids = df_features["entity_id"]
    targets = df_features[["label", "anomaly_type"]]
    features = df_features.drop(
        columns=["entity_id", "label", "anomaly_type"], errors="ignore"
    )

    detector = AnomalyDetector()
    predictions, scores, metrics = detector.run(features, targets)
