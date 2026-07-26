# =============================================================================
# Cybersecurity Anomaly Detection - UEBA System
# Main Entry Point
# =============================================================================
"""
Main orchestration script for the UEBA pipeline.

Runs Phases 1–7 sequentially:
    Phase 1: Data Validation
    Phase 3: Feature Engineering (on raw data, before encoding)
    Phase 4: Baseline Profiling  (on raw enriched data, before encoding)
    Phase 2: Preprocessing       (encoding + scaling on profiled data)
    Phase 5: Anomaly Detection   (Isolation Forest on preprocessed features)
    Phase 6: Attack Classification (multi-class on anomalous events)
    Phase 7: Explainability      (SHAP attributions + natural language narratives)

Usage:
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on the import path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger
from preprocessing.data_validator import DataValidator
from preprocessing.data_preprocessor import DataPreprocessor
from feature_engineering.feature_engineer import FeatureEngineer
from models.baseline.baseline_profiler import BaselineProfiler
from models.anomaly_detection.anomaly_detector import AnomalyDetector
from models.classification.attack_classifier import AttackClassifier
from models.risk_scoring.risk_engine import RiskScoringEngine
from explainability.explainability_engine import ExplainabilityEngine
from models.concept_drift.drift_detector import DriftDetector

logger = get_logger("main")


def main() -> None:
    """Run the UEBA pipeline phases sequentially."""
    logger.info("=" * 70)
    logger.info("  CYBERSECURITY ANOMALY DETECTION - UEBA SYSTEM")
    logger.info("  Honeywell Internship Challenge")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Phase 1: Data Validation
    # ------------------------------------------------------------------
    logger.info("Phase 1: Data Validation - START")
    validator = DataValidator(config_path="config/config.yaml")
    df = validator.run()
    logger.info("Phase 1: Data Validation - COMPLETE")
    logger.info("Dataset ready for preprocessing: %d rows x %d columns", *df.shape)

    # ------------------------------------------------------------------
    # Phase 3: Feature Engineering (runs on RAW data before encoding)
    # ------------------------------------------------------------------
    logger.info("Phase 3: Feature Engineering - START")
    engineer = FeatureEngineer(config_path="config/config.yaml")
    df_enriched = engineer.run(df)
    logger.info("Phase 3: Feature Engineering - COMPLETE")
    logger.info("Enriched dataset: %d rows x %d columns", *df_enriched.shape)

    # ------------------------------------------------------------------
    # Phase 4: Baseline Profiling (runs on raw enriched data)
    #   - Builds per-entity and per-department behavioural profiles
    #   - Computes deviation features (new geo, new device, hour/session
    #     z-scores against personal baseline)
    #   - Cold-start entities use department profiles as fallback
    # ------------------------------------------------------------------
    logger.info("Phase 4: Baseline Profiling - START")
    profiler = BaselineProfiler(config_path="config/config.yaml")
    df_profiled = profiler.run(df_enriched)
    logger.info("Phase 4: Baseline Profiling - COMPLETE")
    logger.info("Profiled dataset: %d rows x %d columns", *df_profiled.shape)

    # ------------------------------------------------------------------
    # Phase 2: Preprocessing (runs on profiled + enriched data)
    #   - Encoding, scaling, and feature/target separation
    # ------------------------------------------------------------------
    logger.info("Phase 2: Preprocessing - START")
    preprocessor = DataPreprocessor(config_path="config/config.yaml")
    features, targets, entity_ids = preprocessor.run(df_profiled)
    logger.info("Phase 2: Preprocessing - COMPLETE")
    logger.info(
        "Feature matrix: %d rows x %d columns", features.shape[0], features.shape[1]
    )

    # ------------------------------------------------------------------
    # Phase 5: Anomaly Detection (Isolation Forest)
    #   - Trains on the full preprocessed feature matrix
    #   - Produces binary predictions (-1 = anomaly, 1 = normal)
    #   - Produces continuous risk scores (0-100)
    #   - Evaluates against ground-truth labels
    # ------------------------------------------------------------------
    logger.info("Phase 5: Anomaly Detection - START")
    detector = AnomalyDetector(config_path="config/config.yaml")
    predictions, anomaly_scores, anomaly_metrics = detector.run(features, targets)
    logger.info("Phase 5: Anomaly Detection - COMPLETE")
    logger.info(
        "Anomalies detected: %d / %d events",
        (predictions == -1).sum(),
        len(predictions),
    )

    # ------------------------------------------------------------------
    # Phase 6: Attack Classification (multi-class)
    #   - Trains ONLY on anomalous events (ground-truth label = Anomaly)
    #   - Compares Random Forest, XGBoost, LightGBM via CV
    #   - Selects and trains the best model on full anomaly set
    #   - Classifies into 7 attack types
    # ------------------------------------------------------------------
    logger.info("Phase 6: Attack Classification - START")
    classifier = AttackClassifier(config_path="config/config.yaml")
    classification_results = classifier.run(features, targets, predictions)
    logger.info("Phase 6: Attack Classification - COMPLETE")

    # ------------------------------------------------------------------
    # Phase 6b: Enterprise Risk Scoring Engine
    #   - Combines IF anomaly scores with business context 
    #   - Generates unified 0-100 Risk Score, Level, and Contributors
    # ------------------------------------------------------------------
    logger.info("Phase 6b: Risk Scoring Engine - START")
    risk_engine = RiskScoringEngine(config_path="config/config.yaml")
    # For attacks, we need an array aligned with events. If classification 
    # was perfect, we can use the original target 'anomaly_type' for demonstration.
    # In a pure inference pipeline, we'd use the model's predictions. 
    attack_preds = targets["anomaly_type"].values if "anomaly_type" in targets.columns else None
    risk_results = risk_engine.run(df_enriched, anomaly_scores, attack_preds)
    logger.info("Phase 6b: Risk Scoring Engine - COMPLETE")

    # ------------------------------------------------------------------
    # Phase 7: Explainability (SHAP + natural language narratives)
    #   - Loads saved Isolation Forest + LightGBM classifier
    #   - Computes global feature importance rankings
    #   - Generates per-event SHAP attributions for anomalous events
    #   - Produces human-readable explanations for SOC analysts
    # ------------------------------------------------------------------
    logger.info("Phase 7: Explainability - START")
    explainer = ExplainabilityEngine(config_path="config/config.yaml")
    explainability_results = explainer.run(features, targets, predictions)
    logger.info("Phase 7: Explainability - COMPLETE")

    # ------------------------------------------------------------------
    # Phase 8: Concept Drift Baseline
    #   - Fits the KS-test baseline distributions on the current dataset
    #   - Allows the dashboard to detect drift on future data
    # ------------------------------------------------------------------
    logger.info("Phase 8: Concept Drift Baseline - START")
    drift_detector = DriftDetector(config_path="config/config.yaml")
    
    # We combine features and risk scores to form the full reference dataset
    drift_ref_df = features.copy()
    drift_ref_df["risk_score"] = risk_results["risk_score"].values
    
    drift_detector.fit(drift_ref_df)
    import joblib
    joblib.dump(drift_detector, PROJECT_ROOT / "saved_models/drift_baseline.joblib")
    logger.info("Phase 8: Concept Drift Baseline - COMPLETE")

    # ------------------------------------------------------------------
    # Pipeline Summary
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("  PIPELINE EXECUTION COMPLETE")
    logger.info("=" * 70)
    logger.info("  Phase 1: Data Validation       [OK]")
    logger.info("  Phase 3: Feature Engineering    [OK]")
    logger.info("  Phase 4: Baseline Profiling     [OK]")
    logger.info("  Phase 2: Preprocessing          [OK]")
    logger.info("  Phase 5: Anomaly Detection      [OK]")
    logger.info("  Phase 6: Attack Classification  [OK]")
    logger.info("  Phase 6b: Risk Scoring          [OK]")
    logger.info("  Phase 7: Explainability         [OK]")
    logger.info("  Phase 8: Concept Drift Baseline [OK]")
    logger.info("-" * 70)
    logger.info("  Total events processed : %d", len(features))
    logger.info("  Feature dimensions     : %d", features.shape[1])
    logger.info("  Entity profiles built  : %d", len(profiler.profiles))
    if anomaly_metrics:
        logger.info("  Anomaly F1 score       : %.4f", anomaly_metrics.get("f1_score", 0))
        logger.info("  Anomaly ROC AUC        : %.4f", anomaly_metrics.get("roc_auc", 0))
    if classification_results:
        report = classification_results.get("classification_report", {})
        macro_f1 = report.get("macro avg", {}).get("f1-score", 0)
        logger.info("  Best classifier        : %s", classification_results.get("model", "N/A"))
        logger.info("  Classification macro F1: %.4f", macro_f1)
    if explainability_results:
        global_summary = explainability_results.get("summary", {})
        logger.info("  Explainability model   : %s", global_summary.get("model_type", "N/A"))
        top_feats = global_summary.get("global_top_features", {})
        if top_feats:
            top_feat_name = next(iter(top_feats))
            logger.info("  Top global feature     : %s", top_feat_name)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Phase 8: Dashboard            (to be added)
    # Phase 9: Evaluation           (to be added)
    # Phase 10: Documentation       (to be added)
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
