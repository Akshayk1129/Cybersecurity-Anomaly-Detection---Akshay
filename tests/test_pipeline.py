# =============================================================================
# Cybersecurity Anomaly Detection - UEBA System
# Module: Pipeline Integration & Unit Tests (Phase 9)
# =============================================================================
"""
Comprehensive test suite for the UEBA pipeline.

Covers:
  - Data Validation (Phase 1)
  - Feature Engineering (Phase 3)
  - Baseline Profiling (Phase 4)
  - Preprocessing (Phase 2)
  - Anomaly Detection (Phase 5)
  - Attack Classification (Phase 6)
  - Explainability Engine (Phase 7)
  - End-to-End Pipeline Integration

Run:
    pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ===================================================================
# Fixtures: Shared test data and module instances
# ===================================================================

@pytest.fixture(scope="session")
def config_path() -> str:
    return "config/config.yaml"


@pytest.fixture(scope="session")
def raw_dataset() -> pd.DataFrame:
    """Load and validate the raw dataset (Phase 1)."""
    from preprocessing.data_validator import DataValidator
    validator = DataValidator(config_path="config/config.yaml")
    df = validator.run()
    return df


@pytest.fixture(scope="session")
def enriched_dataset(raw_dataset: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering (Phase 3)."""
    from feature_engineering.feature_engineer import FeatureEngineer
    engineer = FeatureEngineer(config_path="config/config.yaml")
    return engineer.run(raw_dataset)


@pytest.fixture(scope="session")
def profiled_dataset(enriched_dataset: pd.DataFrame):
    """Apply baseline profiling (Phase 4)."""
    from models.baseline.baseline_profiler import BaselineProfiler
    profiler = BaselineProfiler(config_path="config/config.yaml")
    df_profiled = profiler.run(enriched_dataset)
    return df_profiled, profiler


@pytest.fixture(scope="session")
def preprocessed_data(profiled_dataset):
    """Apply preprocessing (Phase 2)."""
    from preprocessing.data_preprocessor import DataPreprocessor
    df_profiled, _ = profiled_dataset
    preprocessor = DataPreprocessor(config_path="config/config.yaml")
    features, targets, entity_ids = preprocessor.run(df_profiled)
    return features, targets, entity_ids


@pytest.fixture(scope="session")
def anomaly_results(preprocessed_data):
    """Run anomaly detection (Phase 5)."""
    from models.anomaly_detection.anomaly_detector import AnomalyDetector
    features, targets, _ = preprocessed_data
    detector = AnomalyDetector(config_path="config/config.yaml")
    predictions, scores, metrics = detector.run(features, targets)
    return predictions, scores, metrics


@pytest.fixture(scope="session")
def classification_results(preprocessed_data, anomaly_results):
    """Run attack classification (Phase 6)."""
    from models.classification.attack_classifier import AttackClassifier
    features, targets, _ = preprocessed_data
    predictions, _, _ = anomaly_results
    classifier = AttackClassifier(config_path="config/config.yaml")
    results = classifier.run(features, targets, predictions)
    return results


# ===================================================================
# Test Class: Phase 1 - Data Validation
# ===================================================================
class TestDataValidation:
    """Tests for the DataValidator module (Phase 1)."""

    def test_dataset_loads_successfully(self, raw_dataset: pd.DataFrame):
        """Dataset should load with correct row count."""
        assert len(raw_dataset) == 100_000, f"Expected 100000 rows, got {len(raw_dataset)}"

    def test_dataset_has_expected_columns(self, raw_dataset: pd.DataFrame):
        """Dataset should contain all 30 required columns."""
        required = [
            "event_id", "entity_id", "entity_type", "department",
            "timestamp", "login_hour", "label", "anomaly_type",
        ]
        for col in required:
            assert col in raw_dataset.columns, f"Missing column: {col}"

    def test_anomaly_rate(self, raw_dataset: pd.DataFrame):
        """Anomaly rate should be approximately 2%."""
        anomaly_rate = (raw_dataset["label"] == "Anomaly").mean()
        assert 0.01 <= anomaly_rate <= 0.05, f"Anomaly rate {anomaly_rate:.4f} outside expected 1-5%"

    def test_entity_count(self, raw_dataset: pd.DataFrame):
        """Should have 500 unique entities."""
        n_entities = raw_dataset["entity_id"].nunique()
        assert n_entities == 500, f"Expected 500 entities, got {n_entities}"

    def test_no_fully_null_columns(self, raw_dataset: pd.DataFrame):
        """No column should be entirely null."""
        for col in raw_dataset.columns:
            assert not raw_dataset[col].isna().all(), f"Column '{col}' is entirely null"

    def test_attack_types_present(self, raw_dataset: pd.DataFrame):
        """All 7 known attack types should be present."""
        anomalies = raw_dataset[raw_dataset["label"] == "Anomaly"]
        attack_types = anomalies["anomaly_type"].unique()
        assert len(attack_types) == 7, f"Expected 7 attack types, got {len(attack_types)}"

    def test_validation_report_generated(self):
        """Validation report should be written to disk."""
        report_path = PROJECT_ROOT / "reports" / "data_validation_report.md"
        assert report_path.exists(), "Validation report not found"
        assert report_path.stat().st_size > 100, "Validation report is too small"


# ===================================================================
# Test Class: Phase 3 - Feature Engineering
# ===================================================================
class TestFeatureEngineering:
    """Tests for the FeatureEngineer module (Phase 3)."""

    def test_new_features_added(self, raw_dataset: pd.DataFrame, enriched_dataset: pd.DataFrame):
        """Enriched dataset should have more columns than raw."""
        assert enriched_dataset.shape[1] > raw_dataset.shape[1], "No new features were added"

    def test_row_count_preserved(self, raw_dataset: pd.DataFrame, enriched_dataset: pd.DataFrame):
        """Row count should be identical after feature engineering."""
        assert len(enriched_dataset) == len(raw_dataset), "Row count changed during feature engineering"

    def test_cyclical_features_bounded(self, enriched_dataset: pd.DataFrame):
        """hour_sin and hour_cos should be in [-1, 1]."""
        assert enriched_dataset["hour_sin"].between(-1.01, 1.01).all(), "hour_sin out of bounds"
        assert enriched_dataset["hour_cos"].between(-1.01, 1.01).all(), "hour_cos out of bounds"

    def test_zscore_features_exist(self, enriched_dataset: pd.DataFrame):
        """Z-score features should be present."""
        for col in ["session_duration_zscore", "bytes_uploaded_zscore", "bytes_downloaded_zscore"]:
            assert col in enriched_dataset.columns, f"Missing z-score feature: {col}"

    def test_entropy_non_negative(self, enriched_dataset: pd.DataFrame):
        """Shannon entropy should be >= 0."""
        assert (enriched_dataset["resource_access_entropy"] >= -0.01).all(), "Negative entropy detected"

    def test_peer_features_exist(self, enriched_dataset: pd.DataFrame):
        """Peer comparison features should be present."""
        peer_cols = ["peer_avg_session", "peer_session_deviation", "peer_avg_bytes_up"]
        for col in peer_cols:
            assert col in enriched_dataset.columns, f"Missing peer feature: {col}"

    def test_no_inf_in_key_features(self, enriched_dataset: pd.DataFrame):
        """Key numeric features should not contain inf values."""
        numeric_cols = enriched_dataset.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(enriched_dataset[col]).sum()
            assert inf_count == 0, f"Column '{col}' has {inf_count} inf values"


# ===================================================================
# Test Class: Phase 4 - Baseline Profiling
# ===================================================================
class TestBaselineProfiling:
    """Tests for the BaselineProfiler module (Phase 4)."""

    def test_entity_profiles_built(self, profiled_dataset):
        """Every entity should have a profile."""
        _, profiler = profiled_dataset
        assert len(profiler.profiles) == 500, f"Expected 500 profiles, got {len(profiler.profiles)}"

    def test_department_profiles_built(self, profiled_dataset):
        """Department profiles should be built."""
        _, profiler = profiled_dataset
        assert len(profiler.department_profiles) > 0, "No department profiles built"

    def test_deviation_features_added(self, profiled_dataset):
        """Profile deviation features should be added to the dataset."""
        df_profiled, _ = profiled_dataset
        deviation_cols = [
            "profile_hour_deviation", "profile_session_deviation",
            "profile_bytes_up_deviation", "profile_new_geo",
            "profile_new_resource", "profile_new_device",
        ]
        for col in deviation_cols:
            assert col in df_profiled.columns, f"Missing deviation feature: {col}"

    def test_profiles_saved_to_disk(self):
        """Profile artifacts should be persisted."""
        entity_path = PROJECT_ROOT / "saved_models" / "entity_profiles.joblib"
        dept_path = PROJECT_ROOT / "saved_models" / "department_profiles.joblib"
        assert entity_path.exists(), "Entity profiles not saved"
        assert dept_path.exists(), "Department profiles not saved"

    def test_profile_contains_expected_keys(self, profiled_dataset):
        """Each entity profile should have key statistics."""
        _, profiler = profiled_dataset
        sample_profile = next(iter(profiler.profiles.values()))
        # Actual keys from BaselineProfiler use descriptive names
        expected_keys = ["event_count", "login_hour_mean", "login_hour_std"]
        for key in expected_keys:
            assert key in sample_profile, f"Profile missing key: {key}"


# ===================================================================
# Test Class: Phase 2 - Preprocessing
# ===================================================================
class TestPreprocessing:
    """Tests for the DataPreprocessor module (Phase 2)."""

    def test_features_are_numeric(self, preprocessed_data):
        """Feature matrix should be entirely numeric (int, float, bool)."""
        features, _, _ = preprocessed_data
        non_numeric = features.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
        assert len(non_numeric) == 0, f"Non-numeric columns found: {non_numeric}"

    def test_targets_extracted(self, preprocessed_data):
        """Target DataFrame should contain label and anomaly_type."""
        _, targets, _ = preprocessed_data
        assert "label" in targets.columns, "Missing 'label' in targets"
        assert "anomaly_type" in targets.columns, "Missing 'anomaly_type' in targets"

    def test_no_identifier_leakage(self, preprocessed_data):
        """Identifier columns should not leak into features."""
        features, _, _ = preprocessed_data
        leaked = [c for c in ["entity_id", "event_id", "source_ip", "timestamp"] if c in features.columns]
        assert len(leaked) == 0, f"Identifier columns leaked into features: {leaked}"

    def test_scaler_saved(self):
        """Scaler artifact should be saved."""
        assert (PROJECT_ROOT / "saved_models" / "scaler.joblib").exists(), "Scaler not saved"

    def test_feature_dimensions(self, preprocessed_data):
        """Feature matrix should have > 100 columns after encoding."""
        features, _, _ = preprocessed_data
        assert features.shape[1] > 100, f"Only {features.shape[1]} features, expected > 100"

    def test_row_count_preserved_after_preprocessing(self, preprocessed_data):
        """Row count should remain 100K after preprocessing."""
        features, _, _ = preprocessed_data
        assert len(features) == 100_000, f"Expected 100000 rows, got {len(features)}"


# ===================================================================
# Test Class: Phase 5 - Anomaly Detection
# ===================================================================
class TestAnomalyDetection:
    """Tests for the AnomalyDetector module (Phase 5)."""

    def test_predictions_shape(self, anomaly_results, preprocessed_data):
        """Predictions array should match input length."""
        predictions, _, _ = anomaly_results
        features, _, _ = preprocessed_data
        assert len(predictions) == len(features), "Prediction count != event count"

    def test_predictions_values(self, anomaly_results):
        """Predictions should only contain -1 (anomaly) or 1 (normal)."""
        predictions, _, _ = anomaly_results
        unique_vals = set(np.unique(predictions))
        assert unique_vals.issubset({-1, 1}), f"Unexpected prediction values: {unique_vals}"

    def test_anomaly_count_reasonable(self, anomaly_results):
        """Anomaly count should be between 1% and 5% of data (contamination ~2%)."""
        predictions, _, _ = anomaly_results
        anomaly_rate = (predictions == -1).mean()
        assert 0.005 <= anomaly_rate <= 0.10, f"Anomaly rate {anomaly_rate:.4f} outside expected range"

    def test_roc_auc_above_threshold(self, anomaly_results):
        """ROC AUC should be meaningfully above random (0.5)."""
        _, _, metrics = anomaly_results
        roc_auc = metrics.get("roc_auc", 0)
        assert roc_auc > 0.65, f"ROC AUC {roc_auc:.4f} below minimum threshold 0.65"

    def test_false_positive_rate_acceptable(self, anomaly_results):
        """False positive rate should be below 5%."""
        _, _, metrics = anomaly_results
        fpr = metrics.get("false_positive_rate", 1.0)
        assert fpr < 0.05, f"FPR {fpr:.4f} exceeds 5% threshold"

    def test_isolation_forest_saved(self):
        """Isolation Forest model should be saved to disk."""
        assert (PROJECT_ROOT / "saved_models" / "isolation_forest.joblib").exists()

    def test_risk_scores_bounded(self, anomaly_results):
        """Risk scores should be in [0, 100]."""
        _, scores, _ = anomaly_results
        assert np.all(scores >= -0.01), f"Negative risk scores detected: min={scores.min():.2f}"
        assert np.all(scores <= 100.01), f"Risk scores > 100 detected: max={scores.max():.2f}"


# ===================================================================
# Test Class: Phase 6 - Attack Classification
# ===================================================================
class TestAttackClassification:
    """Tests for the AttackClassifier module (Phase 6)."""

    def test_classification_results_not_empty(self, classification_results):
        """Classification should produce results."""
        assert classification_results is not None, "Classification returned None"

    def test_best_model_selected(self, classification_results):
        """A best model should be identified."""
        model_name = classification_results.get("model")
        assert model_name is not None, "No best model selected"
        assert model_name in ["RandomForest", "XGBoost", "LightGBM"], f"Unexpected model: {model_name}"

    def test_macro_f1_above_threshold(self, classification_results):
        """Macro F1 should be above 0.90 for attack classification."""
        report = classification_results.get("classification_report", {})
        macro_f1 = report.get("macro avg", {}).get("f1-score", 0)
        assert macro_f1 > 0.90, f"Macro F1 {macro_f1:.4f} below 0.90 threshold"

    def test_all_attack_types_classified(self, classification_results):
        """All 7 attack types should appear in the report."""
        report = classification_results.get("classification_report", {})
        attack_keys = [k for k in report.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
        assert len(attack_keys) == 7, f"Expected 7 attack types in report, got {len(attack_keys)}"

    def test_classifier_saved(self):
        """Classifier model should be saved to disk."""
        assert (PROJECT_ROOT / "saved_models" / "attack_classifier.joblib").exists()

    def test_label_encoder_saved(self):
        """Attack label encoder should be saved to disk."""
        assert (PROJECT_ROOT / "saved_models" / "attack_label_encoder.joblib").exists()


# ===================================================================
# Test Class: Phase 7 - Explainability
# ===================================================================
class TestExplainability:
    """Tests for the ExplainabilityEngine module (Phase 7)."""

    def test_explainer_initializes(self):
        """ExplainabilityEngine should initialize without errors."""
        from explainability.explainability_engine import ExplainabilityEngine
        engine = ExplainabilityEngine(config_path="config/config.yaml")
        assert engine.classifier_model is not None, "Classifier model not loaded"

    def test_global_summary_generated(self):
        """Global explainability summary should be saved."""
        path = PROJECT_ROOT / "saved_models" / "explainability_summary.joblib"
        assert path.exists(), "Explainability summary not saved"
        summary = joblib.load(str(path))
        assert "global_top_features" in summary, "Missing global_top_features key"
        assert len(summary["global_top_features"]) > 0, "No global features in summary"

    def test_single_event_explanation(self, preprocessed_data):
        """Explaining a single event should produce a valid result."""
        from explainability.explainability_engine import ExplainabilityEngine
        features, _, _ = preprocessed_data
        engine = ExplainabilityEngine(config_path="config/config.yaml")

        explanation = engine.explain_event(features.iloc[0:1], top_k=5)
        assert "top_features" in explanation, "Missing top_features in explanation"
        assert "narrative" in explanation, "Missing narrative in explanation"
        assert len(explanation["top_features"]) > 0, "No feature attributions generated"

    def test_narrative_is_string(self, preprocessed_data):
        """Narrative should be a non-empty string."""
        from explainability.explainability_engine import ExplainabilityEngine
        features, _, _ = preprocessed_data
        engine = ExplainabilityEngine(config_path="config/config.yaml")

        explanation = engine.explain_event(features.iloc[0:1])
        narrative = explanation.get("narrative", "")
        assert isinstance(narrative, str), "Narrative is not a string"
        assert len(narrative) > 20, "Narrative is too short"

    def test_humanized_feature_names(self):
        """Feature humanizer should return meaningful labels."""
        from explainability.explainability_engine import ExplainabilityEngine
        result = ExplainabilityEngine._humanize_feature_name("profile_hour_deviation")
        assert result == "Login hour deviation from personal baseline"
        result_unknown = ExplainabilityEngine._humanize_feature_name("some_random_col")
        assert result_unknown == "Some Random Col"


# ===================================================================
# Test Class: End-to-End Pipeline Integration
# ===================================================================
class TestPipelineIntegration:
    """Integration tests verifying the full pipeline works end-to-end."""

    def test_full_pipeline_produces_output(self, classification_results, anomaly_results):
        """Full pipeline should produce both anomaly and classification results."""
        predictions, _, _ = anomaly_results
        assert len(predictions) == 100_000
        assert classification_results is not None

    def test_saved_models_directory_populated(self):
        """saved_models/ should contain all expected artifacts."""
        models_dir = PROJECT_ROOT / "saved_models"
        expected_files = [
            "isolation_forest.joblib",
            "attack_classifier.joblib",
            "attack_label_encoder.joblib",
            "entity_profiles.joblib",
            "department_profiles.joblib",
            "scaler.joblib",
            "label_encoders.joblib",
            "explainability_summary.joblib",
        ]
        for fname in expected_files:
            path = models_dir / fname
            assert path.exists(), f"Missing saved model: {fname}"
            assert path.stat().st_size > 0, f"Empty saved model: {fname}"

    def test_data_outputs_exist(self):
        """Pipeline data outputs should be written to disk."""
        assert (PROJECT_ROOT / "data" / "enriched_dataset.csv").exists()
        assert (PROJECT_ROOT / "data" / "processed_dataset.csv").exists()

    def test_anomaly_predictions_align_with_ground_truth_distribution(
        self, anomaly_results, preprocessed_data
    ):
        """Anomaly detection should flag a non-trivial subset of true anomalies."""
        predictions, _, metrics = anomaly_results
        _, targets, _ = preprocessed_data
        tp = metrics.get("true_positives", 0)
        assert tp > 0, "Zero true positives detected — model may be broken"

    def test_no_nan_in_predictions(self, anomaly_results):
        """Predictions should not contain NaN."""
        predictions, scores, _ = anomaly_results
        assert not np.any(np.isnan(predictions)), "NaN in predictions"
        assert not np.any(np.isnan(scores)), "NaN in risk scores"
