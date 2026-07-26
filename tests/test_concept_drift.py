import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from models.concept_drift.drift_detector import DriftDetector

class TestConceptDrift:
    @pytest.fixture
    def drift_detector(self):
        # Initialize detector with a dummy config
        detector = DriftDetector(config_path="config/config.yaml")
        # We manually set tracked features to avoid missing column warnings in our dummy data
        detector.tracked_features = ["feature_A", "feature_B"]
        return detector

    @pytest.fixture
    def reference_data(self):
        np.random.seed(42)
        return pd.DataFrame({
            "feature_A": np.random.normal(0, 1, 1000),
            "feature_B": np.random.uniform(0, 10, 1000),
            "ignored_feature": np.random.normal(0, 1, 1000)
        })

    def test_fit_stores_correct_features(self, drift_detector, reference_data):
        drift_detector.fit(reference_data)
        assert len(drift_detector.reference_distributions) == 2
        assert "feature_A" in drift_detector.reference_distributions
        assert "feature_B" in drift_detector.reference_distributions
        assert "ignored_feature" not in drift_detector.reference_distributions

    def test_no_drift_detected_on_same_distribution(self, drift_detector, reference_data):
        drift_detector.fit(reference_data)
        
        # Current data comes from the exact same distribution (or is just a slice of the same)
        current_data = reference_data.sample(500, random_state=1)
        
        results = drift_detector.detect(current_data)
        
        assert len(results) == 2
        assert results["feature_A"]["is_drifting"] is False
        assert results["feature_B"]["is_drifting"] is False
        # p-values should be high (fail to reject null hypothesis)
        assert results["feature_A"]["p_value"] > 0.05

    def test_drift_detected_on_shifted_distribution(self, drift_detector, reference_data):
        drift_detector.fit(reference_data)
        
        # Create current data with a significant mean shift in feature_A
        np.random.seed(99)
        current_data = pd.DataFrame({
            "feature_A": np.random.normal(3, 1, 500), # Mean shifted from 0 to 3
            "feature_B": np.random.uniform(0, 10, 500) # Unchanged distribution
        })
        
        results = drift_detector.detect(current_data)
        
        # feature_A should trigger drift
        assert results["feature_A"]["is_drifting"] is True
        assert results["feature_A"]["p_value"] < 0.05
        
        # feature_B should not trigger drift
        assert results["feature_B"]["is_drifting"] is False
        assert results["feature_B"]["p_value"] > 0.05
