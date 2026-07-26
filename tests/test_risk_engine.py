# =============================================================================
# Cybersecurity Anomaly Detection - UEBA System
# Module: Risk Engine Tests
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.risk_scoring.risk_engine import RiskScoringEngine


@pytest.fixture
def sample_events():
    """Create a sample dataframe of enriched events for risk scoring."""
    return pd.DataFrame([
        {
            "resource_sensitivity": "Critical",
            "privilege_level": "Admin",
            "is_off_hours": True,
            "new_device": True,
            "new_location": True,
            "failed_attempts": 5,
            "geo_velocity_kmph": 600,
        },
        {
            "resource_sensitivity": "Low",
            "privilege_level": "User",
            "is_off_hours": False,
            "new_device": False,
            "new_location": False,
            "failed_attempts": 0,
            "geo_velocity_kmph": 10,
        }
    ])


@pytest.fixture
def risk_engine():
    # Use default config fallback
    return RiskScoringEngine(config_path="config/config.yaml")


def test_risk_score_bounds(risk_engine, sample_events):
    """Risk scores must be between 0 and 100."""
    anomaly_scores = np.array([90.0, 10.0])
    results = risk_engine.run(sample_events, anomaly_scores)
    
    assert all(results["risk_score"] >= 0)
    assert all(results["risk_score"] <= 100)


def test_high_risk_event(risk_engine, sample_events):
    """An event with all risk factors should score Critical."""
    anomaly_scores = np.array([100.0, 0.0])
    attack_preds = np.array(["Brute Force", "Normal"])
    results = risk_engine.run(sample_events, anomaly_scores, attack_preds)
    
    # First event has Critical, Admin, OffHours, NewDev, NewLoc, Fails, Speed + Anomaly + Attack
    assert results.iloc[0]["risk_score"] > 90
    assert results.iloc[0]["risk_level"] == "Critical"
    
    # Second event has nothing
    assert results.iloc[1]["risk_score"] < 20
    assert results.iloc[1]["risk_level"] == "Low"


def test_risk_contributors(risk_engine, sample_events):
    """Risk contributors should reflect the event's attributes."""
    anomaly_scores = np.array([100.0, 0.0])
    attack_preds = np.array(["Brute Force", "Normal"])
    results = risk_engine.run(sample_events, anomaly_scores, attack_preds)
    
    contribs = results.iloc[0]["risk_contributors"]
    assert "High anomaly confidence" in contribs
    assert "Critical resource" in contribs
    assert "Admin privilege" in contribs
    assert "Off-hours login" in contribs
    assert "New device" in contribs
    assert "New location" in contribs
    assert "Multiple failed attempts" in contribs
    assert "Impossible travel" in contribs
    assert "Suspected Brute Force attack" in contribs
