import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.realtime_pipeline import UEBARealTimePipeline

@pytest.fixture(scope="module")
def pipeline():
    # Only load it once for all tests in this module to save time
    return UEBARealTimePipeline("config/config.yaml")

def test_realtime_pipeline_normal_event(pipeline):
    mock_event = {
        "timestamp": "2024-03-20 10:00:00",
        "entity_id": "usr_normal_1",
        "entity_type": "User",
        "department": "Engineering",
        "login_status": "Success",
        "login_hour": 10,
        "is_off_hours": False,
        "geo_location": "US-West",
        "new_location": False,
        "new_device": False,
        "resource_accessed": "CodeRepo",
        "resource_sensitivity": "Medium",
        "privilege_level": "User",
        "session_duration_min": 45.0,
        "bytes_uploaded": 500.0,
        "bytes_downloaded": 10000.0,
        "failed_attempts": 0,
        "geo_velocity_kmph": 0.0,
        "device_fingerprint": "win_chrome_99",
        "protocol": "HTTPS",
        "auth_method": "SSO"
    }
    
    result = pipeline.predict(mock_event)
    
    assert "risk_score" in result
    assert "risk_level" in result
    assert "attack_classification" in result
    assert "explanation" in result
    assert result["is_anomaly"] == False or result["is_anomaly"] == True

def test_realtime_pipeline_attack_event(pipeline):
    mock_event = {
        "timestamp": "2024-03-20 03:00:00",
        "entity_id": "usr_attack_1",
        "entity_type": "User",
        "department": "HR",
        "login_status": "Success",
        "login_hour": 3,
        "is_off_hours": True,
        "geo_location": "RU-Moscow",
        "new_location": True,
        "new_device": True,
        "resource_accessed": "PayrollDB",
        "resource_sensitivity": "Critical",
        "privilege_level": "Admin",
        "session_duration_min": 10.0,
        "bytes_uploaded": 5000000.0,
        "bytes_downloaded": 1000.0,
        "failed_attempts": 5,
        "geo_velocity_kmph": 1200.0,
        "device_fingerprint": "linux_curl",
        "protocol": "SSH",
        "auth_method": "Password"
    }
    
    result = pipeline.predict(mock_event)
    
    assert "risk_score" in result
    assert result["risk_score"] > 50  # Should definitely be high
    assert result["risk_level"] in ["High", "Critical"]
