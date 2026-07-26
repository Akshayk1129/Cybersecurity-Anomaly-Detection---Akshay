import pytest
from incident_response.response_engine import IncidentResponseEngine

class TestIncidentResponseEngine:
    @pytest.fixture
    def engine(self):
        # Initializes with the default config (which we just updated with SOAR playbooks)
        return IncidentResponseEngine(config_path="config/config.yaml")

    def test_default_actions_low_risk(self, engine):
        # Low risk should just log the event, regardless of attack type
        actions = engine.get_actions("Brute Force", "Low")
        assert "Log Event" in actions
        assert len(actions) == 1

    def test_normal_events_get_low_actions(self, engine):
        # Normal event should just log
        actions = engine.get_actions("Normal", "Critical")
        assert "Log Event" in actions
        
        actions2 = engine.get_actions("Normal", "High")
        assert "Log Event" in actions2

    def test_specific_playbook_brute_force(self, engine):
        # Medium Brute Force -> Alert SOC Analyst, Rate Limit IP
        actions_med = engine.get_actions("Brute Force", "Medium")
        assert "Rate Limit IP" in actions_med
        
        # Critical Brute Force -> Disable Account
        actions_crit = engine.get_actions("Brute Force", "Critical")
        assert "Disable Account" in actions_crit
        assert "Block Origin IP" in actions_crit

    def test_specific_playbook_lateral_movement(self, engine):
        actions = engine.get_actions("Lateral Movement", "Critical")
        assert "Isolate Device" in actions
        assert "Revoke Active Sessions" in actions

    def test_missing_playbook_fallback(self, engine):
        # If an attack type isn't defined in playbooks, it should fallback to default actions
        actions = engine.get_actions("Unknown Attack", "High")
        assert "Alert SOC Analyst" in actions
        assert "Trigger MFA Challenge" in actions

    def test_empty_risk_level(self, engine):
        actions = engine.get_actions("Brute Force", "")
        assert actions == []
