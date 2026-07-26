import sys
from pathlib import Path
import yaml
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)


class IncidentResponseEngine:
    """Security Orchestration, Automation, and Response (SOAR) Engine.
    
    Maps anomaly types and risk levels to automated playbook actions.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.config = self._load_config(config_path)
        ir_config = self.config.get("models", {}).get("incident_response", {})
        
        self.default_actions = ir_config.get("default_actions", {
            "Low": ["Log Event"],
            "Medium": ["Alert SOC Analyst"],
            "High": ["Alert SOC Analyst", "Trigger MFA Challenge"],
            "Critical": ["Alert SOC Analyst", "Disable Account", "Block Origin IP"]
        })
        self.playbooks = ir_config.get("playbooks", {})
        
        logger.info("IncidentResponseEngine initialized with %d specific playbooks.", len(self.playbooks))

    @staticmethod
    def _load_config(config_path: str) -> dict:
        resolved = Path(_PROJECT_ROOT) / config_path
        if not resolved.exists():
            resolved = Path(config_path)
        with open(resolved, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_actions(self, anomaly_type: str, risk_level: str) -> List[str]:
        """Get recommended automated actions based on the attack type and risk level.
        
        Args:
            anomaly_type: The classified attack type (e.g., 'Brute Force').
            risk_level: The risk severity ('Low', 'Medium', 'High', 'Critical').
            
        Returns:
            A list of recommended response actions as strings.
        """
        if not risk_level:
            return []
            
        # If normal or low risk, just return default low actions
        if risk_level == "Low" or anomaly_type == "Normal":
            return self.default_actions.get("Low", ["Log Event"])
            
        # Lookup specific playbook
        playbook = self.playbooks.get(anomaly_type)
        
        if playbook and risk_level in playbook:
            return playbook[risk_level]
            
        # Fallback to default actions for that risk level if no specific playbook exists
        return self.default_actions.get(risk_level, [])
