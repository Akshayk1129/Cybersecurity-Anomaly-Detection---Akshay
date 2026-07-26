import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any, List
from scipy.stats import ks_2samp

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """Concept Drift Detector using Kolmogorov-Smirnov (KS) Test.

    Compares distributions of continuous numerical features between a reference
    (baseline) dataset and a current evaluation dataset.

    Attributes:
        config: Full YAML configuration.
        p_value_threshold: Significance level to flag drift (default 0.05).
        tracked_features: List of feature names to monitor.
        reference_distributions: Dictionary holding baseline data for each feature.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.config = self._load_config(config_path)
        
        # Load drift config or fallback to defaults
        drift_cfg = self.config.get("models", {}).get("concept_drift", {})
        self.p_value_threshold = drift_cfg.get("p_value_threshold", 0.05)
        
        # Default features to track if not specified in config
        default_features = [
            "resource_access_entropy",
            "session_duration_min",
            "bytes_up_down_ratio_zscore",
            "failed_logins_streak_cumulative",
            "login_hour_sin",
            "profile_hour_deviation"
        ]
        self.tracked_features = drift_cfg.get("tracked_features", default_features)
        
        self.reference_distributions: Dict[str, np.ndarray] = {}
        logger.info("DriftDetector initialized (threshold=%.3f, tracking %d features).", 
                    self.p_value_threshold, len(self.tracked_features))

    @staticmethod
    def _load_config(config_path: str) -> dict:
        resolved = Path(_PROJECT_ROOT) / config_path
        if not resolved.exists():
            resolved = Path(config_path)
        with open(resolved, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def fit(self, reference_df: pd.DataFrame) -> None:
        """Store the baseline distribution for numerical features.
        
        Args:
            reference_df: The reference dataset (e.g., training data).
        """
        for feature in self.tracked_features:
            if feature in reference_df.columns:
                # Drop NaNs to ensure valid KS test calculation
                self.reference_distributions[feature] = reference_df[feature].dropna().values
            else:
                logger.warning("Feature '%s' not found in reference data. Skipping.", feature)
        
        logger.info("Drift baseline fitted on %d features.", len(self.reference_distributions))

    def detect(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute KS test between current_df and reference baseline.
        
        Args:
            current_df: The new dataset to evaluate for drift.
            
        Returns:
            Dictionary containing drift statistics and flags for each feature.
        """
        drift_results: Dict[str, Any] = {}
        
        for feature, ref_data in self.reference_distributions.items():
            if feature in current_df.columns:
                cur_data = current_df[feature].dropna().values
                
                # If there's no data to compare, skip
                if len(cur_data) == 0 or len(ref_data) == 0:
                    continue
                
                # Round to 4 decimal places to avoid false drift from CSV float precision truncation
                cur_data = np.round(cur_data, 4)
                ref_data_rounded = np.round(ref_data, 4)
                
                # Perform 2-sample Kolmogorov-Smirnov test
                stat, p_value = ks_2samp(ref_data_rounded, cur_data)
                
                is_drifting = bool(p_value < self.p_value_threshold)
                
                drift_results[feature] = {
                    "ks_statistic": float(stat),
                    "p_value": float(p_value),
                    "is_drifting": is_drifting,
                    "reference_mean": float(np.mean(ref_data)),
                    "current_mean": float(np.mean(cur_data)),
                }
            else:
                logger.warning("Feature '%s' not found in current data for drift detection.", feature)
                
        drifting_count = sum(1 for res in drift_results.values() if res["is_drifting"])
        logger.info("Drift detection complete. %d/%d features show drift.", 
                    drifting_count, len(drift_results))
                    
        return drift_results
