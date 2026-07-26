# =============================================================================
# Cybersecurity Anomaly Detection - UEBA System
# Module: Baseline Behaviour Profiler
# =============================================================================
"""
Learns normal behaviour profiles for every entity in the dataset.

Each entity (user, service account, edge device) gets a statistical profile
capturing its typical patterns across multiple behavioural dimensions:

    - Login hours (mean, std, most common)
    - Geo-locations used (set + most common)
    - Devices used (set + most common)
    - Resources accessed (set + most common)
    - Protocols used
    - Session duration (mean, std, median)
    - Authentication methods used
    - Privilege level (most common)
    - Data transfer patterns (upload/download stats)
    - Failed login ratio

Cold-start support: entities with fewer than ``min_events`` use their
department's aggregated profile as a fallback.

Usage:
    from models.baseline.baseline_profiler import BaselineProfiler
    profiler = BaselineProfiler(config_path="config/config.yaml")
    profiles = profiler.run(df_enriched)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)


class BaselineProfiler:
    """Builds statistical behaviour profiles per entity.

    Profiles are stored as a dictionary keyed by ``entity_id``. Each
    profile is itself a dictionary of behavioural statistics.

    Cold-start entities (those with fewer than ``min_events`` events)
    receive their department's aggregated profile as a fallback.

    Attributes:
        config: Full YAML configuration.
        profiles: Dict mapping entity_id -> profile dict.
        department_profiles: Dict mapping department -> aggregated profile.
        min_events: Minimum events before an entity gets its own profile.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        min_events: int = 5,
    ) -> None:
        self.config = self._load_config(config_path)
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.department_profiles: Dict[str, Dict[str, Any]] = {}
        self.min_events = min_events
        logger.info("BaselineProfiler initialized (min_events=%d).", min_events)

    @staticmethod
    def _load_config(config_path: str) -> dict:
        resolved = Path(_PROJECT_ROOT) / config_path
        if not resolved.exists():
            resolved = Path(config_path)
        with open(resolved, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Build a single entity profile from its event group
    # ------------------------------------------------------------------
    @staticmethod
    def _build_profile(group: pd.DataFrame) -> Dict[str, Any]:
        """Compute a behavioural profile from a group of events.

        Args:
            group: DataFrame slice for one entity (or one department).

        Returns:
            Dictionary of behavioural statistics.
        """
        profile: Dict[str, Any] = {}

        profile["event_count"] = len(group)

        # --- Login hours ---
        if "login_hour" in group.columns:
            profile["login_hour_mean"] = float(group["login_hour"].mean())
            profile["login_hour_std"] = float(group["login_hour"].std(ddof=0))
            profile["login_hour_mode"] = int(group["login_hour"].mode().iloc[0])

        # --- Geo-locations ---
        if "geo_location" in group.columns:
            profile["geo_locations"] = set(group["geo_location"].dropna().unique())
            profile["geo_location_mode"] = (
                group["geo_location"].mode().iloc[0]
                if not group["geo_location"].mode().empty
                else "Unknown"
            )

        # --- Devices ---
        if "device_fingerprint" in group.columns:
            profile["devices"] = set(group["device_fingerprint"].dropna().unique())

        # --- Resources accessed ---
        if "resource_accessed" in group.columns:
            profile["resources"] = set(group["resource_accessed"].dropna().unique())
            profile["resource_mode"] = (
                group["resource_accessed"].mode().iloc[0]
                if not group["resource_accessed"].mode().empty
                else "Unknown"
            )

        # --- Protocols ---
        if "protocol" in group.columns:
            profile["protocols"] = set(group["protocol"].dropna().unique())

        # --- Session duration ---
        if "session_duration_min" in group.columns:
            profile["session_mean"] = float(group["session_duration_min"].mean())
            profile["session_std"] = float(group["session_duration_min"].std(ddof=0))
            profile["session_median"] = float(group["session_duration_min"].median())

        # --- Auth methods ---
        if "auth_method" in group.columns:
            profile["auth_methods"] = set(group["auth_method"].dropna().unique())

        # --- Privilege level ---
        if "privilege_level" in group.columns:
            profile["privilege_mode"] = (
                group["privilege_level"].mode().iloc[0]
                if not group["privilege_level"].mode().empty
                else "Unknown"
            )

        # --- Data transfer ---
        if "bytes_uploaded" in group.columns:
            profile["bytes_up_mean"] = float(group["bytes_uploaded"].mean())
            profile["bytes_up_std"] = float(group["bytes_uploaded"].std(ddof=0))
        if "bytes_downloaded" in group.columns:
            profile["bytes_down_mean"] = float(group["bytes_downloaded"].mean())
            profile["bytes_down_std"] = float(group["bytes_downloaded"].std(ddof=0))

        # --- Failed login ratio ---
        if "login_status" in group.columns:
            failed_count = (group["login_status"] == "Failed").sum()
            profile["failed_ratio"] = float(failed_count / len(group))

        return profile

    # ------------------------------------------------------------------
    # Build department-level profiles (cold-start fallback)
    # ------------------------------------------------------------------
    def build_department_profiles(self, df: pd.DataFrame) -> None:
        """Aggregate profiles by department for cold-start entities.

        Args:
            df: Raw enriched DataFrame with department column.
        """
        if "department" not in df.columns:
            logger.warning("No 'department' column. Skipping peer profiles.")
            return

        for dept, group in df.groupby("department"):
            self.department_profiles[dept] = self._build_profile(group)

        logger.info(
            "Built %d department-level profiles.", len(self.department_profiles)
        )

    # ------------------------------------------------------------------
    # Build entity-level profiles
    # ------------------------------------------------------------------
    def build_entity_profiles(self, df: pd.DataFrame) -> None:
        """Build individual profiles for every entity.

        Entities with fewer than ``min_events`` events are marked as
        cold-start and receive their department profile.

        Args:
            df: Raw enriched DataFrame.
        """
        cold_start_count = 0

        for entity_id, group in df.groupby("entity_id"):
            if len(group) >= self.min_events:
                self.profiles[entity_id] = self._build_profile(group)
                self.profiles[entity_id]["cold_start"] = False
            else:
                # Cold-start fallback: use department profile
                dept = group["department"].iloc[0] if "department" in group.columns else None
                if dept and dept in self.department_profiles:
                    self.profiles[entity_id] = self.department_profiles[dept].copy()
                    self.profiles[entity_id]["cold_start"] = True
                    self.profiles[entity_id]["fallback_department"] = dept
                else:
                    # Last resort: minimal self-profile
                    self.profiles[entity_id] = self._build_profile(group)
                    self.profiles[entity_id]["cold_start"] = True
                cold_start_count += 1

        logger.info(
            "Built %d entity profiles (%d cold-start with department fallback).",
            len(self.profiles),
            cold_start_count,
        )

    # ------------------------------------------------------------------
    # Compute deviation scores against profiles
    # ------------------------------------------------------------------
    def compute_profile_deviations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score each event's deviation from its entity's baseline profile.

        Deviation signals (higher = more anomalous):
            - ``profile_hour_deviation``: |login_hour - profile_mean| / std
            - ``profile_session_deviation``: |session - profile_mean| / std
            - ``profile_bytes_up_deviation``: |bytes_up - profile_mean| / std
            - ``profile_new_geo``: 1 if geo_location not in profile set
            - ``profile_new_resource``: 1 if resource not in profile set
            - ``profile_new_device``: 1 if device not in profile set

        Args:
            df: Raw enriched DataFrame (same used for profiling).

        Returns:
            DataFrame with profile deviation columns appended.
        """
        df = df.copy()

        hour_dev = []
        session_dev = []
        bytes_up_dev = []
        new_geo = []
        new_resource = []
        new_device = []

        for _, row in df.iterrows():
            eid = row.get("entity_id")
            prof = self.profiles.get(eid, {})

            # Hour deviation (z-score from profile)
            if "login_hour_mean" in prof and "login_hour" in row.index:
                std = prof.get("login_hour_std", 1) or 1
                hour_dev.append(
                    abs(row["login_hour"] - prof["login_hour_mean"]) / std
                )
            else:
                hour_dev.append(0.0)

            # Session deviation
            if "session_mean" in prof and "session_duration_min" in row.index:
                std = prof.get("session_std", 1) or 1
                session_dev.append(
                    abs(row["session_duration_min"] - prof["session_mean"]) / std
                )
            else:
                session_dev.append(0.0)

            # Bytes upload deviation
            if "bytes_up_mean" in prof and "bytes_uploaded" in row.index:
                std = prof.get("bytes_up_std", 1) or 1
                bytes_up_dev.append(
                    abs(row["bytes_uploaded"] - prof["bytes_up_mean"]) / std
                )
            else:
                bytes_up_dev.append(0.0)

            # New geo-location
            if "geo_locations" in prof and "geo_location" in row.index:
                new_geo.append(
                    1 if row["geo_location"] not in prof["geo_locations"] else 0
                )
            else:
                new_geo.append(0)

            # New resource
            if "resources" in prof and "resource_accessed" in row.index:
                new_resource.append(
                    1 if row["resource_accessed"] not in prof["resources"] else 0
                )
            else:
                new_resource.append(0)

            # New device
            if "devices" in prof and "device_fingerprint" in row.index:
                new_device.append(
                    1 if row["device_fingerprint"] not in prof["devices"] else 0
                )
            else:
                new_device.append(0)

        df["profile_hour_deviation"] = hour_dev
        df["profile_session_deviation"] = session_dev
        df["profile_bytes_up_deviation"] = bytes_up_dev
        df["profile_new_geo"] = new_geo
        df["profile_new_resource"] = new_resource
        df["profile_new_device"] = new_device

        logger.info(
            "Computed 6 profile deviation features for %d events.", len(df)
        )
        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_profiles(self) -> str:
        """Save profiles to disk for later use by dashboard and explainer.

        Returns:
            Path to saved profiles file.
        """
        save_dir = Path(_PROJECT_ROOT) / self.config["paths"]["saved_models_dir"]
        save_dir.mkdir(parents=True, exist_ok=True)

        # Convert sets to lists for JSON serialization via joblib
        serializable = {}
        for eid, prof in self.profiles.items():
            serializable[eid] = {
                k: list(v) if isinstance(v, set) else v
                for k, v in prof.items()
            }

        path = save_dir / "entity_profiles.joblib"
        joblib.dump(serializable, str(path))
        logger.info("Saved %d entity profiles -> %s", len(serializable), path)

        dept_path = save_dir / "department_profiles.joblib"
        dept_serializable = {
            dept: {k: list(v) if isinstance(v, set) else v for k, v in prof.items()}
            for dept, prof in self.department_profiles.items()
        }
        joblib.dump(dept_serializable, str(dept_path))
        logger.info(
            "Saved %d department profiles -> %s",
            len(dept_serializable),
            dept_path,
        )

        return str(path)

    # ------------------------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the baseline profiling pipeline.

        Args:
            df: Raw enriched DataFrame (before encoding).

        Returns:
            DataFrame with profile deviation columns appended.
        """
        logger.info("=" * 70)
        logger.info("  BASELINE PROFILING PIPELINE - START")
        logger.info("=" * 70)

        self.build_department_profiles(df)
        self.build_entity_profiles(df)
        df = self.compute_profile_deviations(df)
        self.save_profiles()

        logger.info("-" * 70)
        logger.info("  BASELINE PROFILING COMPLETE")
        logger.info("  Entity profiles  : %d", len(self.profiles))
        logger.info("  Department profiles: %d", len(self.department_profiles))
        logger.info(
            "  Cold-start entities: %d",
            sum(1 for p in self.profiles.values() if p.get("cold_start")),
        )
        logger.info("=" * 70)

        return df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from preprocessing.data_validator import DataValidator
    from feature_engineering.feature_engineer import FeatureEngineer

    validator = DataValidator()
    df_raw = validator.run()
    engineer = FeatureEngineer()
    df_enriched = engineer.run(df_raw)

    profiler = BaselineProfiler()
    df_profiled = profiler.run(df_enriched)
    print(f"\nProfiled dataset: {df_profiled.shape}")
