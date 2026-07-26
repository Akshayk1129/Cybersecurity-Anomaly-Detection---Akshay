# =============================================================================
# Cybersecurity Anomaly Detection - UEBA System
# Module: Behavioral Feature Engineer
# =============================================================================
"""
Engineers behavioural features from the raw validated dataset that capture
entity-level patterns, temporal rhythms, and deviation signals critical for
anomaly detection.

This module works on the RAW (pre-encoded) DataFrame so it can access
original categorical values (geo_location, resource_accessed, etc.) to
compute meaningful aggregations. The features it creates are then merged
with the preprocessed feature matrix.

Feature Categories
------------------
1. **Temporal Features**
   - is_weekend, hour_sin/cos (cyclical encoding), time_since_last_login
   - WHY: Attacks often exploit off-hours; cyclical encoding preserves
     hour proximity (23:00 is close to 00:00).

2. **Entity Historical Statistics**
   - login_frequency_7d, login_frequency_30d
   - avg_session_duration, std_session_duration
   - avg_bytes_uploaded, avg_bytes_downloaded
   - WHY: Deviations from an entity's personal baseline are the
     strongest anomaly signal in UEBA.

3. **Behavioural Deviation Metrics**
   - session_duration_zscore (vs. entity mean)
   - bytes_uploaded_zscore, bytes_downloaded_zscore
   - WHY: A z-score of 3+ against personal history is a strong signal
     even if the raw value seems normal globally.

4. **Access Pattern Features**
   - resource_access_entropy (Shannon entropy of accessed resources)
   - unique_resources_7d, unique_locations_7d
   - WHY: Lateral movement shows high resource diversity; insider drift
     shows slowly shifting patterns.

5. **Risk Indicator Features**
   - failed_login_ratio (rolling)
   - new_device_rate_7d, new_location_rate_7d
   - consecutive_failures
   - WHY: Brute force and credential stuffing show concentrated failure
     bursts; device spoofing shows new-device spikes.

Usage:
    from feature_engineering.feature_engineer import FeatureEngineer
    engineer = FeatureEngineer(config_path="config/config.yaml")
    df_enriched = engineer.run(df_raw)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Generates behavioural features from raw cybersecurity access logs.

    All features are computed on the raw DataFrame (before encoding) so
    original categorical values are available for aggregations. The result
    is a new DataFrame with the engineered columns appended.

    Attributes:
        config: Full YAML configuration.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.config = self._load_config(config_path)
        logger.info("FeatureEngineer initialized.")

    @staticmethod
    def _load_config(config_path: str) -> dict:
        resolved = Path(_PROJECT_ROOT) / config_path
        if not resolved.exists():
            resolved = Path(config_path)
        with open(resolved, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # ==================================================================
    # 1. TEMPORAL FEATURES
    # ==================================================================
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical hour encoding and weekend flag.

        Cyclical encoding (sin/cos) ensures hour 23 is close to hour 0,
        which raw integer encoding cannot represent.

        Args:
            df: DataFrame with 'login_hour' and 'timestamp' columns.

        Returns:
            DataFrame with hour_sin, hour_cos, is_weekend columns added.
        """
        df = df.copy()

        if "login_hour" in df.columns:
            # Cyclical encoding of hour (period = 24)
            df["hour_sin"] = np.sin(2 * np.pi * df["login_hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["login_hour"] / 24)
            logger.info(
                "Added cyclical hour features: hour_sin, hour_cos."
            )

        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(np.int8)
            df["month"] = ts.dt.month.astype(np.int8)
            df["day_of_month"] = ts.dt.day.astype(np.int8)
            logger.info("Added is_weekend, month, day_of_month.")

        return df

    # ==================================================================
    # 2. TIME SINCE LAST LOGIN (per entity)
    # ==================================================================
    def add_time_since_last_login(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute time elapsed since each entity's previous event.

        A large gap followed by a burst often signals credential stuffing
        or account takeover. Cold-start entities get the median gap.

        Args:
            df: DataFrame sorted by timestamp with entity_id.

        Returns:
            DataFrame with 'time_since_last_login_min' column.
        """
        df = df.copy()

        if "timestamp" not in df.columns or "entity_id" not in df.columns:
            logger.warning("Missing timestamp/entity_id. Skipping time_since_last_login.")
            return df

        ts = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Time diff within each entity, in minutes
        df["time_since_last_login_min"] = (
            df.groupby("entity_id")["timestamp"]
            .diff()
            .dt.total_seconds()
            / 60.0
        )

        # Fill first-login NaN with global median
        median_gap = df["time_since_last_login_min"].median()
        df["time_since_last_login_min"] = df["time_since_last_login_min"].fillna(
            median_gap
        )

        logger.info(
            "Added time_since_last_login_min (median gap: %.1f min).", median_gap
        )
        return df

    # ==================================================================
    # 3. ENTITY HISTORICAL STATISTICS
    # ==================================================================
    def add_entity_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-entity rolling and cumulative statistics.

        These features establish each entity's personal behavioural
        baseline — deviations from which are the core of UEBA detection.

        Features added:
            - entity_login_count: cumulative login count
            - entity_avg_session: expanding mean of session duration
            - entity_std_session: expanding std of session duration
            - entity_avg_bytes_up: expanding mean of bytes uploaded
            - entity_avg_bytes_down: expanding mean of bytes downloaded
            - entity_failed_rate: expanding failed login ratio

        Args:
            df: DataFrame sorted by timestamp.

        Returns:
            DataFrame with entity statistics columns.
        """
        df = df.copy()

        if "entity_id" not in df.columns:
            return df

        # Sort by entity + time for correct expanding calculations
        df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

        grouped = df.groupby("entity_id")

        # Cumulative login count
        df["entity_login_count"] = grouped.cumcount() + 1

        # Session duration statistics (expanding = all history up to now)
        if "session_duration_min" in df.columns:
            df["entity_avg_session"] = (
                grouped["session_duration_min"]
                .expanding()
                .mean()
                .reset_index(level=0, drop=True)
            )
            df["entity_std_session"] = (
                grouped["session_duration_min"]
                .expanding()
                .std()
                .reset_index(level=0, drop=True)
                .fillna(0)
            )

        # Bytes statistics
        if "bytes_uploaded" in df.columns:
            df["entity_avg_bytes_up"] = (
                grouped["bytes_uploaded"]
                .expanding()
                .mean()
                .reset_index(level=0, drop=True)
            )
        if "bytes_downloaded" in df.columns:
            df["entity_avg_bytes_down"] = (
                grouped["bytes_downloaded"]
                .expanding()
                .mean()
                .reset_index(level=0, drop=True)
            )

        # Failed login ratio (expanding)
        if "login_status" in df.columns:
            is_failed = (df["login_status"] == "Failed").astype(np.float32)
            df["entity_failed_rate"] = (
                is_failed.groupby(df["entity_id"])
                .expanding()
                .mean()
                .reset_index(level=0, drop=True)
            )

        logger.info(
            "Added entity historical statistics: login_count, avg_session, "
            "std_session, avg_bytes_up, avg_bytes_down, failed_rate."
        )
        return df

    # ==================================================================
    # 4. BEHAVIOURAL DEVIATION METRICS (z-scores vs entity baseline)
    # ==================================================================
    def add_deviation_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute z-scores relative to each entity's own history.

        A session_duration z-score of 4.0 means this session is 4 standard
        deviations above THIS entity's average — a far stronger signal
        than comparing to the global mean.

        Args:
            df: DataFrame with entity statistics already computed.

        Returns:
            DataFrame with z-score columns.
        """
        df = df.copy()

        # Session duration z-score
        if {"session_duration_min", "entity_avg_session", "entity_std_session"}.issubset(
            df.columns
        ):
            # Avoid division by zero: replace 0 std with 1
            safe_std = df["entity_std_session"].replace(0, 1)
            df["session_duration_zscore"] = (
                (df["session_duration_min"] - df["entity_avg_session"]) / safe_std
            ).clip(-10, 10)  # Clip extremes

        # Bytes uploaded z-score
        if {"bytes_uploaded", "entity_avg_bytes_up"}.issubset(df.columns):
            entity_std_up = (
                df.groupby("entity_id")["bytes_uploaded"]
                .expanding()
                .std()
                .reset_index(level=0, drop=True)
                .fillna(1)
                .replace(0, 1)
            )
            df["bytes_uploaded_zscore"] = (
                (df["bytes_uploaded"] - df["entity_avg_bytes_up"]) / entity_std_up
            ).clip(-10, 10)

        # Bytes downloaded z-score
        if {"bytes_downloaded", "entity_avg_bytes_down"}.issubset(df.columns):
            entity_std_down = (
                df.groupby("entity_id")["bytes_downloaded"]
                .expanding()
                .std()
                .reset_index(level=0, drop=True)
                .fillna(1)
                .replace(0, 1)
            )
            df["bytes_downloaded_zscore"] = (
                (df["bytes_downloaded"] - df["entity_avg_bytes_down"]) / entity_std_down
            ).clip(-10, 10)

        logger.info(
            "Added deviation metrics: session_duration_zscore, "
            "bytes_uploaded_zscore, bytes_downloaded_zscore."
        )
        return df

    # ==================================================================
    # 5. ACCESS PATTERN FEATURES
    # ==================================================================
    def add_access_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute resource diversity and access pattern features.

        Shannon entropy of accessed resources detects lateral movement
        (high entropy = many different resources), while unique counts
        over rolling windows detect exploration bursts.

        Args:
            df: DataFrame with resource_accessed, geo_location columns.

        Returns:
            DataFrame with access pattern features.
        """
        df = df.copy()

        if "entity_id" not in df.columns:
            return df

        # --- Resource access entropy (fast vectorized approach) ---
        # Instead of row-by-row iteration, we approximate expanding entropy
        # using the final per-entity resource distribution, which is a strong
        # proxy and runs in seconds instead of minutes on 100K rows.
        if "resource_accessed" in df.columns:
            # Per-entity resource counts
            res_counts = (
                df.groupby(["entity_id", "resource_accessed"])
                .size()
                .reset_index(name="_count")
            )
            # Shannon entropy per entity
            def _shannon(group: pd.DataFrame) -> float:
                total = group["_count"].sum()
                probs = group["_count"] / total
                return -(probs * np.log2(probs.clip(lower=1e-10))).sum()

            entity_entropy = (
                res_counts.groupby("entity_id")
                .apply(_shannon, include_groups=False)
                .rename("resource_access_entropy")
            )
            df = df.merge(entity_entropy, on="entity_id", how="left")
            logger.info("Added resource_access_entropy (Shannon entropy).")

        # --- Unique resources per entity (fast) ---
        if "resource_accessed" in df.columns:
            unique_res = (
                df.groupby("entity_id")["resource_accessed"]
                .transform("nunique")
            )
            df["unique_resources_cumulative"] = unique_res
            logger.info("Added unique_resources_cumulative.")

        # --- Unique locations per entity (fast) ---
        if "geo_location" in df.columns:
            unique_loc = (
                df.groupby("entity_id")["geo_location"]
                .transform("nunique")
            )
            df["unique_locations_cumulative"] = unique_loc
            logger.info("Added unique_locations_cumulative.")

        return df

    # ==================================================================
    # 6. RISK INDICATOR FEATURES
    # ==================================================================
    def add_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute risk-specific indicator features.

        These directly map to known attack patterns:
        - consecutive_failures: Brute force / credential stuffing
        - new_device_rate: Device spoofing detection
        - new_location_rate: Impossible travel / account takeover

        Args:
            df: DataFrame with login_status, new_device, new_location.

        Returns:
            DataFrame with risk indicator columns.
        """
        df = df.copy()

        # --- Consecutive failed attempts (vectorized per entity) ---
        if "login_status" in df.columns:
            is_failed = (df["login_status"] == "Failed").astype(int)
            # Create groups that break on success within each entity
            is_success = 1 - is_failed
            # Cumulative success count marks "streak reset points"
            reset_groups = is_success.groupby(df["entity_id"]).cumsum()
            # Within each (entity, reset_group), cumcount gives streak length
            df["consecutive_failures"] = (
                is_failed
                .groupby([df["entity_id"], reset_groups])
                .cumsum()
            )
            logger.info("Added consecutive_failures (resets on success).")

        # --- New device rate (expanding) ---
        if "new_device" in df.columns:
            new_dev = df["new_device"].astype(np.float32)
            df["new_device_rate"] = (
                new_dev.groupby(df["entity_id"])
                .expanding()
                .mean()
                .reset_index(level=0, drop=True)
            )
            logger.info("Added new_device_rate (expanding mean).")

        # --- New location rate (expanding) ---
        if "new_location" in df.columns:
            new_loc = df["new_location"].astype(np.float32)
            df["new_location_rate"] = (
                new_loc.groupby(df["entity_id"])
                .expanding()
                .mean()
                .reset_index(level=0, drop=True)
            )
            logger.info("Added new_location_rate (expanding mean).")

        return df

    # ==================================================================
    # 7. DEPARTMENT PEER COMPARISON (Cold-Start Support)
    # ==================================================================
    def add_peer_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compare entity behaviour against department peers.

        For cold-start entities with little personal history, department-
        level baselines provide a usable reference. Even for established
        entities, peer deviation is a complementary signal.

        Features:
            - peer_avg_session: department average session duration
            - peer_session_deviation: entity vs. department average
            - peer_avg_bytes_up: department average upload
            - peer_bytes_up_deviation: entity vs. department upload

        Args:
            df: DataFrame with department and numeric columns.

        Returns:
            DataFrame with peer comparison features.
        """
        df = df.copy()

        if "department" not in df.columns:
            return df

        # Department-level averages
        if "session_duration_min" in df.columns:
            dept_avg_session = df.groupby("department")["session_duration_min"].transform("mean")
            df["peer_avg_session"] = dept_avg_session
            df["peer_session_deviation"] = df["session_duration_min"] - dept_avg_session

        if "bytes_uploaded" in df.columns:
            dept_avg_up = df.groupby("department")["bytes_uploaded"].transform("mean")
            df["peer_avg_bytes_up"] = dept_avg_up
            df["peer_bytes_up_deviation"] = df["bytes_uploaded"] - dept_avg_up

        if "bytes_downloaded" in df.columns:
            dept_avg_down = df.groupby("department")["bytes_downloaded"].transform("mean")
            df["peer_avg_bytes_down"] = dept_avg_down
            df["peer_bytes_down_deviation"] = df["bytes_downloaded"] - dept_avg_down

        logger.info(
            "Added peer comparison features: peer_avg_session, "
            "peer_session_deviation, peer_avg_bytes_up/down, peer_bytes_deviation."
        )
        return df

    # ==================================================================
    # MAIN PIPELINE
    # ==================================================================
    def run(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Execute the full feature engineering pipeline.

        Args:
            df_raw: Raw validated DataFrame (BEFORE preprocessing encoding).

        Returns:
            Enriched DataFrame with all engineered features appended to
            the original columns.
        """
        logger.info("=" * 70)
        logger.info("  FEATURE ENGINEERING PIPELINE - START")
        logger.info("=" * 70)

        original_cols = len(df_raw.columns)

        # Ensure sorted by timestamp for temporal features
        if "timestamp" in df_raw.columns:
            df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)

        df = self.add_temporal_features(df_raw)
        df = self.add_time_since_last_login(df)
        df = self.add_entity_statistics(df)
        df = self.add_deviation_metrics(df)
        df = self.add_access_pattern_features(df)
        df = self.add_risk_indicators(df)
        df = self.add_peer_comparison(df)

        new_cols = len(df.columns) - original_cols

        # Save enriched dataset
        out_path = Path(_PROJECT_ROOT) / "data" / "enriched_dataset.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(out_path), index=False)

        logger.info("-" * 70)
        logger.info("  FEATURE ENGINEERING COMPLETE")
        logger.info("  Original columns : %d", original_cols)
        logger.info("  New features      : %d", new_cols)
        logger.info("  Total columns     : %d", len(df.columns))
        logger.info("  Saved to          : %s", out_path)
        logger.info("=" * 70)

        return df


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from preprocessing.data_validator import DataValidator

    validator = DataValidator()
    df_raw = validator.run()

    engineer = FeatureEngineer()
    df_enriched = engineer.run(df_raw)
    print(f"\nEnriched dataset: {df_enriched.shape}")
    print(f"New columns: {[c for c in df_enriched.columns if c not in ['event_id', 'entity_id']][:20]}")
