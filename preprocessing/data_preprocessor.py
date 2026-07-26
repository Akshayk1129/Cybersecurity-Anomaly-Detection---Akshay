# =============================================================================
# Cybersecurity Anomaly Detection - UEBA System
# Module: Data Preprocessor
# =============================================================================
"""
Cleans, encodes, scales, and transforms the validated dataset into a
model-ready format.

Processing steps (in order):
    1. Missing value handling (browser NaN → Not_Applicable, negative clamp)
    2. Timestamp feature extraction (month, day_of_month, is_weekend)
    3. Ordinal encoding (resource_sensitivity, privilege_level)
    4. Label encoding (binary / low-cardinality columns)
    5. One-hot encoding (nominal categorical columns)
    6. Standard scaling (continuous numeric features)
    7. Persist processed dataset and fitted transformers

All configuration is read from ``config/config.yaml`` so encoding maps,
column lists, and scaling choices can be modified without touching code.

Usage:
    from preprocessing.data_preprocessor import DataPreprocessor
    preprocessor = DataPreprocessor(config_path="config/config.yaml")
    df_processed = preprocessor.run(df_raw)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Enterprise-grade preprocessor for cybersecurity access logs.

    Transforms validated raw data into a clean, numerically-encoded, and
    scaled dataset suitable for anomaly detection and classification models.

    All fitted transformers (encoders, scalers) are persisted to
    ``saved_models/`` so the same transformations can be replayed on new
    data at inference time.

    Attributes:
        config: Parsed YAML configuration dictionary.
        prep_cfg: Shortcut to the ``preprocessing`` section of the config.
        label_encoders: Dict of fitted LabelEncoder instances keyed by
                        column name.
        ordinal_encoder: Fitted OrdinalEncoder for ordinal columns.
        scaler: Fitted StandardScaler for numeric columns.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """Initialize the preprocessor.

        Args:
            config_path: Path to the YAML configuration file.
        """
        self.config = self._load_config(config_path)
        self.prep_cfg: dict = self.config.get("preprocessing", {})

        # Fitted transformers — populated during ``run()``
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.ordinal_encoder: Optional[OrdinalEncoder] = None
        self.scaler: Optional[StandardScaler] = None

        logger.info("DataPreprocessor initialized.")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Load and return the YAML configuration."""
        resolved = Path(_PROJECT_ROOT) / config_path
        if not resolved.exists():
            resolved = Path(config_path)

        with open(resolved, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    # ------------------------------------------------------------------
    # Step 1: Missing Value Handling
    # ------------------------------------------------------------------
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to config-driven strategies.

        - ``browser`` NaN → 'Not_Applicable' (service accounts & edge
          devices do not use browsers).
        - Negative ``session_duration_min`` → clamped to 0.0.

        Args:
            df: Raw DataFrame.

        Returns:
            DataFrame with missing values handled.
        """
        df = df.copy()
        mv_cfg = self.prep_cfg.get("missing_values", {})

        # Browser: fill NaN with entity-type-aware value
        browser_fill = mv_cfg.get("browser", "Not_Applicable")
        if "browser" in df.columns:
            null_count = df["browser"].isna().sum()
            df["browser"] = df["browser"].fillna(browser_fill)
            logger.info(
                "Filled %d missing 'browser' values with '%s'.",
                null_count,
                browser_fill,
            )

        # Session duration: clamp negative values to floor
        clamp_floor = mv_cfg.get("session_duration_min_clamp", 0.0)
        if "session_duration_min" in df.columns:
            neg_count = (df["session_duration_min"] < 0).sum()
            if neg_count > 0:
                df["session_duration_min"] = df["session_duration_min"].clip(
                    lower=clamp_floor
                )
                logger.info(
                    "Clamped %d negative 'session_duration_min' values to %.1f.",
                    neg_count,
                    clamp_floor,
                )

        # Safety net: fill any remaining NaN in numeric columns with 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        remaining_nulls = df[numeric_cols].isna().sum().sum()
        if remaining_nulls > 0:
            df[numeric_cols] = df[numeric_cols].fillna(0)
            logger.info(
                "Filled %d remaining NaN in numeric columns with 0.", remaining_nulls
            )

        return df

    # ------------------------------------------------------------------
    # Step 2: Timestamp Feature Extraction
    # ------------------------------------------------------------------
    def extract_timestamp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from the timestamp column.

        New columns created:
            - ``month`` (1–12)
            - ``day_of_month`` (1–31)
            - ``is_weekend`` (bool: Saturday/Sunday)

        The original ``timestamp`` column is preserved for entity profiling
        in later phases but is not fed to models.

        Args:
            df: DataFrame with a ``timestamp`` column.

        Returns:
            DataFrame with new temporal feature columns.
        """
        df = df.copy()

        if "timestamp" not in df.columns:
            logger.warning("No 'timestamp' column found — skipping extraction.")
            return df

        ts = pd.to_datetime(df["timestamp"], errors="coerce")

        df["month"] = ts.dt.month.astype(np.int8)
        df["day_of_month"] = ts.dt.day.astype(np.int8)
        df["is_weekend"] = ts.dt.dayofweek.isin([5, 6])

        logger.info(
            "Extracted timestamp features: month, day_of_month, is_weekend."
        )
        return df

    # ------------------------------------------------------------------
    # Step 3: Ordinal Encoding
    # ------------------------------------------------------------------
    def encode_ordinal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ordinal-encode columns with a natural ordering.

        The ordering is defined in ``config.yaml → preprocessing →
        ordinal_columns``.  For example, ``resource_sensitivity`` follows
        Low < Medium < High < Critical.

        Args:
            df: DataFrame with ordinal categorical columns.

        Returns:
            DataFrame with ordinal columns replaced by integer codes.
        """
        df = df.copy()
        ordinal_cfg = self.prep_cfg.get("ordinal_columns", {})

        if not ordinal_cfg:
            return df

        cols = []
        categories = []
        for col, order in ordinal_cfg.items():
            if col in df.columns:
                cols.append(col)
                categories.append(order)
                # Ensure no unseen values slip through
                unseen = set(df[col].dropna().unique()) - set(order)
                if unseen:
                    logger.warning(
                        "Ordinal column '%s' has unseen values: %s — they "
                        "will be set to NaN.",
                        col,
                        unseen,
                    )

        if cols:
            self.ordinal_encoder = OrdinalEncoder(
                categories=categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            df[cols] = self.ordinal_encoder.fit_transform(df[cols])
            logger.info("Ordinal-encoded columns: %s", cols)

        return df

    # ------------------------------------------------------------------
    # Step 4: Label Encoding
    # ------------------------------------------------------------------
    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode binary or low-cardinality categorical columns.

        Suitable for columns like ``entity_type`` (3 values),
        ``login_status`` (2 values), and boolean flags.

        Each column gets its own ``LabelEncoder`` stored in
        ``self.label_encoders`` for inference-time reuse.

        Args:
            df: DataFrame with columns listed in ``label_encode_columns``.

        Returns:
            DataFrame with label-encoded columns.
        """
        df = df.copy()
        cols = self.prep_cfg.get("label_encode_columns", [])

        for col in cols:
            if col not in df.columns:
                continue

            # Convert booleans to string for consistent encoding
            if df[col].dtype == bool:
                df[col] = df[col].astype(str)

            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            logger.info(
                "Label-encoded '%s': %s -> %s",
                col,
                list(le.classes_),
                list(range(len(le.classes_))),
            )

        return df

    # ------------------------------------------------------------------
    # Step 5: One-Hot Encoding
    # ------------------------------------------------------------------
    def encode_onehot(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot-encode nominal categorical columns.

        Uses ``pd.get_dummies`` with ``drop_first=True`` to avoid the
        dummy-variable trap.

        Args:
            df: DataFrame with nominal categorical columns.

        Returns:
            DataFrame with one-hot columns replacing the originals.
        """
        df = df.copy()
        cols = self.prep_cfg.get("onehot_encode_columns", [])

        # Only encode columns that still exist
        cols_present = [c for c in cols if c in df.columns]
        if not cols_present:
            return df

        before_cols = len(df.columns)
        df = pd.get_dummies(df, columns=cols_present, drop_first=True, dtype=np.int8)
        after_cols = len(df.columns)

        logger.info(
            "One-hot encoded %d columns -> expanded from %d to %d total columns.",
            len(cols_present),
            before_cols,
            after_cols,
        )
        return df

    # ------------------------------------------------------------------
    # Step 6: Standard Scaling
    # ------------------------------------------------------------------
    def scale_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standard-scale (z-score normalize) continuous numeric columns.

        Scaling is critical for distance-based anomaly detectors like
        Isolation Forest. The fitted ``StandardScaler`` is stored for
        inference-time reuse.

        Args:
            df: DataFrame with numeric columns to scale.

        Returns:
            DataFrame with scaled columns replacing the originals.
        """
        df = df.copy()
        cols = self.prep_cfg.get("scale_columns", [])

        # Only scale columns that still exist
        cols_present = [c for c in cols if c in df.columns]
        if not cols_present:
            return df

        self.scaler = StandardScaler()
        df[cols_present] = self.scaler.fit_transform(df[cols_present])

        logger.info("Standard-scaled %d columns: %s", len(cols_present), cols_present)
        return df

    # ------------------------------------------------------------------
    # Step 7: Drop Columns
    # ------------------------------------------------------------------
    def drop_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Drop identifier and raw-text columns not needed for modelling.

        Target columns (``label``, ``anomaly_type``) are separated into
        their own DataFrame so they're available for supervised training
        but don't leak into features.

        Args:
            df: Full DataFrame.

        Returns:
            Tuple of (features_df, targets_df).
        """
        df = df.copy()

        # Separate targets
        target_cols = self.prep_cfg.get("target_columns", [])
        targets_present = [c for c in target_cols if c in df.columns]
        targets_df = df[targets_present].copy() if targets_present else pd.DataFrame()

        # Drop identifiers + targets
        drop_cols = self.prep_cfg.get("drop_columns", [])
        all_drop = list(set(drop_cols + target_cols))
        cols_to_drop = [c for c in all_drop if c in df.columns]

        # Also drop the raw timestamp (features already extracted)
        if "timestamp" in df.columns:
            cols_to_drop.append("timestamp")

        df = df.drop(columns=cols_to_drop, errors="ignore")
        logger.info("Dropped columns: %s", cols_to_drop)

        return df, targets_df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_transformers(self) -> None:
        """Persist fitted encoders and scaler to disk for inference reuse."""
        save_dir = Path(_PROJECT_ROOT) / self.config["paths"]["saved_models_dir"]
        save_dir.mkdir(parents=True, exist_ok=True)

        artifacts = {
            "label_encoders": self.label_encoders,
            "ordinal_encoder": self.ordinal_encoder,
            "scaler": self.scaler,
        }

        for name, obj in artifacts.items():
            if obj is not None:
                path = save_dir / f"{name}.joblib"
                joblib.dump(obj, str(path))
                logger.info("Saved %s -> %s", name, path)

    def save_dataset(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        entity_ids: pd.Series,
    ) -> str:
        """Save the processed dataset to CSV.

        The entity_id is prepended as the first column (not a feature but
        needed for entity-level profiling in Phase 4).

        Args:
            features: Processed feature DataFrame.
            targets: Target labels DataFrame.
            entity_ids: Entity ID series.

        Returns:
            Path to the saved CSV file.
        """
        out_path = Path(_PROJECT_ROOT) / self.config["paths"]["processed_dataset"]
        out_path.parent.mkdir(parents=True, exist_ok=True)

        combined = pd.concat(
            [entity_ids.reset_index(drop=True),
             features.reset_index(drop=True),
             targets.reset_index(drop=True)],
            axis=1,
        )

        combined.to_csv(str(out_path), index=False)
        logger.info(
            "Processed dataset saved: %s (%d rows × %d columns).",
            out_path,
            combined.shape[0],
            combined.shape[1],
        )
        return str(out_path)

    def load_transformers(self) -> None:
        """Load fitted encoders and scaler from disk for real-time inference."""
        save_dir = Path(_PROJECT_ROOT) / self.config["paths"]["saved_models_dir"]
        
        le_path = save_dir / "label_encoders.joblib"
        if le_path.exists():
            self.label_encoders = joblib.load(str(le_path))
            
        oe_path = save_dir / "ordinal_encoder.joblib"
        if oe_path.exists():
            self.ordinal_encoder = joblib.load(str(oe_path))
            
        sc_path = save_dir / "scaler.joblib"
        if sc_path.exists():
            self.scaler = joblib.load(str(sc_path))
            
        logger.info("Transformers loaded for inference.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pre-fitted transformations to a single event or batch.
        
        Used by the real-time inference pipeline to encode and scale
        events without fitting new encoders.
        """
        df = df.copy()
        
        # 1. Missing values
        df = self.handle_missing_values(df)
        
        # 2. Timestamp features
        df = self.extract_timestamp_features(df)
        
        # 3. Ordinal encoding
        if self.ordinal_encoder is not None:
            ordinal_cfg = self.prep_cfg.get("ordinal_columns", {})
            cols = [c for c in ordinal_cfg.keys() if c in df.columns]
            if cols:
                df[cols] = self.ordinal_encoder.transform(df[cols])
                
        # 4. Label encoding
        if self.label_encoders:
            for col, encoder in self.label_encoders.items():
                if col in df.columns:
                    # Handle unseen labels gracefully by mapping to a default/unknown if possible, 
                    # but LabelEncoder doesn't support unknown out-of-the-box in sklearn easily.
                    # As a safe inference fallback, we can use the first class or a try/except.
                    # For a robust UEBA, unseen labels might be mapped to -1.
                    known_classes = set(encoder.classes_)
                    df[col] = df[col].apply(
                        lambda x: encoder.transform([x])[0] if x in known_classes else -1
                    )
                    
        # 5. One-hot encoding
        df = self.encode_onehot(df)
        
        # 6. Drop non-feature columns
        features, _ = self.drop_columns(df)
        
        # 7. Scale numeric features
        if self.scaler is not None:
            cols = self.prep_cfg.get("scale_columns", [])
            cols_present = [c for c in cols if c in features.columns]
            if cols_present:
                features[cols_present] = self.scaler.transform(features[cols_present])
                
        # Fill any remaining NaNs with 0 (safe fallback for models)
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        return features

    # ------------------------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Execute the full preprocessing pipeline.

        Args:
            df: Raw validated DataFrame from Phase 1.

        Returns:
            Tuple of (features, targets, entity_ids):
                - features: Fully preprocessed numeric feature matrix.
                - targets: Target labels (label, anomaly_type).
                - entity_ids: Entity IDs for profiling.
        """
        logger.info("=" * 70)
        logger.info("  PREPROCESSING PIPELINE — START")
        logger.info("=" * 70)

        # Preserve entity IDs before any transforms
        entity_id_col = self.prep_cfg.get("entity_id_column", "entity_id")
        entity_ids = df[entity_id_col].copy() if entity_id_col in df.columns else pd.Series()

        # Step 1: Missing values
        df = self.handle_missing_values(df)

        # Step 2: Timestamp features
        df = self.extract_timestamp_features(df)

        # Step 3: Ordinal encoding
        df = self.encode_ordinal(df)

        # Step 4: Label encoding
        df = self.encode_labels(df)

        # Step 5: One-hot encoding
        df = self.encode_onehot(df)

        # Step 6: Drop non-feature columns and separate targets
        features, targets = self.drop_columns(df)

        # Step 7: Scale numeric features
        features = self.scale_numeric(features)

        # Persist
        self.save_transformers()
        self.save_dataset(features, targets, entity_ids)

        logger.info("-" * 70)
        logger.info("  PREPROCESSING COMPLETE")
        logger.info(
            "  Feature matrix : %d rows × %d columns",
            features.shape[0],
            features.shape[1],
        )
        logger.info("  Target columns : %s", list(targets.columns))
        logger.info("=" * 70)

        return features, targets, entity_ids


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from preprocessing.data_validator import DataValidator

    validator = DataValidator()
    df_raw = validator.run()

    preprocessor = DataPreprocessor()
    features, targets, entity_ids = preprocessor.run(df_raw)
    print(f"\nFeatures: {features.shape}")
    print(f"Targets: {targets.shape}")
    print(f"Sample feature columns: {list(features.columns[:15])}")
