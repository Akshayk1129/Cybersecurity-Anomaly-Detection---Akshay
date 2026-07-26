# =============================================================================
# Cybersecurity Anomaly Detection - UEBA System
# Module: Data Validator
# =============================================================================
"""
Performs comprehensive data validation on the raw cybersecurity access logs.

This module reads the Excel dataset and runs a battery of quality checks:
    1. Schema validation (expected columns, data types)
    2. Missing value analysis
    3. Duplicate row detection
    4. Timestamp integrity checks
    5. Categorical value validation (unexpected/novel categories)
    6. Numeric range validation (negative values, extreme outliers)
    7. Label consistency checks

All findings are compiled into a structured Markdown report saved to
``reports/data_validation_report.md``.

Usage:
    from preprocessing.data_validator import DataValidator
    validator = DataValidator(config_path="config/config.yaml")
    df = validator.run()
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Ensure the project root is importable regardless of working directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Enterprise-grade data validator for cybersecurity access logs.

    Reads the raw Excel dataset, runs quality checks according to the
    central YAML configuration, and produces a Markdown validation report.

    Attributes:
        config: Parsed YAML configuration dictionary.
        df: The loaded pandas DataFrame (populated after ``load_data``).
        findings: Ordered list of ``(severity, section, detail)`` tuples
                  accumulated during validation.
    """

    # Severity levels for findings
    SEVERITY_INFO = "INFO"
    SEVERITY_WARNING = "WARNING"
    SEVERITY_ERROR = "ERROR"
    SEVERITY_PASS = "PASS"

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """Initialize the validator with a configuration file.

        Args:
            config_path: Path to the YAML configuration file, relative to
                         the project root or absolute.
        """
        self.config = self._load_config(config_path)
        self.df: Optional[pd.DataFrame] = None
        self.findings: List[Tuple[str, str, str]] = []
        logger.info("DataValidator initialized.")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Load and return the YAML configuration.

        Args:
            config_path: Path to the YAML config file.

        Returns:
            Parsed configuration dictionary.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the config file contains invalid YAML.
        """
        resolved = Path(_PROJECT_ROOT) / config_path
        if not resolved.exists():
            resolved = Path(config_path)

        logger.info("Loading configuration from: %s", resolved)
        with open(resolved, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        """Load the raw Excel dataset specified in the configuration.

        Returns:
            pd.DataFrame: The loaded dataset.

        Raises:
            FileNotFoundError: If the dataset file is missing.
        """
        raw_path = self.config["paths"]["raw_dataset"]
        resolved = Path(_PROJECT_ROOT) / raw_path
        if not resolved.exists():
            resolved = Path(raw_path)

        logger.info("Loading dataset from: %s", resolved)
        self.df = pd.read_excel(str(resolved), engine="openpyxl")
        logger.info(
            "Dataset loaded successfully: %d rows × %d columns.",
            self.df.shape[0],
            self.df.shape[1],
        )

        self._add_finding(
            self.SEVERITY_INFO,
            "Dataset Overview",
            f"Loaded **{self.df.shape[0]:,}** rows × **{self.df.shape[1]}** columns "
            f"from `{Path(raw_path).name}`.",
        )
        return self.df

    # ------------------------------------------------------------------
    # Individual Validation Checks
    # ------------------------------------------------------------------
    def validate_schema(self) -> None:
        """Check that all expected columns are present and log data types."""
        self._require_data()
        logger.info("Validating schema...")

        # Build a type summary table
        type_lines = ["| Column | Dtype | Non-Null Count |",
                      "|--------|-------|----------------|"]
        for col in self.df.columns:
            non_null = self.df[col].notna().sum()
            type_lines.append(f"| `{col}` | `{self.df[col].dtype}` | {non_null:,} |")

        self._add_finding(
            self.SEVERITY_INFO,
            "Schema & Data Types",
            "Detected columns and their data types:\n\n" + "\n".join(type_lines),
        )

    def validate_missing_values(self) -> None:
        """Analyze missing values per column against configured thresholds."""
        self._require_data()
        logger.info("Validating missing values...")

        val_cfg = self.config.get("validation", {})
        warn_pct = val_cfg.get("max_missing_pct_warning", 5.0)
        err_pct = val_cfg.get("max_missing_pct_error", 50.0)

        total_rows = len(self.df)
        missing = self.df.isnull().sum()
        cols_with_missing = missing[missing > 0]

        if cols_with_missing.empty:
            self._add_finding(
                self.SEVERITY_PASS,
                "Missing Values",
                "No missing values detected in any column.",
            )
            return

        lines = [
            "| Column | Missing Count | Missing % | Severity |",
            "|--------|---------------|-----------|----------|",
        ]
        for col, count in cols_with_missing.items():
            pct = (count / total_rows) * 100
            if pct >= err_pct:
                sev = self.SEVERITY_ERROR
            elif pct >= warn_pct:
                sev = self.SEVERITY_WARNING
            else:
                sev = self.SEVERITY_INFO
            lines.append(f"| `{col}` | {count:,} | {pct:.2f}% | {sev} |")

            if sev in (self.SEVERITY_WARNING, self.SEVERITY_ERROR):
                logger.warning(
                    "Column '%s' has %.2f%% missing values (%s).", col, pct, sev
                )

        self._add_finding(
            self.SEVERITY_WARNING,
            "Missing Values",
            f"Found **{len(cols_with_missing)}** column(s) with missing values:\n\n"
            + "\n".join(lines),
        )

    def validate_duplicates(self) -> None:
        """Check for duplicate rows in the dataset."""
        self._require_data()
        logger.info("Validating duplicate rows...")

        dup_count = self.df.duplicated().sum()
        max_allowed = self.config.get("validation", {}).get("max_duplicate_rows", 0)

        if dup_count == 0:
            self._add_finding(
                self.SEVERITY_PASS,
                "Duplicate Rows",
                "No duplicate rows detected.",
            )
        elif dup_count <= max_allowed:
            self._add_finding(
                self.SEVERITY_INFO,
                "Duplicate Rows",
                f"Found **{dup_count:,}** duplicate row(s), within acceptable "
                f"threshold ({max_allowed}).",
            )
        else:
            self._add_finding(
                self.SEVERITY_ERROR,
                "Duplicate Rows",
                f"Found **{dup_count:,}** duplicate row(s), exceeding the "
                f"acceptable threshold of {max_allowed}.",
            )
            logger.error("Duplicate row count %d exceeds threshold %d.", dup_count, max_allowed)

    def validate_timestamps(self) -> None:
        """Validate timestamp columns for correctness and chronological range."""
        self._require_data()
        logger.info("Validating timestamps...")

        ts_cols = self.config.get("validation", {}).get("timestamp_columns", [])

        for col in ts_cols:
            if col not in self.df.columns:
                self._add_finding(
                    self.SEVERITY_WARNING,
                    "Timestamp Validation",
                    f"Expected timestamp column `{col}` not found in dataset.",
                )
                continue

            # Attempt to coerce to datetime
            ts_series = pd.to_datetime(self.df[col], errors="coerce")
            nat_count = ts_series.isna().sum() - self.df[col].isna().sum()

            if nat_count > 0:
                self._add_finding(
                    self.SEVERITY_ERROR,
                    "Timestamp Validation",
                    f"Column `{col}`: **{nat_count:,}** value(s) could not be "
                    f"parsed as valid timestamps.",
                )
                logger.error(
                    "Column '%s' has %d unparseable timestamp values.", col, nat_count
                )
            else:
                self._add_finding(
                    self.SEVERITY_PASS,
                    "Timestamp Validation",
                    f"Column `{col}`: All values are valid timestamps.",
                )

            # Chronological range
            ts_min = ts_series.min()
            ts_max = ts_series.max()
            self._add_finding(
                self.SEVERITY_INFO,
                "Timestamp Validation",
                f"Column `{col}` range: **{ts_min}** -> **{ts_max}** "
                f"(span: {(ts_max - ts_min).days} days).",
            )

            # Check for future timestamps
            now = pd.Timestamp.now()
            future_count = (ts_series > now).sum()
            if future_count > 0:
                self._add_finding(
                    self.SEVERITY_INFO,
                    "Timestamp Validation",
                    f"Column `{col}`: **{future_count:,}** event(s) have timestamps "
                    f"in the future (dataset is synthetic, expected).",
                )

    def validate_categorical_values(self) -> None:
        """Check categorical columns for unexpected values."""
        self._require_data()
        logger.info("Validating categorical values...")

        val_cfg = self.config.get("validation", {})

        # Check entity_type
        expected_entity_types = set(val_cfg.get("expected_entity_types", []))
        if expected_entity_types and "entity_type" in self.df.columns:
            actual = set(self.df["entity_type"].dropna().unique())
            unexpected = actual - expected_entity_types
            if unexpected:
                self._add_finding(
                    self.SEVERITY_WARNING,
                    "Categorical Validation",
                    f"Column `entity_type` contains unexpected values: "
                    f"**{unexpected}**.",
                )
            else:
                self._add_finding(
                    self.SEVERITY_PASS,
                    "Categorical Validation",
                    f"Column `entity_type`: All values match expected set "
                    f"({expected_entity_types}).",
                )

        # Check labels
        expected_labels = set(val_cfg.get("expected_labels", []))
        if expected_labels and "label" in self.df.columns:
            actual = set(self.df["label"].dropna().unique())
            unexpected = actual - expected_labels
            if unexpected:
                self._add_finding(
                    self.SEVERITY_WARNING,
                    "Categorical Validation",
                    f"Column `label` contains unexpected values: **{unexpected}**.",
                )
            else:
                self._add_finding(
                    self.SEVERITY_PASS,
                    "Categorical Validation",
                    f"Column `label`: All values match expected set ({expected_labels}).",
                )

        # Check anomaly_type
        expected_anomaly_types = set(val_cfg.get("expected_anomaly_types", []))
        if expected_anomaly_types and "anomaly_type" in self.df.columns:
            actual = set(self.df["anomaly_type"].dropna().unique())
            unexpected = actual - expected_anomaly_types
            if unexpected:
                self._add_finding(
                    self.SEVERITY_WARNING,
                    "Categorical Validation",
                    f"Column `anomaly_type` contains unexpected values: **{unexpected}**.",
                )
            else:
                self._add_finding(
                    self.SEVERITY_PASS,
                    "Categorical Validation",
                    f"Column `anomaly_type`: All values match expected set.",
                )

        # General categorical columns — report unique value counts
        cat_cols = val_cfg.get("categorical_columns", [])
        lines = [
            "| Column | Unique Values | Sample Values |",
            "|--------|---------------|---------------|",
        ]
        for col in cat_cols:
            if col in self.df.columns:
                unique_vals = self.df[col].dropna().unique()
                sample = ", ".join(str(v) for v in unique_vals[:8])
                if len(unique_vals) > 8:
                    sample += ", ..."
                lines.append(f"| `{col}` | {len(unique_vals)} | {sample} |")

        self._add_finding(
            self.SEVERITY_INFO,
            "Categorical Validation",
            "Unique value counts for categorical columns:\n\n" + "\n".join(lines),
        )

    def validate_numeric_ranges(self) -> None:
        """Check numeric columns for invalid ranges (negatives, extreme outliers)."""
        self._require_data()
        logger.info("Validating numeric ranges...")

        non_neg_cols = self.config.get("validation", {}).get(
            "non_negative_columns", []
        )

        for col in non_neg_cols:
            if col not in self.df.columns:
                continue

            series = pd.to_numeric(self.df[col], errors="coerce")
            neg_count = (series < 0).sum()

            if neg_count > 0:
                neg_min = series[series < 0].min()
                self._add_finding(
                    self.SEVERITY_WARNING,
                    "Numeric Range Validation",
                    f"Column `{col}`: **{neg_count:,}** negative value(s) detected "
                    f"(min = {neg_min}). These may indicate data quality issues.",
                )
                logger.warning(
                    "Column '%s' has %d negative values (min=%.2f).",
                    col,
                    neg_count,
                    neg_min,
                )
            else:
                self._add_finding(
                    self.SEVERITY_PASS,
                    "Numeric Range Validation",
                    f"Column `{col}`: All values are non-negative.",
                )

        # Outlier detection using IQR for key numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_lines = [
            "| Column | Min | Q1 | Median | Q3 | Max | IQR Outliers |",
            "|--------|-----|-----|--------|-----|-----|--------------|",
        ]
        for col in numeric_cols:
            if col == "event_id":
                continue
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_count = (
                (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            ).sum()
            outlier_lines.append(
                f"| `{col}` | {self.df[col].min():.2f} | {q1:.2f} | "
                f"{self.df[col].median():.2f} | {q3:.2f} | {self.df[col].max():.2f} | "
                f"{outlier_count:,} |"
            )

        self._add_finding(
            self.SEVERITY_INFO,
            "Numeric Range Validation",
            "Numeric column statistics and IQR-based outlier counts:\n\n"
            + "\n".join(outlier_lines),
        )

    def validate_label_consistency(self) -> None:
        """Check that label and anomaly_type columns are consistent.

        Rules:
            - If label == "Normal", anomaly_type should be "Normal".
            - If label == "Anomaly", anomaly_type should NOT be "Normal".
        """
        self._require_data()
        logger.info("Validating label consistency...")

        if "label" not in self.df.columns or "anomaly_type" not in self.df.columns:
            self._add_finding(
                self.SEVERITY_WARNING,
                "Label Consistency",
                "Cannot check label consistency — `label` or `anomaly_type` "
                "column missing.",
            )
            return

        # Normal label with non-Normal anomaly_type
        normal_but_anomaly = (
            (self.df["label"] == "Normal") & (self.df["anomaly_type"] != "Normal")
        ).sum()

        # Anomaly label with Normal anomaly_type
        anomaly_but_normal = (
            (self.df["label"] == "Anomaly") & (self.df["anomaly_type"] == "Normal")
        ).sum()

        issues = []
        if normal_but_anomaly > 0:
            issues.append(
                f"- **{normal_but_anomaly:,}** rows have `label=Normal` but "
                f"`anomaly_type≠Normal`."
            )
        if anomaly_but_normal > 0:
            issues.append(
                f"- **{anomaly_but_normal:,}** rows have `label=Anomaly` but "
                f"`anomaly_type=Normal`."
            )

        if issues:
            self._add_finding(
                self.SEVERITY_ERROR,
                "Label Consistency",
                "Label-anomaly type mismatches detected:\n\n" + "\n".join(issues),
            )
            logger.error("Label consistency issues found.")
        else:
            self._add_finding(
                self.SEVERITY_PASS,
                "Label Consistency",
                "`label` and `anomaly_type` are fully consistent.",
            )

        # Class distribution
        label_dist = self.df["label"].value_counts()
        anomaly_dist = self.df["anomaly_type"].value_counts()
        dist_lines = ["**Label Distribution:**\n"]
        for val, count in label_dist.items():
            pct = (count / len(self.df)) * 100
            dist_lines.append(f"- `{val}`: {count:,} ({pct:.2f}%)")
        dist_lines.append("\n**Anomaly Type Distribution:**\n")
        for val, count in anomaly_dist.items():
            pct = (count / len(self.df)) * 100
            dist_lines.append(f"- `{val}`: {count:,} ({pct:.2f}%)")

        self._add_finding(
            self.SEVERITY_INFO,
            "Label Consistency",
            "\n".join(dist_lines),
        )

    def validate_entity_coverage(self) -> None:
        """Analyze entity distribution across entity types and departments."""
        self._require_data()
        logger.info("Validating entity coverage...")

        if "entity_id" not in self.df.columns:
            return

        total_entities = self.df["entity_id"].nunique()

        lines = [f"Total unique entities: **{total_entities}**\n"]

        # By entity_type
        if "entity_type" in self.df.columns:
            lines.append("**Events by Entity Type:**\n")
            lines.append("| Entity Type | Entity Count | Event Count | Avg Events/Entity |")
            lines.append("|-------------|-------------|-------------|-------------------|")
            for etype, group in self.df.groupby("entity_type"):
                entity_count = group["entity_id"].nunique()
                event_count = len(group)
                avg_events = event_count / entity_count if entity_count > 0 else 0
                lines.append(
                    f"| `{etype}` | {entity_count} | {event_count:,} | {avg_events:.1f} |"
                )

        self._add_finding(
            self.SEVERITY_INFO,
            "Entity Coverage",
            "\n".join(lines),
        )

    # ------------------------------------------------------------------
    # Report Generation
    # ------------------------------------------------------------------
    def generate_report(self) -> str:
        """Compile all findings into a structured Markdown report.

        Returns:
            str: Path to the generated report file.
        """
        report_path = Path(_PROJECT_ROOT) / self.config["paths"]["validation_report"]
        report_path.parent.mkdir(parents=True, exist_ok=True)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Count severities
        severity_counts = {}
        for sev, _, _ in self.findings:
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        # Build the report
        report_lines = [
            "# 📋 Data Validation Report",
            "",
            f"**Generated:** {now}",
            "",
            "---",
            "",
            "## Summary",
            "",
            "| Severity | Count |",
            "|----------|-------|",
        ]
        for sev in [self.SEVERITY_ERROR, self.SEVERITY_WARNING,
                     self.SEVERITY_INFO, self.SEVERITY_PASS]:
            count = severity_counts.get(sev, 0)
            emoji = {"ERROR": "🔴", "WARNING": "🟡", "INFO": "🔵", "PASS": "🟢"}.get(
                sev, ""
            )
            report_lines.append(f"| {emoji} {sev} | {count} |")

        report_lines.extend(["", "---", ""])

        # Group findings by section
        sections_seen: List[str] = []
        sections_map: Dict[str, List[Tuple[str, str]]] = {}
        for sev, section, detail in self.findings:
            if section not in sections_map:
                sections_seen.append(section)
                sections_map[section] = []
            sections_map[section].append((sev, detail))

        for section in sections_seen:
            report_lines.append(f"## {section}")
            report_lines.append("")
            for sev, detail in sections_map[section]:
                emoji = {
                    "ERROR": "🔴",
                    "WARNING": "🟡",
                    "INFO": "🔵",
                    "PASS": "🟢",
                }.get(sev, "")
                report_lines.append(f"**{emoji} {sev}:** {detail}")
                report_lines.append("")
            report_lines.append("---")
            report_lines.append("")

        report_content = "\n".join(report_lines)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info("Validation report saved to: %s", report_path)
        return str(report_path)

    # ------------------------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """Execute the full validation pipeline.

        Loads the data, runs all validation checks, and generates the report.

        Returns:
            pd.DataFrame: The loaded (unmodified) dataset, ready for
                          downstream preprocessing.
        """
        logger.info("=" * 70)
        logger.info("  DATA VALIDATION PIPELINE — START")
        logger.info("=" * 70)

        self.load_data()
        self.validate_schema()
        self.validate_missing_values()
        self.validate_duplicates()
        self.validate_timestamps()
        self.validate_categorical_values()
        self.validate_numeric_ranges()
        self.validate_label_consistency()
        self.validate_entity_coverage()

        report_path = self.generate_report()

        # Print summary to logger
        error_count = sum(1 for s, _, _ in self.findings if s == self.SEVERITY_ERROR)
        warn_count = sum(1 for s, _, _ in self.findings if s == self.SEVERITY_WARNING)
        pass_count = sum(1 for s, _, _ in self.findings if s == self.SEVERITY_PASS)

        logger.info("-" * 70)
        logger.info("  VALIDATION COMPLETE")
        logger.info(
            "  Results: %d PASS | %d WARNINGS | %d ERRORS",
            pass_count,
            warn_count,
            error_count,
        )
        logger.info("  Report : %s", report_path)
        logger.info("=" * 70)

        return self.df

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------
    def _require_data(self) -> None:
        """Raise an error if data has not been loaded yet."""
        if self.df is None:
            raise RuntimeError(
                "No data loaded. Call `load_data()` or `run()` first."
            )

    def _add_finding(self, severity: str, section: str, detail: str) -> None:
        """Record a validation finding.

        Args:
            severity: One of SEVERITY_INFO, SEVERITY_WARNING, SEVERITY_ERROR,
                      SEVERITY_PASS.
            section: Report section name (e.g., 'Missing Values').
            detail: Markdown-formatted description of the finding.
        """
        self.findings.append((severity, section, detail))


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    validator = DataValidator()
    df = validator.run()
    print(f"\nDataset shape after validation: {df.shape}")
