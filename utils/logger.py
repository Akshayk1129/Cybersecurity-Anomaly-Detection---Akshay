# =============================================================================
# Cybersecurity Anomaly Detection - UEBA System
# Module: Centralized Logging Utility
# =============================================================================
"""
Provides a centralized, reusable logging setup for the entire UEBA system.

All modules import `get_logger` to obtain a named logger instance that writes
to both the console (stdout) and a rotating log file. Configuration is read
from config/config.yaml so log levels and formats can be changed without
touching code.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline started.")
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import yaml


def _load_logging_config() -> dict:
    """Load logging configuration from the central YAML config.

    Returns:
        dict: Logging configuration section, or sensible defaults if the
              config file is unavailable.
    """
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    defaults = {
        "level": "INFO",
        "format": "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "log_file": "logs/ueba_system.log",
    }

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)
        return full_config.get("logging", defaults)
    except (FileNotFoundError, yaml.YAMLError):
        return defaults


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Create or retrieve a named logger with console and file handlers.

    If the logger already has handlers (e.g., from a previous call), it is
    returned as-is to avoid duplicate log lines.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_cfg = _load_logging_config()

    logger = logging.getLogger(name or "ueba")

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt=log_cfg.get("format"),
        datefmt=log_cfg.get("date_format"),
    )

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File handler ---
    log_file = Path(__file__).resolve().parent.parent / log_cfg.get(
        "log_file", "logs/ueba_system.log"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
