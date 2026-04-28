from __future__ import annotations

from multidataset_fraud.config import PATHS as LEGACY_PATHS

from ..core.paths import PROJECT_PATHS


TABULAR_DATASETS = {
    "ieee_cis": "ieee-fraud-detection",
    "sparkov": "kartik2112/fraud-detection",
    "ecommerce": "vbinh002/fraud-ecommerce",
}

TARGET_COLUMNS = {
    "ieee_cis": "isFraud",
    "sparkov": "is_fraud",
    "ecommerce": "class",
}

TABULAR_RAW_DIR = PROJECT_PATHS.data_raw
TABULAR_PROCESSED_DIR = PROJECT_PATHS.data_processed
REPORTS_DIR = PROJECT_PATHS.reports_generated

__all__ = [
    "LEGACY_PATHS",
    "PROJECT_PATHS",
    "REPORTS_DIR",
    "TABULAR_DATASETS",
    "TABULAR_PROCESSED_DIR",
    "TABULAR_RAW_DIR",
    "TARGET_COLUMNS",
]
