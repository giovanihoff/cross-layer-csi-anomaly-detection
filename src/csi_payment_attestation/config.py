from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    experiment_name: str
    transaction_dataset: str
    csi_source: str
    csi_expected_users: int
    csi_subcarriers_mode: str
    target_label: str
    output_dir: Path | None = None
