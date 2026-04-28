from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KaggleReference:
    kind: str
    ref: str


@dataclass(frozen=True)
class SourceFileSpec:
    logical_name: str
    patterns: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True)
class DatasetResearch:
    dataset_key: str
    display_name: str
    uid_choice: str
    uid_rationale: str
    training_prep: str
    references: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class DatasetProfile:
    dataset_key: str
    split_name: str
    rows: int
    columns: int
    frauds: int
    fraud_rate: float
    column_names: list[str]
    dtypes: dict[str, str]
    missing_ratio_top10: dict[str, float]
    identifier_candidates: dict[str, dict[str, Any]]
    source_files: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedArtifacts:
    train_path: Path
    test_path: Path
    extra_paths: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetRunResult:
    dataset_key: str
    display_name: str
    raw_files: dict[str, Path]
    profiles: list[DatasetProfile]
    prepared: PreparedArtifacts
    research: DatasetResearch
    metadata: dict[str, Any] = field(default_factory=dict)
