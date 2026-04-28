from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    reports_generated: Path
    env_file: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        root = Path(__file__).resolve().parents[2]
        return cls(
            root=root,
            data_raw=root / "data" / "raw",
            data_interim=root / "data" / "interim",
            data_processed=root / "data" / "processed",
            reports_generated=root / "reports" / "generated",
            env_file=root / ".env.kaggle",
        )


PATHS = ProjectPaths.discover()

