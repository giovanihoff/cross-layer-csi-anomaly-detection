from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    data_csi: Path
    reports_generated: Path
    docs: Path
    notebooks: Path
    env_file: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        root = Path(__file__).resolve().parents[3]
        return cls(
            root=root,
            data_raw=root / "data" / "raw",
            data_interim=root / "data" / "interim",
            data_processed=root / "data" / "processed",
            data_csi=root / "data" / "csi",
            reports_generated=root / "reports" / "generated",
            docs=root / "docs",
            notebooks=root / "notebooks",
            env_file=root / ".env.kaggle",
        )


PROJECT_PATHS = ProjectPaths.discover()
