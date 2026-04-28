from __future__ import annotations

from pathlib import Path

from ..core.paths import PROJECT_PATHS


CSI_DATA_DIR = PROJECT_PATHS.data_csi

CSI_DATASETS: dict[str, str] = {
    "csi_gdrive": "https://drive.google.com/uc?id=1obqdXmx5yeVCnThakLADc7Ix0SJ5v1Xa",
    "csi_pmc": "https://springernature.figshare.com/ndownloader/files/49119229",
}


def ensure_csi_root(base_dir: Path | None = None) -> Path:
    root = Path(base_dir or CSI_DATA_DIR)
    root.mkdir(parents=True, exist_ok=True)
    return root


__all__ = ["CSI_DATASETS", "CSI_DATA_DIR", "ensure_csi_root"]
