from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import scipy.io as sio
except ImportError:  # pragma: no cover - optional dependency
    sio = None


class CSIAnalyzer:
    def __init__(self, dataset_name: str, path: str | Path) -> None:
        self.dataset_name = dataset_name
        self.path = Path(path)

    def analyze(self, limit: int = 5) -> list[dict[str, Any]]:
        print(f"\n{'=' * 50}\n[CSI STATISTICAL ANALYSIS] Dataset: {self.dataset_name.upper()}\n{'=' * 50}")

        data_files = [
            file_path
            for file_path in self.path.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in {".csv", ".mat", ".txt"}
        ]
        if not data_files:
            print(f"  [WARNING] No .csv, .mat, or .txt file found in {self.path} or subfolders.")
            return []

        analyses: list[dict[str, Any]] = []
        for file_path in data_files[:limit]:
            print(f"-> Analyzing {file_path.name}...")
            try:
                if file_path.suffix.lower() == ".mat":
                    summary = self._analyze_mat(file_path)
                elif file_path.suffix.lower() == ".txt":
                    summary = self._analyze_csv(file_path, sep=None)
                else:
                    summary = self._analyze_csv(file_path)
                analyses.append(summary)
                print("-" * 40)
            except Exception as exc:
                print(f"  [ERROR] Failed to read {file_path.name}: {exc}")

        if len(data_files) > limit:
            print(f"  [INFO] Additional {len(data_files) - limit} files found but not displayed.")
        return analyses

    def _analyze_csv(self, file_path: Path, sep: str | None = ",") -> dict[str, Any]:
        df = pd.read_csv(file_path, sep=sep, engine="python")
        rows, cols = df.shape
        num_cols = df.select_dtypes(include=[np.number]).columns
        sample_cols = list(df.columns[:5]) + ["..."] + list(df.columns[-5:]) if cols > 10 else list(df.columns)

        print(f"  - Total rows (samples): {rows:,}")
        print(f"  - Total columns: {cols:,}")
        print(f"  - Numeric Columns: {len(num_cols):,}")
        print(f"  - Sample column names: {sample_cols}")

        return {
            "file": file_path.name,
            "format": "csv" if sep == "," else "txt",
            "rows": int(rows),
            "columns": int(cols),
            "numeric_columns": int(len(num_cols)),
            "sample_columns": [str(column) for column in sample_cols],
        }

    def _analyze_mat(self, file_path: Path) -> dict[str, Any]:
        if sio is None:
            raise RuntimeError("scipy is not installed. Add the CSI dependencies to inspect .mat files.")
        mat = sio.loadmat(file_path)
        keys: list[dict[str, Any]] = []
        print("  - MAT file loaded. Main keys:")
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            shape = np.shape(value)
            keys.append({"key": key, "shape": tuple(int(dim) for dim in shape)})
            print(f"    * Key: '{key}' | Shape: {shape}")

        return {
            "file": file_path.name,
            "format": "mat",
            "keys": keys,
        }
