from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_SUBCARRIERS = 108


class CSIHarmonizer:
    def __init__(self, base_path: str | Path, target_subcarriers: int = TARGET_SUBCARRIERS) -> None:
        self.base_path = Path(base_path)
        self.target_subcarriers = int(target_subcarriers)
        self.in_base = self.base_path / "smoothed_amplitudes"
        self.out_base = self.base_path / "harmonized_amplitudes"
        self.out_base.mkdir(parents=True, exist_ok=True)

    def _sort_sc_cols(self, df: pd.DataFrame) -> list[str]:
        cols = [column for column in df.columns if str(column).startswith("subcarrier_")]
        return sorted(cols, key=lambda name: int(str(name).split("_")[1]))

    def _resample_matrix(self, matrix: np.ndarray) -> tuple[np.ndarray, int, bool]:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2:
            matrix = matrix.reshape(matrix.shape[0], -1)
        native = int(matrix.shape[1])
        if native == self.target_subcarriers:
            return matrix.astype(np.float32), native, False

        src_grid = np.linspace(0.0, 1.0, native, dtype=np.float32)
        dst_grid = np.linspace(0.0, 1.0, self.target_subcarriers, dtype=np.float32)
        out = np.vstack([np.interp(dst_grid, src_grid, row).astype(np.float32) for row in matrix])
        return out, native, True

    def _to_df(self, matrix: np.ndarray) -> pd.DataFrame:
        cols = [f"subcarrier_{idx:03d}" for idx in range(matrix.shape[1])]
        return pd.DataFrame(matrix, columns=cols)

    def _process_dataframe_files(self, files: list[Path], out_dir: Path, label: str) -> list[dict[str, object]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        stats: list[dict[str, object]] = []

        for file_path in files:
            df = pd.read_parquet(file_path) if file_path.suffix.lower() == ".parquet" else pd.read_csv(file_path)
            sc_cols = self._sort_sc_cols(df)
            if not sc_cols:
                continue
            matrix = df[sc_cols].to_numpy(dtype=np.float32)
            harmonized, native, resampled = self._resample_matrix(matrix)
            out_df = self._to_df(harmonized)
            out_name = file_path.stem + "_harm.parquet"
            out_df.to_parquet(out_dir / out_name, index=False)
            stats.append(
                {
                    "dataset": label,
                    "file": file_path.name,
                    "rows": int(matrix.shape[0]),
                    "native_subcarriers": native,
                    "target_subcarriers": int(harmonized.shape[1]),
                    "resampled": bool(resampled),
                }
            )

        return stats

    def process_all(self) -> pd.DataFrame:
        stats: list[dict[str, object]] = []
        stats.extend(
            self._process_dataframe_files(
                sorted((self.in_base / "csi_gdrive_smoothed").glob("*.csv")),
                self.out_base / "csi_gdrive_harmonized",
                "csi_gdrive",
            )
        )
        stats.extend(
            self._process_dataframe_files(
                sorted((self.in_base / "csi_pmc_smoothed").glob("*.parquet")),
                self.out_base / "csi_pmc_harmonized",
                "csi_pmc",
            )
        )
        return pd.DataFrame(stats)
