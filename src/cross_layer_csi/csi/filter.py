from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


NULL_IDX = list(range(0, 6)) + [63, 64, 65] + list(range(123, 128))
PILOT_IDX = [11, 39, 53, 75, 89, 117]
DROP_IDX_128 = sorted(set(NULL_IDX + PILOT_IDX))

warnings.simplefilter("ignore", PerformanceWarning)


class CSISubcarrierFilter:
    def __init__(self, base_path: str | Path, render_plots: bool = False) -> None:
        self.base_path = Path(base_path)
        self.render_plots = render_plots
        self.out_base = self.base_path / "filtered_amplitudes"
        self.out_base.mkdir(parents=True, exist_ok=True)

    def _rename_numeric_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
        if not num_cols:
            return pd.DataFrame(index=df.index)

        out = df[num_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32).copy()
        out.columns = [f"subcarrier_{idx:03d}" for idx in range(len(num_cols))]
        return out.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    def filter_128_subcarriers(self, df: pd.DataFrame) -> pd.DataFrame:
        csi_cols = [column for column in df.columns if str(column).startswith("subcarrier_")]
        csi_cols = sorted(csi_cols, key=lambda name: int(str(name).split("_")[1]))
        if len(csi_cols) == 128:
            keep_cols = [column for idx, column in enumerate(csi_cols) if idx not in DROP_IDX_128]
            out = df[keep_cols].copy()
            out.columns = [f"subcarrier_{idx:03d}" for idx in range(len(out.columns))]
            return out
        if csi_cols:
            out = df[csi_cols].copy()
            out.columns = [f"subcarrier_{idx:03d}" for idx in range(len(out.columns))]
            return out
        return df.copy()

    def plot_before_after(self, before: pd.DataFrame, after: pd.DataFrame, title: str) -> None:
        if plt is None:
            return
        plt.figure(figsize=(14, 4))
        plt.suptitle(title, fontsize=14, fontweight="bold")

        plt.subplot(1, 2, 1)
        plt.title(f"Before (shape: {before.shape})")
        num_cols = before.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            plt.plot(before[num_cols].iloc[0].values)
        plt.xlabel("Subcarrier index")
        plt.ylabel("Amplitude")
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.subplot(1, 2, 2)
        plt.title(f"After (shape: {after.shape})")
        num_cols = after.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            plt.plot(after[num_cols].iloc[0].values, color="coral")
        plt.xlabel("Subcarrier index")
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.close()

    def process_gdrive(self) -> Path:
        print("\n[FILTER] ITA_CSI: null/pilot removal (128 -> 108)")
        in_dir = self.base_path / "csi_gdrive"
        out_dir = self.out_base / "csi_gdrive_filtered"
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(in_dir.rglob("*.csv"))
        if not files:
            print("  [WARNING] No CSV files found in csi_gdrive.")
            return out_dir

        for idx, file_path in enumerate(files):
            df_before = pd.read_csv(file_path)
            df_before = self._rename_numeric_cols(df_before)
            df_after = self.filter_128_subcarriers(df_before)
            if self.render_plots and idx == 0 and not df_after.empty:
                self.plot_before_after(df_before, df_after, "ITA_CSI (128 -> 108)")
            df_after.to_csv(out_dir / file_path.name, index=False)
        print(f"  -> {len(files)} filtered files saved to {out_dir}")
        return out_dir

    def process_pmc(self) -> Path:
        print("\n[FILTER] PMC_CSI: preserving native useful carriers")
        in_dir = self.base_path / "converted_amplitudes" / "csi_pmc_amp"
        out_dir = self.out_base / "csi_pmc_filtered"
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(file_path for file_path in in_dir.glob("*.parquet") if not file_path.name.startswith("_"))
        if not files:
            print("  [WARNING] No converted parquet files found in csi_pmc_amp.")
            return out_dir

        for idx, file_path in enumerate(files):
            df_before = pd.read_parquet(file_path)
            df_before = self._rename_numeric_cols(df_before)
            if self.render_plots and idx == 0 and not df_before.empty:
                self.plot_before_after(df_before, df_before, "PMC_CSI (native carriers)")
            df_before.to_parquet(out_dir / file_path.name, index=False)
        print(f"  -> {len(files)} filtered files saved to {out_dir}")
        return out_dir
