from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


class CSITemporalSmoother:
    def __init__(self, base_path: str | Path, render_plots: bool = False) -> None:
        self.base_path = Path(base_path)
        self.render_plots = render_plots
        self.in_base = self.base_path / "filtered_amplitudes"
        self.out_base = self.base_path / "smoothed_amplitudes"
        self.out_base.mkdir(parents=True, exist_ok=True)

    def smooth_dataframe(self, df: pd.DataFrame, win: int = 5) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=[np.number]).columns
        out = df.copy()
        out[num_cols] = df[num_cols].rolling(window=win, center=True, min_periods=1).mean()
        return out

    def plot_time_trace(self, before: pd.DataFrame, after: pd.DataFrame, title: str, sc_idx: int = 10) -> None:
        if plt is None:
            return
        num_cols = before.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            return
        if sc_idx >= len(num_cols):
            sc_idx = 0
        trace_before = before[num_cols[sc_idx]].values
        trace_after = after[num_cols[sc_idx]].values

        plt.figure(figsize=(10, 3))
        plt.title(f"{title} | Temporal trace (Subcarrier idx={sc_idx}) | Moving Average")
        plt.plot(trace_before[:400], label="Raw (Filtered)", alpha=0.5)
        plt.plot(trace_after[:400], label="Smooth (Low-Pass win=5)", linewidth=2)
        plt.xlabel("Time (Packets)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.close()

    def process_gdrive(self) -> Path:
        print("\n[SMOOTHING] Processing ITA_CSI...")
        in_dir = self.in_base / "csi_gdrive_filtered"
        out_dir = self.out_base / "csi_gdrive_smoothed"
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(in_dir.glob("*.csv"))
        for idx, file_path in enumerate(files):
            df_before = pd.read_csv(file_path)
            df_after = self.smooth_dataframe(df_before)
            if self.render_plots and idx == 0:
                self.plot_time_trace(df_before, df_after, "ITA_CSI")
            df_after.to_csv(out_dir / file_path.name, index=False)
        print(f" -> {len(files)} smoothed files saved.")
        return out_dir

    def process_pmc(self) -> Path:
        print("\n[SMOOTHING] Processing PMC_CSI...")
        in_dir = self.in_base / "csi_pmc_filtered"
        out_dir = self.out_base / "csi_pmc_smoothed"
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(in_dir.glob("*.parquet"))
        for idx, file_path in enumerate(files):
            df_before = pd.read_parquet(file_path)
            df_after = self.smooth_dataframe(df_before)
            if self.render_plots and idx == 0:
                self.plot_time_trace(df_before, df_after, "PMC_CSI")
            df_after.to_parquet(out_dir / file_path.name, index=False)
        print(f" -> {len(files)} smoothed files saved.")
        return out_dir
