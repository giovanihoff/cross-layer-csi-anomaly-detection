from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class CSIConverter:
    def __init__(self, base_path: str | Path) -> None:
        self.base_path = Path(base_path)
        self.out_base = self.base_path / "converted_amplitudes"
        self.out_base.mkdir(parents=True, exist_ok=True)

    def _parse_pmc_complex(self, val: object) -> complex | float:
        if pd.isna(val):
            return np.nan
        if isinstance(val, str):
            txt = val.strip().strip("()").replace(" ", "").replace("i", "j")
            if txt in {"", "nan", "None"}:
                return np.nan
            try:
                return complex(txt)
            except Exception:
                return np.nan
        return val

    def _sort_candidate_files(self, in_dir: Path) -> tuple[list[Path], str]:
        all_csv = sorted(in_dir.rglob("*.csv"))
        low = [str(path).lower().replace("\\", "/") for path in all_csv]

        raw_amp = [path for path, lowered in zip(all_csv, low) if "raw_amplitudes" in lowered]
        complex_csi = [path for path, lowered in zip(all_csv, low) if "csi_matrices" in lowered]
        iqr_amp = [path for path, lowered in zip(all_csv, low) if "iqr_amplitudes" in lowered]

        if raw_amp:
            return raw_amp, "raw_amplitude_ready"
        if complex_csi:
            return complex_csi, "complex_to_amplitude"
        fallback = [path for path in all_csv if path not in iqr_amp]
        return fallback, "auto_detect"

    def _classify_file_mode(self, file_path: Path, default_mode: str = "auto_detect") -> str:
        if default_mode in {"raw_amplitude_ready", "complex_to_amplitude"}:
            return default_mode

        try:
            sample = pd.read_csv(file_path, engine="python", nrows=5)
            csi_cols = [column for column in sample.columns if str(column).lower() != "timestamp"]
            if not csi_cols:
                return "unknown"
            vals = sample[csi_cols].astype(str).stack().tolist()
            txt = " ".join(vals[:50]).lower()
            if "j" in txt or "(" in txt:
                return "complex_to_amplitude"
            return "raw_amplitude_ready"
        except Exception:
            return "unknown"

    def _convert_ready_amplitude_df(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [column for column in df.columns if str(column).lower() != "timestamp"]
        out = pd.DataFrame(index=df.index)
        for idx, column in enumerate(cols):
            out[f"subcarrier_{idx:03d}"] = pd.to_numeric(df[column], errors="coerce").astype(np.float32)
        return out.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    def _convert_complex_df(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [column for column in df.columns if str(column).lower() != "timestamp"]
        out = pd.DataFrame(index=df.index)
        for idx, column in enumerate(cols):
            series = df[column].apply(self._parse_pmc_complex)
            out[f"subcarrier_{idx:03d}"] = np.abs(series).astype(np.float32)
        return out.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    def convert_pmc(self) -> Path:
        print("[CONVERSION] Processing PMC_CSI...")
        in_dir = self.base_path / "csi_pmc"
        out_dir = self.out_base / "csi_pmc_amp"
        out_dir.mkdir(parents=True, exist_ok=True)

        files, default_mode = self._sort_candidate_files(in_dir)
        if not files:
            print("  [WARNING] No CSV files found in csi_pmc.")
            return out_dir

        print(f"  [INFO] Selected policy for PMC_CSI: {default_mode}")
        audit_rows: list[dict[str, object]] = []

        for file_path in files:
            try:
                mode = self._classify_file_mode(file_path, default_mode)
                if mode == "unknown":
                    print(f"  [WARNING] Could not classify {file_path.name}. Skipping.")
                    continue

                df = pd.read_csv(file_path, engine="python")
                amp_df = self._convert_ready_amplitude_df(df) if mode == "raw_amplitude_ready" else self._convert_complex_df(df)
                if amp_df.empty:
                    print(f"  [WARNING] {file_path.name} became empty after conversion/cleaning.")
                    continue

                out_name = file_path.stem + "_amp.parquet"
                amp_df.to_parquet(out_dir / out_name, index=False)
                values = amp_df.to_numpy(dtype=np.float32)
                audit_rows.append(
                    {
                        "source_file": file_path.name,
                        "input_mode": mode,
                        "source_folder": str(file_path.parent.relative_to(in_dir)),
                        "rows": int(len(amp_df)),
                        "native_subcarriers": int(amp_df.shape[1]),
                        "min_value": float(np.nanmin(values)),
                        "max_value": float(np.nanmax(values)),
                    }
                )
            except Exception as exc:
                print(f"  [ERROR] PMC_CSI | {file_path.name}: {exc}")

        audit_df = pd.DataFrame(audit_rows)
        audit_path = out_dir / "_pmc_conversion_audit.csv"
        audit_df.to_csv(audit_path, index=False)
        print(f"  -> Files saved to: {out_dir}")
        print(f"  -> Audit saved to: {audit_path}")
        return out_dir
