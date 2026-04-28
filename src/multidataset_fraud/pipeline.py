from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from datetime import datetime

import pandas as pd

from .config import PATHS
from .datasets import EcommerceDataset, IEEECISDataset, SparkovDataset
from .models import DatasetRunResult
from .reporting import format_dataset_report
from .utils import ensure_directory, write_json


DATASET_REGISTRY = {
    "ieee_cis": IEEECISDataset,
    "sparkov": SparkovDataset,
    "ecommerce": EcommerceDataset,
}


class BootstrapPipeline:
    def __init__(self, dataset_keys: list[str] | None = None, verbose: bool = True) -> None:
        self.dataset_keys = dataset_keys or list(DATASET_REGISTRY.keys())
        self.verbose = verbose
        ensure_directory(PATHS.reports_generated)

    def log(self, message: str) -> None:
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [bootstrap] {message}", flush=True)

    def run(self) -> list[DatasetRunResult]:
        started_at = perf_counter()
        results: list[DatasetRunResult] = []
        total = len(self.dataset_keys)
        self.log(f"Iniciando pipeline para {total} dataset(s): {', '.join(self.dataset_keys)}")
        for index, dataset_key in enumerate(self.dataset_keys, start=1):
            handler_cls = DATASET_REGISTRY[dataset_key]
            self.log(f"[{index}/{total}] Processando `{dataset_key}`...")
            result = handler_cls(verbose=self.verbose).run()
            results.append(result)
            self.log(f"[{index}/{total}] `{dataset_key}` concluido.")
        self.log("Gerando relatorios consolidados...")
        self._write_reports(results)
        elapsed = perf_counter() - started_at
        self.log(f"Pipeline finalizada em {elapsed:.1f}s.")
        return results

    def _write_reports(self, results: list[DatasetRunResult]) -> None:
        summary_rows: list[dict[str, object]] = []
        profile_payload: list[dict[str, object]] = []
        bootstrap_payload: list[dict[str, object]] = []
        markdown_lines = ["# Dataset Bootstrap Reports", ""]

        for result in results:
            dataset_report_path = PATHS.reports_generated / f"{result.dataset_key}_summary.md"
            dataset_report = format_dataset_report(result)
            dataset_report_path.write_text(dataset_report, encoding="utf-8")
            markdown_lines.append(dataset_report)
            markdown_lines.append("")

            bootstrap_payload.append(
                {
                    "dataset_key": result.dataset_key,
                    "display_name": result.display_name,
                    "data_access_mode": result.metadata.get("data_access_mode"),
                    "raw_files": {name: str(path) for name, path in result.raw_files.items()},
                    "raw_summaries": result.metadata.get("raw_summaries", {}),
                    "uid_summary": result.metadata.get("uid_summary", {}),
                    "research": asdict(result.research),
                    "profiles": [profile.to_dict() for profile in result.profiles],
                    "prepared": {
                        "train_path": str(result.prepared.train_path),
                        "test_path": str(result.prepared.test_path),
                        "extra_paths": {name: str(path) for name, path in result.prepared.extra_paths.items()},
                        "metadata": result.prepared.metadata,
                    },
                    "report_path": str(dataset_report_path),
                }
            )

            for profile in result.profiles:
                summary_rows.append(
                    {
                        "dataset_key": result.dataset_key,
                        "display_name": result.display_name,
                        "data_access_mode": result.metadata.get("data_access_mode"),
                        "split_name": profile.split_name,
                        "rows": profile.rows,
                        "columns": profile.columns,
                        "frauds": profile.frauds,
                        "fraud_rate": profile.fraud_rate,
                        "uid_choice": result.research.uid_choice,
                        "train_path": str(result.prepared.train_path),
                        "test_path": str(result.prepared.test_path),
                        "report_path": str(dataset_report_path),
                    }
                )
                profile_payload.append(profile.to_dict())

        summary_df = pd.DataFrame(summary_rows)
        self.log("Escrevendo dataset_summary.csv...")
        summary_df.to_csv(PATHS.reports_generated / "dataset_summary.csv", index=False)
        self.log("Escrevendo dataset_profiles.json...")
        write_json(PATHS.reports_generated / "dataset_profiles.json", profile_payload)
        self.log("Escrevendo dataset_bootstrap_report.json...")
        write_json(PATHS.reports_generated / "dataset_bootstrap_report.json", bootstrap_payload)
        self.log("Escrevendo dataset_profiles.md...")
        (PATHS.reports_generated / "dataset_profiles.md").write_text(
            "\n".join(markdown_lines),
            encoding="utf-8",
        )
        for result in results:
            self.log(f"Relatorio individual salvo: {(PATHS.reports_generated / f'{result.dataset_key}_summary.md').name}")
