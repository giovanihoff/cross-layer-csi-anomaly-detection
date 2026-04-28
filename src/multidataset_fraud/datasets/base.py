from __future__ import annotations

import abc
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from ..config import PATHS
from ..kaggle import KaggleClient
from ..models import (
    DatasetRunResult,
    KaggleReference,
    PreparedArtifacts,
    SourceFileSpec,
)
from ..profiling import build_profile
from ..reporting import build_split_summary, build_uid_summary, format_split_summary
from ..research import get_research
from ..utils import ensure_directory


class BaseDatasetHandler(abc.ABC):
    dataset_key: str
    display_name: str
    target_col: str
    kaggle_reference: KaggleReference
    local_env_var: str
    source_files: tuple[SourceFileSpec, ...]

    def __init__(self, kaggle_client: KaggleClient | None = None, verbose: bool = True) -> None:
        self.kaggle_client = kaggle_client or KaggleClient()
        self.verbose = verbose
        self.raw_dir = ensure_directory(PATHS.data_raw / self.dataset_key)
        self.processed_dir = ensure_directory(PATHS.data_processed / self.dataset_key)
        self._data_access_mode = "local"

    def log(self, message: str) -> None:
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{self.dataset_key}] {message}", flush=True)

    def log_phase(self, title: str) -> None:
        self.log("=" * 18 + f" {title} " + "=" * 18)

    def candidate_roots(self) -> list[Path]:
        roots = [self.raw_dir]
        extra_root = os.getenv(self.local_env_var)
        if extra_root:
            roots.insert(0, Path(extra_root))
        return roots

    def _find_file(self, patterns: tuple[str, ...]) -> Path | None:
        for root in self.candidate_roots():
            if not root.exists():
                continue
            for pattern in patterns:
                for candidate in root.rglob(pattern):
                    if candidate.is_file():
                        return candidate.resolve()
        return None

    def locate_source_files(self) -> dict[str, Path]:
        located: dict[str, Path] = {}
        missing_required: list[str] = []
        for spec in self.source_files:
            match = self._find_file(spec.patterns)
            if match is not None:
                located[spec.logical_name] = match
            elif spec.required:
                missing_required.append(spec.logical_name)
        if missing_required:
            raise FileNotFoundError(
                f"{self.display_name}: arquivos obrigatorios nao encontrados: {', '.join(missing_required)}"
            )
        return located

    def ensure_raw_available(self) -> dict[str, Path]:
        try:
            self.log("Verificando arquivos locais...")
            self._data_access_mode = "local"
            return self.locate_source_files()
        except FileNotFoundError:
            self.log("Arquivos locais nao encontrados. Iniciando download do Kaggle...")
            self._data_access_mode = "kaggle_download"
            if self.kaggle_reference.kind == "competition":
                self.kaggle_client.download_competition(self.kaggle_reference.ref, self.raw_dir)
            else:
                self.kaggle_client.download_dataset(self.kaggle_reference.ref, self.raw_dir)
            self.log("Download concluido. Validando arquivos extraidos...")
            return self.locate_source_files()

    def run(self) -> DatasetRunResult:
        started_at = perf_counter()
        research = get_research(self.dataset_key)
        self.log("Inicio do processamento do dataset.")
        self.log_phase("DOWNLOAD E CARREGAMENTO")
        raw_files = self.ensure_raw_available()
        self.log(f"Arquivos encontrados: {', '.join(sorted(raw_files.keys()))}")
        self.log(f"Origem dos dados: {self._data_access_mode}")

        self.log("Carregando arquivos brutos em memoria...")
        raw_splits = self.load_raw_splits(raw_files)
        split_summaries = ", ".join(
            f"{split_name}={df.shape[0]}x{df.shape[1]}"
            for split_name, df in raw_splits.items()
        )
        self.log(f"Carga concluida: {split_summaries}")
        raw_summaries = {
            split_name: build_split_summary(
                df,
                split_name=split_name,
                target_col=self.target_col,
                stage="raw",
            )
            for split_name, df in raw_splits.items()
        }
        for split_name, summary in raw_summaries.items():
            self.log(f"Resumo bruto `{split_name}`: {format_split_summary(summary)}")

        self.log_phase("PROXY DE UID E PERFILAMENTO")
        self.log("Gerando perfil dos dados e candidatos de identificadores...")
        profiles = []
        for split_name, df in raw_splits.items():
            if self.target_col not in df.columns:
                continue
            self.log(f"Perfilando split `{split_name}`...")
            profiles.append(
                build_profile(
                    dataset_key=self.dataset_key,
                    split_name=split_name,
                    df=df,
                    target_col=self.target_col,
                    source_files={name: str(path) for name, path in raw_files.items()},
                    identifier_factory=self.identifier_candidates,
                )
            )
        self.log(f"Perfilamento concluido para {len(profiles)} split(s).")
        uid_summary = build_uid_summary(
            profiles=profiles,
            chosen_uid=research.uid_choice,
            rationale=research.uid_rationale,
        )
        self.log(f"UID adotado: {uid_summary['chosen_uid']}")
        for ranking in uid_summary["rankings"]:
            self.log(f"Candidatos avaliados em `{ranking['split_name']}`:")
            for candidate in ranking["candidates"]:
                self.log(
                    "- "
                    f"{candidate['name']}: nunique={candidate['nunique']}, "
                    f"duplicate_ratio={candidate['duplicate_ratio']:.6f}, "
                    f"max_group_size={candidate['max_group_size']}"
                )

        self.log_phase("ENGENHARIA DE ATRIBUTOS")
        self.log("Aplicando preparo inicial para treino...")
        prepared = self.prepare_artifacts(raw_splits, raw_files)
        prepared_summaries = prepared.metadata.get("prepared_summaries", {})
        feature_summary = prepared.metadata.get("feature_engineering", {})
        for split_name, summary in prepared_summaries.items():
            self.log(f"Resumo preparado `{split_name}`: {format_split_summary(summary)}")
        if feature_summary:
            self.log(
                "Transicao de colunas: "
                f"adicionadas={len(feature_summary.get('added_columns', []))} | "
                f"removidas={len(feature_summary.get('removed_columns', []))} | "
                f"finais={len(feature_summary.get('final_columns', []))}"
            )
        elapsed = perf_counter() - started_at
        self.log(
            "Preparacao concluida. "
            f"train={prepared.train_path.name} | test={prepared.test_path.name} | tempo={elapsed:.1f}s"
        )
        self.log_phase("RESUMO DO DATASET")
        self.log(f"Arquivos preparados: train={prepared.train_path.name} | test={prepared.test_path.name}")
        if prepared.extra_paths:
            extras = ", ".join(f"{name}={path.name}" for name, path in prepared.extra_paths.items())
            self.log(f"Artefatos extras: {extras}")
        return DatasetRunResult(
            dataset_key=self.dataset_key,
            display_name=self.display_name,
            raw_files=raw_files,
            profiles=profiles,
            prepared=prepared,
            research=research,
            metadata={
                "data_access_mode": self._data_access_mode,
                "raw_summaries": raw_summaries,
                "uid_summary": uid_summary,
            },
        )

    @staticmethod
    def add_group_stats(
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        prefix: str,
    ) -> pd.DataFrame:
        if group_col not in df.columns or value_col not in df.columns:
            return df
        grouped = pd.to_numeric(df[value_col], errors="coerce").groupby(df[group_col])
        df[f"{prefix}_mean"] = grouped.transform("mean")
        df[f"{prefix}_std"] = grouped.transform("std").fillna(0.0)
        df[f"{prefix}_count"] = grouped.transform("count")
        return df

    @abc.abstractmethod
    def load_raw_splits(self, raw_files: dict[str, Path]) -> dict[str, pd.DataFrame]:
        raise NotImplementedError

    @abc.abstractmethod
    def identifier_candidates(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_artifacts(
        self,
        raw_splits: dict[str, pd.DataFrame],
        raw_files: dict[str, Path],
    ) -> PreparedArtifacts:
        raise NotImplementedError
