from __future__ import annotations

import os
from pathlib import Path
from zipfile import ZipFile

from .config import PATHS


class KaggleClient:
    def __init__(self, env_file: Path | None = None) -> None:
        self.env_file = env_file or PATHS.env_file
        self._api: KaggleApi | None = None

    def _load_env(self) -> None:
        from dotenv import load_dotenv

        if self.env_file.exists():
            load_dotenv(self.env_file, override=False)

        username = os.getenv("KAGGLE_USERNAME")
        key = os.getenv("KAGGLE_KEY")
        if not username or not key:
            raise RuntimeError(
                "Credenciais Kaggle nao encontradas. Defina KAGGLE_USERNAME/KAGGLE_KEY "
                "ou preencha o arquivo .env.kaggle."
            )

    @property
    def api(self):
        if self._api is None:
            from kaggle.api.kaggle_api_extended import KaggleApi

            self._load_env()
            api = KaggleApi()
            api.authenticate()
            self._api = api
        return self._api

    @staticmethod
    def _extract_archives(destination: Path) -> None:
        for archive in destination.rglob("*.zip"):
            with ZipFile(archive, "r") as zipped:
                zipped.extractall(destination)

    def download_competition(self, competition: str, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        self.api.competition_download_files(
            competition=competition,
            path=str(destination),
            force=False,
            quiet=False,
        )
        self._extract_archives(destination)

    def download_dataset(self, dataset: str, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        self.api.dataset_download_files(
            dataset=dataset,
            path=str(destination),
            force=False,
            quiet=False,
        )
        self._extract_archives(destination)
