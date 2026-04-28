from __future__ import annotations

import shutil
from pathlib import Path

try:
    import gdown
except ImportError:  # pragma: no cover - optional dependency
    gdown = None

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:  # pragma: no cover - optional dependency
    KaggleApi = None


class CSIDownloader:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.kaggle_ready = False
        self.api = None
        if KaggleApi is not None:
            try:
                api = KaggleApi()
                api.authenticate()
                self.api = api
                self.kaggle_ready = True
            except Exception:
                self.kaggle_ready = False

    def download_dataset(self, dataset_name: str, url: str) -> Path:
        dataset_path = self.data_dir / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)

        self._extract_archives(dataset_path)

        valid_extensions = {".csv", ".mat", ".txt", ".h5", ".parquet", ".dat", ".npy"}
        existing_files = [path for path in dataset_path.rglob("*") if path.is_file() and path.suffix.lower() in valid_extensions]
        if existing_files:
            print(f"[INFO] Local files found/extracted for {dataset_name}: {[path.name for path in existing_files[:3]]}...")
            return dataset_path

        print(f"[INFO] Files not found for {dataset_name}.")
        try:
            if "drive.google.com" in url:
                if gdown is None:
                    raise RuntimeError("gdown is not installed. Add the optional CSI dependencies first.")
                print("  - Attempting Google Drive download...")
                output = dataset_path / f"{dataset_name}_data.zip"
                if not output.exists():
                    gdown.download(url, str(output), quiet=False)
                self._extract_archives(dataset_path)
                print(f"[INFO] GDrive download and extraction completed for {dataset_name}.")
            elif "kaggle.com" in url and self.kaggle_ready and self.api is not None:
                print("  - Attempting Kaggle download...")
                kaggle_path = url.split("datasets/")[-1]
                self.api.dataset_download_files(kaggle_path, path=str(dataset_path), unzip=True)
                self._extract_archives(dataset_path)
                print(f"[INFO] Kaggle download and extraction completed for {dataset_name}.")
            else:
                print(f"[ATTENTION] Automatic download disabled for '{dataset_name}'.")
                print(f"  Download manually from: {url}")
                print(f"  Place the archive inside: {dataset_path}")
        except Exception as exc:
            print(f"[ERROR] Failed to process {dataset_name}: {exc}")

        return dataset_path

    def download_all(self, datasets: dict[str, str]) -> dict[str, Path]:
        return {name: self.download_dataset(name, url) for name, url in datasets.items()}

    def _extract_archives(self, dataset_path: Path) -> None:
        archive_exts = (".zip", ".tar", ".tar.gz", ".tgz")
        for file_path in dataset_path.iterdir():
            if not file_path.is_file():
                continue
            item = file_path.name.lower()
            if item.endswith(archive_exts):
                print(f"  - Extracting archive: {file_path.name}...")
                try:
                    shutil.unpack_archive(str(file_path), str(dataset_path))
                    file_path.unlink()
                    print(f"  [INFO] Extraction completed. File {file_path.name} removed.")
                except Exception as exc:
                    print(f"  [WARNING] {file_path.name} could not be extracted: {exc}")
