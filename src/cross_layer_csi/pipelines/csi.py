from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.paths import PROJECT_PATHS
from ..csi.config import CSI_DATASETS, CSI_DATA_DIR, ensure_csi_root
from ..csi.converter import CSIConverter
from ..csi.downloader import CSIDownloader
from ..csi.filter import CSISubcarrierFilter
from ..csi.harmonizer import CSIHarmonizer
from ..csi.smoothing import CSITemporalSmoother


@dataclass(frozen=True)
class CSIPreprocessingResult:
    base_dir: Path
    converted_dir: Path
    filtered_dir: Path
    smoothed_dir: Path
    harmonized_dir: Path
    harmonization_report: Path


class CSIPreprocessingPipeline:
    def __init__(self, base_dir: str | Path | None = None, render_plots: bool = False) -> None:
        self.base_dir = ensure_csi_root(Path(base_dir) if base_dir is not None else CSI_DATA_DIR)
        self.render_plots = render_plots

    def run(self, download: bool = False) -> CSIPreprocessingResult:
        if download:
            downloader = CSIDownloader(self.base_dir)
            downloader.download_all(CSI_DATASETS)

        converter = CSIConverter(self.base_dir)
        converted_dir = converter.convert_pmc()

        csi_filter = CSISubcarrierFilter(self.base_dir, render_plots=self.render_plots)
        csi_filter.process_gdrive()
        filtered_dir = csi_filter.process_pmc().parent

        smoother = CSITemporalSmoother(self.base_dir, render_plots=self.render_plots)
        smoother.process_gdrive()
        smoothed_dir = smoother.process_pmc().parent

        harmonizer = CSIHarmonizer(self.base_dir)
        harmonization_df = harmonizer.process_all()
        harmonized_dir = harmonizer.out_base

        PROJECT_PATHS.reports_generated.mkdir(parents=True, exist_ok=True)
        harmonization_report = PROJECT_PATHS.reports_generated / "csi_harmonization_summary.csv"
        harmonization_df.to_csv(harmonization_report, index=False)

        return CSIPreprocessingResult(
            base_dir=self.base_dir,
            converted_dir=converted_dir,
            filtered_dir=filtered_dir,
            smoothed_dir=smoothed_dir,
            harmonized_dir=harmonized_dir,
            harmonization_report=harmonization_report,
        )
