from __future__ import annotations

from dataclasses import dataclass

NOTEBOOK_PROTOCOL_VERSION = "v120.3"
NOTEBOOK_CELL_COUNT = 68


@dataclass(frozen=True)
class ExperimentStage:
    step_id: str
    title: str
    notebook_cells: tuple[int, ...]
    summary: str


EXPERIMENT_STAGES = (
    ExperimentStage(
        step_id="financial-tabular",
        title="Financial transaction bootstrap",
        notebook_cells=tuple(range(2, 14)),
        summary="Acquisition, profiling, splitting, and cleaning of four transaction datasets, including CAIXABANK.",
    ),
    ExperimentStage(
        step_id="csi-preprocessing",
        title="CSI acquisition and harmonization",
        notebook_cells=tuple(range(14, 28)),
        summary="Download, amplitude conversion, subcarrier filtering, temporal smoothing, and 52-to-108 interpolation.",
    ),
    ExperimentStage(
        step_id="csi-segmentation",
        title="Multi-source CSI segmentation",
        notebook_cells=tuple(range(28, 35)),
        summary="Multi-source inventory, 432-feature segmentation, purged anti-leak split, and CSI-only sanity validation.",
    ),
    ExperimentStage(
        step_id="controlled-one-class",
        title="Controlled Tx-only vs Tx+CSI protocol",
        notebook_cells=tuple(range(35, 44)),
        summary="Impostor-only construction, unique held-out donors, claimed-user profiles, and hard leakage guards.",
    ),
    ExperimentStage(
        step_id="selection-and-sensitivity",
        title="Calibration-based selection and sensitivity",
        notebook_cells=tuple(range(44, 58)),
        summary="Paired selection, segment sweep, profile support, evaluation-seed stability, prevalence sensitivity, and negative controls.",
    ),
    ExperimentStage(
        step_id="two-phase-campaign",
        title="Complementary two-phase campaign",
        notebook_cells=tuple(range(58, 62)),
        summary="LightGBM/XGBoost transaction risk combined with one-class CSI coherence.",
    ),
    ExperimentStage(
        step_id="donor-policy-stress-test",
        title="Donor-policy stress test",
        notebook_cells=(62, 63, 64, 65),
        summary="Explicit seed roles, exact random replay guard, and exploratory hard-nearest comparison.",
    ),
    ExperimentStage(
        step_id="reproducibility-and-interpretation",
        title="Reproducibility and interpretation guard",
        notebook_cells=(66, 67),
        summary="Environment manifest and final limits on claims for controlled cross-domain emulation.",
    ),
)
