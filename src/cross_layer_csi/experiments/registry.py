from __future__ import annotations

from dataclasses import dataclass


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
        notebook_cells=tuple(range(1, 13)),
        summary="Acquisition, profiling, splitting, and cleaning of the transaction datasets.",
    ),
    ExperimentStage(
        step_id="csi-preprocessing",
        title="CSI acquisition and harmonization",
        notebook_cells=tuple(range(13, 27)),
        summary="Download, convert amplitudes, filter subcarriers, smooth traces, and harmonize sources.",
    ),
    ExperimentStage(
        step_id="csi-segmentation",
        title="Multi-source CSI segmentation",
        notebook_cells=(27, 28, 29, 30, 31, 32, 33),
        summary="Inventory, visual checks, segmentation, anti-leak split, and CSI-only sanity validation.",
    ),
    ExperimentStage(
        step_id="controlled-one-class",
        title="Controlled Tx-only vs Tx+CSI protocol",
        notebook_cells=(34, 35, 36, 37, 38, 39, 40, 41, 42, 43),
        summary="Feature-level controlled comparison, diagnostics, selection, and segment sweeps.",
    ),
    ExperimentStage(
        step_id="robustness",
        title="Robustness campaign",
        notebook_cells=(44, 45, 46, 47, 48),
        summary="Multi-seed stability, prevalence sensitivity, and CSI control ablations.",
    ),
    ExperimentStage(
        step_id="two-phase-campaign",
        title="Complementary supervised+unsupervised campaign",
        notebook_cells=(49, 50, 51, 52, 53),
        summary="Tx-risk supervision combined with Tx+CSI coherence scoring for selected scenarios.",
    ),
)
