from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from .audits import audit_feature_selection, audit_impostor_protocol, audit_transaction_reuse
from .config import SEED_POLICY_V120_3, SEGMENT_SIZES, SeedPolicy
from .evaluation import evaluate_controlled_one_class
from .fusion import align_transactions_and_csi, inject_csi_impostors
from .reporting import build_paired_result_table, build_random_reproducibility_guard


@dataclass(frozen=True)
class ControlledExperimentResult:
    merged_datasets: dict[str, dict[int, pd.DataFrame]]
    raw_results: pd.DataFrame
    paired_results: pd.DataFrame
    protocol_audit: pd.DataFrame
    feature_audit: pd.DataFrame
    reuse_audit: pd.DataFrame


class ControlledExperimentRunner:
    """Reusable implementation of the v120.3 impostor-only experiment core."""

    def __init__(
        self,
        *,
        seed_policy: SeedPolicy = SEED_POLICY_V120_3,
        target_fraud_rate: float = 0.05,
    ) -> None:
        self.seed_policy = seed_policy
        self.target_fraud_rate = target_fraud_rate

    def construct(
        self,
        transactions: Mapping[str, pd.DataFrame],
        csi_datasets: Mapping[int, pd.DataFrame],
        user_maps: Mapping[str, Mapping[str, str]],
        *,
        segment_sizes: Sequence[int] = SEGMENT_SIZES,
        donor_policy: str = "random",
    ) -> dict[str, dict[int, pd.DataFrame]]:
        constructed: dict[str, dict[int, pd.DataFrame]] = {}
        for dataset_label, transaction_frame in transactions.items():
            if dataset_label not in user_maps:
                raise KeyError(f"Missing Tx-to-CSI user map for {dataset_label}")
            constructed[dataset_label] = {}
            for segment_size in segment_sizes:
                if segment_size not in csi_datasets:
                    raise KeyError(f"Missing CSI segmentation S={segment_size}")
                seeds = self.seed_policy.for_scenario(dataset_label, segment_size)
                merged = align_transactions_and_csi(
                    transaction_frame,
                    csi_datasets[segment_size],
                    user_maps[dataset_label],
                    seed=seeds.merge,
                    target_fraud_rate=self.target_fraud_rate,
                )
                constructed[dataset_label][segment_size] = inject_csi_impostors(
                    merged,
                    seed=seeds.injection,
                    donor_policy=donor_policy,
                )
        return constructed

    def evaluate(
        self,
        constructed: Mapping[str, Mapping[int, pd.DataFrame]],
        *,
        detector_names: Sequence[str] | None = None,
    ) -> ControlledExperimentResult:
        protocol_audit = audit_impostor_protocol(constructed, raise_on_fail=True)
        reuse_audit = audit_transaction_reuse(constructed)
        feature_audit = audit_feature_selection(constructed, raise_on_fail=True)
        rows: list[dict[str, object]] = []
        for dataset_label, by_segment in constructed.items():
            for segment_size, frame in by_segment.items():
                seeds = self.seed_policy.for_scenario(dataset_label, segment_size)
                kwargs = {}
                if detector_names is not None:
                    kwargs["detector_names"] = detector_names
                rows.extend(
                    evaluate_controlled_one_class(
                        frame,
                        dataset_label=dataset_label,
                        segment_size=segment_size,
                        seed=seeds.evaluation,
                        **kwargs,
                    )
                )
        raw = pd.DataFrame(rows)
        paired = build_paired_result_table(raw)
        return ControlledExperimentResult(
            merged_datasets={
                dataset: {segment: frame.copy() for segment, frame in by_segment.items()}
                for dataset, by_segment in constructed.items()
            },
            raw_results=raw,
            paired_results=paired,
            protocol_audit=protocol_audit,
            feature_audit=feature_audit,
            reuse_audit=reuse_audit,
        )

    def donor_policy_stress_test(
        self,
        transactions: Mapping[str, pd.DataFrame],
        csi_datasets: Mapping[int, pd.DataFrame],
        user_maps: Mapping[str, Mapping[str, str]],
        selected_scenarios: pd.DataFrame,
        main_pairs: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Replay random exactly, then evaluate hard-nearest on selected scenarios."""

        rows: list[dict[str, object]] = []
        for spec in (
            selected_scenarios[["Dataset", "S", "Modelo"]].drop_duplicates().to_dict("records")
        ):
            dataset_label = str(spec["Dataset"])
            segment_size = int(spec["S"])
            detector_name = str(spec["Modelo"])
            seeds = self.seed_policy.for_scenario(dataset_label, segment_size)
            for donor_policy in ("random", "hard_nearest"):
                merged = align_transactions_and_csi(
                    transactions[dataset_label],
                    csi_datasets[segment_size],
                    user_maps[dataset_label],
                    seed=seeds.merge,
                    target_fraud_rate=self.target_fraud_rate,
                )
                injected = inject_csi_impostors(
                    merged,
                    seed=seeds.injection,
                    donor_policy=donor_policy,
                )
                audit_impostor_protocol(
                    {dataset_label: {segment_size: injected}},
                    raise_on_fail=True,
                )
                evaluated = evaluate_controlled_one_class(
                    injected,
                    dataset_label=dataset_label,
                    segment_size=segment_size,
                    seed=seeds.evaluation,
                    detector_names=[detector_name],
                )
                for row in evaluated:
                    row.update(
                        {
                            "Donor_Policy": donor_policy,
                            "Merge_Seed": seeds.merge,
                            "Injection_Seed": seeds.injection,
                            "Evaluation_Seed": seeds.evaluation,
                        }
                    )
                rows.extend(evaluated)
        paired = build_paired_result_table(
            pd.DataFrame(rows),
            extra_group_columns=(
                "Donor_Policy",
                "Merge_Seed",
                "Injection_Seed",
                "Evaluation_Seed",
            ),
        )
        guard = build_random_reproducibility_guard(main_pairs, paired, raise_on_fail=True)
        return paired, guard
