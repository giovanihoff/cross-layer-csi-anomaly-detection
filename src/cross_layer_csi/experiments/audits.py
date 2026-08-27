from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .config import DATASET_SEED_OFFSETS, SEED_POLICY_V120_3
from .consistency import (
    CSI_PROFILE_MIN_SAMPLES,
    attach_claimed_user_csi_consistency,
    get_csi_consistency_feature_columns,
)
from .evaluation import DATASET_EVALUATION_CONFIG, build_feature_columns, split_calibration_holdout
from .fusion import get_raw_csi_feature_columns, subset_impostor_cohort

REQUIRED_IMPOSTOR_COLUMNS = {
    "isFraud",
    "split_role",
    "attack_type",
    "csi_impostor_donor_index",
    "csi_impostor_donor_uid",
    "csi_impostor_donor_split",
    "csi_impostor_same_user",
    "uid_csi_mapped",
    "uid_csi",
    "tx_row_id",
    "csi_is_impostor_donor",
}
FORBIDDEN_FEATURE_PATTERNS = (
    "split",
    "source_",
    "attack_type",
    "tx_row_id",
    "csi_impostor",
    "donor",
    "same_user",
)
FORBIDDEN_EXACT_FEATURES = {
    "isFraud",
    "uid",
    "uid_csi",
    "uid_csi_mapped",
    "csi_impostor_donor_index",
    "csi_impostor_donor_uid",
    "csi_impostor_donor_split",
    "csi_impostor_same_user",
    "csi_impostor_donor_policy",
    "csi_is_impostor_donor",
    "csi_donor_for_target_index",
}


def audit_impostor_protocol(
    datasets: Mapping[str, Mapping[int, pd.DataFrame]],
    *,
    raise_on_fail: bool = True,
) -> pd.DataFrame:
    """Run the v120.3 pre-training hard guard over every dataset/segment pair."""

    rows: list[dict[str, object]] = []
    for dataset_label, by_segment in datasets.items():
        for segment_size, frame in by_segment.items():
            row: dict[str, object] = {
                "Dataset": dataset_label,
                "S": int(segment_size),
                "Rows": len(frame),
            }
            violations: list[str] = []
            warnings: list[str] = []
            missing = sorted(REQUIRED_IMPOSTOR_COLUMNS - set(frame.columns))
            row["Missing_Required_Columns"] = ", ".join(missing)
            if missing:
                violations.append(f"missing required columns: {missing}")
            else:
                split = frame["split_role"].astype(str)
                fraud = frame["isFraud"].astype(int)
                attack = frame["attack_type"].astype(str)
                impostor = attack.eq("impostor")
                unsupported = ~attack.isin(["none", "impostor"])
                row["Unsupported_Attack_Rows"] = int(unsupported.sum())
                row["Impostor_Rows"] = int(impostor.sum())
                row["Impostor_Non_Test_Rows"] = int((impostor & ~split.eq("test")).sum())
                row["Impostor_Non_Fraud_Target_Rows"] = int((impostor & fraud.ne(1)).sum())
                row["Impostor_Same_User_Swaps"] = int(
                    frame.loc[impostor, "csi_impostor_same_user"].astype(bool).sum()
                )
                row["Impostor_Unique_Donors"] = int(
                    frame.loc[impostor, "csi_impostor_donor_index"].nunique()
                )
                row["Impostor_Reused_Donors"] = max(
                    0, int(row["Impostor_Rows"]) - int(row["Impostor_Unique_Donors"])
                )
                for description, count in (
                    ("unsupported attack marker found", row["Unsupported_Attack_Rows"]),
                    ("impostor row outside test split", row["Impostor_Non_Test_Rows"]),
                    ("impostor row not attached to fraud", row["Impostor_Non_Fraud_Target_Rows"]),
                    ("CSI donor reused", row["Impostor_Reused_Donors"]),
                    ("same-user CSI swap", row["Impostor_Same_User_Swaps"]),
                ):
                    if int(count):
                        violations.append(description)

                donor_failures = {
                    "Invalid_Donor_Index": 0,
                    "Donor_Non_Test_Rows": 0,
                    "Donor_Fraud_Rows": 0,
                    "Donor_Attack_Rows": 0,
                    "Donor_Split_Metadata_Mismatch": 0,
                    "Donor_Index_Also_Target": 0,
                    "Donor_Not_Reserved": 0,
                }
                target_indices = set(frame.index[impostor])
                donor_ids = pd.to_numeric(
                    frame.loc[impostor, "csi_impostor_donor_index"], errors="coerce"
                ).astype("Int64")
                for target_index, donor_index in zip(frame.index[impostor], donor_ids.tolist()):
                    if pd.isna(donor_index) or int(donor_index) not in frame.index:
                        donor_failures["Invalid_Donor_Index"] += 1
                        continue
                    donor_index = int(donor_index)
                    donor_failures["Donor_Non_Test_Rows"] += int(
                        str(frame.at[donor_index, "split_role"]) != "test"
                    )
                    donor_failures["Donor_Fraud_Rows"] += int(
                        int(frame.at[donor_index, "isFraud"]) != 0
                    )
                    donor_failures["Donor_Attack_Rows"] += int(
                        str(frame.at[donor_index, "attack_type"]) != "none"
                    )
                    donor_failures["Donor_Split_Metadata_Mismatch"] += int(
                        str(frame.at[target_index, "csi_impostor_donor_split"])
                        != str(frame.at[donor_index, "split_role"])
                    )
                    donor_failures["Donor_Index_Also_Target"] += int(donor_index in target_indices)
                    donor_failures["Donor_Not_Reserved"] += int(
                        not bool(frame.at[donor_index, "csi_is_impostor_donor"])
                    )
                row.update(donor_failures)
                for name, count in donor_failures.items():
                    if count:
                        violations.append(name)

                row["Reserved_Donor_Rows"] = int(frame["csi_is_impostor_donor"].astype(bool).sum())
                if "split_coherent" in frame.columns:
                    row["Split_Coherence_Failures"] = int(
                        (~frame["split_coherent"].astype(bool)).sum()
                    )
                    if row["Split_Coherence_Failures"]:
                        violations.append("split_coherent failure")
                else:
                    row["Split_Coherence_Failures"] = np.nan
                    warnings.append("split_coherent unavailable")

                train_ids = set(frame.loc[split.isin(["train", "val"]), "tx_row_id"].astype(str))
                test_ids = set(frame.loc[split.eq("test"), "tx_row_id"].astype(str))
                row["TrainVal_Test_Tx_Row_Overlap"] = len(train_ids & test_ids)
                if row["TrainVal_Test_Tx_Row_Overlap"]:
                    violations.append("tx_row_id overlap between train/val and test")

                replacement = frame.get(
                    "_tx_sampling_replacement_used",
                    pd.Series(False, index=frame.index),
                ).astype(bool)
                row["Replacement_Sampled_Rows"] = int(replacement.sum())
                row["Replacement_Sampled_Rate_pct"] = float(
                    100.0 * replacement.sum() / max(1, len(frame))
                )
                if replacement.any():
                    warnings.append("transaction sampling used replacement")

                prefit_consistency = get_csi_consistency_feature_columns(frame)
                row["CSI_Consistency_Columns_PreEvaluation"] = len(prefit_consistency)
                if prefit_consistency:
                    violations.append("CSI consistency columns present before profile fitting")

            row["Audit_Status"] = "PASS" if not violations else "FAIL"
            row["Violations"] = " | ".join(violations)
            row["Warnings"] = " | ".join(warnings)
            rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["Dataset", "S"]).reset_index(drop=True)
    if raise_on_fail and not result.empty and result["Audit_Status"].ne("PASS").any():
        failed = result[result["Audit_Status"].ne("PASS")][["Dataset", "S", "Violations"]]
        raise RuntimeError("Hard impostor protocol audit failed:\n" + failed.to_string(index=False))
    return result


def audit_transaction_reuse(
    datasets: Mapping[str, Mapping[int, pd.DataFrame]],
) -> pd.DataFrame:
    """Quantify exact transaction reuse, replacement, and donor reservation."""

    rows: list[dict[str, object]] = []
    for dataset_label, by_segment in datasets.items():
        for segment_size, frame in by_segment.items():
            if frame.empty:
                rows.append({"Dataset": dataset_label, "S": segment_size, "Audit_Flag": "EMPTY"})
                continue
            if not {"tx_row_id", "split_role"}.issubset(frame.columns):
                rows.append(
                    {"Dataset": dataset_label, "S": segment_size, "Audit_Flag": "MISSING_COLUMNS"}
                )
                continue
            work = frame.copy()
            work["tx_row_id"] = work["tx_row_id"].astype(str)
            work["split_role"] = work["split_role"].astype(str)
            counts = work["tx_row_id"].value_counts(dropna=False)
            duplicated_rows = int((counts - 1).clip(lower=0).sum())
            train_ids = set(work.loc[work["split_role"].isin(["train", "val"]), "tx_row_id"])
            test = work[work["split_role"].eq("test")]
            test_counts = test["tx_row_id"].value_counts(dropna=False)
            overlap = len(train_ids & set(test_counts.index))
            replacement = work.get(
                "_tx_sampling_replacement_used", pd.Series(False, index=work.index)
            ).astype(bool)
            impostor = (
                work.get("attack_type", pd.Series("none", index=work.index))
                .astype(str)
                .eq("impostor")
            )
            reserved = work.get("csi_is_impostor_donor", pd.Series(False, index=work.index)).astype(
                bool
            )
            flags = []
            if overlap:
                flags.append("TRAIN_TEST_OVERLAP")
            if len(work) and duplicated_rows / len(work) >= 0.30:
                flags.append("HIGH_REUSE_TOTAL")
            if replacement.any():
                flags.append("REPLACEMENT_USED")
            rows.append(
                {
                    "Dataset": dataset_label,
                    "S": int(segment_size),
                    "Rows_Total": len(work),
                    "Unique_tx_row_id": len(counts),
                    "Duplicated_Row_Count": duplicated_rows,
                    "Duplicated_Row_Rate_pct": 100.0 * duplicated_rows / max(1, len(work)),
                    "Max_Reuse_Per_tx_row_id": int(counts.max()) if len(counts) else 0,
                    "Replacement_Sampled_Rows": int(replacement.sum()),
                    "Replacement_Sampled_Rate_pct": 100.0 * replacement.sum() / max(1, len(work)),
                    "Train_Test_tx_row_id_Overlap": overlap,
                    "Test_Rows": len(test),
                    "Test_Unique_tx_row_id": len(test_counts),
                    "Impostor_Target_Rows": int(impostor.sum()),
                    "Unique_CSI_Donors": int(
                        work.loc[impostor, "csi_impostor_donor_index"].nunique()
                    )
                    if impostor.any()
                    else 0,
                    "Reserved_Donor_Rows": int(reserved.sum()),
                    "Audit_Flag": ";".join(flags or ["OK"]),
                }
            )
    result = pd.DataFrame(rows)
    if (
        not result.empty
        and result.get("Train_Test_tx_row_id_Overlap", pd.Series(0, index=result.index))
        .fillna(0)
        .astype(int)
        .sum()
    ):
        raise RuntimeError("tx_row_id leakage detected between train/validation and test")
    return result.sort_values(["Dataset", "S"]).reset_index(drop=True)


def audit_feature_selection(
    datasets: Mapping[str, Mapping[int, pd.DataFrame]],
    *,
    raise_on_fail: bool = True,
) -> pd.DataFrame:
    """Prove Tx-only has no CSI and neither arm receives protocol metadata."""

    rows: list[dict[str, object]] = []
    for dataset_label, by_segment in datasets.items():
        for segment_size, frame in by_segment.items():
            if frame.empty:
                continue
            cohort = subset_impostor_cohort(frame)
            train = cohort[cohort["split_role"].astype(str).isin(["train", "val"])].copy()
            test_pool = cohort[cohort["split_role"].astype(str).eq("test")].copy()
            if train.empty or test_pool.empty:
                continue
            evaluation_seed = (
                SEED_POLICY_V120_3.evaluation_base
                + DATASET_SEED_OFFSETS[dataset_label]
                + int(segment_size)
            )
            calibration, holdout = split_calibration_holdout(
                test_pool,
                calibration_fraction=float(
                    DATASET_EVALUATION_CONFIG[dataset_label]["calibration_fraction"]
                ),
                seed=evaluation_seed,
            )
            train_augmented, _, _, profile_bundle = attach_claimed_user_csi_consistency(
                train,
                calibration,
                holdout,
                min_profile_samples=CSI_PROFILE_MIN_SAMPLES,
            )
            raw_csi = set(get_raw_csi_feature_columns(train_augmented))
            consistency = set(get_csi_consistency_feature_columns(train_augmented))
            for include_csi, scenario in ((False, "Tx-Only"), (True, "Tx+CSI")):
                selected = build_feature_columns(
                    train_augmented,
                    dataset_label,
                    include_csi=include_csi,
                )
                forbidden_exact = sorted(set(selected) & FORBIDDEN_EXACT_FEATURES)
                forbidden_patterns = sorted(
                    column
                    for column in selected
                    if any(pattern in str(column) for pattern in FORBIDDEN_FEATURE_PATTERNS)
                )
                tx_only_csi = (
                    sorted(set(selected) & (raw_csi | consistency)) if not include_csi else []
                )
                consistency_count = len(set(selected) & consistency) if include_csi else 0
                violations = []
                if forbidden_exact:
                    violations.append(f"forbidden exact columns: {forbidden_exact}")
                if forbidden_patterns:
                    violations.append(f"forbidden patterns: {forbidden_patterns}")
                if tx_only_csi:
                    violations.append(f"Tx-Only received CSI: {tx_only_csi[:10]}")
                if include_csi and not consistency_count:
                    violations.append("Tx+CSI has no claimed-user consistency features")
                if include_csi and not profile_bundle:
                    violations.append("No CSI user profiles available")
                rows.append(
                    {
                        "Dataset": dataset_label,
                        "S": int(segment_size),
                        "Scenario_Mode": scenario,
                        "Feature_Count": len(selected),
                        "Raw_CSI_Feature_Count": len(set(selected) & raw_csi),
                        "CSI_Consistency_Feature_Count": consistency_count,
                        "Profile_User_Count": (
                            int(profile_bundle.get("profile_users", 0)) if profile_bundle else 0
                        ),
                        "Feature_Audit_Status": "PASS" if not violations else "FAIL",
                        "Violations": " | ".join(violations),
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["Dataset", "S", "Scenario_Mode"]).reset_index(drop=True)
    if raise_on_fail and not result.empty and result["Feature_Audit_Status"].ne("PASS").any():
        failed = result[result["Feature_Audit_Status"].ne("PASS")][
            ["Dataset", "S", "Scenario_Mode", "Violations"]
        ]
        raise RuntimeError("Feature leakage audit failed:\n" + failed.to_string(index=False))
    return result


audit_impostor_protocol_v118 = audit_impostor_protocol
audit_feature_selection_v118 = audit_feature_selection
