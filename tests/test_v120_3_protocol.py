from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cross_layer_csi.csi.segmentation import segment_matrix
from cross_layer_csi.experiments.audits import audit_impostor_protocol
from cross_layer_csi.experiments.config import SEED_POLICY_V120_3
from cross_layer_csi.experiments.consistency import (
    apply_claimed_user_csi_consistency,
    fit_claimed_user_csi_profiles,
    get_csi_consistency_feature_columns,
)
from cross_layer_csi.experiments.evaluation import (
    build_feature_columns,
    split_calibration_holdout,
)
from cross_layer_csi.experiments.fusion import inject_csi_impostors, subset_impostor_cohort
from cross_layer_csi.experiments.identity import build_label_free_user_map
from cross_layer_csi.experiments.reporting import build_random_reproducibility_guard
from cross_layer_csi.experiments.two_phase import build_two_phase_pair_table
from cross_layer_csi.tabular.datasets.caixabank import CaixaBankDataset


def _merged_frame() -> pd.DataFrame:
    rows = []
    labels = [0, 0, 0, 0, 0, 0, 1, 1]
    users = ["u1", "u2", "u3", "u4", "u1", "u2", "u1", "u2"]
    for index, (label, user) in enumerate(zip(labels, users)):
        rows.append(
            {
                "uid": f"tx-{user}",
                "uid_csi_mapped": user,
                "uid_csi": user,
                "isFraud": label,
                "source_split": "orig_test",
                "split_role": "test",
                "split_source_expected": "orig_test",
                "split_coherent": True,
                "tx_row_id": f"orig_test::{index}",
                "mean_sc000": float(index),
                "std_sc000": float(index) / 10,
                "min_sc000": float(index) - 1,
                "max_sc000": float(index) + 1,
            }
        )
    for index, user in enumerate(("u1", "u2", "u3", "u4"), start=20):
        rows.append(
            {
                "uid": f"tx-{user}",
                "uid_csi_mapped": user,
                "uid_csi": user,
                "isFraud": 0,
                "source_split": "orig_train",
                "split_role": "train",
                "split_source_expected": "orig_train",
                "split_coherent": True,
                "tx_row_id": f"orig_train::{index}",
                "mean_sc000": float(index),
                "std_sc000": float(index) / 10,
                "min_sc000": float(index) - 1,
                "max_sc000": float(index) + 1,
            }
        )
    return pd.DataFrame(rows)


def test_seed_roles_match_the_executed_l9c_manifest() -> None:
    expected = {
        "IEEE_CIS": (67, 190, 58),
        "SPARKOV": (10067, 10190, 10058),
        "ECOMMERCE": (20067, 20190, 20058),
        "CAIXABANK": (30067, 30190, 30058),
    }
    for dataset, values in expected.items():
        seeds = SEED_POLICY_V120_3.for_scenario(dataset, 25)
        assert (seeds.merge, seeds.injection, seeds.evaluation) == values


def test_random_impostor_injection_reserves_unique_different_user_donors() -> None:
    injected = inject_csi_impostors(_merged_frame(), seed=190, donor_policy="random")
    impostors = injected[injected["attack_type"].eq("impostor")]
    assert len(impostors) == 2
    assert impostors["split_role"].eq("test").all()
    assert impostors["isFraud"].eq(1).all()
    assert not impostors["csi_impostor_same_user"].any()
    assert impostors["csi_impostor_donor_index"].nunique() == len(impostors)
    assert injected["csi_is_impostor_donor"].sum() == len(impostors)

    audit = audit_impostor_protocol({"IEEE_CIS": {25: injected}})
    assert audit.loc[0, "Audit_Status"] == "PASS"
    assert audit.loc[0, "TrainVal_Test_Tx_Row_Overlap"] == 0

    cohort = subset_impostor_cohort(injected)
    assert not cohort["csi_is_impostor_donor"].any()


def test_claimed_user_profiles_reject_test_rows_and_add_13_features() -> None:
    train = _merged_frame().query("split_role == 'train'").copy()
    bundle = fit_claimed_user_csi_profiles(train)
    augmented = apply_claimed_user_csi_consistency(train, bundle)
    assert len(get_csi_consistency_feature_columns(augmented)) == 13
    assert augmented["csi_profile_missing"].eq(0).all()

    with pytest.raises(RuntimeError, match="held-out test"):
        fit_claimed_user_csi_profiles(_merged_frame().query("split_role == 'test'"))


def test_tx_only_feature_selection_excludes_raw_and_consistency_csi() -> None:
    train = _merged_frame().query("split_role == 'train'").copy()
    bundle = fit_claimed_user_csi_profiles(train)
    augmented = apply_claimed_user_csi_consistency(train, bundle)
    tx_only = build_feature_columns(augmented, "IEEE_CIS", include_csi=False)
    tx_csi = build_feature_columns(augmented, "IEEE_CIS", include_csi=True)
    assert not any(
        column.startswith(("mean_", "std_", "min_", "max_", "csi_")) for column in tx_only
    )
    assert len([column for column in tx_csi if column.startswith("csi_")]) == 13


def test_grouped_calibration_holdout_has_no_transaction_overlap() -> None:
    rows = []
    for label, count in ((0, 8), (1, 4)):
        for index in range(count):
            tx_id = f"{label}-{index}"
            rows.extend(
                [
                    {"tx_row_id": tx_id, "isFraud": label, "value": index},
                    {"tx_row_id": tx_id, "isFraud": label, "value": index + 100},
                ]
            )
    calibration, holdout = split_calibration_holdout(pd.DataFrame(rows), seed=58)
    assert set(calibration["tx_row_id"]).isdisjoint(set(holdout["tx_row_id"]))
    assert set(calibration["isFraud"]) == {0, 1}
    assert set(holdout["isFraud"]) == {0, 1}


def test_random_replay_guard_fails_on_any_tp_or_fp_change() -> None:
    main = pd.DataFrame(
        [
            {
                "Dataset": "IEEE_CIS",
                "S": 25,
                "Modelo": "LOF_novelty",
                "TP_TxOnly": 57,
                "TP_TxCSI": 88,
                "FP_TxOnly": 909,
                "FP_TxCSI": 437,
            }
        ]
    )
    replay = main.assign(Donor_Policy="random")
    guard = build_random_reproducibility_guard(main, replay)
    assert guard.loc[0, "Random_Reproduces_Main"]

    changed = replay.copy()
    changed.loc[0, "FP_TxCSI"] += 1
    with pytest.raises(RuntimeError, match="did not exactly reproduce"):
        build_random_reproducibility_guard(main, changed)


def test_segmentation_produces_432_features_for_108_subcarriers() -> None:
    matrix = np.arange(50 * 108, dtype=np.float32).reshape(50, 108)
    segmented = segment_matrix(matrix, 25)
    assert segmented.shape == (2, 432)


def test_user_mapping_does_not_depend_on_fraud_labels() -> None:
    transactions = pd.DataFrame(
        {
            "uid": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "source_split": ["orig_train", "orig_test"] * 3,
            "isFraud": [0, 1, 0, 1, 1, 0],
        }
    )
    expected = build_label_free_user_map(transactions, ["csi-a", "csi-b"])
    flipped = transactions.assign(isFraud=1 - transactions["isFraud"])
    assert build_label_free_user_map(flipped, ["csi-a", "csi-b"]) == expected


def test_two_phase_beta_selection_uses_calibration_metrics_only() -> None:
    common = {
        "Dataset": "IEEE_CIS",
        "S": 25,
        "Supervised_Model": "LightGBM",
        "OneClass_CSI_Model": "LOF_novelty",
    }
    results = pd.DataFrame(
        [
            {
                **common,
                "Scenario_Mode": "Tx-Supervised",
                "Fusion_Rule": "TxOnlySupervised",
                "Beta": np.nan,
                "Validation_d": 0.4,
                "Validation_FP": 10,
                "Validation_TP": 5,
                "Validation_PR_AUC": 0.5,
                "Detected Frauds (TP)": 5,
                "False Alarms (FP)": 10,
            },
            {
                **common,
                "Scenario_Mode": "Tx-Supervised+CSI-Coherence",
                "Fusion_Rule": "ScoreFusion_tx_plus_beta_csi",
                "Beta": 0.5,
                "Validation_d": 0.1,
                "Validation_FP": 3,
                "Validation_TP": 6,
                "Validation_PR_AUC": 0.8,
                "Detected Frauds (TP)": 6,
                "False Alarms (FP)": 4,
            },
            {
                **common,
                "Scenario_Mode": "Tx-Supervised+CSI-Coherence",
                "Fusion_Rule": "ScoreFusion_tx_plus_beta_csi",
                "Beta": 1.0,
                "Validation_d": 0.2,
                "Validation_FP": 1,
                "Validation_TP": 7,
                "Validation_PR_AUC": 0.9,
                "Detected Frauds (TP)": 9,
                "False Alarms (FP)": 1,
            },
        ]
    )
    paired, selected = build_two_phase_pair_table(results)
    assert paired.loc[0, "Selected_Beta"] == 0.5
    assert selected.loc[0, "Beta"] == 0.5


def test_caixabank_legitimate_sampling_preserves_every_client_when_possible() -> None:
    legit = pd.DataFrame(
        {
            "client_id": np.repeat([1, 2, 3], [10, 5, 2]),
            "value": np.arange(17),
        }
    )
    handler = object.__new__(CaixaBankDataset)
    sampled = handler._sample_legits_preserving_clients(legit, 9)
    assert len(sampled) == 9
    assert set(sampled["client_id"]) == {1, 2, 3}
