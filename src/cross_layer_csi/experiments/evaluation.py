from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import OneClassSVM

from .config import CALIBRATION_FRACTION
from .consistency import (
    CSI_PROFILE_MIN_SAMPLES,
    attach_claimed_user_csi_consistency,
    get_csi_consistency_feature_columns,
)
from .fusion import get_raw_csi_feature_columns, subset_impostor_cohort

ONE_CLASS_DETECTORS = ("IsolationForest", "OneClassSVM_RBF", "LOF_novelty")
DATASET_EVALUATION_CONFIG = {
    "IEEE_CIS": {"drop_cols": ["uid"], "calibration_fraction": CALIBRATION_FRACTION},
    "SPARKOV": {
        "drop_cols": ["uid", "cc_num", "unix_time"],
        "calibration_fraction": CALIBRATION_FRACTION,
    },
    "ECOMMERCE": {"drop_cols": ["uid"], "calibration_fraction": CALIBRATION_FRACTION},
    "CAIXABANK": {"drop_cols": ["uid"], "calibration_fraction": CALIBRATION_FRACTION},
}

PROTOCOL_METADATA_COLUMNS = {
    "isFraud",
    "uid",
    "uid_csi_mapped",
    "uid_csi",
    "source_split",
    "split_role",
    "split_source_expected",
    "split_coherent",
    "fusion_split",
    "group_id",
    "capture_id",
    "source_file",
    "source_dataset",
    "native_subcarriers",
    "harmonized_subcarriers",
    "n_files_aggregated",
    "attack_type",
    "csi_impostor_donor_index",
    "csi_impostor_donor_uid",
    "csi_impostor_donor_split",
    "csi_impostor_same_user",
    "csi_impostor_donor_policy",
    "csi_is_impostor_donor",
    "csi_donor_for_target_index",
    "tx_row_id",
    "_tx_sampling_replacement_used",
}
FORBIDDEN_FEATURE_PATTERNS = (
    "split",
    "source_",
    "attack_type",
    "tx_row_id",
    "csi_impostor",
    "donor",
    "same_user",
    "_tx_sampling",
)


def build_feature_columns(
    frame: pd.DataFrame,
    dataset_label: str,
    *,
    include_csi: bool,
) -> list[str]:
    if dataset_label not in DATASET_EVALUATION_CONFIG:
        raise ValueError(f"Unknown dataset label: {dataset_label}")

    raw_csi = set(get_raw_csi_feature_columns(frame))
    consistency = set(get_csi_consistency_feature_columns(frame))
    dropped = PROTOCOL_METADATA_COLUMNS | set(
        DATASET_EVALUATION_CONFIG[dataset_label].get("drop_cols", [])
    )
    if not include_csi:
        dropped |= raw_csi | consistency
    selected = [column for column in frame.columns if column not in dropped]
    leaked = [
        column
        for column in selected
        if any(pattern in str(column) for pattern in FORBIDDEN_FEATURE_PATTERNS)
    ]
    if leaked:
        raise RuntimeError(
            f"{dataset_label}: forbidden metadata selected as features: {leaked[:20]}"
        )
    return selected


def coerce_boolean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_bool_dtype(out[column]):
            out[column] = out[column].astype(int)
    return out


def make_preprocessor(training_features: pd.DataFrame) -> Pipeline:
    training_features = coerce_boolean_columns(training_features)
    categorical: list[str] = []
    numeric: list[str] = []
    for column in training_features.columns:
        series = training_features[column]
        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            categorical.append(column)
        else:
            numeric.append(column)

    transformers = []
    if numeric:
        transformers.append(
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric)
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                        (
                            "ordinal",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise RuntimeError("No usable features available for preprocessing")

    return Pipeline(
        [
            ("columns", ColumnTransformer(transformers=transformers, sparse_threshold=0.0)),
            ("scaler", StandardScaler()),
        ]
    )


def build_one_class_estimator(name: str, *, seed: int, nominal_rows: int):
    if nominal_rows < 5:
        raise RuntimeError(f"Too few nominal training rows: {nominal_rows}")
    if name == "IsolationForest":
        return IsolationForest(
            n_estimators=400,
            max_samples="auto",
            contamination="auto",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "OneClassSVM_RBF":
        return OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    if name == "LOF_novelty":
        return LocalOutlierFactor(
            n_neighbors=int(min(35, max(5, nominal_rows - 1))),
            novelty=True,
            contamination="auto",
        )
    raise ValueError(f"Unknown one-class detector: {name}")


def fit_one_class_model(
    train: pd.DataFrame,
    *,
    dataset_label: str,
    include_csi: bool,
    detector_name: str,
    seed: int,
) -> tuple[dict[str, object], Pipeline, list[str]]:
    feature_columns = build_feature_columns(train, dataset_label, include_csi=include_csi)
    if not feature_columns:
        raise RuntimeError(f"{dataset_label}: no feature columns; include_csi={include_csi}")
    nominal = train[train["isFraud"].astype(int).eq(0)].copy()
    if nominal.empty:
        raise RuntimeError(f"{dataset_label}: no nominal rows available")

    nominal_features = coerce_boolean_columns(nominal[feature_columns])
    preprocessor = make_preprocessor(nominal_features)
    transformed = preprocessor.fit_transform(nominal_features)
    estimator = build_one_class_estimator(
        detector_name,
        seed=seed,
        nominal_rows=len(nominal),
    )
    estimator.fit(transformed)
    bundle = {
        "estimator": estimator,
        "estimator_name": detector_name,
        "include_csi": bool(include_csi),
        "n_nominal_train": len(nominal),
    }
    return bundle, preprocessor, feature_columns


def score_one_class_model(
    model_bundle: dict[str, object],
    preprocessor: Pipeline,
    feature_columns: list[str],
    frame: pd.DataFrame,
) -> np.ndarray:
    """Return scores where a larger value means more anomalous."""

    if frame.empty:
        return np.array([], dtype=np.float64)
    transformed = preprocessor.transform(coerce_boolean_columns(frame[feature_columns]))
    estimator = model_bundle["estimator"]
    if hasattr(estimator, "decision_function"):
        normality = estimator.decision_function(transformed)
    elif hasattr(estimator, "score_samples"):
        normality = estimator.score_samples(transformed)
    else:
        raise RuntimeError("One-class estimator exposes neither scoring method")
    return -np.asarray(normality, dtype=np.float64)


def compute_metrics(y_true, scores, threshold: float) -> dict[str, float | int]:
    labels = np.asarray(y_true, dtype=int)
    anomaly_scores = np.asarray(scores, dtype=np.float64)
    predicted = anomaly_scores >= float(threshold)
    positive = labels == 1
    negative = ~positive
    true_positive = int((predicted & positive).sum())
    false_positive = int((predicted & negative).sum())
    false_negative = int((~predicted & positive).sum())
    true_negative = int((~predicted & negative).sum())

    recall = true_positive / (true_positive + false_negative) if positive.any() else 0.0
    precision = true_positive / (true_positive + false_positive) if predicted.any() else 0.0
    fpr = false_positive / (false_positive + true_negative) if negative.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if len(np.unique(labels)) > 1:
        roc_auc = float(roc_auc_score(labels, anomaly_scores))
        pr_auc = float(average_precision_score(labels, anomaly_scores))
    else:
        roc_auc = np.nan
        pr_auc = float(labels.mean()) if len(labels) else np.nan
    distance = float(np.sqrt((1.0 - recall) ** 2 + fpr**2))
    return {
        "Total Transactions (Tx)": len(labels),
        "Total Frauds (Real)": int(positive.sum()),
        "Predicted Frauds": int(predicted.sum()),
        "Detected Frauds (TP)": true_positive,
        "Missed Frauds (FN)": false_negative,
        "False Alarms (FP)": false_positive,
        "True Negatives (TN)": true_negative,
        "FPR": fpr,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "d": distance,
        "Threshold": float(threshold),
    }


def select_operating_threshold(scores, y_true) -> tuple[float, dict[str, float | int]]:
    """Select the same calibration-only min-distance threshold for both arms."""

    labels = np.asarray(y_true, dtype=int)
    anomaly_scores = np.asarray(scores, dtype=np.float64)
    finite = anomaly_scores[np.isfinite(anomaly_scores)]
    if not len(finite):
        metrics = compute_metrics(labels, np.zeros_like(labels, dtype=float), 0.0)
        return 0.0, metrics

    candidates = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 257)))
    candidates = np.r_[finite.min() - 1e-9, candidates, finite.max() + 1e-9]
    table = pd.DataFrame(
        [compute_metrics(labels, anomaly_scores, threshold) for threshold in candidates]
    ).drop_duplicates(subset=["Threshold"])
    table = table.sort_values(
        ["d", "False Alarms (FP)", "FPR", "Recall", "Precision", "Threshold"],
        ascending=[True, True, True, False, False, True],
    ).reset_index(drop=True)
    best = table.iloc[0].to_dict()
    return float(best["Threshold"]), best


def split_calibration_holdout(
    test_pool: pd.DataFrame,
    *,
    calibration_fraction: float = CALIBRATION_FRACTION,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratify by label while keeping every tx_row_id in exactly one subset."""

    if test_pool.empty:
        return test_pool.copy(), test_pool.copy()
    if "tx_row_id" not in test_pool.columns:
        raise RuntimeError("tx_row_id is required for grouped calibration/holdout splitting")
    work = test_pool.copy().reset_index(drop=True)
    work["tx_row_id"] = work["tx_row_id"].astype(str)
    label_counts = work.groupby("tx_row_id", observed=True)["isFraud"].nunique()
    if label_counts.gt(1).any():
        raise RuntimeError("A tx_row_id has conflicting fraud labels")

    labels = work.groupby("tx_row_id", observed=True)["isFraud"].first().astype(int)
    fraud_groups = labels[labels.eq(1)].index.to_numpy(dtype=object)
    legit_groups = labels[labels.eq(0)].index.to_numpy(dtype=object)
    if len(fraud_groups) < 2 or len(legit_groups) < 2:
        raise RuntimeError(
            "Need at least two fraud and two legitimate tx_row_id groups; "
            f"got fraud={len(fraud_groups)}, legit={len(legit_groups)}"
        )
    rng = np.random.default_rng(seed)

    def split_groups(groups: np.ndarray) -> tuple[set[object], set[object]]:
        shuffled = rng.permutation(np.asarray(groups, dtype=object))
        calibration_count = round(len(shuffled) * float(calibration_fraction))
        calibration_count = min(max(1, calibration_count), len(shuffled) - 1)
        return set(shuffled[:calibration_count]), set(shuffled[calibration_count:])

    calibration_fraud, holdout_fraud = split_groups(fraud_groups)
    calibration_legit, holdout_legit = split_groups(legit_groups)
    calibration_ids = calibration_fraud | calibration_legit
    holdout_ids = holdout_fraud | holdout_legit
    if calibration_ids & holdout_ids:
        raise RuntimeError("Calibration and holdout tx_row_id sets overlap")

    calibration = work[work["tx_row_id"].isin(calibration_ids)].copy().reset_index(drop=True)
    holdout = work[work["tx_row_id"].isin(holdout_ids)].copy().reset_index(drop=True)
    calibration["fusion_split"] = "calibration"
    holdout["fusion_split"] = "holdout"
    return calibration, holdout


def apply_csi_negative_control(
    frame: pd.DataFrame,
    *,
    mode: str = "true_csi",
    seed: int = 42,
) -> pd.DataFrame:
    """Apply post-construction ablations; these are not attack contexts."""

    out = frame.copy()
    csi_columns = get_raw_csi_feature_columns(out)
    if mode in {"true_csi", "tx_only"} or not csi_columns:
        return out
    rng = np.random.default_rng(seed)
    if mode == "shuffled_csi":
        for indices in out.groupby(out["split_role"].astype(str)).groups.values():
            indices = list(indices)
            for column in csi_columns:
                out.loc[indices, column] = rng.permutation(out.loc[indices, column].to_numpy())
        return out
    if mode == "noisy_csi":
        train_mask = out["split_role"].astype(str).isin(["train", "val"])
        reference = out.loc[train_mask, csi_columns] if train_mask.any() else out[csi_columns]
        std = reference.std().replace(0, 1.0).fillna(1.0).to_numpy(dtype=np.float64)
        noise = rng.normal(0.0, 1.0, size=(len(out), len(csi_columns))) * std
        out.loc[:, csi_columns] = out[csi_columns].to_numpy(dtype=np.float64) + noise
        return out
    raise ValueError(f"Unknown CSI negative-control mode: {mode}")


def evaluate_controlled_one_class(
    merged: pd.DataFrame,
    *,
    dataset_label: str,
    segment_size: int,
    seed: int,
    csi_mode: str = "true_csi",
    detector_names: Iterable[str] = ONE_CLASS_DETECTORS,
    profile_min_samples: int = CSI_PROFILE_MIN_SAMPLES,
) -> list[dict[str, object]]:
    """Evaluate paired Tx-only and Tx+CSI arms on an untouched holdout."""

    if merged.empty:
        return []
    variant = apply_csi_negative_control(merged, mode=csi_mode, seed=seed)
    cohort = subset_impostor_cohort(variant)
    train = cohort[cohort["split_role"].astype(str).isin(["train", "val"])].copy()
    test_pool = cohort[cohort["split_role"].astype(str).eq("test")].copy()
    if train.empty or test_pool.empty:
        return []

    calibration_fraction = float(DATASET_EVALUATION_CONFIG[dataset_label]["calibration_fraction"])
    calibration, holdout = split_calibration_holdout(
        test_pool,
        calibration_fraction=calibration_fraction,
        seed=seed,
    )
    train, calibration, holdout, profile_bundle = attach_claimed_user_csi_consistency(
        train,
        calibration,
        holdout,
        min_profile_samples=profile_min_samples,
    )
    profile_count = int(profile_bundle.get("profile_users", 0)) if profile_bundle else 0
    profile_rows = int(profile_bundle.get("reference_rows_nominal", 0)) if profile_bundle else 0
    calibration_labels = calibration["isFraud"].astype(int).to_numpy()
    holdout_labels = holdout["isFraud"].astype(int).to_numpy()

    def missing_rate(frame: pd.DataFrame) -> float:
        if frame.empty or "csi_profile_missing" not in frame.columns:
            return np.nan
        return float(pd.to_numeric(frame["csi_profile_missing"], errors="coerce").mean())

    rows: list[dict[str, object]] = []
    for detector_name in map(str, detector_names):
        for scenario, include_csi in (("Tx-Only", False), ("Tx+CSI", True)):
            if include_csi and csi_mode == "tx_only":
                continue
            model, preprocessor, feature_columns = fit_one_class_model(
                train,
                dataset_label=dataset_label,
                include_csi=include_csi,
                detector_name=detector_name,
                seed=seed,
            )
            calibration_scores = score_one_class_model(
                model, preprocessor, feature_columns, calibration
            )
            holdout_scores = score_one_class_model(model, preprocessor, feature_columns, holdout)
            threshold, calibration_metrics = select_operating_threshold(
                calibration_scores, calibration_labels
            )
            holdout_metrics = compute_metrics(holdout_labels, holdout_scores, threshold)
            rows.append(
                {
                    "Dataset": dataset_label,
                    "S": int(segment_size),
                    "Modelo": model["estimator_name"],
                    "Pipeline_Family": "same_oneclass_detector_same_threshold_policy",
                    "Scenario_Mode": scenario,
                    "CSI_Mode": csi_mode if include_csi else "not_used",
                    "Attack_View": "impostor",
                    "Features": "Tx+CSI" if include_csi else "Tx-Only",
                    "Feature_Count": len(feature_columns),
                    "CSI_Consistency_Feature_Count": (
                        len(get_csi_consistency_feature_columns(train)) if include_csi else 0
                    ),
                    "CSI_Feature_Family": (
                        "raw_csi_plus_claimed_user_consistency" if include_csi else "not_used"
                    ),
                    "CSI_Profile_Min_Samples": profile_min_samples if include_csi else 0,
                    "CSI_Profile_User_Count": profile_count if include_csi else 0,
                    "CSI_Profile_Reference_Rows": profile_rows if include_csi else 0,
                    "CSI_Profile_Missing_Rate_Calibration": (
                        missing_rate(calibration) if include_csi else np.nan
                    ),
                    "CSI_Profile_Missing_Rate_Holdout": (
                        missing_rate(holdout) if include_csi else np.nan
                    ),
                    "Training_Rows_Nominal": model["n_nominal_train"],
                    "Calibration_Policy": "same_min_d_on_calibration",
                    **holdout_metrics,
                    "Validation_Threshold": threshold,
                    "Validation_Recall": calibration_metrics["Recall"],
                    "Validation_FPR": calibration_metrics["FPR"],
                    "Validation_TP": calibration_metrics["Detected Frauds (TP)"],
                    "Validation_FP": calibration_metrics["False Alarms (FP)"],
                    "Validation_d": calibration_metrics["d"],
                    "Calibration_Size": len(calibration),
                    "Holdout_Size": len(holdout),
                    "Train_Size": len(train),
                }
            )
    return rows


# Notebook-compatible wrappers retained for result traceability.
def build_feature_columns_v108(frame, dataset_label, include_csi):
    return build_feature_columns(frame, dataset_label, include_csi=include_csi)


make_preprocessor_v108 = make_preprocessor


def build_oneclass_estimator_v108(detector_name, seed, n_nominal):
    return build_one_class_estimator(detector_name, seed=seed, nominal_rows=n_nominal)


def fit_oneclass_controlled_model_v108(train_df, dataset_label, include_csi, detector_name, seed):
    return fit_one_class_model(
        train_df,
        dataset_label=dataset_label,
        include_csi=include_csi,
        detector_name=detector_name,
        seed=seed,
    )


score_oneclass_controlled_model_v108 = score_one_class_model
compute_metrics_from_scores_v108 = compute_metrics
select_operating_threshold_v108 = select_operating_threshold


def split_test_pool_stratified_v108(
    df_test_pool, calibration_fraction=CALIBRATION_FRACTION, seed=42
):
    return split_calibration_holdout(
        df_test_pool,
        calibration_fraction=calibration_fraction,
        seed=seed,
    )


def apply_csi_variant_mode_v108(frame, mode="true_csi", seed=42):
    return apply_csi_negative_control(frame, mode=mode, seed=seed)


def evaluate_controlled_feature_pipeline_v108(
    df_merged,
    dataset_label,
    s_val,
    seed=33,
    csi_mode="true_csi",
    attack_view="impostor",
    detector_names=None,
    profile_min_samples=CSI_PROFILE_MIN_SAMPLES,
):
    if attack_view != "impostor":
        raise ValueError(f"Unsupported attack_view={attack_view!r}; v120.3 is impostor-only")
    kwargs = {}
    if detector_names is not None:
        kwargs["detector_names"] = detector_names
    return evaluate_controlled_one_class(
        df_merged,
        dataset_label=dataset_label,
        segment_size=s_val,
        seed=seed,
        csi_mode=csi_mode,
        profile_min_samples=profile_min_samples,
        **kwargs,
    )
