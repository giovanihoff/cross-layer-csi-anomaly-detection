from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .consistency import (
    CSI_PROFILE_MIN_SAMPLES,
    attach_claimed_user_csi_consistency,
    get_csi_consistency_feature_columns,
)
from .evaluation import (
    DATASET_EVALUATION_CONFIG,
    build_feature_columns,
    coerce_boolean_columns,
    compute_metrics,
    fit_one_class_model,
    make_preprocessor,
    score_one_class_model,
    select_operating_threshold,
    split_calibration_holdout,
)
from .fusion import subset_impostor_cohort

TWO_PHASE_BETA_GRID = (0.25, 0.50, 1.00, 1.50, 2.00)


def _optional_class(module_name: str, class_name: str):
    try:
        return getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError):
        return None


def available_supervised_models() -> list[str]:
    availability = {
        "CatBoost": _optional_class("catboost", "CatBoostClassifier"),
        "XGBoost": _optional_class("xgboost", "XGBClassifier"),
        "LightGBM": _optional_class("lightgbm", "LGBMClassifier"),
    }
    return [name for name, model_class in availability.items() if model_class is not None]


def positive_class_weight(labels) -> float:
    labels = np.asarray(labels, dtype=int)
    positives = max(1, int(labels.sum()))
    negatives = max(1, int(labels.size - labels.sum()))
    return negatives / positives


def build_supervised_tx_estimator(model_name: str, labels, *, seed: int):
    """Build a classifier that receives transaction features only."""

    if model_name == "CatBoost":
        model_class = _optional_class("catboost", "CatBoostClassifier")
        if model_class is None:
            raise RuntimeError("CatBoost is not installed")
        return model_class(
            iterations=500,
            depth=6,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
    if model_name == "XGBoost":
        model_class = _optional_class("xgboost", "XGBClassifier")
        if model_class is None:
            raise RuntimeError("XGBoost is not installed")
        return model_class(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=positive_class_weight(labels),
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "LightGBM":
        model_class = _optional_class("lightgbm", "LGBMClassifier")
        if model_class is None:
            raise RuntimeError("LightGBM is not installed")
        return model_class(
            n_estimators=500,
            num_leaves=31,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(f"Unknown supervised model: {model_name}")


def fit_supervised_tx_model(
    train: pd.DataFrame,
    *,
    dataset_label: str,
    model_name: str,
    seed: int,
) -> tuple[Pipeline, list[str]]:
    feature_columns = build_feature_columns(train, dataset_label, include_csi=False)
    if not feature_columns:
        raise RuntimeError(f"{dataset_label}: no Tx-only features")
    labels = train["isFraud"].astype(int).to_numpy()
    if len(np.unique(labels)) < 2:
        raise RuntimeError(f"{dataset_label}: supervised phase requires both classes")
    features = coerce_boolean_columns(train[feature_columns])
    pipeline = Pipeline(
        [
            ("preprocess", make_preprocessor(features)),
            ("estimator", build_supervised_tx_estimator(model_name, labels, seed=seed)),
        ]
    )
    pipeline.fit(features, labels)
    return pipeline, feature_columns


def score_supervised_tx_model(
    pipeline: Pipeline,
    feature_columns: list[str],
    frame: pd.DataFrame,
) -> np.ndarray:
    if frame.empty:
        return np.array([], dtype=np.float64)
    features = coerce_boolean_columns(frame[feature_columns])
    if hasattr(pipeline, "predict_proba"):
        return np.asarray(pipeline.predict_proba(features)[:, 1], dtype=np.float64)
    raw = pipeline.decision_function(features)
    return 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=np.float64)))


def robust_minmax(reference_scores, values) -> np.ndarray:
    reference = np.asarray(reference_scores, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    reference = reference[np.isfinite(reference)]
    if not len(reference):
        return np.zeros_like(values)
    low, high = np.nanpercentile(reference, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = np.nanmin(reference), np.nanmax(reference)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def evaluate_two_phase_tx_csi(
    merged: pd.DataFrame,
    *,
    dataset_label: str,
    segment_size: int,
    supervised_model_name: str,
    one_class_detector_name: str,
    seed: int,
    beta_grid=TWO_PHASE_BETA_GRID,
    profile_min_samples: int = CSI_PROFILE_MIN_SAMPLES,
) -> list[dict[str, object]]:
    """Train supervised Tx risk and one-class CSI coherence in separate phases."""

    if merged.empty:
        return []
    cohort = subset_impostor_cohort(merged)
    supervised_train = merged[merged["split_role"].astype(str).isin(["train", "val"])].copy()
    csi_train = cohort[cohort["split_role"].astype(str).isin(["train", "val"])].copy()
    test_pool = cohort[cohort["split_role"].astype(str).eq("test")].copy()
    if supervised_train.empty or csi_train.empty or test_pool.empty:
        return []
    calibration, holdout = split_calibration_holdout(
        test_pool,
        calibration_fraction=float(
            DATASET_EVALUATION_CONFIG[dataset_label]["calibration_fraction"]
        ),
        seed=seed,
    )
    csi_train, calibration, holdout, profile_bundle = attach_claimed_user_csi_consistency(
        csi_train,
        calibration,
        holdout,
        min_profile_samples=profile_min_samples,
    )
    calibration_labels = calibration["isFraud"].astype(int).to_numpy()
    holdout_labels = holdout["isFraud"].astype(int).to_numpy()

    tx_pipeline, tx_columns = fit_supervised_tx_model(
        supervised_train,
        dataset_label=dataset_label,
        model_name=supervised_model_name,
        seed=seed,
    )
    tx_calibration_scores = score_supervised_tx_model(tx_pipeline, tx_columns, calibration)
    tx_holdout_scores = score_supervised_tx_model(tx_pipeline, tx_columns, holdout)
    tx_threshold, tx_calibration_metrics = select_operating_threshold(
        tx_calibration_scores,
        calibration_labels,
    )
    tx_holdout_metrics = compute_metrics(holdout_labels, tx_holdout_scores, tx_threshold)

    common = {
        "Dataset": dataset_label,
        "S": int(segment_size),
        "Supervised_Model": supervised_model_name,
        "OneClass_CSI_Model": one_class_detector_name,
        "Total Transactions (Tx)": len(holdout),
        "Total Frauds (Real)": int(holdout_labels.sum()),
        "Calibration_Size": len(calibration),
        "Holdout_Size": len(holdout),
        "Tx_Feature_Count": len(tx_columns),
    }
    rows = [
        {
            **common,
            "Scenario_Mode": "Tx-Supervised",
            "Fusion_Rule": "TxOnlySupervised",
            "Beta": np.nan,
            "TxCSI_Feature_Count": np.nan,
            "CSI_Consistency_Feature_Count": 0,
            "CSI_Profile_Min_Samples": 0,
            "CSI_Profile_User_Count": 0,
            "CSI_Profile_Reference_Rows": 0,
            "Validation_Threshold": tx_threshold,
            "Validation_TP": tx_calibration_metrics["Detected Frauds (TP)"],
            "Validation_FP": tx_calibration_metrics["False Alarms (FP)"],
            "Validation_Recall": tx_calibration_metrics["Recall"],
            "Validation_FPR": tx_calibration_metrics["FPR"],
            "Validation_PR_AUC": tx_calibration_metrics["PR-AUC"],
            "Validation_d": tx_calibration_metrics["d"],
            **tx_holdout_metrics,
        }
    ]

    csi_model, csi_preprocessor, csi_columns = fit_one_class_model(
        csi_train,
        dataset_label=dataset_label,
        include_csi=True,
        detector_name=one_class_detector_name,
        seed=seed,
    )
    csi_calibration_scores = score_one_class_model(
        csi_model, csi_preprocessor, csi_columns, calibration
    )
    csi_holdout_scores = score_one_class_model(csi_model, csi_preprocessor, csi_columns, holdout)
    tx_calibration_norm = robust_minmax(tx_calibration_scores, tx_calibration_scores)
    tx_holdout_norm = robust_minmax(tx_calibration_scores, tx_holdout_scores)
    csi_calibration_norm = robust_minmax(csi_calibration_scores, csi_calibration_scores)
    csi_holdout_norm = robust_minmax(csi_calibration_scores, csi_holdout_scores)

    for beta in map(float, beta_grid):
        fused_calibration = tx_calibration_norm + beta * csi_calibration_norm
        fused_holdout = tx_holdout_norm + beta * csi_holdout_norm
        threshold, validation = select_operating_threshold(
            fused_calibration,
            calibration_labels,
        )
        metrics = compute_metrics(holdout_labels, fused_holdout, threshold)
        rows.append(
            {
                **common,
                "Scenario_Mode": "Tx-Supervised+CSI-Coherence",
                "Fusion_Rule": "ScoreFusion_tx_plus_beta_csi",
                "Beta": beta,
                "TxCSI_Feature_Count": len(csi_columns),
                "CSI_Consistency_Feature_Count": len(
                    get_csi_consistency_feature_columns(csi_train)
                ),
                "CSI_Profile_Min_Samples": profile_min_samples,
                "CSI_Profile_User_Count": (
                    int(profile_bundle.get("profile_users", 0)) if profile_bundle else 0
                ),
                "CSI_Profile_Reference_Rows": (
                    int(profile_bundle.get("reference_rows_nominal", 0)) if profile_bundle else 0
                ),
                "Validation_Threshold": threshold,
                "Validation_TP": validation["Detected Frauds (TP)"],
                "Validation_FP": validation["False Alarms (FP)"],
                "Validation_Recall": validation["Recall"],
                "Validation_FPR": validation["FPR"],
                "Validation_PR_AUC": validation["PR-AUC"],
                "Validation_d": validation["d"],
                **metrics,
            }
        )
    return rows


def build_two_phase_pair_table(raw_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select beta on calibration only, then pair against the Tx-supervised arm."""

    if raw_results.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows, selected_rows = [], []
    group_columns = ["Dataset", "S", "Supervised_Model", "OneClass_CSI_Model"]
    for keys, block in raw_results.groupby(group_columns):
        baseline = block[block["Scenario_Mode"].eq("Tx-Supervised")]
        fused = block[block["Scenario_Mode"].eq("Tx-Supervised+CSI-Coherence")]
        if baseline.empty or fused.empty:
            continue
        tx = baseline.iloc[0]
        selected = fused.sort_values(
            ["Validation_d", "Validation_FP", "Validation_TP", "Validation_PR_AUC", "Beta"],
            ascending=[True, True, False, False, True],
        ).iloc[0]
        fp_tx, fp_fused = int(tx["False Alarms (FP)"]), int(selected["False Alarms (FP)"])
        row = {
            **dict(zip(group_columns, keys)),
            "Selected_Fusion_Rule": str(selected["Fusion_Rule"]),
            "Selected_Beta": float(selected["Beta"]),
            "TP_TxSupervised": int(tx["Detected Frauds (TP)"]),
            "TP_TxSup_CSI": int(selected["Detected Frauds (TP)"]),
            "Delta_TP": int(selected["Detected Frauds (TP)"] - tx["Detected Frauds (TP)"]),
            "FP_TxSupervised": fp_tx,
            "FP_TxSup_CSI": fp_fused,
            "Delta_FP": fp_fused - fp_tx,
            "FP_Reduction_pct": 100.0 * (fp_tx - fp_fused) / max(1, fp_tx),
            "Pareto_Calibration": bool(
                selected["Validation_TP"] >= tx["Validation_TP"]
                and selected["Validation_FP"] <= tx["Validation_FP"]
            ),
            "Pareto_Holdout": bool(
                selected["Detected Frauds (TP)"] >= tx["Detected Frauds (TP)"] and fp_fused <= fp_tx
            ),
        }
        rows.append(row)
        selected_rows.append(selected.to_dict())
    paired = pd.DataFrame(rows)
    for column in ("Supervised_Model", "OneClass_CSI_Model", "Selected_Fusion_Rule"):
        if not paired.empty and paired[column].isna().any():
            raise RuntimeError(f"NaN in required two-phase text column: {column}")
    return paired, pd.DataFrame(selected_rows)


# Notebook-compatible wrappers.
def build_supervised_tx_estimator_v113(model_name, y_train, seed):
    return build_supervised_tx_estimator(model_name, y_train, seed=seed)


def fit_supervised_tx_model_v113(train_df, dataset_label, model_name, seed):
    return fit_supervised_tx_model(
        train_df,
        dataset_label=dataset_label,
        model_name=model_name,
        seed=seed,
    )


score_supervised_tx_model_v113 = score_supervised_tx_model
robust_minmax_from_reference_v113 = robust_minmax


def evaluate_two_phase_tx_csi_v113(
    df_merged,
    dataset_label,
    s_val,
    supervised_model_name,
    oneclass_detector_name,
    seed=33,
    beta_grid=None,
    profile_min_samples=CSI_PROFILE_MIN_SAMPLES,
):
    return evaluate_two_phase_tx_csi(
        df_merged,
        dataset_label=dataset_label,
        segment_size=s_val,
        supervised_model_name=supervised_model_name,
        one_class_detector_name=oneclass_detector_name,
        seed=seed,
        beta_grid=TWO_PHASE_BETA_GRID if beta_grid is None else beta_grid,
        profile_min_samples=profile_min_samples,
    )


build_two_phase_pair_table_v113 = build_two_phase_pair_table
