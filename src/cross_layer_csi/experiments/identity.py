from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def make_proxy_uid(frame: pd.DataFrame, columns: Sequence[str], *, prefix: str) -> pd.Series:
    valid = [column for column in columns if column in frame.columns]
    if not valid:
        return pd.Series(
            [f"{prefix}_{index}" for index in range(len(frame))],
            index=frame.index,
            dtype="object",
        )
    uid = frame[valid[0]].astype("string").fillna("__MISSING__")
    for column in valid[1:]:
        uid = uid + "|" + frame[column].astype("string").fillna("__MISSING__")
    return uid.astype(str)


def fit_apply_numeric_buckets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    column: str,
    *,
    quantiles: int = 6,
) -> tuple[pd.Series, pd.Series]:
    """Fit quantile edges on train only and apply them unchanged to test."""

    if column not in train.columns or column not in test.columns:
        return (
            pd.Series("__NA__", index=train.index, dtype="object"),
            pd.Series("__NA__", index=test.index, dtype="object"),
        )
    train_values = pd.to_numeric(train[column], errors="coerce")
    test_values = pd.to_numeric(test[column], errors="coerce")
    non_missing = train_values.dropna()
    if non_missing.nunique() < 2:
        return (
            train_values.fillna(-999).astype(str),
            test_values.fillna(-999).astype(str),
        )
    try:
        _, edges = pd.qcut(
            non_missing,
            q=min(quantiles, non_missing.nunique()),
            retbins=True,
            duplicates="drop",
        )
        edges = np.unique(np.asarray(edges, dtype=float))
        if len(edges) < 2:
            raise ValueError("insufficient distinct train-only bucket edges")
        edges[0], edges[-1] = -np.inf, np.inf
        return (
            pd.cut(train_values, bins=edges, include_lowest=True)
            .astype("string")
            .fillna("__MISSING__")
            .astype(str),
            pd.cut(test_values, bins=edges, include_lowest=True)
            .astype("string")
            .fillna("__MISSING__")
            .astype(str),
        )
    except (ValueError, TypeError):
        median = float(non_missing.median())
        return (
            train_values.fillna(median).round(2).astype(str),
            test_values.fillna(median).round(2).astype(str),
        )


def evaluate_uid_proxy(
    train_uid: pd.Series,
    test_uid: pd.Series,
    *,
    target_csi_users: int,
) -> dict[str, float | int]:
    """Score UID support and overlap without consulting fraud labels."""

    train_uid = train_uid.astype(str)
    test_uid = test_uid.astype(str)
    support = pd.concat(
        [
            train_uid.value_counts().rename("train_total"),
            test_uid.value_counts().rename("test_total"),
        ],
        axis=1,
    ).fillna(0)
    eligible = support[support["train_total"].ge(1) & support["test_total"].ge(1)]
    largest_group = int(max(train_uid.value_counts().max(), test_uid.value_counts().max()))
    return {
        "eligible_count": len(eligible),
        "overlap": len(eligible),
        "median_train_total": float(eligible["train_total"].median()) if len(eligible) else 0.0,
        "median_test_total": float(eligible["test_total"].median()) if len(eligible) else 0.0,
        "median_joint_support": float(eligible[["train_total", "test_total"]].min(axis=1).median())
        if len(eligible)
        else 0.0,
        "coverage_rows": int(eligible[["train_total", "test_total"]].to_numpy().sum()),
        "distinct_train": train_uid.nunique(),
        "distinct_test": test_uid.nunique(),
        "max_group_share": largest_group / max(1, len(train_uid) + len(test_uid)),
        "target_overlap_gap": abs(min(len(eligible), target_csi_users) - target_csi_users),
    }


def ensure_analytical_uid(
    dataset_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_csi_users: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Select the final analytical UID using metadata only."""

    train = train.copy()
    test = test.copy()
    if dataset_name != "ecommerce":
        if "uid" not in train.columns or "uid" not in test.columns:
            raise RuntimeError(f"{dataset_name}: uid is required")
        train["uid"], test["uid"] = train["uid"].astype(str), test["uid"].astype(str)
        return (
            train,
            test,
            {"strategy": "existing_uid", "label_free_selection": True},
            pd.DataFrame(),
        )

    proxy_train, proxy_test = train.copy(), test.copy()
    for column, output in (
        ("age", "age_bucket"),
        ("purchase_value", "purchase_value_bucket"),
        ("time_diff_seconds", "time_diff_bucket"),
    ):
        proxy_train[output], proxy_test[output] = fit_apply_numeric_buckets(
            proxy_train,
            proxy_test,
            column,
        )
    for frame in (proxy_train, proxy_test):
        for column in ("source", "browser", "sex"):
            if column in frame.columns:
                frame[f"{column}_norm"] = (
                    frame[column].astype("string").fillna("__MISSING__").str.lower()
                )

    candidates = (
        ("existing_uid", ["uid"]),
        ("user_id", ["user_id"]),
        ("device_id", ["device_id"]),
        ("source_browser", ["source_norm", "browser_norm"]),
        ("source_browser_sex", ["source_norm", "browser_norm", "sex_norm"]),
        ("source_browser_age", ["source_norm", "browser_norm", "age_bucket"]),
        ("source_browser_value", ["source_norm", "browser_norm", "purchase_value_bucket"]),
        ("source_browser_time", ["source_norm", "browser_norm", "time_diff_bucket"]),
        ("source_browser_age_sex", ["source_norm", "browser_norm", "age_bucket", "sex_norm"]),
        (
            "source_browser_age_value",
            ["source_norm", "browser_norm", "age_bucket", "purchase_value_bucket"],
        ),
        (
            "source_browser_age_value_sex",
            ["source_norm", "browser_norm", "age_bucket", "purchase_value_bucket", "sex_norm"],
        ),
        ("browser_age_value", ["browser_norm", "age_bucket", "purchase_value_bucket"]),
        ("browser_sex_time", ["browser_norm", "sex_norm", "time_diff_bucket"]),
    )
    rows = []
    for strategy, columns in candidates:
        valid = [
            column
            for column in columns
            if column in proxy_train.columns and column in proxy_test.columns
        ]
        if not valid:
            continue
        train_uid = make_proxy_uid(proxy_train, valid, prefix=strategy)
        test_uid = make_proxy_uid(proxy_test, valid, prefix=strategy)
        rows.append(
            {
                "strategy": strategy,
                "columns": valid,
                "train_uid": train_uid,
                "test_uid": test_uid,
                **evaluate_uid_proxy(
                    train_uid,
                    test_uid,
                    target_csi_users=target_csi_users,
                ),
            }
        )
    diagnostics = pd.DataFrame(rows)
    diagnostics = (
        diagnostics[diagnostics["eligible_count"].gt(0)]
        .sort_values(
            [
                "target_overlap_gap",
                "median_joint_support",
                "coverage_rows",
                "max_group_share",
                "strategy",
            ],
            ascending=[True, False, False, True, True],
        )
        .reset_index(drop=True)
    )
    if diagnostics.empty:
        raise RuntimeError("ECOMMERCE: no UID candidate has train/test support")
    selected = diagnostics.iloc[0]
    train["uid"] = selected["train_uid"].astype(str)
    test["uid"] = selected["test_uid"].astype(str)
    metadata = {
        "strategy": selected["strategy"],
        "columns": selected["columns"],
        "eligible_count": int(selected["eligible_count"]),
        "overlap": int(selected["overlap"]),
        "coverage_rows": int(selected["coverage_rows"]),
        "max_group_share": float(selected["max_group_share"]),
        "label_free_selection": True,
    }
    visible = diagnostics.drop(columns=["train_uid", "test_uid"])
    return train, test, metadata, visible


def load_transaction_xy_artifacts(
    dataset_name: str,
    dataset_dir: str | Path,
    *,
    target_csi_users: int,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    directory = Path(dataset_dir)
    train = pd.read_parquet(directory / "X_train.parquet")
    test = pd.read_parquet(directory / "X_test.parquet")
    y_train = pd.read_parquet(directory / "y_train.parquet").iloc[:, 0].astype(int)
    y_test = pd.read_parquet(directory / "y_test.parquet").iloc[:, 0].astype(int)
    train, test, metadata, diagnostics = ensure_analytical_uid(
        dataset_name,
        train,
        test,
        target_csi_users=target_csi_users,
    )
    full = pd.concat(
        [
            train.assign(isFraud=y_train.to_numpy(), source_split="orig_train"),
            test.assign(isFraud=y_test.to_numpy(), source_split="orig_test"),
        ],
        ignore_index=True,
    )
    return full, metadata, diagnostics


def build_label_free_user_map(
    transactions: pd.DataFrame,
    csi_uids: Iterable[str],
) -> dict[str, str]:
    """Map Tx identities to CSI identities using split support only."""

    csi_uids = list(map(str, csi_uids))
    frame = transactions.copy()
    frame["uid"] = frame["uid"].astype(str)
    frame["source_split"] = frame["source_split"].astype(str)
    support = (
        frame.groupby(["uid", "source_split"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for split in ("orig_train", "orig_test"):
        if split not in support.columns:
            support[split] = 0
    eligible = support[support["orig_train"].ge(1) & support["orig_test"].ge(1)].copy()
    eligible["min_split_support"] = eligible[["orig_train", "orig_test"]].min(axis=1)
    eligible["total_split_support"] = eligible["orig_train"] + eligible["orig_test"]
    eligible = eligible.sort_values(
        ["min_split_support", "total_split_support", "orig_train", "orig_test", "uid"],
        ascending=[False, False, False, False, True],
    )
    transaction_users = eligible.head(len(csi_uids))["uid"].astype(str).tolist()
    return dict(zip(transaction_users, csi_uids[: len(transaction_users)]))


def build_user_map_legacy(full_df, csi_uids, dataset_label=None):
    del dataset_label
    return build_label_free_user_map(full_df, csi_uids)
