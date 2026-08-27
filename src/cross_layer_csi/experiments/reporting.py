from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

PAIR_METRICS = ("TP_TxOnly", "TP_TxCSI", "FP_TxOnly", "FP_TxCSI")


def build_paired_result_table(
    raw_results: pd.DataFrame,
    *,
    extra_group_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Convert Tx-only/Tx+CSI result rows into one paired comparison row."""

    if raw_results is None or raw_results.empty:
        return pd.DataFrame()
    group_columns = ["Dataset", "S", "Modelo", *extra_group_columns]
    rows: list[dict[str, object]] = []
    for keys, block in raw_results.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        tx_only = block[block["Scenario_Mode"].astype(str).eq("Tx-Only")]
        tx_csi = block[block["Scenario_Mode"].astype(str).eq("Tx+CSI")]
        if tx_only.empty or tx_csi.empty:
            continue
        tx = tx_only.iloc[0]
        fused = tx_csi.iloc[0]
        fp_tx = int(tx["False Alarms (FP)"])
        fp_fused = int(fused["False Alarms (FP)"])
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "TP_TxOnly": int(tx["Detected Frauds (TP)"]),
                "TP_TxCSI": int(fused["Detected Frauds (TP)"]),
                "Delta_TP": int(fused["Detected Frauds (TP)"] - tx["Detected Frauds (TP)"]),
                "FP_TxOnly": fp_tx,
                "FP_TxCSI": fp_fused,
                "Delta_FP": fp_fused - fp_tx,
                "FP_Reduction_pct": 100.0 * (fp_tx - fp_fused) / max(1, fp_tx),
                "PR_AUC_TxOnly": float(tx["PR-AUC"]),
                "PR_AUC_TxCSI": float(fused["PR-AUC"]),
                "Delta_PR_AUC": float(fused["PR-AUC"] - tx["PR-AUC"]),
                "Profile_User_Count": int(fused.get("CSI_Profile_User_Count", 0)),
                "Profile_Missing_Rate_Holdout": float(
                    fused.get("CSI_Profile_Missing_Rate_Holdout", np.nan)
                ),
                "Pareto_Holdout": bool(
                    fused["Detected Frauds (TP)"] >= tx["Detected Frauds (TP)"]
                    and fp_fused <= fp_tx
                ),
            }
        )
    return pd.DataFrame(rows)


def build_random_reproducibility_guard(
    main_pairs: pd.DataFrame,
    donor_policy_pairs: pd.DataFrame,
    *,
    raise_on_fail: bool = True,
) -> pd.DataFrame:
    """Require the L9C random branch to exactly replay main TP/FP counts."""

    key_columns = ["Dataset", "S", "Modelo"]
    missing_main = sorted(set(key_columns + list(PAIR_METRICS)) - set(main_pairs.columns))
    missing_stress = sorted(
        set(key_columns + ["Donor_Policy", *PAIR_METRICS]) - set(donor_policy_pairs.columns)
    )
    if missing_main or missing_stress:
        raise ValueError(f"Missing guard columns: main={missing_main}, donor={missing_stress}")

    main = (
        main_pairs[[*key_columns, *PAIR_METRICS]]
        .drop_duplicates()
        .rename(columns={metric: f"{metric}_Main" for metric in PAIR_METRICS})
    )
    replay = (
        donor_policy_pairs[donor_policy_pairs["Donor_Policy"].astype(str).eq("random")][
            [*key_columns, *PAIR_METRICS]
        ]
        .drop_duplicates()
        .rename(columns={metric: f"{metric}_L9C_Random" for metric in PAIR_METRICS})
    )
    guard = main.merge(replay, on=key_columns, how="outer", validate="one_to_one", indicator=True)
    match_columns = []
    for metric in PAIR_METRICS:
        delta_column = f"{metric}_L9C_minus_Main"
        match_column = f"{metric}_Match"
        guard[delta_column] = pd.to_numeric(
            guard[f"{metric}_L9C_Random"], errors="coerce"
        ) - pd.to_numeric(guard[f"{metric}_Main"], errors="coerce")
        guard[match_column] = guard[delta_column].eq(0)
        match_columns.append(match_column)
    guard["Random_Reproduces_Main"] = guard["_merge"].eq("both") & guard[match_columns].all(axis=1)
    if raise_on_fail and (
        len(guard) != len(main) or not bool(guard["Random_Reproduces_Main"].all())
    ):
        failed = guard[~guard["Random_Reproduces_Main"]]
        raise RuntimeError(
            "Random donor policy did not exactly reproduce the main TP/FP results:\n"
            + failed.to_string(index=False)
        )
    return guard


def summarize_runs_with_ci(
    raw_results: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metric_columns: Iterable[str],
) -> pd.DataFrame:
    """Report mean, sample standard deviation, and 95% t-interval half-width."""

    rows: list[dict[str, object]] = []
    for keys, block in raw_results.groupby(list(group_columns)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, object] = dict(zip(group_columns, keys))
        row["Runs"] = len(block)
        for metric in metric_columns:
            values = pd.to_numeric(block[metric], errors="coerce").dropna().to_numpy(float)
            if not len(values):
                mean = standard_deviation = ci95 = np.nan
            else:
                mean = float(np.mean(values))
                standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                ci95 = (
                    float(
                        stats.t.ppf(0.975, len(values) - 1)
                        * standard_deviation
                        / np.sqrt(len(values))
                    )
                    if len(values) > 1
                    else 0.0
                )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = standard_deviation
            row[f"{metric}_ci95"] = ci95
        rows.append(row)
    return pd.DataFrame(rows)


def build_pair_table_from_results_v118(raw_results, extra_group_cols=None):
    return build_paired_result_table(
        raw_results,
        extra_group_columns=tuple(extra_group_cols or ()),
    )


def summarize_runs_with_ci_v108(raw_results, group_cols, metric_cols):
    return summarize_runs_with_ci(
        raw_results,
        group_columns=tuple(group_cols),
        metric_columns=tuple(metric_cols),
    )
