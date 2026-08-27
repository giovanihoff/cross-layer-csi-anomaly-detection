from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
import pandas as pd

CSI_FEATURE_PREFIXES = ("mean_", "std_", "min_", "max_")
CSI_METADATA_COLUMNS = {
    "uid_csi",
    "split",
    "group_id",
    "capture_id",
    "source_file",
    "source_dataset",
    "native_subcarriers",
    "harmonized_subcarriers",
    "n_files_aggregated",
}


def get_raw_csi_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if str(column).startswith(CSI_FEATURE_PREFIXES)]


def fractional_fraud_budget(
    target_rows: int,
    target_fraud_rate: float,
    rng: np.random.Generator,
) -> int:
    """Round an expected fraud count without systematically rounding down."""

    if target_rows <= 0 or target_fraud_rate is None:
        return 0
    expected = float(target_rows) * float(target_fraud_rate)
    base = int(np.floor(expected))
    if rng.random() < expected - base:
        base += 1
    return int(min(max(base, 0), target_rows))


def sample_transactions_for_bucket(
    frame: pd.DataFrame,
    target_rows: int,
    *,
    seed: int = 42,
    fixed_fraud_count: int | None = None,
    target_fraud_rate: float = 0.05,
) -> pd.DataFrame:
    """Sample one split-local transaction per CSI segment.

    Replacement is allowed only when a class-specific pool lacks capacity and
    is recorded row by row in ``_tx_sampling_replacement_used``.
    """

    replacement_column = "_tx_sampling_replacement_used"
    if target_rows == 0:
        return pd.DataFrame(columns=[*frame.columns, replacement_column])
    if "isFraud" not in frame.columns:
        raise ValueError("sample_transactions_for_bucket requires an isFraud column")

    frame = frame.copy()
    frauds = frame[frame["isFraud"].astype(int).eq(1)]
    legits = frame[frame["isFraud"].astype(int).eq(0)]
    rng = np.random.default_rng(seed)

    fraud_count = (
        int(fixed_fraud_count)
        if fixed_fraud_count is not None
        else fractional_fraud_budget(target_rows, target_fraud_rate, rng)
    )
    if frauds.empty:
        fraud_count = 0
    fraud_count = min(fraud_count, target_rows)
    legit_count = target_rows - fraud_count

    def draw(pool: pd.DataFrame, count: int, *, replace: bool) -> pd.DataFrame:
        if count <= 0 or pool.empty:
            return pd.DataFrame(columns=[*frame.columns, replacement_column])
        sampled = pool.sample(
            n=count,
            replace=replace,
            random_state=int(rng.integers(0, 1_000_000_000)),
        ).copy()
        sampled[replacement_column] = bool(replace)
        return sampled

    parts: list[pd.DataFrame] = []
    if fraud_count:
        parts.append(draw(frauds, fraud_count, replace=len(frauds) < fraud_count))
    if legit_count and not legits.empty:
        parts.append(draw(legits, legit_count, replace=len(legits) < legit_count))
    elif legit_count and not frauds.empty:
        parts.append(draw(frauds, legit_count, replace=True))

    parts = [part for part in parts if not part.empty]
    if not parts:
        return pd.DataFrame(columns=[*frame.columns, replacement_column])

    result = pd.concat(parts, ignore_index=True)
    if len(result) < target_rows and not frame.empty:
        result = pd.concat(
            [result, draw(frame, target_rows - len(result), replace=True)],
            ignore_index=True,
        )

    result = (
        result.sample(
            frac=1.0,
            random_state=int(rng.integers(0, 1_000_000_000)),
        )
        .head(target_rows)
        .reset_index(drop=True)
    )
    result[replacement_column] = result[replacement_column].fillna(False).astype(bool)
    return result


def align_transactions_and_csi(
    transactions: pd.DataFrame,
    csi_segments: pd.DataFrame,
    user_map: Mapping[str, str],
    *,
    seed: int = 42,
    target_fraud_rate: float = 0.05,
) -> pd.DataFrame:
    """Build split-coherent transaction/CSI pairs for the controlled protocol.

    CSI ``train`` and ``val`` segments can receive only ``orig_train``
    transactions; CSI ``test`` segments can receive only ``orig_test`` rows.
    ``tx_row_id`` is retained so subsequent grouped splits and audits can prove
    that no transaction crosses the train/test boundary.
    """

    required_tx = {"uid", "source_split", "isFraud"}
    required_csi = {"uid_csi", "split"}
    missing_tx = sorted(required_tx - set(transactions.columns))
    missing_csi = sorted(required_csi - set(csi_segments.columns))
    if missing_tx or missing_csi:
        raise ValueError(f"Missing columns: transactions={missing_tx}, csi={missing_csi}")

    mapped_users = {str(key): str(value) for key, value in user_map.items()}
    tx = transactions[transactions["uid"].astype(str).isin(mapped_users)].copy()
    tx["uid"] = tx["uid"].astype(str)
    tx["uid_csi_mapped"] = tx["uid"].map(mapped_users).astype(str)
    tx["source_split"] = tx["source_split"].astype(str)
    if "tx_row_id" not in tx.columns:
        tx["tx_row_id"] = tx["source_split"] + "::" + tx.index.astype(str)

    csi_feature_columns = [
        column for column in csi_segments.columns if column not in CSI_METADATA_COLUMNS
    ]
    mapped_csi_users = set(mapped_users.values())
    parts: list[pd.DataFrame] = []

    for csi_uid in sorted(csi_segments["uid_csi"].astype(str).unique()):
        if csi_uid not in mapped_csi_users:
            continue
        user_csi = csi_segments[csi_segments["uid_csi"].astype(str).eq(csi_uid)].copy()
        user_tx = tx[tx["uid_csi_mapped"].eq(csi_uid)].copy()
        if user_csi.empty or user_tx.empty:
            continue

        for split_role, split_csi in user_csi.groupby("split", sort=True):
            split_role = str(split_role)
            source_split = "orig_test" if split_role == "test" else "orig_train"
            tx_pool = user_tx[user_tx["source_split"].eq(source_split)].copy()
            if tx_pool.empty:
                continue

            digits = re.sub(r"[^0-9]", "", csi_uid)
            split_offset = {"train": 11, "val": 17, "test": 23}.get(split_role, 31)
            user_seed = int(seed + int((digits or "0") + "99") + split_offset)
            sampled_tx = sample_transactions_for_bucket(
                tx_pool,
                len(split_csi),
                seed=user_seed,
                target_fraud_rate=target_fraud_rate,
            )
            if len(sampled_tx) != len(split_csi):
                continue

            sampled_tx = sampled_tx.reset_index(drop=True)
            split_csi = split_csi.reset_index(drop=True)
            csi_for_merge = split_csi[[*csi_feature_columns, "split"]].rename(
                columns={"split": "split_role"}
            )
            merged = pd.concat([sampled_tx, csi_for_merge], axis=1)
            merged["uid_csi"] = csi_uid
            merged["split_source_expected"] = source_split
            merged["split_coherent"] = merged["source_split"].eq(source_split)
            parts.append(merged)

    if not parts:
        return pd.DataFrame()

    merged = pd.concat(parts, ignore_index=True)
    potential_columns = [
        *transactions.columns,
        "tx_row_id",
        "_tx_sampling_replacement_used",
        *csi_feature_columns,
        "uid_csi_mapped",
        "uid_csi",
        "split_role",
        "split_source_expected",
        "split_coherent",
    ]
    keep = list(dict.fromkeys(column for column in potential_columns if column in merged.columns))
    return merged[keep].copy()


def inject_csi_impostors(
    merged: pd.DataFrame,
    *,
    seed: int = 42,
    require_different_user: bool = True,
    donor_policy: str = "random",
) -> pd.DataFrame:
    """Replace test-fraud CSI with unique real CSI from held-out legit users.

    ``random`` is the primary v120.3 condition. ``hard_nearest`` is an
    exploratory stress test and must not be interpreted as a causal comparison
    between identical cohorts because donor reservation can change eligibility.
    """

    if merged.empty:
        return merged.copy()
    if donor_policy not in {"random", "hard_nearest"}:
        raise ValueError(f"Unsupported donor_policy={donor_policy!r}")

    out = merged.copy()
    csi_columns = get_raw_csi_feature_columns(out)
    out["attack_type"] = "none"
    out["csi_impostor_donor_index"] = -1
    out["csi_impostor_donor_uid"] = None
    out["csi_impostor_donor_split"] = None
    out["csi_impostor_same_user"] = False
    out["csi_impostor_donor_policy"] = "none"
    out["csi_is_impostor_donor"] = False
    out["csi_donor_for_target_index"] = -1
    if not csi_columns:
        return out

    rng = np.random.default_rng(seed)
    test = out["split_role"].astype(str).eq("test")
    target_indices = out.index[test & out["isFraud"].astype(int).eq(1)].to_numpy()
    donor_indices = out.index[test & out["isFraud"].astype(int).eq(0)].to_numpy()
    if not len(target_indices) or not len(donor_indices):
        return out

    rng.shuffle(target_indices)
    rng.shuffle(donor_indices)
    uid_column = (
        "uid_csi_mapped"
        if "uid_csi_mapped" in out.columns
        else "uid_csi"
        if "uid_csi" in out.columns
        else None
    )
    available_donors = list(donor_indices)
    donor_scale = (
        out.loc[donor_indices, csi_columns]
        .apply(pd.to_numeric, errors="coerce")
        .std(axis=0)
        .replace(0, np.nan)
        .fillna(1.0)
        .to_numpy(dtype=np.float64)
    )

    for target_index in target_indices:
        if not available_donors:
            break
        candidate_positions = list(range(len(available_donors)))
        if uid_column is not None and require_different_user:
            target_uid = str(out.at[target_index, uid_column])
            candidate_positions = [
                position
                for position, donor_index in enumerate(available_donors)
                if str(out.at[donor_index, uid_column]) != target_uid
            ]
        if not candidate_positions:
            continue

        if donor_policy == "random":
            selected_position = candidate_positions[0]
        else:
            candidate_ids = [available_donors[position] for position in candidate_positions]
            target_values = out.loc[target_index, csi_columns].to_numpy(dtype=np.float64)
            candidate_values = out.loc[candidate_ids, csi_columns].to_numpy(dtype=np.float64)
            distances = np.nanmean(((candidate_values - target_values) / donor_scale) ** 2, axis=1)
            selected_position = candidate_positions[int(np.nanargmin(distances))]

        donor_index = available_donors.pop(selected_position)
        out.loc[target_index, csi_columns] = out.loc[donor_index, csi_columns].to_numpy(
            dtype=np.float32
        )
        out.loc[target_index, "attack_type"] = "impostor"
        out.loc[target_index, "csi_impostor_donor_index"] = int(donor_index)
        out.loc[target_index, "csi_impostor_donor_uid"] = (
            str(out.at[donor_index, uid_column]) if uid_column is not None else None
        )
        out.loc[target_index, "csi_impostor_donor_split"] = str(out.at[donor_index, "split_role"])
        out.loc[target_index, "csi_impostor_donor_policy"] = donor_policy
        out.loc[donor_index, "csi_is_impostor_donor"] = True
        out.loc[donor_index, "csi_donor_for_target_index"] = int(target_index)
        if uid_column is not None:
            out.loc[target_index, "csi_impostor_same_user"] = str(
                out.at[target_index, uid_column]
            ) == str(out.at[donor_index, uid_column])

    return out


def subset_impostor_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the impostor-only evaluation cohort, excluding reserved donors."""

    reserved = frame.get("csi_is_impostor_donor", pd.Series(False, index=frame.index)).astype(bool)
    if "attack_type" not in frame.columns:
        return frame.loc[~reserved].copy()
    cohort = frame["isFraud"].astype(int).eq(0) | frame["attack_type"].astype(str).eq("impostor")
    return frame.loc[~reserved & cohort].copy()


# Notebook-compatible wrappers retained for traceability.
def align_and_merge_variable(
    df_tx_full,
    df_csi,
    user_map_dict,
    seed=42,
    target_fraud_rate=0.05,
):
    return align_transactions_and_csi(
        df_tx_full,
        df_csi,
        user_map_dict,
        seed=seed,
        target_fraud_rate=target_fraud_rate,
    )


def inject_csi_impostors_variable(
    df_merged,
    seed=42,
    require_different_user=True,
    donor_policy="random",
):
    return inject_csi_impostors(
        df_merged,
        seed=seed,
        require_different_user=require_different_user,
        donor_policy=donor_policy,
    )


def subset_by_attack_view_v108(frame, attack_view="impostor"):
    if attack_view != "impostor":
        raise ValueError(f"Unsupported attack_view={attack_view!r}; v120.3 is impostor-only")
    return subset_impostor_cohort(frame)
