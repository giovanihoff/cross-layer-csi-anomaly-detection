from __future__ import annotations

import numpy as np
import pandas as pd

from .fusion import get_raw_csi_feature_columns

CSI_CONSISTENCY_PREFIXES = (
    "csi_claimed_",
    "csi_nearest_",
    "csi_second_",
    "csi_margin_",
    "csi_profile_",
)
CSI_PROFILE_MIN_SAMPLES = 1


def get_csi_consistency_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if str(column).startswith(CSI_CONSISTENCY_PREFIXES)]


def get_claimed_uid_column(frame: pd.DataFrame) -> str | None:
    if "uid_csi_mapped" in frame.columns:
        return "uid_csi_mapped"
    if "uid_csi" in frame.columns:
        return "uid_csi"
    return None


def fit_claimed_user_csi_profiles(
    reference: pd.DataFrame,
    *,
    min_profile_samples: int = CSI_PROFILE_MIN_SAMPLES,
    eps: float = 1e-6,
) -> dict[str, object] | None:
    """Fit user CSI profiles from nominal train/validation rows only."""

    if reference is None or reference.empty:
        return None
    uid_column = get_claimed_uid_column(reference)
    csi_columns = get_raw_csi_feature_columns(reference)
    if uid_column is None or not csi_columns:
        return None

    ref = reference.copy()
    if "split_role" in ref.columns and ref["split_role"].astype(str).eq("test").any():
        raise RuntimeError("CSI profile fitting received held-out test rows")
    if "attack_type" in ref.columns and ref["attack_type"].astype(str).eq("impostor").any():
        raise RuntimeError("CSI profile fitting received injected impostor rows")
    if "isFraud" in ref.columns:
        ref = ref[ref["isFraud"].astype(int).eq(0)].copy()
    if ref.empty:
        return None

    features = (
        ref[csi_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    )
    global_median = features.median(axis=0).fillna(0.0)
    features = features.fillna(global_median)
    global_std = features.std(axis=0).replace(0, np.nan).fillna(1.0)
    standardized = (features - global_median) / (global_std + eps)

    profiled = pd.concat(
        [ref[[uid_column]].astype(str).reset_index(drop=True), standardized.reset_index(drop=True)],
        axis=1,
    )
    grouped = profiled.groupby(uid_column, sort=True)
    counts = grouped.size().astype(int)
    valid_users = counts[counts >= int(min_profile_samples)].index.tolist()
    if not valid_users:
        return None

    centroids = grouped[csi_columns].mean().loc[valid_users]
    diagonal_std = grouped[csi_columns].std().loc[valid_users].replace(0, np.nan).fillna(1.0)
    return {
        "uid_col": uid_column,
        "csi_cols": csi_columns,
        "global_median": global_median,
        "global_std": global_std,
        "centroids": centroids,
        "diag_std": diagonal_std,
        "counts": counts.loc[valid_users],
        "eps": float(eps),
        "min_profile_samples": int(min_profile_samples),
        "reference_rows_nominal": len(ref),
        "profile_users": len(valid_users),
    }


def apply_claimed_user_csi_consistency(
    frame: pd.DataFrame,
    profile_bundle: dict[str, object] | None,
) -> pd.DataFrame:
    """Attach the 13 v120.3 claimed-user physical-coherence features."""

    out = frame.copy()
    stale_columns = get_csi_consistency_feature_columns(out)
    if stale_columns:
        out = out.drop(columns=stale_columns)
    if profile_bundle is None or out.empty:
        return out

    uid_column = str(profile_bundle["uid_col"])
    csi_columns = [
        column
        for column in profile_bundle["csi_cols"]
        if column in out.columns  # type: ignore[union-attr]
    ]
    if uid_column not in out.columns or not csi_columns:
        return out

    eps = float(profile_bundle.get("eps", 1e-6))
    median = profile_bundle["global_median"].reindex(csi_columns).fillna(0.0)  # type: ignore[union-attr]
    std = (
        profile_bundle["global_std"]  # type: ignore[union-attr]
        .reindex(csi_columns)
        .replace(0, np.nan)
        .fillna(1.0)
    )
    features = (
        out[csi_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(median)
    )
    standardized = ((features - median) / (std + eps)).to_numpy(dtype=np.float64)

    centroids_df = (
        profile_bundle["centroids"].reindex(columns=csi_columns).fillna(0.0)  # type: ignore[union-attr]
    )
    diagonal_std_df = (
        profile_bundle["diag_std"]  # type: ignore[union-attr]
        .reindex(index=centroids_df.index, columns=csi_columns)
        .replace(0, np.nan)
        .fillna(1.0)
    )
    users = centroids_df.index.astype(str).to_numpy()
    centroids = centroids_df.to_numpy(dtype=np.float64)
    diagonal_std = diagonal_std_df.to_numpy(dtype=np.float64)
    counts = (
        profile_bundle["counts"].reindex(centroids_df.index).fillna(0).to_numpy(dtype=np.float64)  # type: ignore[union-attr]
    )

    row_count = standardized.shape[0]
    profile_count = centroids.shape[0]
    claimed_l2 = np.full(row_count, np.nan, dtype=np.float64)
    claimed_l1 = np.full(row_count, np.nan, dtype=np.float64)
    claimed_cosine = np.full(row_count, np.nan, dtype=np.float64)
    claimed_mahalanobis = np.full(row_count, np.nan, dtype=np.float64)
    claimed_rank = np.full(row_count, np.nan, dtype=np.float64)
    claimed_is_top1 = np.zeros(row_count, dtype=np.float64)
    claimed_is_top3 = np.zeros(row_count, dtype=np.float64)
    claimed_count = np.zeros(row_count, dtype=np.float64)
    missing_profile = np.ones(row_count, dtype=np.float64)
    nearest_l2 = np.full(row_count, np.nan, dtype=np.float64)
    second_l2 = np.full(row_count, np.nan, dtype=np.float64)
    margin_to_nearest = np.full(row_count, np.nan, dtype=np.float64)
    ratio_to_nearest = np.full(row_count, np.nan, dtype=np.float64)

    if profile_count:
        differences = standardized[:, None, :] - centroids[None, :, :]
        distances = np.sqrt(np.mean(differences**2, axis=2))
        sorted_distances = np.sort(distances, axis=1)
        nearest_l2 = sorted_distances[:, 0]
        second_l2 = sorted_distances[:, 1] if profile_count > 1 else sorted_distances[:, 0]

        claimed_users = out[uid_column].astype(str).to_numpy()
        user_positions = {user: position for position, user in enumerate(users)}
        row_norms = np.linalg.norm(standardized, axis=1) + eps
        centroid_norms = np.linalg.norm(centroids, axis=1) + eps
        for row_index, claimed_uid in enumerate(claimed_users):
            position = user_positions.get(str(claimed_uid))
            if position is None:
                continue
            missing_profile[row_index] = 0.0
            delta = standardized[row_index] - centroids[position]
            claimed_l2[row_index] = float(distances[row_index, position])
            claimed_l1[row_index] = float(np.mean(np.abs(delta)))
            cosine_similarity = float(
                np.dot(standardized[row_index], centroids[position])
                / (row_norms[row_index] * centroid_norms[position])
            )
            claimed_cosine[row_index] = 1.0 - np.clip(cosine_similarity, -1.0, 1.0)
            claimed_mahalanobis[row_index] = float(
                np.sqrt(np.mean((delta / (diagonal_std[position] + eps)) ** 2))
            )
            claimed_rank[row_index] = float(
                1 + np.sum(distances[row_index] < distances[row_index, position])
            )
            claimed_is_top1[row_index] = float(claimed_rank[row_index] <= 1)
            claimed_is_top3[row_index] = float(claimed_rank[row_index] <= min(3, profile_count))
            claimed_count[row_index] = float(counts[position])
            margin_to_nearest[row_index] = claimed_l2[row_index] - nearest_l2[row_index]
            ratio_to_nearest[row_index] = claimed_l2[row_index] / (nearest_l2[row_index] + eps)

    feature_values = {
        "csi_claimed_l2_z": claimed_l2,
        "csi_claimed_l1_z": claimed_l1,
        "csi_claimed_cosine_distance": claimed_cosine,
        "csi_claimed_mahalanobis_diag": claimed_mahalanobis,
        "csi_claimed_rank": claimed_rank,
        "csi_claimed_is_top1": claimed_is_top1,
        "csi_claimed_is_top3": claimed_is_top3,
        "csi_profile_missing": missing_profile,
        "csi_profile_count_log1p": np.log1p(np.maximum(claimed_count, 0.0)),
        "csi_nearest_l2_z": nearest_l2,
        "csi_second_nearest_l2_z": second_l2,
        "csi_margin_claimed_minus_nearest": margin_to_nearest,
        "csi_margin_claimed_ratio_nearest": ratio_to_nearest,
    }
    for name, values in feature_values.items():
        array = np.asarray(values, dtype=np.float64)
        array[~np.isfinite(array)] = np.nan
        out[name] = array.astype(np.float32)
    return out


def attach_claimed_user_csi_consistency(
    train: pd.DataFrame,
    *frames: pd.DataFrame,
    min_profile_samples: int = CSI_PROFILE_MIN_SAMPLES,
) -> tuple[object, ...]:
    """Fit on nominal training rows, then apply the same bundle to every split."""

    nominal_reference = (
        train[train["isFraud"].astype(int).eq(0)].copy()
        if "isFraud" in train.columns
        else train.copy()
    )
    bundle = fit_claimed_user_csi_profiles(
        nominal_reference,
        min_profile_samples=min_profile_samples,
    )
    train_out = apply_claimed_user_csi_consistency(train, bundle)
    other_outputs = [apply_claimed_user_csi_consistency(frame, bundle) for frame in frames]
    return (train_out, *other_outputs, bundle)


# Notebook-compatible wrappers retained for result traceability.
get_csi_consistency_feature_cols = get_csi_consistency_feature_columns


def fit_claimed_user_csi_profiles_v116(reference_df, min_profile_samples=1, eps=1e-6):
    return fit_claimed_user_csi_profiles(
        reference_df,
        min_profile_samples=min_profile_samples,
        eps=eps,
    )


apply_claimed_user_csi_consistency_v116 = apply_claimed_user_csi_consistency


def attach_claimed_user_csi_consistency_v116(
    train_df,
    *frames,
    min_profile_samples=CSI_PROFILE_MIN_SAMPLES,
):
    return attach_claimed_user_csi_consistency(
        train_df,
        *frames,
        min_profile_samples=min_profile_samples,
    )
