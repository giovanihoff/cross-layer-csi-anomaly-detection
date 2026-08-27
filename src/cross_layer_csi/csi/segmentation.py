from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from cross_layer_csi.experiments.config import SEGMENT_SIZES, TARGET_HARMONIZED_SUBCARRIERS

PURGE_SEGMENTS = 1


def build_segment_feature_columns(subcarrier_count: int) -> list[str]:
    return [
        f"{statistic}_sc{subcarrier:03d}"
        for statistic in ("mean", "std", "min", "max")
        for subcarrier in range(subcarrier_count)
    ]


def segment_matrix(matrix: np.ndarray, segment_size: int) -> np.ndarray:
    """Return mean/std/min/max CSI features without mixing source files."""

    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D CSI matrix, got shape={values.shape}")
    if values.shape[1] != TARGET_HARMONIZED_SUBCARRIERS:
        raise ValueError(
            f"Expected {TARGET_HARMONIZED_SUBCARRIERS} harmonized subcarriers, "
            f"got {values.shape[1]}"
        )
    if segment_size not in SEGMENT_SIZES:
        raise ValueError(f"Unsupported segment size {segment_size}; expected {SEGMENT_SIZES}")

    segment_count = len(values) // segment_size
    if not segment_count:
        return np.empty((0, values.shape[1] * 4), dtype=np.float32)
    sliced = values[: segment_count * segment_size]
    segments = sliced.reshape(segment_count, segment_size, values.shape[1])
    return np.concatenate(
        [
            segments.mean(axis=1),
            segments.std(axis=1),
            segments.min(axis=1),
            segments.max(axis=1),
        ],
        axis=1,
    ).astype(np.float32)


def build_user_segment_rows(
    record: Mapping[str, object],
    segment_size: int,
) -> pd.DataFrame:
    """Segment each capture file independently and preserve packet provenance."""

    rows: list[pd.DataFrame] = []
    paths = [Path(path) for path in record["paths"]]  # type: ignore[arg-type]
    for path in sorted(paths):
        source = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        amplitude_columns = sorted(
            [column for column in source.columns if str(column).startswith("subcarrier_")],
            key=lambda column: int(str(column).split("_")[1]),
        )
        matrix = (
            source[amplitude_columns]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=np.float32)
        )
        features = segment_matrix(matrix, segment_size)
        if not len(features):
            continue
        part = pd.DataFrame(
            features,
            columns=build_segment_feature_columns(matrix.shape[1]),
        )
        part.insert(0, "uid_csi", str(record["uid_csi"]))
        part["source_file"] = path.name
        part["source_dataset"] = str(record["source_dataset"])
        part["native_subcarriers"] = int(record["native_subcarriers"])
        part["harmonized_subcarriers"] = int(record["harmonized_subcarriers"])
        part["n_files_aggregated"] = len(paths)
        part["_file_segment_idx"] = np.arange(len(part), dtype=int)
        part["_packet_start"] = part["_file_segment_idx"] * segment_size
        part["_packet_end"] = part["_packet_start"] + segment_size - 1
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    out = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["source_file", "_file_segment_idx"])
        .reset_index(drop=True)
    )
    out.insert(1, "capture_id", np.arange(len(out), dtype=int))
    out["group_id"] = (
        out["uid_csi"].astype(str)
        + "|"
        + out["source_file"].astype(str)
        + "|"
        + out["_file_segment_idx"].astype(str)
    )
    return out


def calculate_split_sizes(total: int) -> tuple[int, int, int]:
    if total < 2:
        return 1, 0, 0
    if total <= 3:
        return max(1, total - 1), 0, 1
    test = max(1, round(total * 0.182))
    validation = max(1, round(total * 0.091))
    if test + validation >= total:
        test, validation = 1, 0
    train = total - test - validation
    if train < 1:
        train = max(1, total - test)
        validation = max(0, total - train - test)
    return int(train), int(validation), int(test)


def choose_holdout_files(
    file_counts: Mapping[str, int],
    validation_target: int,
    test_target: int,
) -> tuple[str | None, str | None]:
    items = [(str(name), int(count)) for name, count in file_counts.items() if int(count) > 0]
    if len(items) < 2:
        return None, None
    if len(items) == 2:
        test_file = min(
            items,
            key=lambda item: (abs(item[1] - test_target), item[1], item[0]),
        )[0]
        return None, test_file

    best: tuple[float, str, str] | None = None
    for validation_file, validation_count in items:
        for test_file, test_count in items:
            if validation_file == test_file:
                continue
            candidate = (
                abs(validation_count - validation_target)
                + abs(test_count - test_target)
                + 0.01 * (validation_count + test_count),
                validation_file,
                test_file,
            )
            if best is None or candidate < best:
                best = candidate
    return (best[1], best[2]) if best else (None, None)


def assign_multi_file_split(user_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    out = user_rows.copy()
    file_counts = out.groupby("source_file")["capture_id"].count().sort_values().to_dict()
    _, validation_target, test_target = calculate_split_sizes(len(out))
    validation_file, test_file = choose_holdout_files(
        file_counts,
        validation_target,
        test_target,
    )
    train_files = [name for name in file_counts if name not in {validation_file, test_file}]
    if not train_files:
        largest = max(file_counts.items(), key=lambda item: item[1])[0]
        train_files = [largest]
        remaining = [name for name in file_counts if name != largest]
        test_file = remaining[0] if remaining else None
        validation_file = remaining[1] if len(remaining) > 1 else None

    out["split"] = "train"
    if validation_file is not None:
        out.loc[out["source_file"].eq(validation_file), "split"] = "val"
    if test_file is not None:
        out.loc[out["source_file"].eq(test_file), "split"] = "test"
    return out, {
        "strategy": "file_holdout",
        "n_files": len(file_counts),
        "purged": 0,
        "train_files": len(train_files),
        "val_files": int(validation_file is not None),
        "test_files": int(test_file is not None),
    }


def assign_single_file_split(
    user_rows: pd.DataFrame,
    *,
    purge_segments: int = PURGE_SEGMENTS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    out = user_rows.sort_values(["source_file", "_file_segment_idx"]).reset_index(drop=True)
    total = len(out)
    out["split"] = "train"
    if total <= 1:
        return out, {"strategy": "single_file_all_train", "n_files": 1, "purged": 0}
    if total <= 3:
        out.loc[out.index[-1], "split"] = "test"
        return out, {"strategy": "single_file_tail_test_no_purge", "n_files": 1, "purged": 0}

    _, validation_target, test_target = calculate_split_sizes(total)
    validation = max(1, validation_target) if total >= 8 and validation_target > 0 else 0
    test = max(1, test_target)
    purge_count = purge_segments + (purge_segments if validation else 0)
    while total - test - validation - purge_count < 1:
        if validation:
            validation -= 1
            purge_count = purge_segments + (purge_segments if validation else 0)
        elif test > 1:
            test -= 1
            purge_count = purge_segments
        else:
            break
    train = total - test - validation - purge_count
    if train < 1:
        train = max(1, total - test - validation)

    cursor = train
    first_purge = list(range(cursor, min(cursor + purge_segments, total)))
    cursor += len(first_purge)
    validation_indices: list[int] = []
    second_purge: list[int] = []
    if validation:
        validation_indices = list(range(cursor, min(cursor + validation, total)))
        cursor += len(validation_indices)
        second_purge = list(range(cursor, min(cursor + purge_segments, total)))
        cursor += len(second_purge)
    test_indices = list(range(cursor, min(cursor + test, total)))

    out.loc[first_purge, "split"] = "purged"
    out.loc[validation_indices, "split"] = "val"
    out.loc[second_purge, "split"] = "purged"
    out.loc[test_indices, "split"] = "test"
    return out, {
        "strategy": "single_file_contiguous_purged",
        "n_files": 1,
        "purged": int(out["split"].eq("purged").sum()),
    }


def apply_anti_leak_split(
    frame: pd.DataFrame,
    *,
    purge_segments: int = PURGE_SEGMENTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use file holdout when possible; otherwise use a purged contiguous tail."""

    parts: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for uid, user_rows in frame.groupby("uid_csi", sort=True):
        file_count = user_rows["source_file"].nunique()
        if file_count >= 2:
            assigned, info = assign_multi_file_split(user_rows)
        else:
            assigned, info = assign_single_file_split(
                user_rows,
                purge_segments=purge_segments,
            )
        summaries.append(
            {
                "uid_csi": uid,
                "total_caps": len(assigned),
                "train_caps": int(assigned["split"].eq("train").sum()),
                "val_caps": int(assigned["split"].eq("val").sum()),
                "test_caps": int(assigned["split"].eq("test").sum()),
                "purged_caps": int(assigned["split"].eq("purged").sum()),
                "n_files": file_count,
                "strategy": info["strategy"],
            }
        )
        parts.append(assigned)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(summaries)


def build_segmented_datasets(
    registry: Iterable[Mapping[str, object]],
    *,
    segment_sizes: Iterable[int] = SEGMENT_SIZES,
) -> dict[int, pd.DataFrame]:
    """Build and split all canonical CSI segment sizes."""

    datasets: dict[int, pd.DataFrame] = {}
    for segment_size in segment_sizes:
        parts = [build_user_segment_rows(record, segment_size) for record in registry]
        parts = [part for part in parts if not part.empty]
        if not parts:
            raise RuntimeError(f"No valid CSI segmentation for S={segment_size}")
        segmented = pd.concat(parts, ignore_index=True)
        split, _ = apply_anti_leak_split(segmented)
        usable = split[split["split"].isin(["train", "val", "test"])].copy()
        feature_columns = [
            column
            for column in usable.columns
            if str(column).startswith(("mean_sc", "std_sc", "min_sc", "max_sc"))
        ]
        metadata = [
            "uid_csi",
            "capture_id",
            "group_id",
            "source_file",
            "source_dataset",
            "native_subcarriers",
            "harmonized_subcarriers",
            "n_files_aggregated",
            "split",
        ]
        datasets[int(segment_size)] = usable[[*metadata, *feature_columns]].reset_index(drop=True)
    return datasets
