from __future__ import annotations

from typing import Callable

import pandas as pd

from .models import DatasetProfile
from .utils import top_missing_ratios


def _candidate_stats(series: pd.Series) -> dict[str, float | int]:
    non_null = series.dropna()
    nunique = int(non_null.nunique(dropna=True))
    rows = int(series.shape[0])
    duplicate_rows = int(non_null.duplicated().sum())
    max_group = int(non_null.value_counts(dropna=True).max()) if not non_null.empty else 0
    return {
        "non_null": int(non_null.shape[0]),
        "nunique": nunique,
        "duplicate_rows": duplicate_rows,
        "duplicate_ratio": float(duplicate_rows / rows) if rows else 0.0,
        "max_group_size": max_group,
    }


def build_profile(
    dataset_key: str,
    split_name: str,
    df: pd.DataFrame,
    target_col: str,
    source_files: dict[str, str],
    identifier_factory: Callable[[pd.DataFrame], dict[str, pd.Series]],
) -> DatasetProfile:
    target = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
    identifier_candidates = {
        name: _candidate_stats(candidate)
        for name, candidate in identifier_factory(df).items()
    }

    return DatasetProfile(
        dataset_key=dataset_key,
        split_name=split_name,
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        frauds=int(target.sum()),
        fraud_rate=float(target.mean()),
        column_names=[str(column) for column in df.columns.tolist()],
        dtypes={str(column): str(dtype) for column, dtype in df.dtypes.items()},
        missing_ratio_top10=top_missing_ratios(df),
        identifier_candidates=identifier_candidates,
        source_files=source_files,
    )
