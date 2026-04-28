from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stringify_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype("string")
    return df


def combine_columns(df: pd.DataFrame, columns: list[str], fill_value: str = "missing") -> pd.Series:
    safe = []
    for column in columns:
        if column in df.columns:
            values = df[column].astype("string").fillna(fill_value)
        else:
            values = pd.Series(fill_value, index=df.index, dtype="string")
        safe.append(values.astype("string"))
    return pd.Series(safe[0], index=df.index).str.cat(safe[1:], sep="_")


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def top_missing_ratios(df: pd.DataFrame, limit: int = 10) -> dict[str, float]:
    ratios = df.isna().mean().sort_values(ascending=False).head(limit)
    return {column: float(value) for column, value in ratios.items()}


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def haversine_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    radius_km = 6371.0
    lat1_rad = np.radians(lat1.astype(float))
    lon1_rad = np.radians(lon1.astype(float))
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(a))
