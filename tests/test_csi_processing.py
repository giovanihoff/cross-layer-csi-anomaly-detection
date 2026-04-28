from __future__ import annotations

import numpy as np
import pandas as pd

from cross_layer_csi.csi.converter import CSIConverter
from cross_layer_csi.csi.filter import CSISubcarrierFilter
from cross_layer_csi.csi.harmonizer import CSIHarmonizer


def test_converter_renames_numeric_columns_and_drops_timestamp(tmp_path) -> None:
    converter = CSIConverter(tmp_path)
    df = pd.DataFrame(
        {
            "timestamp": [1, 2],
            "c0": ["1.0", "2.0"],
            "c1": ["3.0", "4.0"],
        }
    )

    converted = converter._convert_ready_amplitude_df(df)

    assert list(converted.columns) == ["subcarrier_000", "subcarrier_001"]
    assert converted.shape == (2, 2)


def test_filter_drops_null_and_pilot_subcarriers_for_128_grid(tmp_path) -> None:
    manager = CSISubcarrierFilter(tmp_path)
    df = pd.DataFrame(
        {
            f"subcarrier_{idx:03d}": np.arange(4, dtype=np.float32) + idx
            for idx in range(128)
        }
    )

    filtered = manager.filter_128_subcarriers(df)

    assert filtered.shape == (4, 108)
    assert filtered.columns[0] == "subcarrier_000"
    assert filtered.columns[-1] == "subcarrier_107"


def test_harmonizer_resamples_native_matrix_to_target_grid(tmp_path) -> None:
    harmonizer = CSIHarmonizer(tmp_path, target_subcarriers=108)
    native = np.arange(104, dtype=np.float32).reshape(2, 52)

    resampled, native_count, used_resampling = harmonizer._resample_matrix(native)

    assert resampled.shape == (2, 108)
    assert native_count == 52
    assert used_resampling is True
