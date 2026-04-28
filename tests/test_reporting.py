from __future__ import annotations

from pathlib import Path

import pandas as pd

from cross_layer_csi.tabular.models import (
    DatasetProfile,
    DatasetResearch,
    DatasetRunResult,
    PreparedArtifacts,
)
from cross_layer_csi.tabular.reporting import (
    build_column_transition,
    build_split_summary,
    build_uid_summary,
    format_dataset_report,
)


def test_build_split_summary_tracks_basic_dataset_metrics() -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1, 0],
            "amount": [10.0, 20.0, None],
            "device": ["a", "b", None],
        }
    )

    summary = build_split_summary(df, split_name="train", target_col="target", stage="raw")

    assert summary["rows"] == 3
    assert summary["columns"] == 3
    assert summary["frauds"] == 1
    assert summary["fraud_rate"] == 1 / 3
    assert summary["column_names"] == ["target", "amount", "device"]
    assert "float64" in summary["dtype_counts"]


def test_build_column_transition_separates_added_removed_and_kept_columns() -> None:
    raw_df = pd.DataFrame({"a": [1], "b": [2], "drop_me": [3]})
    prepared_df = pd.DataFrame({"a": [1.0], "b": [2], "new_feature": [5]})

    transition = build_column_transition(
        raw_df,
        prepared_df,
        raw_split_name="train",
        prepared_split_name="train",
        added_reason="Added by feature engineering.",
        removed_reason="Removed by cleanup.",
        kept_reason="Kept from the raw frame.",
        final_reason="Final prepared columns.",
    )

    assert transition["added_columns"] == ["new_feature"]
    assert transition["removed_columns"] == ["drop_me"]
    assert transition["kept_columns"] == ["a", "b"]
    assert transition["final_columns"] == ["a", "b", "new_feature"]
    assert transition["dtype_changes"] == [{"column": "a", "from": "int64", "to": "float64"}]


def test_format_dataset_report_includes_stage_sections() -> None:
    profile = DatasetProfile(
        dataset_key="demo",
        split_name="train",
        rows=3,
        columns=2,
        frauds=1,
        fraud_rate=1 / 3,
        column_names=["target", "amount"],
        dtypes={"target": "int64", "amount": "float64"},
        missing_ratio_top10={"amount": 0.0, "target": 0.0},
        identifier_candidates={
            "event_id": {
                "non_null": 3,
                "nunique": 3,
                "duplicate_rows": 0,
                "duplicate_ratio": 0.0,
                "max_group_size": 1,
            }
        },
        source_files={"train": "train.csv"},
    )
    research = DatasetResearch(
        dataset_key="demo",
        display_name="Demo Dataset",
        uid_choice="uid = customer_id",
        uid_rationale="Unique enough for demos.",
        training_prep="Minimal prep.",
        references=({"label": "Demo", "url": "https://example.com"},),
    )
    result = DatasetRunResult(
        dataset_key="demo",
        display_name="Demo Dataset",
        raw_files={"train": Path("train.csv")},
        profiles=[profile],
        prepared=PreparedArtifacts(
            train_path=Path("train_prepared.parquet"),
            test_path=Path("test_prepared.parquet"),
            metadata={
                "prepared_summaries": {
                    "train": build_split_summary(
                        pd.DataFrame({"target": [0, 1], "uid": ["a", "b"]}),
                        split_name="train",
                        target_col="target",
                        stage="prepared",
                    )
                },
                "feature_engineering": build_column_transition(
                    pd.DataFrame({"target": [0, 1]}),
                    pd.DataFrame({"target": [0, 1], "uid": ["a", "b"]}),
                    raw_split_name="train",
                    prepared_split_name="train",
                    added_reason="Generated uid.",
                    removed_reason="No removals.",
                    kept_reason="Raw columns kept.",
                    final_reason="Prepared output columns.",
                ),
            },
        ),
        research=research,
        metadata={
            "data_access_mode": "local",
            "raw_summaries": {
                "train": build_split_summary(
                    pd.DataFrame({"target": [0, 1], "amount": [10.0, 20.0]}),
                    split_name="train",
                    target_col="target",
                    stage="raw",
                )
            },
            "uid_summary": build_uid_summary([profile], research.uid_choice, research.uid_rationale),
        },
    )

    report = format_dataset_report(result)

    assert "## Download e carregamento" in report
    assert "## Proxy de UID" in report
    assert "## Engenharia de atributos" in report
    assert "RELATORIO DETALHADO DE COLUNAS" in report
    assert "[ADICIONADAS] Engenharia de atributos (1):" in report
