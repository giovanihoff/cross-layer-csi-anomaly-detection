from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pandas as pd

from .models import DatasetProfile, DatasetRunResult
from .utils import top_missing_ratios


def build_split_summary(
    df: pd.DataFrame,
    split_name: str,
    target_col: str,
    stage: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "split_name": split_name,
        "stage": stage,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [str(column) for column in df.columns.tolist()],
        "dtypes": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
        "dtype_counts": dict(Counter(str(dtype) for dtype in df.dtypes.values)),
        "missing_ratio_top10": top_missing_ratios(df),
        "frauds": None,
        "fraud_rate": None,
    }
    if target_col in df.columns:
        target = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
        summary["frauds"] = int(target.sum())
        summary["fraud_rate"] = float(target.mean()) if len(target) else 0.0
    return summary


def build_uid_summary(
    profiles: list[DatasetProfile],
    chosen_uid: str,
    rationale: str,
) -> dict[str, Any]:
    rankings: list[dict[str, Any]] = []
    for profile in profiles:
        candidates = [
            {
                "name": name,
                **stats,
            }
            for name, stats in profile.identifier_candidates.items()
        ]
        candidates.sort(
            key=lambda item: (
                item["duplicate_ratio"],
                -item["nunique"],
                item["max_group_size"],
                -item["non_null"],
            )
        )
        rankings.append(
            {
                "split_name": profile.split_name,
                "candidates": candidates,
            }
        )
    return {
        "chosen_uid": chosen_uid,
        "rationale": rationale,
        "rankings": rankings,
    }


def build_column_transition(
    raw_df: pd.DataFrame,
    prepared_df: pd.DataFrame,
    *,
    raw_split_name: str,
    prepared_split_name: str,
    added_reason: str,
    removed_reason: str,
    kept_reason: str,
    final_reason: str,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    raw_columns = [str(column) for column in raw_df.columns.tolist()]
    prepared_columns = [str(column) for column in prepared_df.columns.tolist()]
    raw_set = set(raw_columns)
    prepared_set = set(prepared_columns)

    removed_columns = [column for column in raw_columns if column not in prepared_set]
    kept_columns = [column for column in raw_columns if column in prepared_set]
    added_columns = [column for column in prepared_columns if column not in raw_set]

    dtype_changes = []
    common_columns = [column for column in prepared_columns if column in raw_set]
    for column in common_columns:
        raw_dtype = str(raw_df[column].dtype)
        prepared_dtype = str(prepared_df[column].dtype)
        if raw_dtype != prepared_dtype:
            dtype_changes.append(
                {
                    "column": column,
                    "from": raw_dtype,
                    "to": prepared_dtype,
                }
            )

    return {
        "raw_split_name": raw_split_name,
        "prepared_split_name": prepared_split_name,
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "kept_columns": kept_columns,
        "final_columns": prepared_columns,
        "dtype_changes": dtype_changes,
        "added_reason": added_reason,
        "removed_reason": removed_reason,
        "kept_reason": kept_reason,
        "final_reason": final_reason,
        "notes": notes or [],
    }


def format_split_summary(summary: dict[str, Any]) -> str:
    parts = [
        f"linhas={summary['rows']}",
        f"colunas={summary['columns']}",
    ]
    if summary.get("frauds") is not None:
        fraud_rate = summary.get("fraud_rate", 0.0) or 0.0
        parts.extend(
            [
                f"fraudes={summary['frauds']}",
                f"fraud_rate={fraud_rate:.6f}",
            ]
        )
    else:
        parts.append("fraudes=n/a")
    return " | ".join(parts)


def humanize_data_access_mode(mode: str | None) -> str:
    if mode == "kaggle_download":
        return "download do Kaggle"
    if mode == "local":
        return "arquivos locais"
    return "nao identificado"


def format_dataset_report(result: DatasetRunResult) -> str:
    raw_summaries = result.metadata.get("raw_summaries", {})
    uid_summary = result.metadata.get("uid_summary", {})
    prepared_summaries = result.prepared.metadata.get("prepared_summaries", {})
    feature_summary = result.prepared.metadata.get("feature_engineering", {})

    lines = [f"# {result.display_name}", ""]

    lines.extend(
        [
            "## Download e carregamento",
            "",
            f"- Origem dos arquivos: {humanize_data_access_mode(result.metadata.get('data_access_mode'))}",
            f"- Arquivos localizados: {', '.join(sorted(result.raw_files.keys()))}",
            "",
        ]
    )
    for split_name, summary in raw_summaries.items():
        lines.append(f"- Split bruto `{split_name}`: {format_split_summary(summary)}")
    lines.append("")

    lines.extend(
        [
            "## Proxy de UID",
            "",
            f"- UID adotado: `{uid_summary.get('chosen_uid', result.research.uid_choice)}`",
            f"- Racional: {uid_summary.get('rationale', result.research.uid_rationale)}",
            "",
        ]
    )
    for ranking in uid_summary.get("rankings", []):
        lines.append(f"### Candidatos avaliados em `{ranking['split_name']}`")
        lines.append("")
        for candidate in ranking["candidates"]:
            lines.append(
                "- "
                f"`{candidate['name']}`: non_null={candidate['non_null']}, "
                f"nunique={candidate['nunique']}, duplicate_ratio={candidate['duplicate_ratio']:.6f}, "
                f"max_group_size={candidate['max_group_size']}"
            )
        lines.append("")

    lines.extend(
        [
            "## Engenharia de atributos",
            "",
        ]
    )
    for split_name, summary in prepared_summaries.items():
        lines.append(f"- Split preparado `{split_name}`: {format_split_summary(summary)}")
    for note in feature_summary.get("notes", []):
        lines.append(f"- Nota: {note}")
    lines.append("")

    lines.extend(
        [
            "==================================================",
            "RELATORIO DETALHADO DE COLUNAS",
            "==================================================",
            "",
        ]
    )
    for profile in result.profiles:
        lines.append(f"### Perfil bruto `{profile.split_name}`")
        lines.append("")
        lines.append(
            f"- Linhas={profile.rows} | Colunas={profile.columns} | Fraudes={profile.frauds} | "
            f"Fraud rate={profile.fraud_rate:.6f}"
        )
        dtype_counts = dict(Counter(profile.dtypes.values()))
        lines.append(f"- Tipos de dados: {json.dumps(dtype_counts, ensure_ascii=False, sort_keys=True)}")
        lines.append("- Missing ratio top 10:")
        for column, ratio in profile.missing_ratio_top10.items():
            lines.append(f"  - {column}: {ratio:.6f}")
        lines.append(f"- Colunas brutas ({len(profile.column_names)}):")
        lines.append("```text")
        lines.append(json.dumps(profile.column_names, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    if feature_summary:
        lines.extend(
            [
                f"[REMOVIDAS] Limpeza/normalizacao ({len(feature_summary.get('removed_columns', []))}):",
                f"Motivo: {feature_summary.get('removed_reason', 'Colunas removidas no preparo.')}",
                "```text",
                json.dumps(feature_summary.get("removed_columns", []), ensure_ascii=False, indent=2),
                "```",
                "",
                f"[ADICIONADAS] Engenharia de atributos ({len(feature_summary.get('added_columns', []))}):",
                f"Motivo: {feature_summary.get('added_reason', 'Colunas geradas no preparo.')}",
                "```text",
                json.dumps(feature_summary.get("added_columns", []), ensure_ascii=False, indent=2),
                "```",
                "",
                f"[MANTIDAS] Colunas brutas preservadas ({len(feature_summary.get('kept_columns', []))}):",
                f"Motivo: {feature_summary.get('kept_reason', 'Colunas brutas preservadas no artefato final.')}",
                "```text",
                json.dumps(feature_summary.get("kept_columns", []), ensure_ascii=False, indent=2),
                "```",
                "",
                f"[FINAIS] Colunas finais do artefato preparado ({len(feature_summary.get('final_columns', []))}):",
                f"Motivo: {feature_summary.get('final_reason', 'Colunas presentes no artefato final.')}",
                "```text",
                json.dumps(feature_summary.get("final_columns", []), ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
        if feature_summary.get("dtype_changes"):
            lines.append(f"- Tipos alterados ({len(feature_summary['dtype_changes'])}):")
            for change in feature_summary["dtype_changes"]:
                lines.append(f"  - `{change['column']}`: {change['from']} -> {change['to']}")
            lines.append("")

    lines.extend(["## Fontes", ""])
    for reference in result.research.references:
        lines.append(f"- [{reference['label']}]({reference['url']})")
    lines.append("")

    return "\n".join(lines)
