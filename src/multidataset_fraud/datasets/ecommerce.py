from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import KaggleReference, PreparedArtifacts, SourceFileSpec
from ..reporting import build_column_transition, build_split_summary
from ..utils import combine_columns, stringify_columns
from .base import BaseDatasetHandler


class EcommerceDataset(BaseDatasetHandler):
    dataset_key = "ecommerce"
    display_name = "Fraud E-commerce"
    target_col = "class"
    kaggle_reference = KaggleReference(kind="dataset", ref="vbinh002/fraud-ecommerce")
    local_env_var = "ECOMMERCE_LOCAL_DIR"
    source_files = (
        SourceFileSpec("fraud_data", ("Fraud_Data.csv", "fraud_data.csv")),
        SourceFileSpec(
            "ip_country",
            ("IpAddress_to_Country.csv", "ipaddress_to_country.csv"),
            required=False,
        ),
    )

    def load_raw_splits(self, raw_files: dict[str, Path]) -> dict[str, pd.DataFrame]:
        self.log("Lendo `Fraud_Data.csv`...")
        return {"full": pd.read_csv(raw_files["fraud_data"], low_memory=False)}

    def identifier_candidates(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        return {
            "user_id": df["user_id"].astype("string"),
            "device_id": df["device_id"].astype("string"),
            "ip_address": pd.to_numeric(df["ip_address"], errors="coerce").round().astype("Int64").astype("string"),
            "user_device": combine_columns(df, ["user_id", "device_id"]),
        }

    @staticmethod
    def _event_id(df: pd.DataFrame) -> pd.Series:
        hash_columns = ["user_id", "device_id", "purchase_time", "purchase_value", "ip_address"]
        present = [column for column in hash_columns if column in df.columns]
        hashed = pd.util.hash_pandas_object(df[present].fillna("missing"), index=False).astype("uint64")
        return "fraudecom_" + hashed.astype("string")

    @staticmethod
    def _attach_ip_country(df: pd.DataFrame, ip_country_df: pd.DataFrame | None) -> pd.DataFrame:
        if ip_country_df is None:
            return df

        ip_map = ip_country_df.copy()
        ip_map.columns = [column.strip() for column in ip_map.columns]
        lower_col = "lower_bound_ip_address"
        upper_col = "upper_bound_ip_address"
        country_col = "country"
        if not {lower_col, upper_col, country_col}.issubset(ip_map.columns):
            return df

        ip_map[lower_col] = pd.to_numeric(ip_map[lower_col], errors="coerce")
        ip_map[upper_col] = pd.to_numeric(ip_map[upper_col], errors="coerce")
        ip_map = ip_map.sort_values(lower_col).reset_index(drop=True)

        enriched = df.copy()
        enriched["_original_order"] = np.arange(len(enriched))
        enriched["ip_address_int"] = pd.to_numeric(enriched["ip_address"], errors="coerce")
        merged = pd.merge_asof(
            enriched.sort_values("ip_address_int"),
            ip_map[[lower_col, upper_col, country_col]],
            left_on="ip_address_int",
            right_on=lower_col,
            direction="backward",
        )
        valid = merged["ip_address_int"] <= merged[upper_col]
        merged["country"] = merged[country_col].where(valid, "unknown").fillna("unknown")
        merged = merged.sort_values("_original_order").drop(columns=["_original_order"])
        return merged

    def _prepare_full_frame(self, df: pd.DataFrame, ip_country_df: pd.DataFrame | None) -> pd.DataFrame:
        out = df.copy()
        out["event_id"] = self._event_id(out)
        out["uid"] = out["device_id"].astype("string")
        out["account_id"] = out["user_id"].astype("string")
        out["ip_address_int"] = pd.to_numeric(out["ip_address"], errors="coerce").round()

        purchase_time = pd.to_datetime(out["purchase_time"], errors="coerce")
        signup_time = pd.to_datetime(out["signup_time"], errors="coerce")
        out["event_timestamp"] = purchase_time
        out["time_since_signup_sec"] = (purchase_time - signup_time).dt.total_seconds()
        out["event_hour"] = purchase_time.dt.hour.astype("Int64")
        out["event_dayofweek"] = purchase_time.dt.dayofweek.astype("Int64")
        out["event_month"] = purchase_time.dt.month.astype("Int64")
        out["purchase_value_log1p"] = np.log1p(pd.to_numeric(out["purchase_value"], errors="coerce"))

        out = self._attach_ip_country(out, ip_country_df)
        out = self.add_group_stats(out, "uid", "purchase_value", "device_purchase_value")
        out = self.add_group_stats(out, "account_id", "purchase_value", "user_purchase_value")

        ip_group = out["ip_address_int"].round().astype("Int64").astype("string")
        out["ip_group"] = ip_group
        out = self.add_group_stats(out, "ip_group", "purchase_value", "ip_purchase_value")

        categorical_columns = ["source", "browser", "sex", "country", "device_id"]
        stringify_columns(out, [column for column in categorical_columns if column in out.columns])
        return out.sort_values("event_timestamp").reset_index(drop=True)

    def prepare_artifacts(
        self,
        raw_splits: dict[str, pd.DataFrame],
        raw_files: dict[str, Path],
    ) -> PreparedArtifacts:
        self.log("Preparando base completa do Fraud E-commerce...")
        ip_country_df = (
            pd.read_csv(raw_files["ip_country"], low_memory=False)
            if "ip_country" in raw_files
            else None
        )
        if ip_country_df is not None:
            self.log("Tabela de IP para pais encontrada e carregada.")
        full_prepared = self._prepare_full_frame(raw_splits["full"], ip_country_df)

        split_index = int(full_prepared.shape[0] * 0.8)
        train_prepared = full_prepared.iloc[:split_index, :].copy()
        train_prepared["source_split"] = "train"
        test_prepared = full_prepared.iloc[split_index:, :].copy()
        test_prepared["source_split"] = "test"

        train_path = self.processed_dir / "train_prepared.parquet"
        test_path = self.processed_dir / "test_prepared.parquet"
        full_path = self.processed_dir / "full_prepared.parquet"
        self.log(f"Salvando {train_path.name}, {test_path.name} e {full_path.name}...")
        train_prepared.to_parquet(train_path, index=False)
        test_prepared.to_parquet(test_path, index=False)
        full_prepared.to_parquet(full_path, index=False)

        prepared_summaries = {
            "full": build_split_summary(
                full_prepared,
                split_name="full",
                target_col=self.target_col,
                stage="prepared",
            ),
            "train": build_split_summary(
                train_prepared,
                split_name="train",
                target_col=self.target_col,
                stage="prepared",
            ),
            "test": build_split_summary(
                test_prepared,
                split_name="test",
                target_col=self.target_col,
                stage="prepared",
            ),
        }
        feature_engineering = build_column_transition(
            raw_splits["full"],
            full_prepared,
            raw_split_name="full",
            prepared_split_name="full",
            added_reason=(
                "Colunas derivadas de tempo, tempo desde cadastro, pais por IP e "
                "agregacoes por dispositivo, usuario e IP."
            ),
            removed_reason="Nao ha remocoes estruturais no split completo; o preparo preserva a base bruta.",
            kept_reason="Colunas brutas preservadas na base completa preparada.",
            final_reason="Colunas presentes no artefato final completo apos a engenharia de atributos.",
            notes=[
                "A divisao treino/teste e feita a partir da base completa preparada em proporcao 80/20.",
            ],
        )

        return PreparedArtifacts(
            train_path=train_path,
            test_path=test_path,
            extra_paths={"full_path": full_path},
            metadata={
                "prepared_summaries": prepared_summaries,
                "feature_engineering": feature_engineering,
            },
        )
