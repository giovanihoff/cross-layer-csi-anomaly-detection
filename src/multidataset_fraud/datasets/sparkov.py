from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import KaggleReference, PreparedArtifacts, SourceFileSpec
from ..reporting import build_column_transition, build_split_summary
from ..utils import combine_columns, haversine_km, stringify_columns
from .base import BaseDatasetHandler


class SparkovDataset(BaseDatasetHandler):
    dataset_key = "sparkov"
    display_name = "Sparkov Simulated Credit Card Transactions"
    target_col = "is_fraud"
    kaggle_reference = KaggleReference(kind="dataset", ref="kartik2112/fraud-detection")
    local_env_var = "SPARKOV_LOCAL_DIR"
    source_files = (
        SourceFileSpec("train", ("fraudTrain.csv",)),
        SourceFileSpec("test", ("fraudTest.csv",)),
    )

    def load_raw_splits(self, raw_files: dict[str, Path]) -> dict[str, pd.DataFrame]:
        self.log("Lendo `fraudTrain.csv`...")
        train_df = pd.read_csv(raw_files["train"], low_memory=False)
        self.log("Lendo `fraudTest.csv`...")
        test_df = pd.read_csv(raw_files["test"], low_memory=False)
        return {
            "train": train_df,
            "test": test_df,
        }

    def identifier_candidates(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        return {
            "trans_num": df["trans_num"].astype("string"),
            "cc_num": df["cc_num"].astype("string"),
            "merchant": df["merchant"].astype("string"),
            "cc_num__merchant": combine_columns(df, ["cc_num", "merchant"]),
        }

    def _prepare_frame(self, df: pd.DataFrame, split_label: str) -> pd.DataFrame:
        out = df.copy()
        if "Unnamed: 0" in out.columns:
            out = out.drop(columns=["Unnamed: 0"])

        out["event_id"] = out["trans_num"].astype("string")
        out["uid"] = out["cc_num"].astype("string")
        out["merchant_uid"] = out["merchant"].astype("string")

        transaction_time = pd.to_datetime(out["trans_date_trans_time"], errors="coerce")
        dob = pd.to_datetime(out["dob"], errors="coerce")
        out["event_timestamp"] = transaction_time
        out["event_hour"] = transaction_time.dt.hour.astype("Int64")
        out["event_dayofweek"] = transaction_time.dt.dayofweek.astype("Int64")
        out["event_month"] = transaction_time.dt.month.astype("Int64")
        out["age_at_transaction_years"] = ((transaction_time - dob).dt.days / 365.25).astype("float32")
        out["amt_log1p"] = np.log1p(pd.to_numeric(out["amt"], errors="coerce"))

        if {"lat", "long", "merch_lat", "merch_long"}.issubset(out.columns):
            out["distance_km"] = haversine_km(out["lat"], out["long"], out["merch_lat"], out["merch_long"])

        out = self.add_group_stats(out, "uid", "amt", "uid_amt")
        out = self.add_group_stats(out, "merchant_uid", "amt", "merchant_amt")

        categorical_columns = [
            "merchant",
            "category",
            "gender",
            "job",
            "city",
            "state",
        ]
        stringify_columns(out, [column for column in categorical_columns if column in out.columns])
        out["source_split"] = split_label
        return out

    def prepare_artifacts(
        self,
        raw_splits: dict[str, pd.DataFrame],
        raw_files: dict[str, Path],
    ) -> PreparedArtifacts:
        self.log("Preparando split de treino do Sparkov...")
        train_prepared = self._prepare_frame(raw_splits["train"], "train")
        self.log("Preparando split de teste do Sparkov...")
        test_prepared = self._prepare_frame(raw_splits["test"], "test")

        train_path = self.processed_dir / "train_prepared.parquet"
        test_path = self.processed_dir / "test_prepared.parquet"
        self.log(f"Salvando {train_path.name}...")
        train_prepared.to_parquet(train_path, index=False)
        self.log(f"Salvando {test_path.name}...")
        test_prepared.to_parquet(test_path, index=False)
        prepared_summaries = {
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
            raw_splits["train"],
            train_prepared,
            raw_split_name="train",
            prepared_split_name="train",
            added_reason="Colunas derivadas de timestamp, idade, distancia e agregacoes por cartao e merchant.",
            removed_reason="Colunas residuais de indice exportadas do CSV sao descartadas durante o preparo.",
            kept_reason="Colunas brutas preservadas no split preparado de treino.",
            final_reason="Colunas presentes no artefato final de treino apos a engenharia de atributos.",
        )
        return PreparedArtifacts(
            train_path=train_path,
            test_path=test_path,
            metadata={
                "prepared_summaries": prepared_summaries,
                "feature_engineering": feature_engineering,
            },
        )
