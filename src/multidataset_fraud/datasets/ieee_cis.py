from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..models import KaggleReference, PreparedArtifacts, SourceFileSpec
from ..reporting import build_column_transition, build_split_summary
from ..utils import combine_columns, numeric_series, stringify_columns
from .base import BaseDatasetHandler


class IEEECISDataset(BaseDatasetHandler):
    dataset_key = "ieee_cis"
    display_name = "IEEE-CIS Fraud Detection"
    target_col = "isFraud"
    kaggle_reference = KaggleReference(kind="competition", ref="ieee-fraud-detection")
    local_env_var = "IEEE_CIS_LOCAL_DIR"
    source_files = (
        SourceFileSpec("train_transaction", ("train_transaction.csv",)),
        SourceFileSpec("train_identity", ("train_identity.csv",)),
        SourceFileSpec("test_transaction", ("test_transaction.csv",), required=False),
        SourceFileSpec("test_identity", ("test_identity.csv",), required=False),
        SourceFileSpec("sample_submission", ("sample_submission.csv",), required=False),
    )

    @staticmethod
    def _merge_identity(transaction_df: pd.DataFrame, identity_df: pd.DataFrame | None) -> pd.DataFrame:
        if identity_df is None:
            return transaction_df.copy()
        return transaction_df.merge(identity_df, on="TransactionID", how="left")

    @staticmethod
    def winner_uid(df: pd.DataFrame) -> pd.Series:
        card_addr = combine_columns(df, ["card1", "addr1"])
        day = numeric_series(df, "TransactionDT") / 86400.0
        d1 = numeric_series(df, "D1").fillna(-1)
        day_anchor = np.floor(day - d1).astype("Int64").astype("string")
        return card_addr.str.cat(day_anchor, sep="_")

    def load_raw_splits(self, raw_files: dict[str, Path]) -> dict[str, pd.DataFrame]:
        self.log("Lendo `train_transaction.csv`...")
        train_tx = pd.read_csv(raw_files["train_transaction"], low_memory=False)
        self.log("Lendo `train_identity.csv`...")
        train_id = pd.read_csv(raw_files["train_identity"], low_memory=False)
        self.log("Mesclando transacoes e identidades de treino...")
        train_df = self._merge_identity(train_tx, train_id)

        splits = {"train": train_df}
        if "test_transaction" in raw_files:
            self.log("Lendo `test_transaction.csv`...")
            test_tx = pd.read_csv(raw_files["test_transaction"], low_memory=False)
            self.log("Lendo `test_identity.csv`...")
            test_id = pd.read_csv(raw_files["test_identity"], low_memory=False) if "test_identity" in raw_files else None
            self.log("Mesclando transacoes e identidades de teste...")
            splits["competition_test"] = self._merge_identity(test_tx, test_id)
        return splits

    def identifier_candidates(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        return {
            "TransactionID": df["TransactionID"].astype("string"),
            "uid_winner_card1_addr1_d1": self.winner_uid(df),
            "uid_card_pack": combine_columns(df, ["card1", "card2", "card3", "card5", "addr1", "addr2"]),
            "uid_card_pack_email": combine_columns(
                df,
                ["card1", "card2", "card3", "card5", "addr1", "addr2", "P_emaildomain"],
            ),
        }

    def _prepare_frame(self, df: pd.DataFrame, split_label: str) -> pd.DataFrame:
        out = df.copy()
        out["event_id"] = out["TransactionID"].astype("Int64").astype("string")
        out["uid"] = self.winner_uid(out)
        out["uid_card_pack_email"] = combine_columns(
            out,
            ["card1", "card2", "card3", "card5", "addr1", "addr2", "P_emaildomain"],
        )

        transaction_dt = numeric_series(out, "TransactionDT")
        out["transaction_day"] = transaction_dt / 86400.0
        out["transaction_hour"] = ((transaction_dt // 3600) % 24).astype("Int64")
        out["transaction_week"] = (transaction_dt // (86400 * 7)).astype("Int64")
        out["transaction_month"] = (transaction_dt // (86400 * 30)).astype("Int64")
        out["email_match"] = (
            out.get("P_emaildomain", pd.Series(index=out.index, dtype="string")).astype("string").fillna("missing")
            == out.get("R_emaildomain", pd.Series(index=out.index, dtype="string")).astype("string").fillna("missing")
        ).astype("int8")

        for idx in range(1, 16):
            column = f"D{idx}"
            if column in out.columns:
                out[f"{column}_normalized"] = numeric_series(out, column) - out["transaction_day"]

        out = self.add_group_stats(out, "uid", "TransactionAmt", "uid_transaction_amt")
        if "card1" in out.columns:
            out["card1_group"] = out["card1"].astype("string").fillna("missing")
            out = self.add_group_stats(out, "card1_group", "TransactionAmt", "card1_transaction_amt")

        categorical_columns = [
            "ProductCD",
            "card4",
            "card6",
            "P_emaildomain",
            "R_emaildomain",
            "DeviceType",
            "DeviceInfo",
            "id_30",
            "id_31",
        ]
        stringify_columns(out, [column for column in categorical_columns if column in out.columns])
        out["source_split"] = split_label
        return out

    def prepare_artifacts(
        self,
        raw_splits: dict[str, pd.DataFrame],
        raw_files: dict[str, Path],
    ) -> PreparedArtifacts:
        self.log("Preparando split de treino do IEEE-CIS...")
        train_prepared = self._prepare_frame(raw_splits["train"], "train")
        train_path = self.processed_dir / "train_prepared.parquet"
        self.log(f"Salvando {train_path.name}...")
        train_prepared.to_parquet(train_path, index=False)

        if "competition_test" in raw_splits:
            self.log("Preparando split de teste do IEEE-CIS...")
            test_prepared = self._prepare_frame(raw_splits["competition_test"], "competition_test")
        else:
            test_prepared = pd.DataFrame()
        test_path = self.processed_dir / "test_prepared.parquet"
        self.log(f"Salvando {test_path.name}...")
        test_prepared.to_parquet(test_path, index=False)

        prepared_summaries = {
            "train": build_split_summary(
                train_prepared,
                split_name="train",
                target_col=self.target_col,
                stage="prepared",
            ),
            "competition_test": build_split_summary(
                test_prepared,
                split_name="competition_test",
                target_col=self.target_col,
                stage="prepared",
            ),
        }
        feature_engineering = build_column_transition(
            raw_splits["train"],
            train_prepared,
            raw_split_name="train",
            prepared_split_name="train",
            added_reason=(
                "Colunas derivadas de tempo, normalizacao de D*, agregacoes por uid/cartao "
                "e proxies auxiliares de identidade."
            ),
            removed_reason="Nao ha remocoes estruturais no split de referencia; o preparo preserva a base bruta.",
            kept_reason="Colunas brutas preservadas no split preparado de treino.",
            final_reason="Colunas presentes no artefato final de treino apos a engenharia de atributos.",
            notes=[
                "O split competition_test nao possui a coluna-alvo `isFraud`.",
            ],
        )

        return PreparedArtifacts(
            train_path=train_path,
            test_path=test_path,
            metadata={
                "prepared_summaries": prepared_summaries,
                "feature_engineering": feature_engineering,
            },
        )
