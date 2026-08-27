from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from multidataset_fraud.models import KaggleReference, PreparedArtifacts, SourceFileSpec
from multidataset_fraud.reporting import build_column_transition, build_split_summary

from .base import BaseDatasetHandler


class CaixaBankDataset(BaseDatasetHandler):
    """Controlled 50k/~500-fraud subset used by the v120.3 experiment."""

    dataset_key = "caixabank"
    display_name = "CaixaBank Transactions Fraud"
    target_col = "is_fraud"
    kaggle_reference = KaggleReference(
        kind="dataset",
        ref="computingvictor/transactions-fraud-datasets",
    )
    local_env_var = "CAIXABANK_LOCAL_DIR"
    source_files = (
        SourceFileSpec(
            "transactions",
            ("transactions_data.csv", "transactions.csv", "transaction_data.csv"),
        ),
        SourceFileSpec(
            "labels",
            ("train_fraud_labels.json", "fraud_labels.json", "labels.json"),
        ),
        SourceFileSpec("users", ("users_data.csv", "users.csv", "user_data.csv"), required=False),
        SourceFileSpec(
            "cards",
            ("cards_data.csv", "cards.csv", "card_data.csv", "cards_dat.csv"),
            required=False,
        ),
    )
    target_rows = 50_000
    target_frauds = 500
    subset_seed = 49

    @staticmethod
    def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out.columns = [str(column).strip().lower() for column in out.columns]
        return out

    @staticmethod
    def _load_labels(path: Path) -> pd.DataFrame:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("target"), dict):
            raise TypeError("Expected CaixaBank labels in {'target': {...}} format")
        labels = payload["target"]
        return pd.DataFrame(
            {
                "id": [str(key).strip() for key in labels],
                "is_fraud": [int(str(value).strip().lower() == "yes") for value in labels.values()],
            }
        )

    def _build_client_stats(self, transaction_path: Path, fraud_ids: set[str]) -> pd.DataFrame:
        aggregates: dict[int, dict[str, int]] = {}
        for chunk in pd.read_csv(
            transaction_path,
            usecols=["id", "client_id"],
            chunksize=250_000,
        ):
            chunk = self._normalize_columns(chunk)
            chunk["id"] = chunk["id"].astype(str).str.strip()
            chunk["client_id"] = pd.to_numeric(chunk["client_id"], errors="coerce")
            chunk = chunk.dropna(subset=["client_id"])
            chunk["client_id"] = chunk["client_id"].astype(int)
            chunk["is_fraud"] = chunk["id"].isin(fraud_ids).astype(int)
            grouped = chunk.groupby("client_id")["is_fraud"].agg(
                tx_count="count", fraud_count="sum"
            )
            for client_id, values in grouped.iterrows():
                target = aggregates.setdefault(int(client_id), {"tx_count": 0, "fraud_count": 0})
                target["tx_count"] += int(values["tx_count"])
                target["fraud_count"] += int(values["fraud_count"])
        stats = pd.DataFrame(
            [{"client_id": client_id, **values} for client_id, values in aggregates.items()]
        )
        if stats.empty:
            raise RuntimeError("Could not build CaixaBank client statistics")
        stats["legit_count"] = stats["tx_count"] - stats["fraud_count"]
        return stats

    def _select_clients(self, stats: pd.DataFrame) -> list[int]:
        candidates = stats[stats["fraud_count"].gt(0) & stats["tx_count"].ge(20)].copy()
        if candidates.empty:
            candidates = stats[stats["fraud_count"].gt(0)].copy()
        candidates = candidates.sort_values(
            ["tx_count", "fraud_count"], ascending=[False, False]
        ).head(300)
        candidate_ids = candidates["client_id"].astype(int).tolist()
        if not candidate_ids:
            raise RuntimeError("No CaixaBank clients with labelled fraud were found")
        rng = np.random.default_rng(self.subset_seed)
        weights = np.sqrt(candidates["tx_count"].to_numpy(float))
        weights /= weights.sum()
        best: tuple[float, list[int]] | None = None
        minimum = min(6, len(candidate_ids))
        maximum = min(30, len(candidate_ids))
        for _ in range(4_000):
            count = int(rng.integers(minimum, maximum + 1))
            chosen = rng.choice(candidate_ids, size=count, replace=False, p=weights)
            subset = candidates[candidates["client_id"].isin(chosen)]
            tx_count = int(subset["tx_count"].sum())
            fraud_count = int(subset["fraud_count"].sum())
            score = abs(tx_count - self.target_rows) / self.target_rows
            score += 1.4 * abs(fraud_count - self.target_frauds) / self.target_frauds
            if float(subset["tx_count"].mean()) < 100:
                score += 0.25
            proposal = (score, sorted(map(int, chosen)))
            if best is None or proposal[0] < best[0]:
                best = proposal
        return best[1] if best else candidate_ids

    def _sample_legits_preserving_clients(
        self,
        legit: pd.DataFrame,
        target_rows: int,
    ) -> pd.DataFrame:
        """Sample legitimate rows approximately in proportion to each client."""
        if target_rows <= 0:
            return legit.iloc[0:0].copy()
        if len(legit) <= target_rows:
            return legit.copy()

        rng = np.random.default_rng(self.subset_seed)
        indexed = legit.reset_index(drop=False).rename(columns={"index": "_source_index"})
        counts = indexed.groupby("client_id").size().sort_values(ascending=False)
        allocation = np.floor(counts / counts.sum() * target_rows).astype(int)
        allocation[counts.gt(0) & allocation.eq(0)] = 1
        allocation = allocation.clip(upper=counts)

        while int(allocation.sum()) > target_rows:
            reducible = allocation[allocation.gt(1)]
            if reducible.empty:
                break
            allocation.loc[reducible.idxmax()] -= 1

        sampled_parts: list[pd.DataFrame] = []
        sampled_indices: set[int] = set()
        for client_id, count in allocation.items():
            if count <= 0:
                continue
            group = indexed[indexed["client_id"].eq(client_id)]
            chosen = group.sample(
                n=min(int(count), len(group)),
                replace=False,
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
            sampled_parts.append(chosen)
            sampled_indices.update(chosen["_source_index"].astype(int))

        sampled = pd.concat(sampled_parts, ignore_index=True)
        remaining = target_rows - len(sampled)
        if remaining > 0:
            pool = indexed[~indexed["_source_index"].isin(sampled_indices)]
            extra = pool.sample(
                n=min(remaining, len(pool)),
                replace=False,
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
            sampled = pd.concat([sampled, extra], ignore_index=True)
        return sampled.drop(columns="_source_index")

    def _build_subset(self, raw_files: dict[str, Path]) -> pd.DataFrame:
        cache = (
            self.processed_dir / f"caixabank_subset_{self.target_rows}_{self.target_frauds}.parquet"
        )
        if cache.exists():
            return pd.read_parquet(cache)
        labels = self._load_labels(raw_files["labels"])
        fraud_ids = set(labels.loc[labels["is_fraud"].eq(1), "id"].astype(str))
        stats = self._build_client_stats(raw_files["transactions"], fraud_ids)
        selected_clients = set(self._select_clients(stats))
        usecols = [
            "id",
            "date",
            "client_id",
            "card_id",
            "amount",
            "use_chip",
            "merchant_id",
            "merchant_city",
            "merchant_state",
            "zip",
            "mcc",
            "errors",
        ]
        chunks = []
        for chunk in pd.read_csv(
            raw_files["transactions"],
            usecols=usecols,
            chunksize=250_000,
        ):
            chunk = self._normalize_columns(chunk)
            chunk["client_id"] = pd.to_numeric(chunk["client_id"], errors="coerce")
            chunk = chunk[chunk["client_id"].isin(selected_clients)].copy()
            if chunk.empty:
                continue
            chunk["client_id"] = chunk["client_id"].astype(int)
            chunk["id"] = chunk["id"].astype(str).str.strip()
            chunks.append(chunk.merge(labels, on="id", how="left"))
        if not chunks:
            raise RuntimeError("No transactions found for selected CaixaBank clients")
        full = pd.concat(chunks, ignore_index=True)
        full["is_fraud"] = pd.to_numeric(full["is_fraud"], errors="coerce").fillna(0).astype(int)
        full["date"] = pd.to_datetime(full["date"], errors="coerce")
        fraud = full[full["is_fraud"].eq(1)]
        legit = full[full["is_fraud"].eq(0)]
        target_legit = max(self.target_rows - len(fraud), 0)
        legit = self._sample_legits_preserving_clients(legit, target_legit)
        subset = pd.concat([fraud, legit], ignore_index=True).sort_values("date")

        for logical_name, left_key, renamed_id in (
            ("users", "client_id", "user_table_id"),
            ("cards", "card_id", "card_table_id"),
        ):
            if logical_name not in raw_files:
                continue
            auxiliary = self._normalize_columns(pd.read_csv(raw_files[logical_name]))
            if "id" in auxiliary.columns:
                auxiliary = auxiliary.rename(columns={"id": renamed_id})
                subset = subset.merge(
                    auxiliary,
                    left_on=left_key,
                    right_on=renamed_id,
                    how="left",
                    suffixes=("", f"_{logical_name}"),
                )
        subset = subset.reset_index(drop=True)
        subset.to_parquet(cache, index=False)
        return subset

    def load_raw_splits(self, raw_files: dict[str, Path]) -> dict[str, pd.DataFrame]:
        subset = self._build_subset(raw_files).sort_values("date").reset_index(drop=True)
        split = int(len(subset) * 0.8)
        return {"train": subset.iloc[:split].copy(), "test": subset.iloc[split:].copy()}

    def identifier_candidates(self, frame: pd.DataFrame) -> dict[str, pd.Series]:
        client = frame["client_id"].astype("string")
        card = frame["card_id"].astype("string")
        return {"client_id": client, "client_card": client + "|" + card}

    @staticmethod
    def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["uid"] = out["client_id"].astype("string") + "|" + out["card_id"].astype("string")
        for column in (
            "amount",
            "credit_limit",
            "per_capita_income",
            "yearly_income",
            "total_debt",
        ):
            if column in out.columns:
                out[column] = pd.to_numeric(
                    out[column].astype(str).str.replace(r"[\$,]", "", regex=True),
                    errors="coerce",
                ).astype("float32")
        timestamp = pd.to_datetime(out["date"], errors="coerce")
        out["tx_hour"] = timestamp.dt.hour.astype("float32")
        out["tx_dayofweek"] = timestamp.dt.dayofweek.astype("float32")
        out["tx_month"] = timestamp.dt.month.astype("float32")
        out = out.drop(columns=["date"], errors="ignore")
        for source, prefix in (("acct_open_date", "acct_open"), ("expires", "expires")):
            if source in out.columns:
                parsed = pd.to_datetime(out[source], format="%m/%Y", errors="coerce")
                out[f"{prefix}_year"] = parsed.dt.year.astype("float32")
                out[f"{prefix}_month"] = parsed.dt.month.astype("float32")
                out = out.drop(columns=[source])
        out = out.drop(
            columns=[
                "id",
                "client_id",
                "card_id",
                "user_table_id",
                "card_table_id",
                "card_number",
                "cvv",
                "address",
            ],
            errors="ignore",
        )
        numeric = [
            column
            for column in out.select_dtypes(include=[np.number]).columns
            if column != "is_fraud"
        ]
        out[numeric] = out[numeric].astype("float32").fillna(-999.0)
        for column in out.select_dtypes(include=["object"]).columns:
            if column != "uid":
                out[column] = out[column].astype(str).replace({"nan": "MISSING", "None": "MISSING"})
        out["uid"] = out["uid"].fillna("MISSING").astype(str)
        return out

    def prepare_artifacts(
        self,
        raw_splits: dict[str, pd.DataFrame],
        raw_files: dict[str, Path],
    ) -> PreparedArtifacts:
        train = self._prepare_frame(raw_splits["train"])
        test = self._prepare_frame(raw_splits["test"])
        shared = [column for column in train.columns if column in test.columns]
        train, test = train[shared], test[shared]
        y_train, y_test = train.pop(self.target_col), test.pop(self.target_col)
        x_train_path = self.processed_dir / "X_train.parquet"
        x_test_path = self.processed_dir / "X_test.parquet"
        y_train_path = self.processed_dir / "y_train.parquet"
        y_test_path = self.processed_dir / "y_test.parquet"
        train.to_parquet(x_train_path, index=False)
        test.to_parquet(x_test_path, index=False)
        y_train.to_frame(self.target_col).to_parquet(y_train_path, index=False)
        y_test.to_frame(self.target_col).to_parquet(y_test_path, index=False)
        prepared_train = train.assign(**{self.target_col: y_train.to_numpy()})
        prepared_test = test.assign(**{self.target_col: y_test.to_numpy()})
        return PreparedArtifacts(
            train_path=x_train_path,
            test_path=x_test_path,
            extra_paths={"y_train": y_train_path, "y_test": y_test_path},
            metadata={
                "prepared_summaries": {
                    "train": build_split_summary(
                        prepared_train,
                        split_name="train",
                        target_col=self.target_col,
                        stage="prepared",
                    ),
                    "test": build_split_summary(
                        prepared_test,
                        split_name="test",
                        target_col=self.target_col,
                        stage="prepared",
                    ),
                },
                "feature_engineering": build_column_transition(
                    raw_splits["train"],
                    prepared_train,
                    raw_split_name="train",
                    prepared_split_name="train",
                    added_reason="Controlled subset UID, temporal, account, and monetary features.",
                    removed_reason="Raw identifiers and unparsed date fields are removed after derivation.",
                    kept_reason="Transaction attributes retained after cleaning.",
                    final_reason="Columns written to the v120.3 X/y artifacts.",
                ),
            },
        )
