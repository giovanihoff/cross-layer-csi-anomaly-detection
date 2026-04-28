from __future__ import annotations

import pandas as pd

from cross_layer_csi.tabular.datasets.ecommerce import EcommerceDataset
from cross_layer_csi.tabular.datasets.ieee_cis import IEEECISDataset


def test_ieee_winner_uid_uses_card_addr_and_day_anchor() -> None:
    df = pd.DataFrame(
        {
            "card1": [1000, 1000],
            "addr1": [200, 200],
            "TransactionDT": [86400 * 10, 86400 * 12],
            "D1": [3, 5],
        }
    )
    uid = IEEECISDataset.winner_uid(df)
    assert uid.iloc[0] == "1000_200_7"
    assert uid.iloc[1] == "1000_200_7"


def test_ecommerce_event_id_is_deterministic() -> None:
    df = pd.DataFrame(
        {
            "user_id": [1, 2],
            "device_id": ["d1", "d2"],
            "purchase_time": ["2015-01-01 00:00:00", "2015-01-01 00:00:01"],
            "purchase_value": [10.0, 12.0],
            "ip_address": [100.0, 101.0],
        }
    )
    event_id_a = EcommerceDataset._event_id(df)
    event_id_b = EcommerceDataset._event_id(df)
    assert event_id_a.tolist() == event_id_b.tolist()
