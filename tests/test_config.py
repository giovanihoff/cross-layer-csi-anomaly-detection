from csi_payment_attestation.config import ExperimentConfig


def test_config_dataclass_fields() -> None:
    cfg = ExperimentConfig(
        experiment_name="demo",
        transaction_dataset="ieee_cis",
        csi_source="raspberry_pi",
        csi_expected_users=72,
        csi_subcarriers_mode="data_only_108",
        target_label="attack_event",
    )
    assert cfg.transaction_dataset == "ieee_cis"
