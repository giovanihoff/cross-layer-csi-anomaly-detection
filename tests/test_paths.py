from csi_payment_attestation.paths import DATA_DIR, CONFIG_DIR


def test_core_directories_are_named() -> None:
    assert DATA_DIR.name == "data"
    assert CONFIG_DIR.name == "configs"
