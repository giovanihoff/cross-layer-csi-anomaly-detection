# Suggested repository structure

```text
cross-layer-csi-fraud-detection/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── ieee_cis_raspberrypi.yaml
│   ├── ieee_cis_esp32.yaml
│   ├── sparkov_raspberrypi.yaml
│   ├── sparkov_esp32.yaml
│   ├── ecommerce_raspberrypi.yaml
│   └── ecommerce_esp32.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── docs/
│   ├── paper/
│   └── review_notes/
├── notebooks/
│   ├── 01_transaction_baselines.ipynb
│   ├── 02_csi_preprocessing.ipynb
│   ├── 03_csi_user_identification.ipynb
│   ├── 04_tx_csi_fusion.ipynb
│   └── 05_results_analysis.ipynb
├── reports/
│   ├── figures/
│   ├── tables/
│   └── logs/
├── artifacts/
├── scripts/
│   ├── run_ieee_cis.sh
│   ├── run_sparkov.sh
│   └── run_ecommerce.sh
├── src/
│   └── csi_payment_attestation/
│       ├── __init__.py
│       ├── config.py
│       ├── paths.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── download_ieee_cis.py
│       │   ├── prepare_ieee_cis.py
│       │   ├── prepare_sparkov.py
│       │   ├── prepare_ecommerce.py
│       │   └── load_csi.py
│       ├── features/
│       │   ├── __init__.py
│       │   ├── transaction_features.py
│       │   ├── csi_preprocessing.py
│       │   ├── csi_sharding.py
│       │   └── fusion_features.py
│       ├── simulation/
│       │   ├── __init__.py
│       │   ├── threat_model.py
│       │   ├── attack_event.py
│       │   └── negative_controls.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── tx_baselines.py
│       │   ├── csi_baselines.py
│       │   ├── anomaly_detectors.py
│       │   └── ensemble.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── robustness.py
│       │   └── reporting.py
│       └── pipelines/
│           ├── __init__.py
│           ├── run_ieee_cis.py
│           ├── run_sparkov.py
│           └── run_ecommerce.py
└── tests/
    ├── test_config.py
    └── test_paths.py
```
