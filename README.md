# Cross-Layer Physical Presence Attestation with Wi-Fi CSI

This repository organizes the research artifacts for a cross-layer fraud detection workflow that combines transactional data with Wi-Fi Channel State Information (CSI) as contextual evidence of physical presence.

## Purpose

The project evaluates whether CSI can provide additional signal for anomaly detection in in-person payment scenarios by fusing transactional datasets with CSI measurements collected from heterogeneous hardware platforms.

## Scope

The repository is structured to support:

- transactional baselines for IEEE-CIS, Sparkov, and Ecommerce datasets;
- CSI ingestion and preprocessing for Raspberry Pi/Nexmon and ESP32 acquisitions;
- controlled transaction-to-CSI association and attack-event simulation;
- anomaly detection, robustness analysis, and experiment reporting;
- reproducible experiment configuration for GitHub-based collaboration.

## Datasets and scenarios

Planned experiment tracks include:

- IEEE-CIS + Raspberry Pi CSI
- IEEE-CIS + ESP32 CSI
- Sparkov + Raspberry Pi CSI
- Sparkov + ESP32 CSI
- Ecommerce + Raspberry Pi CSI
- Ecommerce + ESP32 CSI

## Repository notes

- Raw datasets are not versioned in Git.
- Secrets such as Kaggle credentials must be provided through environment variables or local `.env` files.
- Generated outputs should be written to the `reports/` and `artifacts/` directories.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then edit the dataset paths and run the experiment entry points under `src/csi_payment_attestation/pipelines/`.
