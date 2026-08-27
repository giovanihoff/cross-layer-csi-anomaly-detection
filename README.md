# Cross-Layer CSI Anomaly Detection

Reproducible research code for controlled cross-layer physical-presence emulation in in-person
payment anomaly detection. The project compares the same one-class detector under two paired
feature views:

- **Tx-only:** transactional attributes;
- **Tx+CSI:** the same attributes plus 432 raw Wi-Fi CSI segment features and 13 claimed-user
  physical-coherence features.

The current implementation tracks the fully executed **v120.3 camera-ready protocol**.

## Experimental scope

The transaction layer covers IEEE-CIS, Sparkov, E-commerce Fraud, and the controlled 50k-row
CAIXABANK subset. The CSI layer combines ITA_CSI (108 native subcarriers) and PMC_CSI (52 native
subcarriers, linearly interpolated to a normalized 108-position grid).

Only one attack context is evaluated: a fraud-labelled held-out transaction receives an
unmodified real CSI segment from a different held-out user. Donors are unique, reserved, and
excluded from calibration and holdout evaluation. No remote-mismatch context or artificially
perturbed attack sample belongs to the primary protocol.

## v120.3 safeguards

- CSI `train`/`val` pairs only with original transaction train rows; CSI `test` only with original
  transaction test rows.
- Calibration and holdout are split by `tx_row_id`, with no group overlap.
- One-class models train only on nominal rows.
- CSI user profiles fit only nominal train/validation rows.
- Tx-only receives zero raw or derived CSI features.
- Merge (`42`), injection (`merge + 123`), and evaluation (`33`) seed roles are explicit.
- The L9C random branch must reproduce the main TP/FP counts exactly before the exploratory
  `hard_nearest` stress test is accepted.

## Primary result

For the selected `S=25` scenarios:

| Dataset | Detector | TP Tx→Tx+CSI | FP Tx→Tx+CSI |
|---|---|---:|---:|
| CAIXABANK | OneClassSVM RBF | 6→6 | 82→16 |
| ECOMMERCE | OneClassSVM RBF | 88→122 | 1,420→185 |
| IEEE-CIS | LOF novelty | 57→88 | 909→437 |
| SPARKOV | OneClassSVM RBF | 21→30 | 837→181 |
| **Total** | — | **172→246** | **3,248→819** |

This is a 74.78% reduction in false positives with a Pareto improvement in all four controlled
dataset evaluations. Versioned result tables and interpretation limits are under
[`reports/v120_3/`](reports/v120_3/README.md).

## Project structure

- `src/cross_layer_csi/tabular/`: dataset bootstrap, including CAIXABANK.
- `src/cross_layer_csi/csi/`: conversion, filtering, smoothing, 52→108 harmonization,
  segmentation, and anti-leak splitting.
- `src/cross_layer_csi/experiments/`: identity mapping, controlled fusion, CSI consistency
  profiles, paired one-class evaluation, audits, seed guards, and reporting.
- `reports/v120_3/`: canonical compact results extracted from the executed notebook.
- `notebooks/01_bootstrap_datasets.ipynb`: dataset bootstrap helper.
- `tests/`: synthetic invariants for processing, donor reservation, leakage prevention, feature
  separation, seed roles, and result replay.

See [`docs/architecture.md`](docs/architecture.md) and
[`docs/protocol_v120_3.md`](docs/protocol_v120_3.md) for the notebook-to-package map and the exact
experimental flow.

## Installation and checks

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env.kaggle
python -m pytest -q
```

Optional two-phase dependencies:

```bash
python -m pip install -e ".[two-phase]"
```

Bootstrap and preprocess data:

```bash
cross-layer-csi tabular-bootstrap --datasets all
cross-layer-csi csi-preprocess --no-plots
cross-layer-csi protocol-manifest --segment-size 25
```

The reusable controlled experiment entry point is
`cross_layer_csi.experiments.ControlledExperimentRunner`. It accepts prepared transaction frames,
segmented CSI frames, and label-free Tx-to-CSI user maps, then runs the hard audits before fitting
any detector.

## Data, secrets, and interpretation

Raw data and generated artifacts are intentionally not versioned. Supply Kaggle credentials via
environment variables, the ignored local `.env.kaggle` file, or the standard Kaggle configuration file; never add
credentials to source code or notebooks.

The implementation is a controlled cross-domain integration, not a synchronized real-world
Tx/CSI dataset. The detectors are one-class with label-informed calibration. The multi-seed table
measures evaluation-seed stability on a fixed constructed dataset, and `shuffled_csi`/`noisy_csi`
are negative controls rather than attack contexts.
