# v120.3 camera-ready result snapshot

These tables are the compact, versioned result snapshot extracted from the fully executed
`Untitled25_v120_3_camera_ready_executed(2).ipynb` notebook. The notebook remains the
computational provenance record; the reusable package implements its controlled protocol.

## Primary result

The primary condition is a controlled impostor emulation with `S=25`. A fraud-labelled held-out
transaction receives an unmodified real CSI segment from a different held-out user. Each donor is
used at most once and is excluded from calibration and holdout evaluation.

Across CAIXABANK, ECOMMERCE, IEEE-CIS, and SPARKOV, Tx+CSI changes TP from 172 to 246 and FP from
3,248 to 819 (74.78% fewer false positives), with a Pareto improvement in all four datasets.

## Interpretation limits

- The detector is one-class, with label-informed calibration; the full pipeline is not fully
  unsupervised.
- The ten-seed analysis varies calibration/holdout assignment and estimator randomness on one
  fixed constructed dataset. It is evaluation-seed stability, not an end-to-end reconstruction.
- `shuffled_csi` and `noisy_csi` are post-construction negative controls, not attack contexts.
- `hard_nearest` is an exploratory stress test. Donor reservation can change the eligible cohort,
  so the difference from `random` is not a causal effect of impostor proximity alone.
- The work is controlled cross-domain integration, not a synchronized real-world Tx/CSI dataset.

## Files

- `main_results.csv`: selected `S=25` one-class comparison.
- `segment_sweep.csv`: aggregate selected result for every segment size.
- `evaluation_seed_stability.csv`: ten-run mean and 95% t-interval half-width.
- `donor_policy_sensitivity.csv`: random and hard-nearest paired results.
- `seed_manifest.csv`: explicit merge, injection, and evaluation seeds.
- `two_phase_results.csv`: complementary LightGBM/XGBoost results at `S=25`.
- `environment_manifest.csv`: versions recorded in the executed notebook.
