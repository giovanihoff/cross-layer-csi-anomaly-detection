# Architecture and notebook traceability

The reusable package follows the 68-cell v120.3 camera-ready notebook while separating data
bootstrap, CSI processing, controlled fusion, evaluation, and result reporting.

| Notebook cells | Responsibility | Package location |
|---|---|---|
| 2–13 | Four transaction datasets and X/y artifacts | `cross_layer_csi.tabular` |
| 14–27 | CSI acquisition, conversion, filtering, smoothing, harmonization | `cross_layer_csi.csi` |
| 28–34 | 432-feature segmentation and purged split | `cross_layer_csi.csi.segmentation` |
| 35–40 | Label-free identity mapping, Tx/CSI pairing, impostor injection, audits | `cross_layer_csi.experiments.identity`, `fusion`, `audits` |
| 41–43 | Claimed-user profiles, paired one-class evaluation, feature guard | `cross_layer_csi.experiments.consistency`, `evaluation`, `audits` |
| 44–57 | Pairing, selection, segment/prevalence/profile sensitivity, evaluation seeds, controls | `cross_layer_csi.experiments.reporting` and versioned result tables |
| 58–61 | Complementary two-phase LightGBM/XGBoost campaign | `cross_layer_csi.experiments.two_phase`; canonical output in `reports/v120_3` |
| 62–65 | Explicit seed roles and donor-policy stress test | `config`, `pipeline`, `reporting` |
| 66–67 | Environment and interpretation guard | `reports/v120_3` and `docs/protocol_v120_3.md` |

## Runtime flow

1. Tabular handlers create stable train/test transaction artifacts and analytical UIDs.
2. CSI handlers create harmonized 108-subcarrier amplitude frames.
3. `build_segmented_datasets` creates 432-feature segments and applies the anti-leak CSI split.
4. `build_label_free_user_map` selects recurring transaction identities without fraud labels.
5. `ControlledExperimentRunner.construct` builds split-coherent pairs and injects unique test-only
   CSI impostors using the explicit merge/injection seeds.
6. `ControlledExperimentRunner.evaluate` executes the protocol, reuse, and feature audits before
   fitting any one-class model.
7. Calibration chooses a threshold; holdout produces the reported paired metrics.
8. `donor_policy_stress_test` first enforces exact random replay, then evaluates `hard_nearest`.

## Compatibility boundary

`multidataset_fraud` remains in the tree as the original tabular bootstrap implementation. New
cross-layer work should import `cross_layer_csi`. Notebook-compatible function aliases are kept in
the experiment modules only where they help trace a published output back to its original cell.
