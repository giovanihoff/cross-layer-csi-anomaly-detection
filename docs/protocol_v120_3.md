# Controlled protocol v120.3

## Inputs

- Transaction datasets: IEEE-CIS, Sparkov, E-commerce Fraud, and CAIXABANK.
- CSI sources: ITA_CSI and PMC_CSI.
- Segment sizes: `S ∈ {25, 50, 75, 100, 200}`.
- CSI representation: mean, standard deviation, minimum, and maximum over each of 108 harmonized
  subcarriers, for 432 raw features per segment.

PMC_CSI vectors with 52 subcarriers are linearly interpolated row by row from a normalized
52-position grid to a normalized 108-position grid. This harmonizes dimensionality; it does not
claim physical equivalence between the acquisition frequencies or hardware.

## Construction

1. Preserve each transaction dataset's original train/test boundary.
2. Construct or select the analytical UID without using fraud labels.
3. Associate transaction UIDs with CSI UIDs using train/test support only.
4. Segment each CSI source file independently; files are never mixed inside a segment.
5. Use file-level CSI holdout when multiple capture files exist. For a single file, use a
   contiguous tail holdout with a purged segment at each boundary.
6. Pair CSI `train`/`val` only with `orig_train` transactions and CSI `test` only with
   `orig_test` transactions.
7. Preserve `tx_row_id` and record every row sampled with replacement.
8. For each fraud-labelled test row, replace CSI with one unmodified test CSI segment from a
   different legitimate user's row.
9. Reserve every donor row and exclude it from calibration and holdout evaluation. Donors cannot
   be reused.

## Model and calibration

The paired arms use the same detector and threshold policy. The candidate detectors are
Isolation Forest, one-class SVM with an RBF kernel, and LOF in novelty mode. Each model is fit only
on nominal train/validation rows.

Tx+CSI also fits a claimed-user CSI profile on nominal train/validation rows. It adds 13 features:

- claimed-user L1, L2, cosine, and diagonal-Mahalanobis distances;
- claimed-user rank and top-1/top-3 indicators;
- profile availability and log support;
- nearest and second-nearest profile distances;
- claimed-minus-nearest margin and claimed/nearest ratio.

The test pool is stratified by label and grouped by `tx_row_id` into 40% calibration and 60%
holdout. The operating threshold minimizes Euclidean distance to `(FNR=0, FPR=0)` on calibration,
with deterministic tie-breakers. The selected threshold is applied once to holdout.

## Random roles

For dataset offset `o` and segment size `S`:

- merge seed: `42 + o + S`;
- injection seed: `merge seed + 123`;
- evaluation seed: `33 + o + S`.

Offsets are 0 for IEEE-CIS, 10,000 for SPARKOV, 20,000 for ECOMMERCE, and 30,000 for CAIXABANK.
The L9C random-policy replay must exactly match the selected main TP/FP counts.

## Hard audits

Before fitting, the package rejects:

- any attack marker other than `none` or `impostor`;
- impostors outside test or attached to legitimate rows;
- missing, invalid, reused, fraudulent, attacked, or non-test donors;
- same-user swaps or donors not marked reserved;
- split incoherence or `tx_row_id` overlap between train/validation and test;
- precomputed consistency features before profile fitting;
- protocol metadata in model features;
- raw or derived CSI in Tx-only;
- Tx+CSI without claimed-user consistency features.

## Sensitivity and limits

- Ten seeds vary calibration/holdout assignment and estimator randomness over a fixed constructed
  dataset. They do not repeat construction end to end.
- Profile support values 1, 3, and 5 produced the same selected aggregate result.
- Shuffled and noisy CSI are post-construction negative controls.
- `hard_nearest` is an exploratory donor-policy stress test. Because donor reservation can alter
  the eligible cohort, it is not an isolated causal estimate of impostor proximity.
- The study emulates cross-domain integration and does not claim synchronized deployment data.
