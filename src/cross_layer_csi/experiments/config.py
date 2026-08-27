from __future__ import annotations

from dataclasses import dataclass

DATASET_ORDER = (
    ("ieee_cis", "IEEE_CIS"),
    ("sparkov", "SPARKOV"),
    ("ecommerce", "ECOMMERCE"),
    ("caixabank", "CAIXABANK"),
)

DATASET_SEED_OFFSETS = {
    "IEEE_CIS": 0,
    "SPARKOV": 10_000,
    "ECOMMERCE": 20_000,
    "CAIXABANK": 30_000,
}

SEGMENT_SIZES = (25, 50, 75, 100, 200)
TARGET_HARMONIZED_SUBCARRIERS = 108
RAW_CSI_FEATURE_COUNT = TARGET_HARMONIZED_SUBCARRIERS * 4
CSI_CONSISTENCY_FEATURE_COUNT = 13
PRIMARY_TARGET_FRAUD_RATE = 0.05
CALIBRATION_FRACTION = 0.40
ROBUSTNESS_EVALUATION_SEEDS = (11, 19, 27, 35, 43, 51, 59, 67, 75, 83)


@dataclass(frozen=True)
class ScenarioSeeds:
    """The three independent random roles used by the v120.3 protocol."""

    merge: int
    injection: int
    evaluation: int


@dataclass(frozen=True)
class SeedPolicy:
    """Canonical v120.3 seed policy.

    Data construction and impostor injection intentionally use a different base
    from calibration/model evaluation.  Keeping these roles explicit prevents
    the global-state bug fixed in STEP L9C.
    """

    merge_base: int = 42
    evaluation_base: int = 33
    injection_offset: int = 123

    def for_scenario(self, dataset_label: str, segment_size: int) -> ScenarioSeeds:
        if dataset_label not in DATASET_SEED_OFFSETS:
            allowed = ", ".join(DATASET_SEED_OFFSETS)
            raise ValueError(f"Unknown dataset label {dataset_label!r}; expected one of: {allowed}")
        if segment_size not in SEGMENT_SIZES:
            raise ValueError(
                f"Unsupported segment size {segment_size}; expected one of {SEGMENT_SIZES}"
            )

        offset = DATASET_SEED_OFFSETS[dataset_label] + int(segment_size)
        merge = self.merge_base + offset
        return ScenarioSeeds(
            merge=merge,
            injection=merge + self.injection_offset,
            evaluation=self.evaluation_base + offset,
        )


SEED_POLICY_V120_3 = SeedPolicy()
