from __future__ import annotations

from multidataset_fraud.pipeline import BootstrapPipeline as _LegacyBootstrapPipeline
from multidataset_fraud.pipeline import DATASET_REGISTRY


class TabularBootstrapPipeline(_LegacyBootstrapPipeline):
    """Cross-layer wrapper around the existing tabular fraud bootstrap pipeline."""


BootstrapPipeline = TabularBootstrapPipeline

__all__ = ["BootstrapPipeline", "DATASET_REGISTRY", "TabularBootstrapPipeline"]
