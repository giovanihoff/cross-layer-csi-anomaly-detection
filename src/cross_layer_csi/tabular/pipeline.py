from __future__ import annotations

from multidataset_fraud.pipeline import DATASET_REGISTRY
from multidataset_fraud.pipeline import BootstrapPipeline as _LegacyBootstrapPipeline

from .datasets.caixabank import CaixaBankDataset

DATASET_REGISTRY["caixabank"] = CaixaBankDataset


class TabularBootstrapPipeline(_LegacyBootstrapPipeline):
    """Cross-layer wrapper around the existing tabular fraud bootstrap pipeline."""


BootstrapPipeline = TabularBootstrapPipeline

__all__ = ["DATASET_REGISTRY", "BootstrapPipeline", "TabularBootstrapPipeline"]
