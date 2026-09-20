"""Cross-sectional features: values compared ACROSS instruments on the same date, restricted to the members of
the universe on that date (the membership mask). Row t depends on row t only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from predict_stock.features.registry import CrossSectionalFeature, RegistryError, register_feature


def masked_rank(values: pd.DataFrame, mask: pd.DataFrame, min_count: int) -> pd.DataFrame:
    """Percentile rank in (0, 1] of each value among the members with a valid value on that date (average rank for
    ties). Non-members and invalid values get NaN; dates with fewer than ``min_count`` valid members get NaN."""
    v = values.where(mask.reindex_like(values).fillna(False).astype(bool))
    ranks = v.rank(axis=1, pct=True, method="average")
    return ranks.where(v.notna().sum(axis=1) >= min_count, np.nan)


@register_feature
class CrossSectionalRank(CrossSectionalFeature):
    """Rank of each input column among universe members on the date. Output column: ``<input>_csrank``."""

    name, version, group = "cs_rank", 1, "common"
    DEFAULT_PARAMS = {"inputs": [], "min_count": 5}

    def validate(self):
        if not self.params["inputs"]:
            raise RegistryError("cs_rank needs a non-empty 'inputs' list of feature columns")
        if not isinstance(self.params["min_count"], int) or self.params["min_count"] < 1:
            raise RegistryError("min_count must be a positive integer")

    def inputs(self): return list(self.params["inputs"])
    def columns(self): return [f"{c}_csrank" for c in self.params["inputs"]]

    def compute(self, inputs, mask):
        return {f"{c}_csrank": masked_rank(inputs[c], mask, self.params["min_count"]) for c in self.params["inputs"]}
