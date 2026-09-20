"""Fundamental data hook. OFF by default: DNSE's public endpoint has no fundamentals and every strategy must run on
prices alone. A provider can be plugged in later without touching anything else.

Contract for a provider (this is what makes fundamentals point-in-time):
    ``as_of(when, instrument_ids)`` returns a frame indexed by instrument_id with one column per field, holding ONLY
    what had been PUBLISHED on or before ``when`` (report date + publication lag, latest restatement known then).
    It must never return a value for a period that was published after ``when``.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from predict_stock.features.registry import PanelFeature, RegistryError, register_feature


class FundamentalsDisabled(RuntimeError):
    pass


@runtime_checkable
class FundamentalProvider(Protocol):
    enabled: bool

    def as_of(self, when: pd.Timestamp, instrument_ids: list[int]) -> pd.DataFrame: ...


class NullFundamentalProvider:
    """The default: fundamentals are switched off."""

    enabled = False

    def as_of(self, when, instrument_ids):
        raise FundamentalsDisabled("fundamentals are disabled (features.fundamentals.enabled = false); "
                                   "the strategy must be built from price features only")


@register_feature
class Fundamentals(PanelFeature):
    """Pass-through of provider fields, one column ``fund_<field>`` each. Only usable with an enabled provider."""

    name, version, group = "fundamentals", 1, "invest"
    DEFAULT_PARAMS = {"fields": []}

    def validate(self):
        if not self.params["fields"]:
            raise RegistryError("fundamentals needs a non-empty 'fields' list")

    def columns(self): return [f"fund_{f}" for f in self.params["fields"]]

    def compute(self, dates, instrument_ids, provider):
        if provider is None or not getattr(provider, "enabled", False):
            raise FundamentalsDisabled("this feature set uses fundamentals but no fundamental provider is enabled")
        out = {f"fund_{f}": pd.DataFrame(index=dates, columns=instrument_ids, dtype=float) for f in self.params["fields"]}
        for d in dates:
            frame = provider.as_of(d, list(instrument_ids)).reindex(instrument_ids)
            for f in self.params["fields"]:
                out[f"fund_{f}"].loc[d] = frame[f].to_numpy(dtype=float) if f in frame.columns else float("nan")
        return out
