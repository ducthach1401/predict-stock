"""Portfolio construction: top-K selection, equal / inverse-volatility weights, a per-instrument cap, a rebalance threshold
(the threshold itself lives in the engine's ``EngineConfig.rebalance_threshold``: it needs current holdings).

Ties are broken by instrument id, so the same scores always give the same portfolio."""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_top_k(scores: pd.Series, k: int, ascending: bool = False) -> list[int]:
    """Instrument ids of the ``k`` best scores (NaN scores are never selected). ``ascending=True`` picks the lowest."""
    s = scores.dropna()
    if s.empty or k <= 0:
        return []
    order = pd.DataFrame({"score": s, "id": s.index}).sort_values(["score", "id"], ascending=[ascending, True], kind="mergesort")
    return [int(i) for i in order["id"].iloc[:k]]


def apply_cap(weights: pd.Series, cap: float | None) -> pd.Series:
    """Limit each weight to ``cap`` and hand the excess to the uncapped names in proportion to their weight.
    If every name is capped the remainder stays in cash (weights then sum to less than the original total)."""
    w = weights.astype(float).copy()
    if cap is None or w.empty:
        return w
    for _ in range(len(w) + 1):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = ~over & (w < cap - 1e-12)
        if not free.any() or w[free].sum() <= 0:
            break
        w[free] += excess * w[free] / w[free].sum()
    return w


def build_weights(scores: pd.Series, k: int, *, scheme: str = "equal", vol: pd.Series | None = None, cap: float | None = None,
                  gross: float = 1.0, ascending: bool = False) -> pd.Series:
    """Weights of the top-K instruments by ``scores``.

    scheme "equal": 1/K each; "inverse_vol": proportional to 1/vol (instruments without a positive volatility are
    dropped, and if none has one it falls back to equal). Weights sum to ``gross`` before the cap."""
    ids = select_top_k(scores, k, ascending)
    if not ids:
        return pd.Series(dtype=float)
    if scheme == "inverse_vol" and vol is not None:
        v = vol.reindex(ids)
        v = v[(v > 0) & v.notna()]
        w = (1.0 / v) / (1.0 / v).sum() * gross if len(v) else pd.Series(gross / len(ids), index=ids)
    elif scheme in ("equal", "inverse_vol"):
        w = pd.Series(gross / len(ids), index=ids)
    else:
        raise ValueError(f"unknown weighting scheme {scheme!r}")
    return apply_cap(w, cap)
