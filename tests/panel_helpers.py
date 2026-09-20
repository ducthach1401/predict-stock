"""Synthetic price panels for feature/label tests (no database)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from predict_stock.features.data import Panel


def make_calendar(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n, name="trade_date")


def bars_from_close(close, *, spread: float = 0.01, volume=1_000_000.0, index=None) -> pd.DataFrame:
    """Deterministic OHLCV around a close path: open = previous close, high/low a fixed spread around the range."""
    c = pd.Series(np.asarray(close, dtype=float), index=index if index is not None else make_calendar(len(close)))
    o = c.shift().fillna(c.iloc[0])
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + spread), "low": np.minimum(o, c) * (1 - spread),
                         "close": c, "volume": pd.Series(volume, index=c.index, dtype=float) if np.isscalar(volume) else pd.Series(volume, index=c.index, dtype=float)})


def random_panel(n_days: int = 320, n_inst: int = 6, seed: int = 0, gaps: int = 6, bench: bool = True, first_id: int = 1) -> Panel:
    """Random-walk panel with a few missing bars per instrument (suspension-like) and a benchmark."""
    rng = np.random.default_rng(seed)
    cal = make_calendar(n_days)
    ids = list(range(first_id, first_id + n_inst))
    frames = {k: pd.DataFrame(index=cal, columns=ids, dtype=float) for k in ("open", "high", "low", "close", "volume")}
    for iid in ids:
        r = rng.normal(0.0004, 0.018, n_days)
        c = 20000 * np.exp(np.cumsum(r))
        o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.004, n_days))
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, n_days)))
        l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, n_days)))
        v = rng.integers(200_000, 3_000_000, n_days).astype(float)
        for k, a in zip(("open", "high", "low", "close", "volume"), (o, h, l, c, v)):
            frames[k][iid] = a
    for iid in ids:                                                      # missing bars (not on the first/last 40 sessions)
        for j in rng.choice(np.arange(40, n_days - 40), size=gaps, replace=False):
            for k in frames:
                frames[k].loc[cal[j], iid] = np.nan
    b = None
    if bench:
        b = pd.Series(900 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, n_days))), index=cal)
    return Panel(cal, frames["open"], frames["high"], frames["low"], frames["close"], frames["volume"], bench=b)


def panel_from_arrays(close, high=None, low=None, open_=None, volume=None, ids=None, start="2024-01-01") -> Panel:
    """Explicit panel: arrays are (dates x instruments)."""
    c = np.atleast_2d(np.asarray(close, dtype=float))
    if c.shape[0] == 1 and c.shape[1] > 1 and np.asarray(close).ndim == 1:
        c = c.T
    n, m = c.shape
    cal = make_calendar(n, start)
    ids = ids or list(range(1, m + 1))
    mk = lambda a, default: pd.DataFrame(np.asarray(default if a is None else a, dtype=float).reshape(n, m), index=cal, columns=ids)
    return Panel(cal, mk(open_, c), mk(high, c), mk(low, c), mk(c, c), mk(volume, np.ones((n, m))))


def all_members(panel: Panel) -> pd.DataFrame:
    return pd.DataFrame(True, index=panel.calendar, columns=panel.instrument_ids)
