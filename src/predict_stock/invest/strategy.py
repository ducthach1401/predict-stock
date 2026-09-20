"""Rebalance signals for the INVEST portfolios: scheduled top-K, a weighting scheme with a cap, tranches, an optional regime filter.

Every scheduled rebalance moves the portfolio from the previous target to the new one in ``tranches`` equal steps ``tranche_spacing`` sessions apart
(steps that would run into the next rebalance are dropped). Orders execute at the next open through the engine, whose relative rebalance band skips
trades that are too small. The regime filter multiplies the stock weights by ``equity_share`` when the index closed below its SMA on the decision date
(the rest stays in cash)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from predict_stock.backtest.baselines import rebalance_sessions
from predict_stock.backtest.engine import Signal, SignalItem
from predict_stock.backtest.portfolio import select_top_k
from predict_stock.invest.weights import target_weights


def regime_off(index_close: dict[str, pd.Series], calendar: pd.DatetimeIndex, symbol: str, fallback: str, sma_window: int) -> pd.Series:
    """True on dates where the index closed below its SMA(``sma_window``): ``symbol`` where it has a full average, else ``fallback``. Point-in-time."""
    def state(sym: str) -> pd.Series:
        px = index_close.get(sym)
        if px is None:
            return pd.Series(np.nan, index=calendar)
        px = px.reindex(calendar)
        sma = px.rolling(sma_window, min_periods=sma_window).mean()
        return (px < sma).where(sma.notna() & px.notna())
    a, b = state(symbol), state(fallback)
    return a.where(a.notna(), b).fillna(False).astype(bool)


def invest_signals(pred: pd.DataFrame, calendar: pd.DatetimeIndex, returns: pd.DataFrame | None, first: int, last: int, *, rebalance: str, k: int,
                   weighting: str, cap: float | None, cov_window: int, tranches: int, spacing: int, off: pd.Series | None = None,
                   equity_share: float = 1.0) -> dict[int, Signal]:
    """``pred`` needs trade_date, instrument_id, score. ``returns`` = daily returns (date x instrument) for the covariance-based schemes."""
    by_date = {d: g for d, g in pred.groupby("trade_date")}
    sched = rebalance_sessions(calendar, rebalance, first, last)
    signals: dict[int, Signal] = {}
    prev: dict[int, float] = {}
    for n, i in enumerate(sched):
        rows = by_date.get(calendar[i])
        if rows is None or rows.empty:
            continue
        ids = select_top_k(pd.Series(rows["score"].to_numpy(), index=rows["instrument_id"].to_numpy()), k)
        window = returns.iloc[max(0, i - cov_window + 1): i + 1] if returns is not None and weighting != "equal" else None
        w = target_weights(weighting, ids, window, cap)
        if off is not None and bool(off.iloc[i]):
            w = w * equity_share
        target = {int(j): float(x) for j, x in w.items()}
        stop_at = sched[n + 1] if n + 1 < len(sched) else last + 1
        cur = prev
        for j in range(tranches):
            idx = i + j * spacing
            if idx >= stop_at or idx > last:
                break
            frac = (j + 1) / tranches
            cur = {a: prev.get(a, 0.0) + (target.get(a, 0.0) - prev.get(a, 0.0)) * frac for a in set(prev) | set(target)}
            signals[idx] = Signal([SignalItem(a, x) for a, x in sorted(cur.items()) if x > 1e-9], full_rebalance=True)
        prev = cur
    return signals
