"""Turning the model's predictions into Signals for the Phase 4 engine.

A  (primary, pre-registered)  top-K by the rank score, equal weight, rebalanced weekly, full rebalance - exactly like the baselines, so the
                              comparison is like for like. ``min_prob_multiple`` > 0 keeps only names whose calibrated probability is at least
                              that multiple of the base rate (fewer than K names -> the rest stays in cash).
B  (secondary)                barrier trades: every day the best-ranked names that pass the probability filter and are not already held enter at the
                              next open with a stop / target of ``stop_mult`` / ``target_mult`` ATR (percent of the signal-day close, the same
                              ATR14 as the label) and a time-stop after ``max_hold`` sessions; at most ``max_positions`` at once, equal weight 1 / max_positions.
Signals use only the prediction of the decision date (features known at that close)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from predict_stock.backtest.baselines import rebalance_sessions
from predict_stock.backtest.engine import Signal, SignalItem
from predict_stock.backtest.portfolio import apply_cap, select_top_k


def with_probability_filter(pred: pd.DataFrame, base_rate: float | None, min_prob_multiple: float) -> pd.DataFrame:
    """Score set to NaN where the calibrated probability is below ``min_prob_multiple * base_rate`` (NaN scores are never selected).
    ``base_rate=None`` uses the ``calib_base_rate`` column of ``pred``: the event rate of the validation rows the model's calibrator was fitted on, which is
    what its calibrated probabilities are centred on (using the training rate instead left the strategy idle whenever the validation window was calmer)."""
    if min_prob_multiple <= 0:
        return pred
    keep = pred["proba"] >= min_prob_multiple * (pred["calib_base_rate"] if base_rate is None else base_rate)
    return pred.assign(score=pred["score"].where(keep))


def topk_signals(pred: pd.DataFrame, calendar: pd.DatetimeIndex, first: int, last: int, *, k: int, max_weight: float | None, rebalance: str = "weekly",
                 base_rate: float | None = None, min_prob_multiple: float = 0.0) -> dict[int, Signal]:
    """Weight 1/K per selected name (the same as the baselines when K names are available); fewer selected names leave the rest in cash and
    an empty selection sells everything."""
    by_date = {d: g for d, g in with_probability_filter(pred, base_rate, min_prob_multiple).groupby("trade_date")}
    signals: dict[int, Signal] = {}
    for i in rebalance_sessions(calendar, rebalance, first, last):
        rows = by_date.get(calendar[i])
        if rows is None or rows.empty:
            continue
        ids = select_top_k(pd.Series(rows["score"].to_numpy(), index=rows["instrument_id"].to_numpy()), k)
        w = apply_cap(pd.Series(1.0 / k, index=ids, dtype=float), max_weight) if ids else pd.Series(dtype=float)
        signals[i] = Signal([SignalItem(int(j), float(x)) for j, x in w.items()], full_rebalance=True)
    return signals


def barrier_signals(pred: pd.DataFrame, calendar: pd.DatetimeIndex, first: int, last: int, *, max_positions: int, target_mult: float, stop_mult: float,
                    max_hold: int, base_rate: float | None = None, min_prob_multiple: float = 1.0) -> dict[int, Signal]:
    """``pred`` needs trade_date, instrument_id, score, proba and atr_pct_14."""
    filtered = with_probability_filter(pred, base_rate, min_prob_multiple)
    by_date = {d: g for d, g in filtered.groupby("trade_date")}
    weight = 1.0 / max_positions
    signals: dict[int, Signal] = {}
    for i in range(first, last + 1):
        rows = by_date.get(calendar[i])
        if rows is None or rows.empty:
            continue
        rows = rows[rows["atr_pct_14"].notna() & (rows["atr_pct_14"] > 0)]
        ids = select_top_k(pd.Series(rows["score"].to_numpy(), index=rows["instrument_id"].to_numpy()), max_positions)
        if not ids:
            continue
        atr = pd.Series(rows["atr_pct_14"].to_numpy(), index=rows["instrument_id"].to_numpy())
        items = [SignalItem(int(j), weight, stop_pct=float(min(stop_mult * atr[j], 0.5)), target_pct=float(target_mult * atr[j]), max_hold=max_hold,
                            only_if_flat=True) for j in ids]
        signals[i] = Signal(items, full_rebalance=False)
    return signals
