"""Baselines: the yardsticks any model must beat AFTER costs. Signals use only the dataset row of the decision date (features
known at that close) and become orders for the next open (the engine guarantees that).

portfolio baselines (top-K by a score, rebalanced on a schedule, full rebalance through the engine so lots, ticks, bands, T+2 and costs apply)
  equal_weight      every universe member, equal weight, monthly
  random_weekly     K random members, weekly   } noise floor: what luck plus the same turnover and costs produce
  random_monthly    K random members, monthly  }
  mom_short         top K by 10-session return (short-term momentum), weekly
  mean_reversion    K most oversold by the 20-session z-score, weekly
  mom_long          top K by the mean rank of 6- and 12-month momentum (skipping the last month), monthly
  mom_long_invvol   the same, weights proportional to 1 / 126-session volatility
index benchmarks (buy & hold, not tradable directly): VNINDEX and VN30 (from 2020-05-11)

The scores are cross-sectional facts already in the Phase 3 datasets, so nothing is refitted and nothing is tuned: a baseline
has no parameter chosen by looking at results (K = 10 and the schedules are conventions, not optimised).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from predict_stock.backtest.engine import Signal, SignalItem
from predict_stock.backtest.portfolio import build_weights


@dataclass(frozen=True)
class BaselineSpec:
    key: str
    title: str
    description: str
    kind: str = "portfolio"                 # portfolio | index
    dataset: str = "swing"                  # which dataset supplies the columns: swing | invest
    score: str = "const"                    # a column, "mom_long" (mean of two ranks) or "random"
    ascending: bool = False
    k: int | None = None                    # None = every member
    weighting: str = "equal"
    vol_col: str | None = None
    rebalance: str = "monthly"              # weekly | monthly
    seed: int | None = None
    index_symbol: str | None = None


BASELINES: dict[str, BaselineSpec] = {b.key: b for b in [
    BaselineSpec("equal_weight", "Equal-weight universe", "All members, equal weights, rebalanced monthly.", k=None, rebalance="monthly"),
    BaselineSpec("random_weekly", "Random top-K, weekly", "K random members each week (seeded): the noise floor for high-turnover strategies.",
                 score="random", k=10, rebalance="weekly", seed=20240229),
    BaselineSpec("random_monthly", "Random top-K, monthly", "K random members each month (seeded): the noise floor for low-turnover strategies.",
                 score="random", k=10, rebalance="monthly", seed=20240301),
    BaselineSpec("mom_short", "Short-term momentum", "Top K by the 10-session return, weekly.", score="ret_10", k=10, rebalance="weekly"),
    BaselineSpec("mean_reversion", "Short-term mean reversion", "K most oversold by the 20-session z-score, weekly.", score="zscore_20", ascending=True, k=10, rebalance="weekly"),
    BaselineSpec("mom_long", "Momentum 6-12 months", "Top K by the mean rank of 6- and 12-month momentum (skipping the last month), monthly.",
                 dataset="invest", score="mom_long", k=10, rebalance="monthly"),
    BaselineSpec("mom_long_invvol", "Momentum 6-12 months, inverse-vol", "As mom_long, weights proportional to 1 / 126-session volatility.",
                 dataset="invest", score="mom_long", k=10, weighting="inverse_vol", vol_col="vol_126", rebalance="monthly"),
    BaselineSpec("bh_vnindex", "Buy & hold VNINDEX", "The index bought once and held (price index; not tradable directly).", kind="index", index_symbol="VNINDEX"),
    BaselineSpec("bh_vn30", "Buy & hold VN30", "The index bought once and held (price index; exists from 2020-05-11).", kind="index", index_symbol="VN30"),
]}
PORTFOLIO_KEYS = [k for k, b in BASELINES.items() if b.kind == "portfolio"]
INDEX_KEYS = [k for k, b in BASELINES.items() if b.kind == "index"]


def rebalance_sessions(calendar: pd.DatetimeIndex, freq: str, first: int, last: int) -> list[int]:
    """Session indices in [first, last] on which a rebalance decision is made: the first session of each ISO week / month / calendar quarter."""
    out, prev = [], None
    for i in range(first, last + 1):
        d = calendar[i]
        if freq == "weekly":
            key = (d.isocalendar().year, d.isocalendar().week)
        elif freq == "quarterly":
            key = (d.year, (d.month - 1) // 3)
        else:
            key = (d.year, d.month)
        if key != prev:
            out.append(i)
            prev = key
    return out


def scores_for(spec: BaselineSpec, rows: pd.DataFrame, session_index: int) -> tuple[pd.Series, pd.Series | None]:
    """(score by instrument_id, volatility by instrument_id or None) for one decision date."""
    idx = rows["instrument_id"].to_numpy()
    if spec.score == "random":
        rng = np.random.default_rng([spec.seed or 0, session_index])
        score = pd.Series(rng.random(len(rows)), index=idx)
    elif spec.score == "mom_long":
        score = pd.Series(((rows["mom_6m_csrank"] + rows["mom_12m_csrank"]) / 2).to_numpy(), index=idx)
    elif spec.score == "const":
        score = pd.Series(1.0, index=idx)
    else:
        score = pd.Series(rows[spec.score].to_numpy(), index=idx)
    vol = pd.Series(rows[spec.vol_col].to_numpy(), index=idx) if spec.vol_col else None
    return score, vol


def make_signals(spec: BaselineSpec, frame: pd.DataFrame, calendar: pd.DatetimeIndex, first: int, last: int, max_weight: float | None) -> dict[int, Signal]:
    """Signals at the CLOSE of each rebalance session, from that date's dataset rows (members with a bar and enough history)."""
    by_date = {d: g for d, g in frame.groupby("trade_date")}
    signals: dict[int, Signal] = {}
    for i in rebalance_sessions(calendar, spec.rebalance, first, last):
        rows = by_date.get(calendar[i])
        if rows is None or rows.empty:
            continue
        score, vol = scores_for(spec, rows, i)
        k = spec.k or len(score.dropna())
        w = build_weights(score, k, scheme=spec.weighting, vol=vol, cap=max_weight, ascending=spec.ascending)
        signals[i] = Signal([SignalItem(int(iid), float(x)) for iid, x in w.items()], full_rebalance=True)
    return signals
