"""Performance metrics from an equity curve and the trades.

Conventions (all stated, none hidden): 252 sessions a year; returns are simple daily returns of the equity curve; CAGR uses
calendar days / 365.25; the risk-free rate is a parameter (default 0, so Sharpe here is a plain mean/std ratio); Sortino uses
the root mean square of negative returns (target 0) over ALL days; MDD is the deepest peak-to-trough fall of the curve;
turnover = (buys + sells) / 2 / average equity / years; a "trade" is a closed position lifecycle (first buy to full exit),
positions still open at the end are counted separately and excluded from win rate / profit factor / expectancy."""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = 252


def _returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min()) if len(equity) else float("nan")


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else float("nan")


def sharpe(r: pd.Series, rf_annual: float = 0.0) -> float:
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float((r.mean() - rf_annual / PERIODS) / r.std(ddof=1) * np.sqrt(PERIODS))


def sortino(r: pd.Series, rf_annual: float = 0.0) -> float:
    if len(r) < 2:
        return float("nan")
    excess = r - rf_annual / PERIODS
    dd = np.sqrt((np.minimum(excess, 0.0) ** 2).mean())
    return float(excess.mean() / dd * np.sqrt(PERIODS)) if dd > 0 else float("nan")


def trade_stats(trips: pd.DataFrame | None) -> dict:
    if trips is None or len(trips) == 0:
        return {"trades": 0, "win_rate": float("nan"), "profit_factor": float("nan"), "expectancy": float("nan"), "avg_holding_sessions": float("nan"),
                "avg_win": float("nan"), "avg_loss": float("nan")}
    wins, losses = trips[trips["pnl"] > 0], trips[trips["pnl"] < 0]
    gp, gl = float(wins["pnl"].sum()), float(-losses["pnl"].sum())
    return {"trades": int(len(trips)), "win_rate": float((trips["pnl"] > 0).mean()),
            "profit_factor": gp / gl if gl > 0 else (float("inf") if gp > 0 else float("nan")),
            "expectancy": float(trips["net_return"].mean()),                       # average net return per closed trade
            "avg_holding_sessions": float(trips["sessions"].mean()),
            "avg_win": float(wins["net_return"].mean()) if len(wins) else float("nan"),
            "avg_loss": float(losses["net_return"].mean()) if len(losses) else float("nan")}


def compute_metrics(equity: pd.Series, *, trips: pd.DataFrame | None = None, fills: pd.DataFrame | None = None,
                    exposure: pd.Series | None = None, rf_annual: float = 0.0) -> dict:
    r = _returns(equity)
    years = (equity.index[-1] - equity.index[0]).days / 365.25 if len(equity) > 1 else float("nan")
    mdd = max_drawdown(equity)
    c = cagr(equity)
    out = {"start": str(equity.index[0].date()), "end": str(equity.index[-1].date()), "sessions": int(len(equity)),
           "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1), "cagr": c, "volatility": float(r.std(ddof=1) * np.sqrt(PERIODS)) if len(r) > 1 else float("nan"),
           "sharpe": sharpe(r, rf_annual), "sortino": sortino(r, rf_annual), "max_drawdown": mdd,
           "calmar": float(c / abs(mdd)) if mdd and mdd < 0 and np.isfinite(c) else float("nan")}
    if fills is not None and len(fills) and years and years > 0:
        traded = float(fills["value"].sum())
        out["turnover_annual"] = traded / 2 / float(equity.mean()) / years
        out["costs_paid"] = float(fills["fee"].sum() + fills["tax"].sum())
        out["slippage_paid"] = float(fills["slippage_cost"].sum())
    elif fills is not None:
        out["turnover_annual"], out["costs_paid"], out["slippage_paid"] = 0.0, 0.0, 0.0
    if exposure is not None:
        out["avg_exposure"] = float(exposure.mean())
    if trips is not None:
        out.update(trade_stats(trips))
    return out


# ---- sub-periods ---------------------------------------------------------------------------------------------------
def classify_regimes(index_close: pd.Series, window: int = 126, threshold: float = 0.10) -> pd.Series:
    """Label each session 'up' / 'down' / 'sideways' by the benchmark's return over consecutive, non-overlapping windows of
    ``window`` sessions (a short final window is merged into the previous one). This is an EX-POST description used only
    to REPORT results by market condition; nothing in a strategy may use it."""
    n = len(index_close)
    bounds = list(range(0, n, window))
    if len(bounds) > 1 and n - bounds[-1] < window // 2:
        bounds.pop()
    bounds.append(n)
    labels = pd.Series(index=index_close.index, dtype=object)
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = index_close.iloc[a:b]
        ret = seg.iloc[-1] / seg.iloc[0] - 1
        labels.iloc[a:b] = "up" if ret > threshold else ("down" if ret < -threshold else "sideways")
    return labels


def _annualised(r: pd.Series) -> float:
    return float((1 + r).prod() ** (PERIODS / len(r)) - 1) if len(r) else float("nan")


def slice_metrics(equity: pd.Series, mask: pd.Series, rf_annual: float = 0.0) -> dict:
    """Metrics over the sessions where ``mask`` is True, computed on those sessions' returns only (non-contiguous days are
    simply concatenated), so a regime's figures are those of the days spent in it."""
    r = _returns(equity)
    r = r[mask.reindex(r.index).fillna(False).to_numpy(bool)]
    if len(r) < 2:
        return {"sessions": int(len(r))}
    curve = (1 + r).cumprod()
    return {"sessions": int(len(r)), "annualised_return": _annualised(r), "sharpe": sharpe(r, rf_annual), "max_drawdown": float((curve / curve.cummax() - 1).min()),
            "hit_rate_days": float((r > 0).mean())}


def by_regime(equity: pd.Series, regimes: pd.Series, rf_annual: float = 0.0) -> dict:
    return {k: slice_metrics(equity, regimes == k, rf_annual) for k in ("up", "sideways", "down") if (regimes == k).any()}


def by_year(equity: pd.Series, rf_annual: float = 0.0) -> dict:
    out = {}
    for y in sorted(set(equity.index.year)):
        mask = pd.Series(equity.index.year == y, index=equity.index)
        m = slice_metrics(equity, mask, rf_annual)
        seg = equity[equity.index.year == y]
        m["return"] = float(seg.iloc[-1] / seg.iloc[0] - 1) if len(seg) > 1 else float("nan")
        out[int(y)] = m
    return out
