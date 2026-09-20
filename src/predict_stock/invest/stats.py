"""Small-sample statistics for the INVEST evaluation: stationary block bootstrap (confidence intervals for Sharpe / CAGR / drawdown and for their
DIFFERENCE against a benchmark on the same resampled days), White's reality check over several candidates, rolling-window outperformance, a DCA
simulation from a return series, and the information content of the thesis-break flags.

Why the bootstrap is by blocks: daily returns of a portfolio are autocorrelated (overlapping holdings, volatility clusters); an i.i.d. resample would
give intervals that are too narrow. Stationary bootstrap (Politis-Romano): random block lengths with mean ``block``, circular."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def bootstrap_indices(n_obs: int, resamples: int, block: int, seed: int) -> np.ndarray:
    """(resamples, n_obs) integer indices of a circular stationary bootstrap with mean block length ``block``. Deterministic for a seed."""
    rng = np.random.default_rng(seed)
    start = rng.integers(0, n_obs, size=(resamples, n_obs))
    restart = rng.random((resamples, n_obs)) < 1.0 / max(block, 1)
    restart[:, 0] = True
    idx = np.empty((resamples, n_obs), dtype=np.int64)
    idx[:, 0] = start[:, 0]
    for t in range(1, n_obs):
        idx[:, t] = np.where(restart[:, t], start[:, t], (idx[:, t - 1] + 1) % n_obs)
    return idx


def sharpe_rows(r: np.ndarray) -> np.ndarray:
    sd = r.std(axis=-1, ddof=1)
    return np.where(sd > 0, r.mean(axis=-1) / np.where(sd > 0, sd, 1.0) * np.sqrt(TRADING_DAYS), np.nan)


def cagr_rows(r: np.ndarray) -> np.ndarray:
    return np.exp(np.log1p(r).sum(axis=-1) * TRADING_DAYS / r.shape[-1]) - 1.0


def mdd_rows(r: np.ndarray) -> np.ndarray:
    curve = np.cumprod(1.0 + r, axis=-1)
    return (curve / np.maximum.accumulate(curve, axis=-1) - 1.0).min(axis=-1)


METRICS = {"sharpe": sharpe_rows, "cagr": cagr_rows, "max_drawdown": mdd_rows}


def _ci(values: np.ndarray, level: float) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.nanquantile(values, a)), float(np.nanquantile(values, 1 - a))


def paired_bootstrap(returns: pd.Series, bench: pd.Series | None, *, resamples: int, block: int, level: float, seed: int) -> dict:
    """Point estimate and interval of each metric for ``returns``; if ``bench`` is given also for the difference (same resampled days).
    The two series are aligned on their common dates."""
    if bench is not None:
        both = pd.concat([returns, bench], axis=1, join="inner").dropna()
        r, b = both.iloc[:, 0].to_numpy(float), both.iloc[:, 1].to_numpy(float)
    else:
        r, b = returns.dropna().to_numpy(float), None
    idx = bootstrap_indices(len(r), resamples, block, seed)
    out = {"n_days": int(len(r)), "resamples": resamples, "block": block, "level": level}
    for name, f in METRICS.items():
        boot = f(r[idx])
        out[name] = {"point": float(f(r)), "lo": _ci(boot, level)[0], "hi": _ci(boot, level)[1]}
        if b is not None:
            d = boot - f(b[idx])
            out[f"{name}_diff"] = {"point": float(f(r) - f(b)), "lo": _ci(d, level)[0], "hi": _ci(d, level)[1], "share_positive": float(np.nanmean(d > 0))}
    return out


def reality_check(candidates: dict[str, pd.Series], bench: pd.Series, *, resamples: int, block: int, seed: int) -> dict:
    """White's reality check on the Sharpe difference against ``bench``. H0: no candidate is better than the benchmark. Statistic = the largest observed
    difference; its null distribution = the maximum over candidates of the resampled differences centred on their observed values. The p-value corrects
    for having looked at several candidates. Also returns each candidate's own (uncorrected) share of resamples with a positive difference."""
    names = list(candidates)
    frame = pd.concat([bench.rename("__bench__")] + [candidates[n].rename(n) for n in names], axis=1, join="inner").dropna()
    b = frame["__bench__"].to_numpy(float)
    idx = bootstrap_indices(len(frame), resamples, block, seed)
    obs = np.array([sharpe_rows(frame[n].to_numpy(float)) - sharpe_rows(b) for n in names])
    boot = np.array([sharpe_rows(frame[n].to_numpy(float)[idx]) - sharpe_rows(b[idx]) for n in names])           # (candidates, resamples)
    stat = obs.max()
    null_max = (boot - obs[:, None]).max(axis=0)
    return {"candidates": names, "observed": {n: float(o) for n, o in zip(names, obs)}, "best": names[int(obs.argmax())], "statistic": float(stat),
            "p_value": float((null_max >= stat).mean()), "n_days": int(len(frame)), "resamples": resamples, "block": block}


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def rolling_outperformance(equity: pd.Series, bench: pd.Series, window: int) -> dict:
    """Total return over every rolling window of ``window`` sessions, for the strategy and the benchmark, and how often the strategy is ahead."""
    both = pd.concat([equity.rename("s"), bench.rename("b")], axis=1, join="inner").dropna()
    if len(both) <= window:
        return {"window": window, "n_windows": 0}
    ret = both / both.shift(window) - 1.0
    ret = ret.dropna()
    ex = ret["s"] - ret["b"]
    return {"window": window, "n_windows": int(len(ret)), "share_ahead": float((ex > 0).mean()), "median_excess": float(ex.median()), "worst_excess": float(ex.min()),
            "best_excess": float(ex.max()), "median_return": float(ret["s"].median()), "worst_return": float(ret["s"].min()), "share_negative": float((ret["s"] < 0).mean())}


def dca(equity: pd.Series, monthly: float, initial: float = 0.0) -> dict:
    """A recurring contribution on the first session of each month, invested at the strategy's daily return (an approximation: no lot rounding on the
    marginal contribution; costs are already inside the return series). Gives the final value, the amount contributed and the money-weighted annual return."""
    r = equity.pct_change().fillna(0.0)
    months = pd.Series(r.index.to_period("M"), index=r.index)
    first_of_month = months != months.shift(1)
    value, paid, flows = initial, initial, []
    if initial:
        flows.append((0, initial))
    for i, (d, ret) in enumerate(r.items()):
        value *= 1.0 + ret
        if first_of_month.iloc[i]:
            value += monthly
            paid += monthly
            flows.append((i, monthly))
    return {"final_value": float(value), "contributed": float(paid), "gain": float(value - paid), "irr_annual": _irr(flows, float(value), len(r))}


def _irr(flows: list[tuple[int, float]], final: float, n: int) -> float:
    """Annual money-weighted return: the rate at which the contributions grow to ``final`` (bisection on the daily rate)."""
    if not flows or final <= 0:
        return float("nan")
    t = np.array([n - 1 - i for i, _ in flows], float)
    c = np.array([a for _, a in flows], float)
    lo, hi = -0.02, 0.05
    f = lambda rate: (c * (1 + rate) ** t).sum() - final
    if f(lo) * f(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
    return float((1 + (lo + hi) / 2) ** TRADING_DAYS - 1)


def thesis_flags(rows: pd.DataFrame, drawdown_break: float, rs_floor: float) -> pd.DataFrame:
    """Boolean thesis-break conditions from features known at the close: below the 200-day average; relative strength (cross-sectional rank of
    6-month momentum) under ``rs_floor``; drawdown from the 252-session high deeper than ``drawdown_break``."""
    return pd.DataFrame({"below_sma200": rows["sma_ratio_200"] < 0, "weak_relative_strength": rows["mom_6m_csrank"] < rs_floor,
                         "deep_drawdown": rows["dd_252"] < -drawdown_break}, index=rows.index)


def thesis_information(rows: pd.DataFrame, flags: pd.DataFrame, ret_col: str) -> dict:
    """For the names a portfolio would hold: forward return when a flag is raised against when it is not (the value of the warning, not a trading rule)."""
    out = {}
    y = rows[ret_col]
    for c in list(flags.columns) + ["any"]:
        f = flags.any(axis=1) if c == "any" else flags[c]
        ok = y.notna()
        a, b = y[ok & f], y[ok & ~f]
        out[c] = {"n_flagged": int(len(a)), "n_clear": int(len(b)), "mean_flagged": float(a.mean()) if len(a) else None, "mean_clear": float(b.mean()) if len(b) else None,
                  "share_negative_flagged": float((a < 0).mean()) if len(a) else None, "share_negative_clear": float((b < 0).mean()) if len(b) else None}
    return out
