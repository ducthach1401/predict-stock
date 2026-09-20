"""Ranking and quantile metrics of the SWING model.

Rank IC = the cross-sectional Spearman correlation between the score at the close of day t and the realised forward return, one value per
day, then summarised over days. Consecutive days share most of their forward window (h = 5 sessions), so the daily ICs are autocorrelated: the
t-statistic uses ``n_days / h`` as the effective number of observations (conservative, no model of the autocorrelation)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_rank_ic(frame: pd.DataFrame, score: str, target: str, min_names: int = 8) -> pd.Series:
    """One Spearman correlation per ``trade_date`` between ``score`` and ``target`` (rows with a NaN in either are dropped; days with fewer
    than ``min_names`` names are skipped)."""
    df = frame[["trade_date", score, target]].dropna()
    r = df.groupby("trade_date")[[score, target]].rank()
    r["trade_date"] = df["trade_date"]
    out = {}
    for d, g in r.groupby("trade_date"):
        if len(g) >= min_names and g[score].nunique() > 1 and g[target].nunique() > 1:
            out[d] = float(np.corrcoef(g[score], g[target])[0, 1])
    return pd.Series(out, dtype=float, name=f"ic_{score}")


def ic_summary(ic: pd.Series, horizon: int = 5) -> dict:
    ic = ic.dropna()
    n = len(ic)
    if n < 2:
        return {"n_days": n, "mean": float("nan"), "std": float("nan"), "icir": float("nan"), "t_stat": float("nan"), "hit_rate": float("nan")}
    mean, std = float(ic.mean()), float(ic.std(ddof=1))
    n_eff = max(n / horizon, 1.0)
    return {"n_days": n, "mean": mean, "std": std, "icir": mean / std if std > 0 else float("nan"),
            "t_stat": mean / (std / np.sqrt(n_eff)) if std > 0 else float("nan"), "hit_rate": float((ic > 0).mean())}


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def quantile_report(pred: pd.DataFrame, target: str, quantiles=(0.1, 0.5, 0.9)) -> dict:
    """Coverage of each quantile and of the q10-q90 interval, and the pinball loss against the unconditional quantile of the same sample
    (that constant sees the whole sample, so it is a slightly optimistic yardstick for the model to beat)."""
    ok = pred[target].notna()
    y = pred.loc[ok, target].to_numpy(float)
    out = {"n": int(ok.sum()), "coverage": {}, "pinball": {}, "pinball_constant": {}}
    for a in quantiles:
        col = f"q{int(round(a * 100))}"
        qv = pred.loc[ok, col].to_numpy(float)
        out["coverage"][col] = float(np.mean(y <= qv))
        out["pinball"][col] = pinball(y, qv, a)
        out["pinball_constant"][col] = pinball(y, np.full_like(y, np.quantile(y, a)), a)
    lo, hi = f"q{int(round(quantiles[0] * 100))}", f"q{int(round(quantiles[-1] * 100))}"
    out["interval_coverage"] = float(np.mean((y >= pred.loc[ok, lo]) & (y <= pred.loc[ok, hi])))
    out["nominal_interval"] = quantiles[-1] - quantiles[0]
    out["mean_interval_width"] = float((pred.loc[ok, hi] - pred.loc[ok, lo]).mean())
    return out


def holding_report(pred: pd.DataFrame, realised: str = "tb_time", buckets: int = 5) -> dict:
    """Predicted holding time against what happened, by bucket of the predicted median: n, mean predicted median / p75, realised median / p75."""
    ok = pred[realised].notna() & pred["hold_median"].notna()
    df = pred.loc[ok]
    key = pd.qcut(df["hold_median"].rank(method="first"), min(buckets, df["hold_median"].nunique()), labels=False)
    rows = []
    for b, g in df.groupby(key):
        rows.append({"bucket": int(b), "n": int(len(g)), "pred_median": float(g["hold_median"].mean()), "pred_p75": float(g["hold_p75"].mean()),
                     "realised_median": float(g[realised].median()), "realised_p75": float(g[realised].quantile(0.75))})
    return {"n": int(len(df)), "mae_median": float((df["hold_median"] - df[realised]).abs().mean()),
            "mae_constant": float((df[realised].median() - df[realised]).abs().mean()), "buckets": rows}
