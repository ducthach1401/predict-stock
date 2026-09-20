"""Weighting schemes for the INVEST portfolios, all long-only and point-in-time (covariance from the trailing window ending at the decision date).

equal | inverse_vol | risk_parity (equal risk contribution) | min_variance (Ledoit-Wolf covariance, weights within [0, cap]) | hrp (hierarchical risk parity,
Lopez de Prado 2016). The per-name cap is applied last (the excess goes to the uncapped names in proportion to their weight); if K x cap < 1 the cap is
raised to 1 / K, because a fully invested portfolio could not otherwise exist."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from predict_stock.backtest.portfolio import apply_cap

SCHEMES = ("equal", "inverse_vol", "risk_parity", "min_variance", "hrp")
MIN_OBS = 60


def clean_returns(window: pd.DataFrame) -> pd.DataFrame:
    """Returns of the selected names; a name with fewer than MIN_OBS observations gets the cross-sectional average return series (so it is
    neither dropped nor given a fake low volatility); other gaps (suspensions) count as 0."""
    w = window.copy()
    thin = w.notna().sum() < MIN_OBS
    if thin.any():
        avg = w.loc[:, ~thin].mean(axis=1) if (~thin).any() else pd.Series(0.0, index=w.index)
        for c in w.columns[thin]:
            w[c] = avg
    return w.fillna(0.0)


def shrunk_covariance(returns: pd.DataFrame) -> np.ndarray:
    return LedoitWolf().fit(returns.to_numpy(float)).covariance_


def inverse_vol(cov: np.ndarray) -> np.ndarray:
    v = 1.0 / np.sqrt(np.diag(cov))
    return v / v.sum()


def risk_parity(cov: np.ndarray, iters: int = 2000, tol: float = 1e-12) -> np.ndarray:
    w = inverse_vol(cov)
    for _ in range(iters):
        rc = w * (cov @ w)
        new = w * (rc.mean() / rc) ** 0.5
        new /= new.sum()
        if np.abs(new - w).max() < tol:
            w = new
            break
        w = new
    return w


def min_variance(cov: np.ndarray, cap: float | None) -> np.ndarray:
    n = len(cov)
    hi = 1.0 if cap is None else max(cap, 1.0 / n)
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1.0 / n), jac=lambda w: 2 * cov @ w, bounds=[(0.0, hi)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}], method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0.0, hi)
    return w / w.sum()


def hrp(cov: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(sd, sd), -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)
    order = list(leaves_list(linkage(squareform(dist, checks=False), method="single")))
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [c[i:j] for c in clusters for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for k in range(0, len(clusters), 2):
            a, b = clusters[k], clusters[k + 1]
            va, vb = _cluster_var(cov, a), _cluster_var(cov, b)
            alpha = 1.0 - va / (va + vb)
            w[a] *= alpha
            w[b] *= 1.0 - alpha
    out = np.zeros(len(cov))
    out[w.index.to_numpy()] = w.to_numpy()
    return out / out.sum()


def _cluster_var(cov: np.ndarray, idx: list[int]) -> float:
    sub = cov[np.ix_(idx, idx)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def target_weights(scheme: str, ids: list[int], window: pd.DataFrame | None, cap: float | None) -> pd.Series:
    """Fully invested weights of ``ids`` (before any cash overlay). ``window`` = trailing daily returns with those ids as columns."""
    n = len(ids)
    if n == 0:
        return pd.Series(dtype=float)
    if scheme not in SCHEMES:
        raise ValueError(f"unknown weighting scheme {scheme!r}")
    if scheme == "equal" or n == 1 or window is None or len(window) < MIN_OBS:          # too little history for a covariance: equal weights
        w = np.full(n, 1.0 / n)
    else:
        cov = shrunk_covariance(clean_returns(window[ids]))
        var = np.diag(cov).copy()
        if not np.isfinite(var).all() or (var <= 0).all():
            w = np.full(n, 1.0 / n)
        else:
            var[var <= 0] = var[var > 0].mean()                                       # e.g. a name that did not trade in the whole window
            cov = cov + np.diag(var - np.diag(cov))
            w = {"inverse_vol": inverse_vol, "risk_parity": risk_parity, "hrp": hrp}.get(scheme, lambda c: min_variance(c, cap))(cov)
            if not np.isfinite(w).all() or w.sum() <= 0:
                w = np.full(n, 1.0 / n)
    out = pd.Series(w, index=ids, dtype=float)
    if scheme == "min_variance":
        return out                                                              # the cap is already inside the optimisation
    return apply_cap(out, max(cap, 1.0 / n) if cap is not None else None)
