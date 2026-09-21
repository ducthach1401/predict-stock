"""Reference distribution of the training features (kept with the model) and the drift measures against it: PSI and the KS distance.

The profile stores, per feature, the quantiles of the TRAINING rows (101 points by default) plus mean / std / share of missing values. PSI compares the share of recent
rows in the training deciles with the expected 10% each (ties collapse the deciles); KS compares the recent empirical distribution with the one the quantiles describe.
Both use features only: no return, no label, no outcome, so they can be computed at any time, also over the period the research keeps out of reach."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4


def profile(frame: pd.DataFrame, features: list[str], points: int = 101) -> dict:
    out = {}
    grid = np.linspace(0, 1, points)
    for f in features:
        v = frame[f].to_numpy(float)
        ok = v[np.isfinite(v)]
        if len(ok) < 50:
            continue
        out[f] = {"q": [float(x) for x in np.quantile(ok, grid)], "mean": float(ok.mean()), "std": float(ok.std()), "n": int(len(ok)), "nan_share": float(1 - len(ok) / len(v))}
    return out


def psi(ref: dict, values: np.ndarray, bins: int = 10) -> float:
    """Population stability index of ``values`` against a reference profile entry. 0 = identical; < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return float("nan")
    q = np.asarray(ref["q"], float)
    edges = np.unique(np.quantile(q, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                                                     # a (nearly) constant feature: a shift is any change of the value
        return float(abs(v.mean() - ref["mean"]) > 1e-9 + 1e-3 * max(ref["std"], 1e-9)) * 1.0
    inner = edges[1:-1]
    expected = np.diff(np.searchsorted(np.sort(q), np.r_[-np.inf, inner, np.inf], side="right")) / len(q)
    actual = np.bincount(np.searchsorted(inner, v, side="right"), minlength=len(inner) + 1) / len(v)
    expected, actual = np.maximum(expected, EPS), np.maximum(actual, EPS)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def ks(ref: dict, values: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance between the recent values and the training distribution described by the profile's quantiles."""
    v = np.sort(np.asarray(values, float)[np.isfinite(values)])
    if len(v) < 20:
        return float("nan")
    q = np.asarray(ref["q"], float)
    levels = np.linspace(0, 1, len(q))
    keep = np.r_[True, np.diff(q) > 0]                                     # np.interp needs strictly increasing x
    f_ref = np.interp(v, q[keep], levels[keep], left=0.0, right=1.0)
    f_cur = np.arange(1, len(v) + 1) / len(v)
    return float(max(np.max(np.abs(f_cur - f_ref)), np.max(np.abs(f_cur - 1.0 / len(v) - f_ref))))


def drift(profile_: dict, frame: pd.DataFrame) -> dict[str, dict]:
    """{feature: {psi, ks}} of the rows of ``frame`` against the profile."""
    return {f: {"psi": psi(p, frame[f].to_numpy(float)), "ks": ks(p, frame[f].to_numpy(float))} for f, p in profile_.items() if f in frame.columns}


def window_baseline(frame: pd.DataFrame, profile_: dict, window: int, step: int = 10, q: float = 0.95) -> dict:
    """Adds to every feature of ``profile_`` its own yardstick: the ``q`` quantile of the PSI and of the KS distance of every ``window``-session slice of the TRAINING period (every ``step``
    sessions) against the whole training profile. The 50 stocks share one market state, so a 60-session slice is far less than 3000 independent rows: a level feature such as a 12-month
    momentum moves as a block with the market and its PSI against pooled history is large even in ordinary times. A shift only means something beyond what the training period itself did."""
    d = pd.to_datetime(frame["trade_date"])
    sessions = np.sort(d.unique())
    psis: dict[str, list[float]] = {f: [] for f in profile_}
    kss: dict[str, list[float]] = {f: [] for f in profile_}
    for i in range(window - 1, len(sessions), max(step, 1)):
        rows = frame[(d >= sessions[i - window + 1]) & (d <= sessions[i])]
        for f, r in drift(profile_, rows).items():
            if np.isfinite(r["psi"]):
                psis[f].append(r["psi"])
            if np.isfinite(r["ks"]):
                kss[f].append(r["ks"])
    for f, p in profile_.items():
        p["ref_window"] = int(window)
        p["psi_ref"] = float(np.quantile(psis[f], q)) if len(psis[f]) >= 5 else None
        p["ks_ref"] = float(np.quantile(kss[f], q)) if len(kss[f]) >= 5 else None
        p["ref_slices"] = int(len(psis[f]))
    return profile_
