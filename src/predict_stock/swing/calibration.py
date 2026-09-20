"""Probability calibration and its measurement.

Isotonic regression is stored as its two threshold arrays and applied with ``np.interp`` (monotone, clipped at both ends), so a saved
model needs no scikit-learn object. Platt scaling is a logistic regression on the log-odds of the raw probability.
ECE uses equal-FREQUENCY bins: the probabilities live in a narrow band (the event rate is ~30%), equal-width bins would be mostly empty."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_isotonic(p_raw: np.ndarray, y: np.ndarray, min_bin: int = 1) -> dict:
    """Isotonic fit. With ``min_bin`` > 1 the points are first pooled into equal-count bins of at least that many rows (weighted by the count):
    isotonic regression on raw points reaches probability 1 (or 0) wherever a handful of the highest (lowest) scores happen to be all events (non-events)."""
    x, t = np.asarray(p_raw, float), np.asarray(y, float)
    w = None
    if min_bin > 1 and len(x) >= 2 * min_bin:
        order = np.argsort(x, kind="stable")
        groups = np.array_split(order, len(x) // min_bin)
        x, t, w = (np.array([x[g].mean() for g in groups]), np.array([t[g].mean() for g in groups]), np.array([len(g) for g in groups], float))
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(x, t, sample_weight=w)
    return {"kind": "isotonic", "x": [float(v) for v in iso.X_thresholds_], "y": [float(v) for v in iso.y_thresholds_]}


def fit_platt(p_raw: np.ndarray, y: np.ndarray) -> dict:
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(_logit(p_raw).reshape(-1, 1), np.asarray(y, int))
    return {"kind": "platt", "a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}


def apply_calibrator(cal: dict, p_raw: np.ndarray) -> np.ndarray:
    p_raw = np.asarray(p_raw, float)
    if cal["kind"] == "isotonic":
        return np.interp(p_raw, cal["x"], cal["y"])
    if cal["kind"] == "platt":
        return 1.0 / (1.0 + np.exp(-(cal["a"] * _logit(p_raw) + cal["b"])))
    raise ValueError(f"unknown calibrator {cal['kind']!r}")


def brier(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def log_loss(y, p) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    """Equal-frequency bins (ties kept together) of the predicted probability: n, mean predicted, observed frequency (the calibration curve)."""
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)}).dropna()
    if df.empty:
        return pd.DataFrame(columns=["bin", "n", "mean_pred", "observed"])
    edges = np.unique(np.quantile(df["p"], np.linspace(0, 1, min(bins, len(df)) + 1)[1:-1]))      # tied probabilities always share a bin
    df["bin"] = np.searchsorted(edges, df["p"].to_numpy(), side="right")
    g = df.groupby("bin")
    return pd.DataFrame({"n": g.size(), "mean_pred": g["p"].mean(), "observed": g["y"].mean()}).reset_index()


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error: the count-weighted mean |observed - predicted| over the reliability bins."""
    r = reliability(y, p, bins)
    return float((r["n"] * (r["observed"] - r["mean_pred"]).abs()).sum() / r["n"].sum()) if len(r) else float("nan")


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, float)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def summary(y, p, bins: int = 10) -> dict:
    y = np.asarray(y, float)
    return {"n": int(len(y)), "base_rate": float(np.mean(y)), "mean_pred": float(np.mean(p)), "brier": brier(y, p), "log_loss": log_loss(y, p),
            "ece": ece(y, p, bins), "auc": auc(y, p), "brier_constant": brier(y, np.full_like(y, y.mean())), "log_loss_constant": log_loss(y, np.full_like(y, y.mean()))}
