"""The INVEST candidates. Each maps a frame of features known at a close to a ranking score and gives additive per-feature contributions:

  factor      rule-based: mean of cross-sectional ranks with fixed signs; nothing is fitted; contribution of a factor = (its rank - 0.5) / n_factors
  ridge       standardised features, target = cross-sectional rank of the forward return at the horizon; contribution = coefficient x standardised value
  elasticnet  the same with an L1 part (sparse)
  lgbm        4 leaves, depth 2: TreeSHAP contributions

plus a ScenarioModel: q10/q50/q90 of the forward return at the horizon (bear / base / bull) from three shallow quantile boosters.
Everything serialises to plain JSON (boosters as LightGBM model strings) with a byte-stable encoding, so the same data and config give the same sha256."""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge

from predict_stock.config import InvestConfig

FORMAT_VERSION = 1


class InvestModelError(RuntimeError):
    pass


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Non-key, non-label columns that are not constant in ``frame`` are decided by the caller; this lists all candidates."""
    return [c for c in frame.columns if c not in ("trade_date", "instrument_id", "label_end_max") and not c.startswith(("fwd_", "tb_"))]


def _x(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise InvestModelError(f"missing feature columns: {missing}")
    return frame[cols].to_numpy(float)


# ---- factor score ---------------------------------------------------------------------------------------------------------------------
@dataclass
class FactorModel:
    factors: list[tuple[str, int]]
    name: str = "factor"

    def _terms(self, frame: pd.DataFrame) -> np.ndarray:
        cols = []
        for c, sign in self.factors:
            v = frame[c].to_numpy(float)
            cols.append(v if sign > 0 else 1.0 - v)
        return np.column_stack(cols)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        t = self._terms(frame)
        ok = np.isfinite(t).sum(axis=1) >= max(1, (len(self.factors) + 1) // 2)         # at least half of the factors must exist
        out = np.full(len(frame), np.nan)
        out[ok] = np.nanmean(t[ok], axis=1)
        return out

    def contributions(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        t = np.nan_to_num(self._terms(frame), nan=0.5)
        return (t - 0.5) / len(self.factors), [f"{'-' if s < 0 else ''}{c}" for c, s in self.factors]                  # the inputs ARE the cross-sectional ranks

    def inputs(self) -> list[str]:
        return [c for c, _ in self.factors]

    def to_dict(self) -> dict:
        return {"kind": "factor", "factors": [[c, s] for c, s in self.factors]}

    @classmethod
    def from_dict(cls, d: dict) -> "FactorModel":
        return cls([(c, int(s)) for c, s in d["factors"]])


# ---- linear models ------------------------------------------------------------------------------------------------------------------------
@dataclass
class LinearModel:
    kind: str                       # ridge | elasticnet
    features: list[str]
    median: list[float]
    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float
    hyper: dict
    name: str = ""

    def _z(self, frame: pd.DataFrame) -> np.ndarray:
        x = _x(frame, self.features)
        x = np.where(np.isfinite(x), x, np.asarray(self.median))
        return (x - np.asarray(self.mean)) / np.asarray(self.scale)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self._z(frame) @ np.asarray(self.coef) + self.intercept

    def contributions(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        return self._z(frame) * np.asarray(self.coef), list(self.features)

    def inputs(self) -> list[str]:
        return list(self.features)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "features": self.features, "median": self.median, "mean": self.mean, "scale": self.scale, "coef": self.coef,
                "intercept": self.intercept, "hyper": self.hyper}

    @classmethod
    def from_dict(cls, d: dict) -> "LinearModel":
        return cls(d["kind"], d["features"], d["median"], d["mean"], d["scale"], d["coef"], d["intercept"], d["hyper"], d["kind"])


def fit_linear(kind: str, fit: pd.DataFrame, target: str, hyper: dict, seed: int = 0) -> LinearModel:
    feats = [c for c in feature_columns(fit) if fit[c].nunique(dropna=True) > 1]           # constant columns carry nothing
    ok = fit[target].notna().to_numpy()
    if ok.sum() < 500:
        raise InvestModelError(f"too few labelled rows to fit: {int(ok.sum())}")
    X = fit.loc[ok, feats].to_numpy(float)
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isfinite(X), X, med)
    mean, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mean) / sd
    y = fit.loc[ok, target].to_numpy(float)
    est = Ridge(alpha=hyper["alpha"], random_state=seed) if kind == "ridge" else ElasticNet(alpha=hyper["alpha"], l1_ratio=hyper["l1_ratio"], max_iter=20000, random_state=seed)
    est.fit(Z, y)
    return LinearModel(kind, feats, [float(v) for v in med], [float(v) for v in mean], [float(v) for v in sd], [float(v) for v in est.coef_], float(est.intercept_), dict(hyper), kind)


# ---- shallow LightGBM ---------------------------------------------------------------------------------------------------------------------
def _train(cfg: InvestConfig, obj: dict, X, y, Xv, yv) -> lgb.Booster:
    params = {"verbosity": -1, "seed": cfg.seed, "deterministic": True, "force_row_wise": True, "num_threads": cfg.num_threads, **cfg.lgbm, **obj}
    dtr = lgb.Dataset(X, y, free_raw_data=False)
    dva = lgb.Dataset(Xv, yv, reference=dtr, free_raw_data=False)
    return lgb.train(params, dtr, num_boost_round=cfg.n_estimators, valid_sets=[dva], callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)])


@dataclass
class LgbmModel:
    features: list[str]
    booster: lgb.Booster
    best_iteration: int
    name: str = "lgbm"

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(_x(frame, self.features))

    def contributions(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        return self.booster.predict(_x(frame, self.features), pred_contrib=True)[:, :-1], list(self.features)

    def inputs(self) -> list[str]:
        return list(self.features)

    def to_dict(self) -> dict:
        return {"kind": "lgbm", "features": self.features, "best_iteration": self.best_iteration, "model": self.booster.model_to_string(num_iteration=self.best_iteration)}

    @classmethod
    def from_dict(cls, d: dict) -> "LgbmModel":
        return cls(d["features"], lgb.Booster(model_str=d["model"]), d["best_iteration"])


def fit_lgbm(cfg: InvestConfig, fit: pd.DataFrame, val: pd.DataFrame, target: str) -> LgbmModel:
    feats = [c for c in feature_columns(fit) if fit[c].nunique(dropna=True) > 1]
    mf, mv = fit[target].notna().to_numpy(), val[target].notna().to_numpy()
    if mf.sum() < 500 or mv.sum() < 100:
        raise InvestModelError(f"too few labelled rows: fit {int(mf.sum())}, validation {int(mv.sum())}")
    b = _train(cfg, {"objective": "regression", "metric": "l2"}, fit.loc[mf, feats].to_numpy(float), fit.loc[mf, target].to_numpy(float),
               val.loc[mv, feats].to_numpy(float), val.loc[mv, target].to_numpy(float))
    return LgbmModel(feats, b, int(b.best_iteration or b.current_iteration()))


@dataclass
class ScenarioModel:
    """q10 / q50 / q90 of the forward return at the horizon, sorted per row so that bear <= base <= bull."""
    features: list[str]
    boosters: dict[str, lgb.Booster]
    best_iteration: dict[str, int]
    quantiles: list[float]

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = _x(frame, self.features)
        q = np.sort(np.column_stack([self.boosters[_qname(a)].predict(X) for a in self.quantiles]), axis=1)
        return pd.DataFrame(q, index=frame.index, columns=[_qname(a) for a in self.quantiles])

    def to_dict(self) -> dict:
        return {"features": self.features, "quantiles": self.quantiles, "best_iteration": self.best_iteration,
                "boosters": {k: b.model_to_string(num_iteration=self.best_iteration[k]) for k, b in sorted(self.boosters.items())}}

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioModel":
        return cls(d["features"], {k: lgb.Booster(model_str=s) for k, s in d["boosters"].items()}, d["best_iteration"], d["quantiles"])


def _qname(a: float) -> str:
    return f"q{int(round(a * 100))}"


def fit_scenario(cfg: InvestConfig, fit: pd.DataFrame, val: pd.DataFrame, target: str) -> ScenarioModel:
    feats = [c for c in feature_columns(fit) if fit[c].nunique(dropna=True) > 1]
    mf, mv = fit[target].notna().to_numpy(), val[target].notna().to_numpy()
    if mf.sum() < 500 or mv.sum() < 100:
        raise InvestModelError(f"too few labelled rows: fit {int(mf.sum())}, validation {int(mv.sum())}")
    boosters, best = {}, {}
    for a in cfg.quantiles:
        b = _train(cfg, {"objective": "quantile", "alpha": a, "metric": "quantile"}, fit.loc[mf, feats].to_numpy(float), fit.loc[mf, target].to_numpy(float),
                   val.loc[mv, feats].to_numpy(float), val.loc[mv, target].to_numpy(float))
        boosters[_qname(a)], best[_qname(a)] = b, int(b.best_iteration or b.current_iteration())
    return ScenarioModel(feats, boosters, best, list(cfg.quantiles))


# ---- the set of candidates of one fold -------------------------------------------------------------------------------------------------------
@dataclass
class Candidates:
    preset: str
    horizon: int
    models: dict[str, object]
    scenario: ScenarioModel
    meta: dict = field(default_factory=dict)

    def artifact(self, name: str) -> bytes:
        """The bytes of one candidate (+ the shared scenario model): what is stored under artifacts/ and hashed."""
        doc = {"format": FORMAT_VERSION, "preset": self.preset, "horizon": self.horizon, "candidate": self.models[name].to_dict(), "scenario": self.scenario.to_dict(), "meta": self.meta}
        return gzip.compress(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode(), compresslevel=6, mtime=0)


def load_candidate(raw: bytes):
    """(model, scenario, doc) from artifact bytes."""
    doc = json.loads(gzip.decompress(raw))
    if doc.get("format") != FORMAT_VERSION:
        raise InvestModelError(f"unsupported model format {doc.get('format')!r}")
    c = doc["candidate"]
    model = {"factor": FactorModel, "ridge": LinearModel, "elasticnet": LinearModel, "lgbm": LgbmModel}[c["kind"]].from_dict(c)
    return model, ScenarioModel.from_dict(doc["scenario"]), doc


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()
