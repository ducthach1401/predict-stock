"""The SWING model: five LightGBM boosters on the same features plus the post-processing that makes their outputs usable.

    rank   regression on the cross-sectional rank of the 5-session forward return  -> the ranking score
    event  binary: the target is touched before the stop within H sessions        -> raw probability -> calibrated (isotonic; Platt kept for comparison)
    q10/q50/q90  quantile objective on the 5-session forward return                -> then sorted per row so the quantiles never cross
    hold   table of the sessions-to-barrier of similar past signals (bucket of the calibrated probability)
    shap   TreeSHAP contributions (``pred_contrib``) of the rank and event boosters

Everything is trained on the fit rows (early stopping on the validation rows); calibration and the holding-time table use the
validation rows only, which the boosters did not fit. Training is deterministic (seed, ``deterministic``, ``force_row_wise``, fixed threads)
and the saved file is byte-stable (gzip mtime 0, sorted JSON), so the same data and config give the same sha256.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from predict_stock.config import SwingConfig
from predict_stock.swing import calibration as cal
from predict_stock.swing.holding import HoldTable, fit_hold_table

FORMAT_VERSION = 1
HEADS = ("rank", "event", "q10", "q50", "q90")
MIN_HOLD_N = 100


class SwingModelError(RuntimeError):
    pass


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Every column that is neither a key nor a label. ``instrument_id`` / ``trade_date`` are keys, never features."""
    return [c for c in frame.columns if c not in ("trade_date", "instrument_id", "label_end_max") and not c.startswith(("fwd_", "tb_"))]


def head_targets(cfg: SwingConfig) -> dict[str, str]:
    q = {f"q{int(round(x * 100))}": cfg.return_target for x in cfg.quantiles}
    return {"rank": cfg.rank_target, "event": cfg.event_label, **q}


def _lgb_params(cfg: SwingConfig, head: str, tree: dict) -> dict:
    base = {"verbosity": -1, "seed": cfg.seed, "deterministic": True, "force_row_wise": True, "num_threads": cfg.num_threads, **tree}
    if head == "rank":
        return {**base, "objective": "regression", "metric": "l2"}
    if head == "event":
        return {**base, "objective": "binary", "metric": "binary_logloss"}
    alpha = int(head[1:]) / 100
    return {**base, "objective": "quantile", "alpha": alpha, "metric": "quantile"}


def _target(frame: pd.DataFrame, head: str, cfg: SwingConfig) -> np.ndarray:
    col = head_targets(cfg)[head]
    y = frame[col].to_numpy(float)
    return (y == 1.0).astype(float) if head == "event" else y      # tb_label: 1 = target first, 0 = time-out, -1 = stop first (both non-events)


def _has_target(frame: pd.DataFrame, head: str, cfg: SwingConfig) -> np.ndarray:
    return frame[head_targets(cfg)[head]].notna().to_numpy()


@dataclass
class SwingBundle:
    features: list[str]
    boosters: dict[str, lgb.Booster]
    best_iteration: dict[str, int]
    isotonic: dict
    platt: dict
    calibration_used: str
    hold: HoldTable
    base_rate: float           # event rate of the rows the boosters were fitted on
    tree_params: dict
    meta: dict = field(default_factory=dict)

    # ---- prediction -------------------------------------------------------------------------------------------
    @property
    def calibration_base_rate(self) -> float:
        """Event rate of the validation rows the calibrator was fitted on: the calibrated probabilities are centred on it, so 'above average' means above THIS."""
        return float(self.meta["calibration_base_rate"])

    def _x(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in frame.columns]
        if missing:
            raise SwingModelError(f"missing feature columns: {missing}")
        return frame[self.features].to_numpy(float)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = self._x(frame)
        out = pd.DataFrame(index=frame.index)
        out["score"] = self.boosters["rank"].predict(X)
        raw = self.boosters["event"].predict(X)
        out["proba_raw"] = raw
        out["proba_isotonic"] = cal.apply_calibrator(self.isotonic, raw)
        out["proba_platt"] = cal.apply_calibrator(self.platt, raw)
        out["proba"] = out["proba_isotonic" if self.calibration_used == "isotonic" else "proba_platt"]
        q = np.column_stack([self.boosters[h].predict(X) for h in ("q10", "q50", "q90")])
        q = np.sort(q, axis=1)                                             # monotone rearrangement: q10 <= q50 <= q90 on every row
        out["q10"], out["q50"], out["q90"] = q[:, 0], q[:, 1], q[:, 2]
        med, p75, n = self.hold.lookup(out["proba"].to_numpy())
        out["hold_median"], out["hold_p75"], out["hold_n"] = med, p75, n
        return out

    def contributions(self, frame: pd.DataFrame, head: str = "rank") -> tuple[np.ndarray, np.ndarray]:
        """(contributions, base value): TreeSHAP of one booster, raw-output units (log-odds for ``event``). Rows sum to the raw prediction."""
        c = self.boosters[head].predict(self._x(frame), pred_contrib=True)
        return c[:, :-1], c[:, -1]

    # ---- persistence ------------------------------------------------------------------------------------------
    def to_bytes(self) -> bytes:
        doc = {"format": FORMAT_VERSION, "features": self.features, "best_iteration": self.best_iteration,
               "boosters": {h: b.model_to_string(num_iteration=self.best_iteration[h]) for h, b in sorted(self.boosters.items())},
               "isotonic": self.isotonic, "platt": self.platt, "calibration_used": self.calibration_used, "hold": self.hold.to_dict(),
               "base_rate": self.base_rate, "tree_params": self.tree_params, "meta": self.meta}
        return gzip.compress(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode(), compresslevel=6, mtime=0)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "SwingBundle":
        doc = json.loads(gzip.decompress(raw))
        if doc.get("format") != FORMAT_VERSION:
            raise SwingModelError(f"unsupported model format {doc.get('format')!r}")
        return cls(doc["features"], {h: lgb.Booster(model_str=s) for h, s in doc["boosters"].items()}, doc["best_iteration"], doc["isotonic"], doc["platt"],
                   doc["calibration_used"], HoldTable.from_dict(doc["hold"]), doc["base_rate"], doc["tree_params"], doc["meta"])

    def save(self, path: Path) -> str:
        raw = self.to_bytes()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_bytes() != raw:
            path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()


def load_bundle(path: Path, sha256: str | None = None) -> SwingBundle:
    raw = Path(path).read_bytes()
    if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
        raise SwingModelError(f"{path}: sha256 does not match the recorded value (file changed or damaged)")
    return SwingBundle.from_bytes(raw)


def fit_bundle(fit: pd.DataFrame, val: pd.DataFrame, cfg: SwingConfig, tree_params: dict | None = None, meta: dict | None = None) -> SwingBundle:
    """Train the five boosters on ``fit`` (early stopping on ``val``), then calibrate and build the holding-time table on ``val``."""
    tree = {**cfg.lgbm, **(tree_params or {})}
    feats = feature_columns(fit)
    if not feats:
        raise SwingModelError("no feature columns")
    boosters, best = {}, {}
    for head in HEADS:
        if head.startswith("q") and int(head[1:]) / 100 not in cfg.quantiles:
            continue
        mf, mv = _has_target(fit, head, cfg), _has_target(val, head, cfg)
        if mf.sum() < 200 or mv.sum() < 50:
            raise SwingModelError(f"too few rows for {head}: fit {int(mf.sum())}, validation {int(mv.sum())}")
        dtr = lgb.Dataset(fit.loc[mf, feats].to_numpy(float), _target(fit, head, cfg)[mf], free_raw_data=False)
        dva = lgb.Dataset(val.loc[mv, feats].to_numpy(float), _target(val, head, cfg)[mv], reference=dtr, free_raw_data=False)
        b = lgb.train(_lgb_params(cfg, head, tree), dtr, num_boost_round=cfg.n_estimators, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)])
        boosters[head], best[head] = b, int(b.best_iteration or b.current_iteration())
    mv = _has_target(val, "event", cfg) & val[cfg.hold_column].notna().to_numpy()
    xv = val.loc[mv, feats].to_numpy(float)
    y = _target(val, "event", cfg)[mv]
    if y.sum() < 20 or (1 - y).sum() < 20:
        raise SwingModelError(f"the validation rows hold {int(y.sum())} events and {int((1 - y).sum())} non-events: too few of one class to calibrate a probability")
    raw = boosters["event"].predict(xv, num_iteration=best["event"])
    iso, platt = cal.fit_isotonic(raw, y, cfg.isotonic_min_bin), cal.fit_platt(raw, y)
    used = iso if cfg.calibration == "isotonic" else platt
    hold = fit_hold_table(cal.apply_calibrator(used, raw), val.loc[mv, cfg.hold_column].to_numpy(float), cfg.horizon, cfg.holding_buckets, MIN_HOLD_N)
    base_rate = float(np.mean(_target(fit, "event", cfg)[_has_target(fit, "event", cfg)]))
    return SwingBundle(feats, boosters, best, iso, platt, cfg.calibration, hold, base_rate, tree,
                       {"n_fit": int(len(fit)), "n_val": int(len(val)), "lightgbm": lgb.__version__, "calibration_base_rate": float(y.mean()), **(meta or {})})
