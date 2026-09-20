"""Walk-forward for the INVEST presets: folds whose embargo equals the label horizon, purging by the end date of THAT horizon's label,
hyper-parameter grid once on the first fold, the candidates fitted per fold, predictions on the test window only.

Evaluation rows are restricted to those whose forward return ends BEFORE the held-out period starts, so no held-out price enters any figure
through a label (63 / 126-session labels of the last development months would otherwise reach into it)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from predict_stock.backtest.walkforward import Fold, make_folds
from predict_stock.config import InvestConfig, InvestPreset
from predict_stock.invest.models import (
    Candidates, FactorModel, InvestModelError, fit_lgbm, fit_linear, fit_scenario,
)
from predict_stock.swing.folds import FoldData, split_fold
from predict_stock.swing.metrics import daily_rank_ic

CARRY = ["sma_ratio_200", "mom_6m_csrank", "dd_252", "vol_126", "mom_12m", "mom_6m"]


def targets(preset: InvestPreset) -> tuple[str, str, str]:
    h = preset.horizon
    return f"fwd_rank_{h}", f"fwd_ret_{h}", f"fwd_end_{h}"


def prepare(frame: pd.DataFrame, preset: InvestPreset) -> pd.DataFrame:
    """Adds ``label_end_max`` = the end date of this preset's label (what purging uses)."""
    return frame.assign(label_end_max=pd.to_datetime(frame[targets(preset)[2]]))


class _FoldsAdapter:
    """Presents an InvestConfig + preset as the object ``swing.folds.split_fold`` expects (``.folds.val_sessions``)."""
    def __init__(self, cfg: InvestConfig):
        self.folds = cfg.folds


def plan_folds(frame: pd.DataFrame, cfg: InvestConfig, preset: InvestPreset, end: pd.Timestamp) -> list[Fold]:
    f = cfg.folds
    return make_folds(frame["trade_date"].unique(), scheme="expanding", train_sessions=f.train_min_sessions, test_sessions=f.test_sessions,
                      step_sessions=f.step_sessions, embargo_sessions=preset.horizon, end=end)


def split(frame: pd.DataFrame, fold: Fold, cfg: InvestConfig) -> FoldData:
    return split_fold(frame, fold, _FoldsAdapter(cfg))


def _val_ic(model_predict: Callable[[pd.DataFrame], np.ndarray], val: pd.DataFrame, ret: str) -> float:
    scored = val[["trade_date", ret]].assign(score=model_predict(val))
    ic = daily_rank_ic(scored, "score", ret)
    return float(ic.mean()) if len(ic) else float("nan")


def grid(cfg: InvestConfig) -> list[tuple[str, dict]]:
    g = [("ridge", {"alpha": a}) for a in cfg.ridge_alphas]
    g += [("elasticnet", {"alpha": a, "l1_ratio": l}) for a in cfg.enet_alphas for l in cfg.enet_l1_ratios]
    return g


def tune(first: FoldData, cfg: InvestConfig, preset: InvestPreset, sink: Callable[[dict], None]) -> dict[str, dict]:
    """Every grid point is fitted on the first fold's fit rows and scored by the mean daily rank IC on its validation rows; the best point of each
    family is frozen for all folds. Every point (also a failed one) goes to ``sink``."""
    rank, ret, _ = targets(preset)
    best: dict[str, tuple[float, dict]] = {}
    for i, (kind, hyper) in enumerate(grid(cfg)):
        rec = {"number": i, "kind": kind, "hyper": hyper, "value": None, "state": "COMPLETE", "error": None}
        try:
            m = fit_linear(kind, first.fit, rank, hyper, cfg.seed)
            v = _val_ic(m.predict, first.val, ret)
            if not np.isfinite(v):
                raise ValueError("no valid daily rank IC on the validation rows")
            rec["value"] = v
            if kind not in best or v > best[kind][0]:
                best[kind] = (v, hyper)
        except Exception as exc:
            rec["state"], rec["error"] = "FAIL", f"{type(exc).__name__}: {exc}"[:500]
        sink(rec)
    missing = {"ridge", "elasticnet"} - set(best)
    if missing:
        raise InvestModelError(f"no valid hyper-parameters for {sorted(missing)}")
    return {k: {**v[1], "validation_ic": v[0]} for k, v in best.items()}


def fit_candidates(data: FoldData, cfg: InvestConfig, preset_key: str, preset: InvestPreset, hypers: dict[str, dict]) -> Candidates:
    rank, ret, _ = targets(preset)
    models: dict[str, object] = {}
    for name in cfg.candidates:
        if name == "factor":
            models[name] = FactorModel([(c, int(s)) for c, s in cfg.factors])
        elif name in ("ridge", "elasticnet"):
            h = {k: v for k, v in hypers[name].items() if k != "validation_ic"}
            models[name] = fit_linear(name, data.fit, rank, h, cfg.seed)
        elif name == "lgbm":
            models[name] = fit_lgbm(cfg, data.fit, data.val, rank)
        else:
            raise InvestModelError(f"unknown candidate {name!r}")
    scenario = fit_scenario(cfg, data.fit, data.val, ret)
    f = data.fold
    return Candidates(preset_key, preset.horizon, models, scenario,
                      {"fold": f.index, "train": [str(f.train_start.date()), str(f.train_end.date())], "test": [str(f.test_start.date()), str(f.test_end.date())],
                       "n_fit": len(data.fit), "n_val": len(data.val), "hypers": hypers})


def predict_frames(cands: Candidates, rows: pd.DataFrame, preset: InvestPreset) -> dict[str, pd.DataFrame]:
    """One frame per candidate: keys, score, the scenario quantiles, the realised labels (evaluation only) and a few carried feature columns."""
    rank, ret, end = targets(preset)
    q = cands.scenario.predict(rows)
    base = pd.concat([rows[["trade_date", "instrument_id"]], q, rows[[c for c in (rank, ret, end, *CARRY) if c in rows.columns]]], axis=1)
    out = {}
    for name, m in cands.models.items():
        df = base.assign(score=m.predict(rows))
        df["rank_in_universe"] = df.groupby("trade_date")["score"].rank(ascending=False, method="first")
        out[name] = df
    return out


@dataclass
class FoldRun:
    fold: Fold
    data: FoldData
    cands: Candidates
    preds: dict[str, pd.DataFrame]


def run_walk_forward(frame: pd.DataFrame, cfg: InvestConfig, preset_key: str, preset: InvestPreset, hypers: dict[str, dict], end: pd.Timestamp,
                     on_fold: Callable[[FoldRun], None] | None = None) -> list[FoldRun]:
    """``frame`` needs ``label_end_max`` (``prepare``)."""
    runs = []
    for fold in plan_folds(frame, cfg, preset, end):
        d = split(frame, fold, cfg)
        cands = fit_candidates(d, cfg, preset_key, preset, hypers)
        run = FoldRun(fold, d, cands, predict_frames(cands, d.test, preset))
        if on_fold is not None:
            on_fold(run)
        runs.append(run)
    return runs


def final_fold(frame: pd.DataFrame, preset: InvestPreset, holdout_start: pd.Timestamp, holdout_end: pd.Timestamp) -> Fold:
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(frame["trade_date"]).unique()))
    t = int(sessions.searchsorted(pd.Timestamp(holdout_start)))
    return Fold(-1, "expanding", sessions[0], sessions[t - preset.horizon - 1], pd.Timestamp(holdout_start), pd.Timestamp(holdout_end), preset.horizon, int(t - preset.horizon), 0)


def fit_final(frame: pd.DataFrame, cfg: InvestConfig, preset_key: str, preset: InvestPreset, hypers: dict[str, dict], holdout_start: pd.Timestamp,
              holdout_end: pd.Timestamp) -> tuple[Candidates, FoldData]:
    """Candidates trained on all development data (labels purged against the held-out start); never scored on the held-out period here."""
    fold = final_fold(frame, preset, holdout_start, holdout_end)
    d = split(frame, fold, cfg)
    d = FoldData(fold, d.fit, d.val, d.test.iloc[0:0], d.purged, d.unlabelled, d.fit_dropped)
    return fit_candidates(d, cfg, preset_key, preset, hypers), d


def concat(runs: list[FoldRun], name: str) -> pd.DataFrame:
    return pd.concat([r.preds[name].assign(fold=r.fold.index) for r in runs], ignore_index=True)


def evaluable(preds: pd.DataFrame, preset: InvestPreset, holdout_start: pd.Timestamp) -> pd.DataFrame:
    """Rows whose label is known and ends before the held-out period."""
    end = pd.to_datetime(preds[targets(preset)[2]])
    return preds[end.notna() & (end < pd.Timestamp(holdout_start))]
