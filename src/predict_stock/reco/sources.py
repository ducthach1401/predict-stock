"""Where the cards get their numbers: the market context of each name at a close (from the price panel the engine uses), the walk-forward predictions of the
stored SWING / INVEST models, the similar-signal statistics of each fold (from its validation rows: point-in-time) and the evidence about how far the
probabilities can be trusted (from PAST out-of-sample predictions only)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sqlalchemy import Engine, select

from predict_stock.backtest.engine import MarketData
from predict_stock.backtest.market import MarketRules
from predict_stock.backtest.runner import Setup
from predict_stock.config import PROJECT_ROOT, AppConfig
from predict_stock.db.models import InstrumentSymbolHistory as SymbolRow
from predict_stock.db.models import Model
from predict_stock.db.session import session_scope
from predict_stock.features.labels import triple_barrier_arrays
from predict_stock.invest import walkforward as IW
from predict_stock.invest.models import load_candidate, sha256
from predict_stock.invest.strategy import regime_off
from predict_stock.reco.builders import Ctx, Pred
from predict_stock.swing.folds import plan_folds, split_fold, with_ends
from predict_stock.swing.model import SwingBundle, load_bundle
from predict_stock.swing.walkforward import predict_rows

CONTEXT_FEATURES = ("rsi_14", "vol_spike_20", "rs_5", "rs_10", "atr_pct_14")


class Symbols:
    """instrument_id -> ticker as of a date (the symbol history)."""
    def __init__(self, engine: Engine, ids: list[int]):
        with session_scope(engine) as s:
            rows = s.execute(select(SymbolRow.instrument_id, SymbolRow.symbol, SymbolRow.valid_from, SymbolRow.valid_to).where(SymbolRow.instrument_id.in_(ids))).all()
        self.rows: dict[int, list] = {}
        for iid, sym, vf, vt in rows:
            self.rows.setdefault(iid, []).append((pd.Timestamp(vf), pd.Timestamp(vt) if vt is not None else pd.Timestamp.max, sym))

    def at(self, iid: int, d) -> str:
        d = pd.Timestamp(d)
        best = None
        for vf, vt, sym in self.rows.get(iid, []):
            if vf <= d < vt:
                return sym
            if vf <= d:
                best = sym
        return best or (self.rows[iid][-1][2] if self.rows.get(iid) else f"#{iid}")


class Market:
    """Wide price arrays and rolling levels for the cards' context; the same prices the engine simulates."""
    def __init__(self, setup: Setup, cfg: AppConfig):
        md: MarketData = setup.data
        self.cal, self.rules = setup.calendar, setup.rules
        self.md = md
        self.close, self.high = md.close, md.high
        self.hi20 = self.high.rolling(20, min_periods=20).max()
        self.hi60 = self.high.rolling(cfg.reco.swing.resistance_lookback, min_periods=20).max()
        self.hi252 = self.high.rolling(252, min_periods=120).max()
        self.sma50 = self.close.rolling(50, min_periods=50).mean()
        self.sma200 = self.close.rolling(200, min_periods=200).mean()
        iv = cfg.invest.regime
        self.regime_off = regime_off(setup.index_close, self.cal, iv.symbol, iv.fallback, iv.sma_window)
        self.pos = {d: i for i, d in enumerate(self.cal)}
        self._close = self.close.to_numpy(float)
        self.col = {iid: j for j, iid in enumerate(self.close.columns)}
        self._bands = md.bands.reindex(index=self.cal, columns=self.close.columns).to_numpy(float)

    def level(self, wide: pd.DataFrame, i: int, iid: int) -> float | None:
        v = wide.iat[i, self.col[iid]]
        return None if not np.isfinite(v) else float(v)

    def band_prices(self, i: int, iid: int) -> tuple[float, float]:
        """Floor and ceiling of the session after ``i`` from the close at ``i`` (the reference price) and the band the engine will use."""
        j = self.col[iid]
        b = self._bands[min(i + 1, len(self.cal) - 1), j]
        if not np.isfinite(b):
            b = self._bands[i, j] if np.isfinite(self._bands[i, j]) else self.rules.bands[0]
        c = self._close[i, j]
        return float(self.rules.floor(c, b)), float(self.rules.ceiling(c, b))

    def ctx(self, i: int, iid: int, symbol: str, atr: float | None, feats: dict) -> Ctx | None:
        j = self.col[iid]
        c = self._close[i, j]
        if not np.isfinite(c) or c <= 0:
            return None
        floor, ceiling = self.band_prices(i, iid)
        g = lambda k: feats.get(k) if feats.get(k) is not None and np.isfinite(feats.get(k)) else None
        return Ctx(symbol=symbol, instrument_id=iid, as_of=self.cal[i], calendar=self.cal, close=float(c), atr=float(atr) if atr is not None and np.isfinite(atr) else float("nan"),
                   floor=floor, ceiling=ceiling, hi20=self.level(self.hi20, i, iid), hi60=self.level(self.hi60, i, iid), hi252=self.level(self.hi252, i, iid),
                   sma50=self.level(self.sma50, i, iid), sma200=self.level(self.sma200, i, iid), rsi14=g("rsi_14"), vol_spike=g("vol_spike_20"), rs5=g("rs_5"), rs10=g("rs_10"),
                   regime_off=bool(self.regime_off.iloc[i]), feats={k: feats.get(k) for k in ("mom_6m_csrank", "dd_252", "mom_6m", "vol_126", "sma_ratio_200")})


def _latest_model(engine: Engine, name: str) -> tuple[int, str, str]:
    with session_scope(engine) as s:
        row = s.scalars(select(Model).where(Model.name == name).order_by(Model.version.desc())).first()
        if row is None:
            raise LookupError(f"model {name!r} is not registered: run `swing run` / `invest run` first")
        return row.id, row.artifact_path, row.artifact_sha256


# ---- SWING -------------------------------------------------------------------------------------------------------------------------------------------
@dataclass
class SwingHistory:
    preds: pd.DataFrame                       # one row per (date, instrument) of the out-of-sample test windows
    contributions: dict                       # (date, instrument_id) -> [[feature, value, contribution], ...] for the top-ranked rows
    similar: dict                             # fold -> DataFrame indexed by bucket
    bundles: dict[int, SwingBundle]
    models: dict[int, tuple[int, str]]        # fold -> (model_id, name)
    first_test: pd.Timestamp = None
    last_test: pd.Timestamp = None


def barrier_1_1(setup: Setup, atr_pct_wide: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """Outcome (1 = +1 ATR first, -1 = -1 ATR first, 0 = neither) of a 1 ATR / 1 ATR barrier from every close, same rules as the label (stop wins ties)."""
    md = setup.data
    A = (atr_pct_wide.reindex(index=md.close.index, columns=md.close.columns) * md.close).to_numpy(float)
    lab, *_ = triple_barrier_arrays(md.open.to_numpy(float), md.high.to_numpy(float), md.low.to_numpy(float), md.close.to_numpy(float), A, horizon, 1.0, 1.0)
    return pd.DataFrame(lab, index=md.close.index, columns=md.close.columns)


def load_swing_history(engine: Engine, cfg: AppConfig, setup: Setup, universe_id: int, top_rows: int = 10) -> SwingHistory:
    sw = cfg.swing
    frame = with_ends(setup.frames["swing"])
    folds = plan_folds(frame, sw, setup.holdout.start)
    atr_pct = setup.frames["swing"].pivot(index="trade_date", columns="instrument_id", values="atr_pct_14")
    t1 = barrier_1_1(setup, atr_pct, sw.horizon)
    preds, contrib, similar, bundles, models = [], {}, {}, {}, {}
    for f in folds:
        mid, path, sha = _latest_model(engine, f"swing_lgbm_f{f.index:02d}")
        b = load_bundle(PROJECT_ROOT / path, sha)
        bundles[f.index], models[f.index] = b, (mid, f"swing_lgbm_f{f.index:02d}")
        d = split_fold(frame, f, sw)
        p = predict_rows(b, d.test)
        extra = d.test[[c for c in CONTEXT_FEATURES if c in d.test.columns and c not in p.columns]]
        p = p.join(extra, how="left")
        p["fold"] = f.index
        preds.append(p)
        top = p[p["rank_in_universe"] <= top_rows]
        rows = d.test.loc[top.index]
        c, _ = b.contributions(rows, "rank")
        for k, (idx, r) in enumerate(top.iterrows()):
            order = np.argsort(-np.abs(c[k]), kind="stable")[:5]
            contrib[(pd.Timestamp(r["trade_date"]), int(r["instrument_id"]))] = [[b.features[j], _finite(rows.iloc[k][b.features[j]]), float(c[k][j])] for j in order]
        similar[f.index] = _similar_table(b, d.val, t1)
    P = pd.concat(preds, ignore_index=True)
    P["n_universe"] = P.groupby("trade_date")["instrument_id"].transform("size")
    return SwingHistory(P, contrib, similar, bundles, models, folds[0].test_start, folds[-1].test_end)


def _finite(v):
    return None if v is None or not np.isfinite(v) else float(v)


def _similar_table(bundle: SwingBundle, val: pd.DataFrame, t1: pd.DataFrame) -> pd.DataFrame:
    """Outcomes of the validation-window signals grouped by the calibrated-probability bucket of the model (the same buckets as its holding-time table)."""
    p = bundle.predict(val)
    ok = val["tb_label"].notna().to_numpy()
    v = val.loc[ok, ["trade_date", "instrument_id", "tb_label", "tb_time"]].assign(bucket=bundle.hold.bucket(p.loc[ok, "proba"].to_numpy()))
    v["t1"] = [t1.at[d, i] if (d in t1.index and i in t1.columns) else np.nan for d, i in zip(v["trade_date"], v["instrument_id"])]
    g = v.groupby("bucket")
    out = pd.DataFrame({"n": g.size(), "win_rate": g["tb_label"].apply(lambda s: (s == 1).mean()), "stop_rate": g["tb_label"].apply(lambda s: (s == -1).mean()),
                        "timeout_rate": g["tb_label"].apply(lambda s: (s == 0).mean()), "t1_rate": g["t1"].apply(lambda s: (s.dropna() == 1).mean() if s.notna().any() else np.nan),
                        "hold_median": g["tb_time"].median()})
    overall = pd.DataFrame([{"n": len(v), "win_rate": (v["tb_label"] == 1).mean(), "stop_rate": (v["tb_label"] == -1).mean(), "timeout_rate": (v["tb_label"] == 0).mean(),
                             "t1_rate": (v["t1"].dropna() == 1).mean(), "hold_median": v["tb_time"].median()}], index=["all"])
    return pd.concat([out, overall])


def similar_for(hist: SwingHistory, fold: int, proba: float) -> dict | None:
    """Past signals in the same calibrated-probability bucket as this one (validation rows of the fold's model), with their n."""
    b, tab = hist.bundles[fold], hist.similar[fold]
    k = int(b.hold.bucket(np.array([proba]))[0])
    if k not in tab.index:
        return None
    r = tab.loc[k]
    return {"n": int(r["n"]), "win_rate": float(r["win_rate"]), "t1_rate": None if not np.isfinite(r["t1_rate"]) else float(r["t1_rate"]), "stop_rate": float(r["stop_rate"]),
            "timeout_rate": float(r["timeout_rate"]), "hold_median": float(r["hold_median"]), "basis": "validation window of the fold's model, same calibrated-probability bucket"}


class Evidence:
    """How far the calibrated probability could be trusted on PAST out-of-sample predictions whose labels had already ended. Recomputed at the first
    session of each month (point-in-time). AUC = the mean over folds of each fold's AUC of the raw probability (pooling folds would mix their different
    probability levels); calibration gap = mean displayed probability minus the realised event rate over the same rows."""
    def __init__(self, preds: pd.DataFrame, min_fold_rows: int = 500):
        self.p = preds[preds["tb_label"].notna() & preds["tb_end"].notna()][["trade_date", "fold", "proba_raw", "proba", "tb_label", "tb_end"]].copy()
        self.p["y"] = (self.p["tb_label"] == 1).astype(float)
        self.p["tb_end"] = pd.to_datetime(self.p["tb_end"])
        self.min_fold_rows = min_fold_rows
        self.cache: dict = {}

    def at(self, d) -> dict:
        d = pd.Timestamp(d)
        key = (d.year, d.month)
        if key in self.cache:
            return self.cache[key]
        start = pd.Timestamp(d.year, d.month, 1)
        known = self.p[self.p["tb_end"] < start]
        aucs, weights = [], []
        for _, g in known.groupby("fold"):
            if len(g) >= self.min_fold_rows and 0 < g["y"].sum() < len(g):
                aucs.append(roc_auc_score(g["y"], g["proba_raw"]))
                weights.append(len(g))
        ev = {"n": int(len(known)), "auc": float(np.average(aucs, weights=weights)) if aucs else None,
              "calibration_gap": float(known["proba"].mean() - known["y"].mean()) if len(known) else None, "realised_rate": float(known["y"].mean()) if len(known) else None,
              "folds_used": len(aucs), "as_of": str(start.date())}
        self.cache[key] = ev
        return ev


def swing_pred(hist: SwingHistory, row: pd.Series) -> Pred:
    mid, name = hist.models[int(row["fold"])]
    return Pred(score=float(row["score"]), rank=int(row["rank_in_universe"]), n_universe=int(row["n_universe"]), model_id=mid, model_name=name, universe_id=None,
                proba_raw=float(row["proba_raw"]), proba=float(row["proba"]), q10=float(row["q10"]), q50=float(row["q50"]), q90=float(row["q90"]), hold_median=float(row["hold_median"]),
                hold_p75=float(row["hold_p75"]), hold_n=int(row["hold_n"]), contributions=hist.contributions.get((pd.Timestamp(row["trade_date"]), int(row["instrument_id"])), []))


# ---- INVEST ------------------------------------------------------------------------------------------------------------------------------------------------
@dataclass
class InvestHistory:
    key: str
    preds: pd.DataFrame                       # out-of-sample rows of the card model: score, scenario quantiles, features, realised forward labels
    models: dict[int, tuple[int, str, object]]     # fold -> (model_id, name, candidate model)
    first_test: pd.Timestamp = None
    last_test: pd.Timestamp = None


def load_invest_history(engine: Engine, cfg: AppConfig, setup: Setup, key: str) -> InvestHistory:
    iv = cfg.invest
    preset, cand = iv.presets[key], cfg.reco.card_model[key]
    frame = IW.prepare(setup.frames["invest"], preset)
    folds = IW.plan_folds(frame, iv, preset, setup.holdout.start)
    if not folds:
        raise LookupError(f"invest {key}: no walk-forward fold fits inside the development data")
    rank, ret, end = IW.targets(preset)
    keep = ["trade_date", "instrument_id", rank, ret, end, "mom_6m_csrank", "dd_252", "sma_ratio_200", "mom_6m", "mom_12m", "vol_126"]
    preds, models = [], {}
    for f in folds:
        name = f"invest_{key}_{cand}_f{f.index:02d}"
        mid, path, sha = _latest_model(engine, name)
        raw = (PROJECT_ROOT / path).read_bytes()
        if sha256(raw) != sha:
            raise ValueError(f"{path}: sha256 does not match the recorded value")
        model, scen, _ = load_candidate(raw)
        models[f.index] = (mid, name, model)
        rows = IW.split(frame, f, iv).test
        q = scen.predict(rows)
        cols = list(dict.fromkeys([c for c in keep + model.inputs() if c in rows.columns]))
        p = rows[cols].assign(score=model.predict(rows), fold=f.index).join(q)
        p["rank_in_universe"] = p.groupby("trade_date")["score"].rank(ascending=False, method="first")
        p["n_universe"] = p.groupby("trade_date")["score"].transform("size")
        preds.append(p)
    return InvestHistory(key, pd.concat(preds), models, folds[0].test_start, folds[-1].test_end)


def invest_similar(hist: InvestHistory, as_of, top_k: int, horizon: int) -> dict | None:
    """Past top-K picks of this candidate whose forward return had already been realised before ``as_of``: how often they were positive / beat the median."""
    p = hist.preds
    ret, rank, end = f"fwd_ret_{horizon}", f"fwd_rank_{horizon}", f"fwd_end_{horizon}"
    known = p[(p["rank_in_universe"] <= top_k) & p[end].notna() & (pd.to_datetime(p[end]) < pd.Timestamp(as_of))]
    if known.empty:
        return None
    return {"n": int(len(known)), "win_rate": float((known[ret] > 0).mean()), "beat_median_rate": float((known[rank] > 0.5).mean()), "mean_return": float(known[ret].mean()),
            "basis": "past top-K picks of this model with a realised forward return; observations overlap heavily, so the number of independent cases is far smaller than n"}
