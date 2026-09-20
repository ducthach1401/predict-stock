"""Cards of one session for the paper trader: SWING every day, INVEST only on rebalance and tranche dates (the target book of the sleeve is kept in `sleeve_targets`).

Same builders, rules and models as the backtest of the cards (Phase 7); the only differences are that the models are the FINAL ones and that the INVEST plan is
driven by the stored target book instead of being computed in one pass over history."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import Engine, select

from predict_stock.backtest.baselines import rebalance_sessions
from predict_stock.backtest.runner import Setup
from predict_stock.config import PROJECT_ROOT, AppConfig
from predict_stock.db.models import Experiment
from predict_stock.db.session import session_scope
from predict_stock.invest.models import load_candidate, sha256
from predict_stock.paper import state as ST
from predict_stock.reco import aggregate as AG
from predict_stock.reco import builders as B
from predict_stock.reco import sources as S
from predict_stock.reco.backtest import _invest_pred
from predict_stock.reco.cards import Card
from predict_stock.reco.job import next_schedule_date, stored_verdicts, universe_id
from predict_stock.swing.folds import split_fold, with_ends
from predict_stock.swing.model import load_bundle
from predict_stock.swing.walkforward import final_fold, predict_rows


@dataclass
class DayContext:
    cfg: AppConfig
    engine: Engine
    setup: Setup
    market: S.Market
    symbols: S.Symbols
    uid: int
    verdicts: dict
    i: int
    d: pd.Timestamp


def build_day(engine: Engine, cfg: AppConfig, setup: Setup, as_of: pd.Timestamp) -> DayContext:
    cal = setup.calendar
    d = cal[cal <= pd.Timestamp(as_of)][-1]
    return DayContext(cfg, engine, setup, S.Market(setup, cfg), S.Symbols(engine, list(setup.data.close.columns)), universe_id(engine, setup.universe), stored_verdicts(cfg),
                      int(cal.get_loc(d)), d)


def get_evidence(dc: DayContext) -> dict | None:
    """Past out-of-sample evidence for the display grade, cached per month in `experiments` (recomputing needs every fold model: ~20 s)."""
    key = f"{dc.d.year}-{dc.d.month:02d}"
    with session_scope(dc.engine) as s:
        row = s.scalars(select(Experiment).where(Experiment.name == "reco:evidence").order_by(Experiment.id.desc())).first()
        rows = [r for r in s.scalars(select(Experiment).where(Experiment.name == "reco:evidence")) if (r.params or {}).get("month") == key]
        if rows:
            return rows[-1].summary
    try:
        hist = S.load_swing_history(dc.engine, dc.cfg, dc.setup, dc.uid)
        ev = S.Evidence(hist.preds).at(dc.d)
    except LookupError:
        return None
    from predict_stock.swing.registry import log_experiment
    log_experiment(dc.engine, dc.cfg, "reco:evidence", key=f"evidence:{key}", params={"month": key}, summary=ev, status="success", description="Past out-of-sample evidence for the card's display grade")
    return ev


def swing_cards(dc: DayContext, holdings: dict[str, dict[int, float]], evidence: dict | None) -> list[Card]:
    cfg, rc, sw, setup = dc.cfg, dc.cfg.reco, dc.cfg.swing, dc.setup
    frame = with_ends(setup.frames["swing"])
    mid, path, sha = S._latest_model(dc.engine, "swing_lgbm_final")
    bundle = load_bundle(PROJECT_ROOT / path, sha)
    rows = frame[pd.to_datetime(frame["trade_date"]) == dc.d]
    if rows.empty:
        return []
    pred = predict_rows(bundle, rows)
    pred = pred.join(rows[[c for c in S.CONTEXT_FEATURES if c in rows.columns and c not in pred.columns]])
    pred["n_universe"] = len(pred)
    top = pred.sort_values("rank_in_universe").head(rc.swing.max_positions + 4)
    contrib, _ = bundle.contributions(rows.loc[top.index], "rank")
    fold = final_fold(frame, sw, setup.holdout.start, setup.holdout.end)
    d = split_fold(frame, fold, sw)
    atr_pct = setup.frames["swing"].pivot(index="trade_date", columns="instrument_id", values="atr_pct_14")
    live = S.SwingHistory(pred, {}, {-1: S._similar_table(bundle, d.val, S.barrier_1_1(setup, atr_pct, sw.horizon))}, {-1: bundle}, {-1: (mid, "swing_lgbm_final")})
    out: list[Card] = []
    buys: list[Card] = []
    for k, (idx, r) in enumerate(top.iterrows()):
        iid = int(r["instrument_id"])
        order = np.argsort(-np.abs(contrib[k]), kind="stable")[:5]
        cl = [[bundle.features[j], S._finite(rows.loc[idx, bundle.features[j]]), float(contrib[k][j])] for j in order]
        atr = float(r["atr_pct_14"]) * float(dc.market._close[dc.i, dc.market.col[iid]])
        ctx = dc.market.ctx(dc.i, iid, dc.symbols.at(iid, dc.d), atr, {c: r.get(c) for c in S.CONTEXT_FEATURES})
        if ctx is None:
            continue
        pr = B.Pred(score=float(r["score"]), rank=int(r["rank_in_universe"]), n_universe=int(r["n_universe"]), model_id=mid, model_name="swing_lgbm_final", universe_id=dc.uid,
                    proba_raw=float(r["proba_raw"]), proba=float(r["proba"]), q10=float(r["q10"]), q50=float(r["q50"]), q90=float(r["q90"]), hold_median=float(r["hold_median"]),
                    hold_p75=float(r["hold_p75"]), hold_n=int(r["hold_n"]), contributions=cl)
        card = B.build_swing_card(ctx, pr, S.similar_for(live, -1, pr.proba), evidence, dc.verdicts.get("swing"), rc, setup.rules, watch=k >= rc.swing.max_positions)
        (buys if card.action == "BUY" else out).append(card)
    agg = AG.aggregate(buys, rc, setup.rules, holdings=holdings)
    return out + agg.accepted + agg.rejected


def _invest_model(dc: DayContext, key: str):
    cand = dc.cfg.reco.card_model[key]
    name = f"invest_{key}_{cand}_final"
    mid, path, sha = S._latest_model(dc.engine, name)
    raw = (PROJECT_ROOT / path).read_bytes()
    if sha256(raw) != sha:
        raise ValueError(f"{path}: sha256 does not match the recorded value")
    model, scen, _ = load_candidate(raw)
    return mid, name, model, scen


@dataclass
class InvestStep:
    kind: str | None                 # rebalance | tranche | None (nothing due today)
    n: int = 0
    of: int = 1
    rebalance_date: pd.Timestamp | None = None


def invest_step_due(dc: DayContext, key: str) -> InvestStep:
    """Is today a rebalance date of this preset (first session of the month / quarter) or a tranche date (rebalance + k x spacing sessions)?"""
    preset = dc.cfg.invest.presets[key]
    cal, i = dc.setup.calendar, dc.i
    sched = rebalance_sessions(cal, preset.rebalance, max(0, i - 400), i)
    last_r = max(sched)
    if last_r == i:
        return InvestStep("rebalance", 1, preset.tranches, dc.d)
    off = i - last_r
    if preset.tranche_spacing > 0 and off % preset.tranche_spacing == 0 and 0 < off // preset.tranche_spacing < preset.tranches:
        return InvestStep("tranche", off // preset.tranche_spacing + 1, preset.tranches, cal[last_r])
    return InvestStep(None)


def invest_cards(dc: DayContext, key: str, run_id: int | None) -> tuple[list[Card], dict | None]:
    """Cards of one INVEST sleeve for today, if a rebalance / tranche is due; records the step in the sleeve's target book (immutable per date, kind, tranche)."""
    cfg, iv, rc, setup = dc.cfg, dc.cfg.invest, dc.cfg.reco, dc.setup
    preset, sleeve = iv.presets[key], f"invest_{key}"
    step = invest_step_due(dc, key)
    if step.kind is None:
        return [], None
    d = dc.d.date()
    existing = ST.get_target(dc.engine, sleeve, d, step.kind, step.n)
    frame = setup.frames["invest"]
    rows = frame[pd.to_datetime(frame["trade_date"]) == dc.d]
    if rows.empty:
        return [], None
    mid, name, model, scen = _invest_model(dc, key)
    p = rows.assign(score=model.predict(rows)).join(scen.predict(rows))
    p["rank_in_universe"] = p["score"].rank(ascending=False, method="first")
    p["n_universe"] = len(p)
    p = p.sort_values("rank_in_universe")
    budget = AG.sleeve_budgets(rc)[sleeve]
    if step.kind == "rebalance":
        prev0 = {k: v for k, v in ST._key((ST.latest_target(dc.engine, sleeve, before=d) or _Empty()).current).items() if v > 1e-9}
        target = {int(r.instrument_id): budget / iv.top_k for r in p.head(iv.top_k).itertuples()}
    else:
        base = ST.get_target(dc.engine, sleeve, step.rebalance_date.date(), "rebalance", 1)
        if base is None:
            return [], None                                            # the rebalance itself was never run: nothing to continue
        prev0, target = ST._key(base.target["from"]), ST._key(base.target["final"])
    frac = step.n / step.of
    prev_step = prev0 if step.kind == "rebalance" else ST._key((ST.latest_target(dc.engine, sleeve, before=d) or _Empty()).current)
    cur = {a: prev0.get(a, 0.0) + (target.get(a, 0.0) - prev0.get(a, 0.0)) * frac for a in set(prev0) | set(target)}
    if step.kind == "tranche":                                          # a stock that left the universe since the rebalance keeps what it has: not topped up, not sold either
        members = ST.members_on(dc.engine, cfg.universe.training_code, d)
        for a in set(cur) | set(prev_step):
            if a not in members:
                cur[a] = prev_step.get(a, 0.0)
    cur = {a: w for a, w in cur.items() if w > 1e-9}
    nxt_review = next_schedule_date(dc.d, preset.rebalance)
    row = existing or ST.save_target(dc.engine, sleeve, d, step.kind, step.n, step.of, cur, {"final": {str(k): v for k, v in target.items()}, "from": {str(k): v for k, v in prev0.items()}},
                                     nxt_review.date(), run_id)
    cur = ST._key(row.current)
    cards: list[Card] = []
    try:                                                               # the walk-forward fold models give the record of past top-K picks; without them there is none
        similar = S.invest_similar(S.load_invest_history(dc.engine, cfg, setup, key), dc.d, iv.top_k, preset.horizon)
    except LookupError:
        similar = None
    for _, r in p.iterrows():
        iid = int(r["instrument_id"])
        w = cur.get(iid)
        if w is None or w <= prev_step.get(iid, 0.0) + 1e-12:
            continue
        ctx = dc.market.ctx(dc.i, iid, dc.symbols.at(iid, dc.d), None, {c: r.get(c) for c in ("mom_6m_csrank", "dd_252", "mom_6m", "vol_126", "sma_ratio_200")})
        if ctx is None:
            continue
        pr = _invest_pred(r, model, mid, name)
        pr.universe_id = dc.uid
        card = B.build_invest_card(ctx, pr, key, preset, similar, None, dc.verdicts.get(sleeve), rc, setup.rules, weight=w, next_review=nxt_review, top_k=iv.top_k,
                                   rs_floor=iv.thesis_rs_floor, tranche=(step.n, step.of, target.get(iid, 0.0)))
        card.card_id = f"{card.strategy}:{card.symbol}:{d}:t{step.n}"
        cards.append(card)
    for r in p.iloc[iv.top_k: iv.top_k + rc.invest.watch_ranks].iterrows() if step.kind == "rebalance" else []:
        rr = r[1]
        iid = int(rr["instrument_id"])
        ctx = dc.market.ctx(dc.i, iid, dc.symbols.at(iid, dc.d), None, {c: rr.get(c) for c in ("mom_6m_csrank", "dd_252", "mom_6m", "vol_126", "sma_ratio_200")})
        if ctx is not None:
            cards.append(B.build_invest_card(ctx, _invest_pred(rr, model, mid, name), key, preset, None, None, dc.verdicts.get(sleeve), rc, setup.rules, weight=budget / iv.top_k,
                                             next_review=nxt_review, top_k=iv.top_k, rs_floor=iv.thesis_rs_floor, watch=True))
    return cards, {"sleeve": sleeve, "kind": step.kind, "tranche": step.n, "of": step.of, "names": len(cur), "reused": existing is not None}


class _Empty:
    current: dict = {}
