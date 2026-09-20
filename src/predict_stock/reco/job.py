"""The `reco` jobs: backtest of the cards exactly as issued (`reco backtest`) and today's cards (`reco generate`)."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
from sqlalchemy import Engine, delete, select

from predict_stock.backtest import metrics as M
from predict_stock.backtest.baselines import BASELINES, BaselineSpec, make_signals
from predict_stock.backtest.report import save_equity_artifacts
from predict_stock.backtest.runner import Setup, clean, evaluate_index, load_setup
from predict_stock.config import PROJECT_ROOT, AppConfig
from predict_stock.db.models import Model, Recommendation, Universe
from predict_stock.db.session import session_scope
from predict_stock.features.registry import spec_hash
from predict_stock.invest import stats as IS
from predict_stock.reco import aggregate as AG
from predict_stock.reco import analyze as AN
from predict_stock.reco import backtest as BT
from predict_stock.reco import builders as B
from predict_stock.reco import sources as S
from predict_stock.reco.cards import Card
from predict_stock.runs import tracked_run
from predict_stock.swing.evaluate import run_signals
from predict_stock.swing.registry import log_experiment, save_predictions

SLEEVES = ("swing", "invest_b1", "invest_b2")


def universe_id(engine: Engine, code: str) -> int:
    with session_scope(engine) as s:
        return s.scalar(select(Universe.id).where(Universe.code == code))


def stored_verdicts(cfg: AppConfig) -> dict[str, dict | None]:
    """The pre-registered verdict of each sleeve's model, from its latest stored development run (what the cards must show)."""
    from predict_stock.invest.report import latest_payloads
    from predict_stock.swing.report import latest_payload_path
    out: dict[str, dict | None] = {"swing": None, "invest_b1": None, "invest_b2": None}
    try:
        out["swing"] = json.loads(latest_payload_path(cfg).read_text(encoding="utf-8"))["decision"]
    except (FileNotFoundError, KeyError):
        pass
    try:
        for k, p in latest_payloads(cfg).items():
            out[f"invest_{k}"] = p["decision"]
    except FileNotFoundError:
        pass
    return out


def _metrics(res, cfg: AppConfig) -> dict:
    return clean(M.compute_metrics(res.equity, trips=res.round_trips, fills=res.fills, exposure=res.exposure, rf_annual=cfg.backtest.risk_free_annual))


def run_backtest_job(engine: Engine, cfg: AppConfig, *, write: bool = True) -> dict:
    rc = cfg.reco
    with tracked_run(engine, "reco_backtest", cfg, {"kill_switch": rc.kill_switch}) as (run_id, stats):
        setup = load_setup(engine, cfg)
        uid = universe_id(engine, setup.universe)
        verdicts = stored_verdicts(cfg)
        prereg = {"reco": rc.model_dump(), "verdicts": {k: (v or {}).get("passed") for k, v in verdicts.items()}, "swing_dataset": setup.dataset_hashes["swing"], "invest_dataset": setup.dataset_hashes["invest"]}
        log_experiment(engine, cfg, "reco:preregistration", key=spec_hash(prereg), params={"universe": setup.universe}, summary=clean(prereg), status="registered",
                       description="Card rules, portfolio limits and display policy fixed before the cards were backtested", run_id=run_id)
        swing = S.load_swing_history(engine, cfg, setup, uid)
        invests = {k: S.load_invest_history(engine, cfg, setup, k) for k in ("b1", "b2")}
        market, symbols = S.Market(setup, cfg), S.Symbols(engine, list(setup.data.close.columns))
        evidence = S.Evidence(swing.preds)
        first = int(setup.calendar.get_loc(max(swing.first_test, *(h.first_test for h in invests.values()))))
        last = int(setup.calendar.get_loc(min(swing.last_test, *(h.last_test for h in invests.values()))))
        local = replace(setup, start_idx=first, dev_end_idx=last)
        plan = BT.build_plan(cfg, setup, market, symbols, swing, evidence, invests, verdicts, first, last, universe_id=uid)
        vmd = BT.virtual_market(setup.data, list(SLEEVES))
        signals = BT.to_signals(plan)
        mults = sorted(set(cfg.backtest.cost_multipliers) | {0.0, 1.0, 2.0})
        runs = {m: BT.run_book(setup, cfg, vmd, signals, first, last, m, kill=True) for m in mults}
        net, gross, stress = runs[1.0], runs[0.0], runs[2.0]
        no_kill = BT.run_book(setup, cfg, vmd, signals, first, last, 1.0, kill=False)
        sleeve_runs = {}
        for sl in SLEEVES:                                                     # attribution: the same cards, one sleeve at a time, no kill-switch
            sleeve_runs[sl] = BT.run_book(setup, cfg, vmd, _sleeve_signals(plan, sl), first, last, 1.0, kill=False)
        # ---- benchmarks over the same window ------------------------------------------------------------------------------------------------
        bench = _benchmarks(local, first, last, cfg)
        # ---- statistics -------------------------------------------------------------------------------------------------------------------------
        bs = cfg.invest.bootstrap
        kw = dict(resamples=bs.resamples, block=bs.block, level=bs.level, seed=bs.seed)
        r_net = IS.daily_returns(net.equity)
        ci = {"vs_equal_weight": IS.paired_bootstrap(r_net, IS.daily_returns(bench["equal_weight"]["equity"]), **kw), "alone": IS.paired_bootstrap(r_net, None, **kw)}
        for b in ("bh_vn30", "bh_vnindex"):
            if b in bench:
                ci[f"vs_{b}"] = IS.paired_bootstrap(r_net, IS.daily_returns(bench[b]["equity"]), **kw)
        rolling = {b: {str(w): IS.rolling_outperformance(net.equity, bench[b]["equity"], w) for w in cfg.invest.rolling_windows} for b in bench}
        # ---- what happened to the cards -----------------------------------------------------------------------------------------------------------
        labels = {(pd.Timestamp(d), int(i)): t for d, i, t in zip(swing.preds["trade_date"], swing.preds["instrument_id"], swing.preds["tb_label"])}
        sw_out = AN.swing_outcomes(net, plan.cards, labels)
        inv_out = {}
        for k, h in invests.items():
            preset = cfg.invest.presets[k]
            end = pd.to_datetime(h.preds[f"fwd_end_{preset.horizon}"])
            ok = h.preds[end.notna() & (end < setup.holdout.start) & h.preds[f"fwd_ret_{preset.horizon}"].notna()]
            card_keys = {(pd.Timestamp(c.as_of), c.instrument_id) for c in plan.cards.values() if c.strategy == f"INVEST_{k.upper()}"}
            q = ok[[(pd.Timestamp(d), int(i)) in card_keys for d, i in zip(ok["trade_date"], ok["instrument_id"])]]
            inv_out[k] = AN.invest_outcomes(net, plan.cards, k, q.assign(y=q[f"fwd_ret_{preset.horizon}"])[["y", "q10", "q50", "q90"]] if len(q) else None, preset.horizon)
        # ---- overlaps between sleeves -----------------------------------------------------------------------------------------------------------
        overlap = _overlap(plan, first, last)
        payload = clean({
            "window": [str(setup.calendar[first].date()), str(setup.calendar[last].date())], "universe": setup.universe, "holdout": [str(setup.holdout.start.date()), str(setup.holdout.end.date())],
            "rules": rc.model_dump(), "verdicts": {k: (None if v is None else {"passed": v["passed"], "failed": [c["name"] for c in v["criteria"] if not c["ok"]]}) for k, v in verdicts.items()},
            "plan": plan.stats, "cards": {"buy": len(plan.cards), "watch": len(plan.watch), "no_trade": len(plan.no_trade)},
            "portfolio": {"net": _metrics(net, cfg), "gross": _metrics(gross, cfg), "stress2": _metrics(stress, cfg), "no_kill_switch": _metrics(no_kill, cfg),
                          "sensitivity": {str(m): {k: v for k, v in _metrics(r, cfg).items() if k in ("cagr", "sharpe", "max_drawdown", "turnover_annual")} for m, r in runs.items()},
                          "exposure": AN.exposure_stats(net, plan), "exposure_no_kill": AN.exposure_stats(no_kill, plan)},
            "sleeves": {sl: {**_metrics(r, cfg), "pnl_pct_capital": float(r.equity.iloc[-1] / r.equity.iloc[0] - 1), "avg_exposure": float(r.exposure.mean())} for sl, r in sleeve_runs.items()},
            "sleeves_in_book": {"with_kill_switch": _book_pnl(net, rc.portfolio.capital), "without_kill_switch": _book_pnl(no_kill, rc.portfolio.capital)},
            "benchmarks": {k: {kk: vv for kk, vv in v.items() if kk != "equity"} for k, v in bench.items()}, "bootstrap": ci, "rolling": rolling,
            "swing": sw_out, "invest": inv_out, "overlap": overlap, "run_key": spec_hash({"prereg": prereg, "window": [first, last]})[:16]})
        equities = {"portfolio_net": net.equity, "portfolio_gross": gross.equity, "portfolio_no_kill": no_kill.equity, **{f"sleeve_{k}": r.equity for k, r in sleeve_runs.items()},
                    **{k: v["equity"] for k, v in bench.items()}}
        arts = save_equity_artifacts(cfg, f"reco_{payload['run_key']}", {k: {"net": e} for k, e in equities.items()})
        payload["artifacts"] = arts
        out_dir = PROJECT_ROOT / rc.artifacts_dir / "runs" / payload["run_key"]
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_samples(plan, out_dir)
        (out_dir / "payload.json").write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        log_experiment(engine, cfg, f"reco:backtest:{payload['run_key']}", key=payload["run_key"], params={"window": payload["window"]}, summary=payload, status="success",
                       description="Backtest of the recommendation cards exactly as issued", run_id=run_id)
        stats.update(run_key=payload["run_key"], cards=payload["cards"])
    if write:
        from predict_stock.reco.report import write_report
        write_report(cfg, payload, {k: {"net": e} for k, e in equities.items()}, engine)
    return {"payload": payload, "equities": equities, "plan": plan}


def _book_pnl(res, capital: float) -> dict:
    """P&L of each sleeve INSIDE the shared book, % of starting capital: cash flows of its fills (net of fees and tax) plus the value of what it still holds."""
    f = res.fills.copy()
    f["sleeve"] = f["instrument_id"].map(lambda v: BT.unvid(v)[0])
    f["cash"] = np.where(f["side"] == "buy", -(f["value"] + f["fee"]), f["value"] - f["fee"] - f["tax"])
    out = f.groupby("sleeve")["cash"].sum()
    op = res.open_positions
    if len(op):
        held = op.assign(sleeve=op["instrument_id"].map(lambda v: BT.unvid(v)[0])).groupby("sleeve")["value"].sum()
        out = out.add(held, fill_value=0.0)
    return {k: float(v / capital) for k, v in out.items()}


def _sleeve_signals(plan: BT.CardPlan, sleeve: str) -> dict:
    """The combined plan's signals restricted to one sleeve (INVEST sleeves): items whose virtual id belongs to it."""
    out = {}
    for i, items in plan.signals.items():
        keep = [it for it in items if BT.unvid(it.instrument_id)[0] == sleeve]
        if keep:
            out[i] = BT.Signal(keep, full_rebalance=False)
    return out


def _benchmarks(local: Setup, first: int, last: int, cfg: AppConfig) -> dict:
    bt = cfg.backtest
    out = {}
    for key in ("equal_weight", "mom_long"):
        spec = BASELINES[key]
        spec_k = spec if spec.k is None else BaselineSpec(**{**spec.__dict__, "k": bt.top_k})
        sig = make_signals(spec_k, local.frames[spec.dataset], local.calendar, first, last, bt.max_weight)
        res = {m: run_signals(local, sig, first, m) for m in (0.0, 1.0)}
        out[key] = {"title": spec.title, "net": _metrics(res[1.0], cfg), "gross": _metrics(res[0.0], cfg), "equity": res[1.0].equity}
    for key in ("bh_vn30", "bh_vnindex"):
        spec = BASELINES[key]
        if spec.index_symbol in local.index_close:
            summary, eq = evaluate_index(local, spec)
            out[key] = {"title": spec.title, "net": summary["net"], "gross": summary["gross"], "equity": eq["net"]}
    return out


def _overlap(plan: BT.CardPlan, first: int, last: int) -> dict:
    """Days on which the same stock is a target of two sleeves, and the largest combined target weight of a name (SWING cards issued that day + INVEST targets)."""
    by_name: dict[tuple[int, int], float] = {}
    for sl, t in plan.invest_targets.items():
        for i, w in t.items():
            for iid, x in w.items():
                by_name[(i, iid)] = by_name.get((i, iid), 0.0) + x
    both = {}
    for i, items in plan.signals.items():
        for it in items:
            sl, iid = BT.unvid(it.instrument_id)
            if sl == "swing" and it.weight > 0 and (i, iid) in by_name:
                both[(i, iid)] = by_name[(i, iid)] + it.weight
    return {"days_with_same_stock_in_swing_and_invest": len({i for i, _ in both}), "cards_overlapping_invest_targets": len(both),
            "max_combined_target_weight": float(max(both.values())) if both else None,
            "cap": float(0.15)}


def _save_samples(plan: BT.CardPlan, out_dir) -> None:
    def pick(pred, n=2):
        return [c.to_dict() for c in list(filter(pred, plan.cards.values()))[:: max(1, len(plan.cards) // 200)][:n]]
    sample = {"swing_buy": pick(lambda c: c.strategy == "SWING"), "invest_b1_buy": pick(lambda c: c.strategy == "INVEST_B1"), "invest_b2_buy": pick(lambda c: c.strategy == "INVEST_B2"),
              "no_trade": [c.to_dict() for c in plan.no_trade[:5]]}
    (out_dir / "sample_cards.json").write_text(json.dumps(sample, indent=1, ensure_ascii=False), encoding="utf-8")


# ---- live cards -----------------------------------------------------------------------------------------------------------------------------------------
def next_schedule_date(as_of: pd.Timestamp, freq: str) -> pd.Timestamp:
    """First weekday of the next month / quarter after ``as_of`` (public holidays are not known in advance)."""
    d = pd.Timestamp(as_of)
    if freq == "quarterly":
        q = (d.month - 1) // 3
        nxt = pd.Timestamp(d.year + (1 if q == 3 else 0), 1 if q == 3 else (q + 1) * 3 + 1, 1)
    elif freq == "monthly":
        nxt = pd.Timestamp(d.year + (1 if d.month == 12 else 0), 1 if d.month == 12 else d.month + 1, 1)
    else:
        nxt = d + pd.offsets.Week(weekday=0)
    return pd.Timestamp(np.busday_offset(nxt.date(), 0, roll="forward"))


def generate_live(engine: Engine, cfg: AppConfig, as_of: str | None = None, *, write_db: bool = True, write_files: bool = True) -> dict:
    """Today's cards: SWING from the final SWING model, INVEST B1 / B2 from the final candidate; aggregated with the sleeve budgets and the per-name cap; BUY / WATCH cards
    are written to `recommendations`, NO_TRADE cards (with their reasons) only to the report. Uses no return after ``as_of``: nothing here evaluates the held-out period."""
    from predict_stock.swing.folds import split_fold, with_ends
    from predict_stock.swing.model import load_bundle
    from predict_stock.swing.walkforward import final_fold, predict_rows
    rc, iv, sw = cfg.reco, cfg.invest, cfg.swing
    with tracked_run(engine, "reco_generate", cfg, {"as_of": as_of}) as (run_id, stats):
        setup = load_setup(engine, cfg)
        uid = universe_id(engine, setup.universe)
        cal = setup.calendar
        d = pd.Timestamp(as_of) if as_of else cal[-1]
        d = cal[cal <= d][-1]
        i = int(cal.get_loc(d))
        verdicts = stored_verdicts(cfg)
        market, symbols = S.Market(setup, cfg), S.Symbols(engine, list(setup.data.close.columns))
        rules = setup.rules
        cards: list[Card] = []
        # ---- INVEST (targets first: they limit what SWING may add) ------------------------------------------------------------------------------------
        holdings: dict[str, dict[int, float]] = {}
        frame_inv = setup.frames["invest"]
        rows_inv = frame_inv[pd.to_datetime(frame_inv["trade_date"]) == d]
        budgets = AG.sleeve_budgets(rc)
        for key, preset in iv.presets.items():
            sleeve, cand = f"invest_{key}", rc.card_model[key]
            mid, path, sha = S._latest_model(engine, f"invest_{key}_{cand}_final")
            from predict_stock.invest.models import load_candidate, sha256
            raw = (PROJECT_ROOT / path).read_bytes()
            assert sha256(raw) == sha, f"{path}: sha256 mismatch"
            model, scen, _ = load_candidate(raw)
            ih = S.load_invest_history(engine, cfg, setup, key)
            sim = S.invest_similar(ih, d, iv.top_k, preset.horizon)
            p = rows_inv.assign(score=model.predict(rows_inv)).join(scen.predict(rows_inv))
            p["rank_in_universe"] = p["score"].rank(ascending=False, method="first")
            p["n_universe"] = len(p)
            p = p.sort_values("rank_in_universe")
            nxt = next_schedule_date(d, preset.rebalance)
            w = budgets[sleeve] / iv.top_k
            holdings[sleeve] = {}
            for k, (_, r) in enumerate(p.head(iv.top_k + rc.invest.watch_ranks).iterrows()):
                iid = int(r["instrument_id"])
                ctx = market.ctx(i, iid, symbols.at(iid, d), None, {c: r.get(c) for c in ("mom_6m_csrank", "dd_252", "mom_6m", "vol_126", "sma_ratio_200")})
                if ctx is None:
                    continue
                pr = BT._invest_pred(r, model, mid, f"invest_{key}_{cand}_final")
                pr.universe_id = uid
                watch = k >= iv.top_k
                card = B.build_invest_card(ctx, pr, key, preset, sim, None, verdicts.get(sleeve), rc, rules, weight=w / preset.tranches if not watch else w, next_review=nxt, top_k=iv.top_k,
                                           rs_floor=iv.thesis_rs_floor, watch=watch, tranche=(1, preset.tranches, w))
                cards.append(card)
                if not watch and card.action == "BUY":
                    holdings[sleeve][iid] = w
        # ---- SWING -----------------------------------------------------------------------------------------------------------------------------------
        frame = with_ends(setup.frames["swing"])
        mid, path, sha = S._latest_model(engine, "swing_lgbm_final")
        bundle = load_bundle(PROJECT_ROOT / path, sha)
        rows = frame[pd.to_datetime(frame["trade_date"]) == d]
        pred = predict_rows(bundle, rows)
        pred = pred.join(rows[[c for c in S.CONTEXT_FEATURES if c in rows.columns and c not in pred.columns]])
        pred["n_universe"] = len(pred)
        top = pred.sort_values("rank_in_universe").head(sw_top(rc))
        contrib, _ = bundle.contributions(rows.loc[top.index], "rank")
        fold = final_fold(frame, sw, setup.holdout.start, setup.holdout.end)
        dsplit = split_fold(frame, fold, sw)
        atr_pct = setup.frames["swing"].pivot(index="trade_date", columns="instrument_id", values="atr_pct_14")
        t1 = S.barrier_1_1(setup, atr_pct, sw.horizon)
        live = S.SwingHistory(pred, {}, {-1: S._similar_table(bundle, dsplit.val, t1)}, {-1: bundle}, {-1: (mid, "swing_lgbm_final")})
        hist = S.load_swing_history(engine, cfg, setup, uid)
        evidence = S.Evidence(hist.preds).at(d)
        buys: list[Card] = []
        for k, (idx, r) in enumerate(top.iterrows()):
            iid = int(r["instrument_id"])
            order = np.argsort(-np.abs(contrib[k]), kind="stable")[:5]
            cl = [[bundle.features[j], S._finite(rows.loc[idx, bundle.features[j]]), float(contrib[k][j])] for j in order]
            atr = float(r["atr_pct_14"]) * float(market._close[i, market.col[iid]])
            ctx = market.ctx(i, iid, symbols.at(iid, d), atr, {c: r.get(c) for c in S.CONTEXT_FEATURES})
            if ctx is None:
                continue
            pr = B.Pred(score=float(r["score"]), rank=int(r["rank_in_universe"]), n_universe=int(r["n_universe"]), model_id=mid, model_name="swing_lgbm_final", universe_id=uid,
                        proba_raw=float(r["proba_raw"]), proba=float(r["proba"]), q10=float(r["q10"]), q50=float(r["q50"]), q90=float(r["q90"]), hold_median=float(r["hold_median"]),
                        hold_p75=float(r["hold_p75"]), hold_n=int(r["hold_n"]), contributions=cl)
            card = B.build_swing_card(ctx, pr, S.similar_for(live, -1, pr.proba), evidence, verdicts.get("swing"), rc, rules, watch=k >= rc.swing.max_positions)
            (buys if card.action == "BUY" else cards).append(card)
        agg = AG.aggregate(buys, rc, rules, holdings=holdings)
        cards += agg.accepted + agg.rejected
        # ---- output -------------------------------------------------------------------------------------------------------------------------------------
        result = {"as_of": str(d.date()), "cards": cards, "summary": agg.summary, "evidence": evidence}
        if write_db:
            from predict_stock.swing.registry import build_details
            c_r, _ = bundle.contributions(rows, "rank")
            c_e, _ = bundle.contributions(rows, "event")
            pp = pred.assign(details=build_details(pred, c_r, c_e, bundle.features, rows[bundle.features].to_numpy(float)))
            save_predictions(engine, mid, uid, sw.horizon, pp, run_id)
            save_live_cards(engine, cfg, [c for c in cards if c.action in ("BUY", "WATCH")], pred, mid, uid, run_id)
        if write_files:
            out = PROJECT_ROOT / rc.artifacts_dir / "live" / str(d.date())
            out.mkdir(parents=True, exist_ok=True)
            for c in cards:
                (out / f"{c.card_id.replace(':', '_')}.json").write_text(c.to_json(), encoding="utf-8")
                (out / f"{c.card_id.replace(':', '_')}.txt").write_text(c.text_vi(), encoding="utf-8")
        stats.update(as_of=str(d.date()), buy=sum(c.action == "BUY" for c in cards), watch=sum(c.action == "WATCH" for c in cards), no_trade=sum(c.action == "NO_TRADE" for c in cards))
    return result


def sw_top(rc) -> int:
    return rc.swing.max_positions + 4


def save_live_cards(engine: Engine, cfg: AppConfig, cards: list[Card], swing_pred: pd.DataFrame, swing_model_id: int, uid: int, run_id: int | None) -> int:
    """BUY / WATCH cards into `recommendations` (idempotent per strategy, instrument, date and action); the SWING cards are linked to their stored prediction."""
    from predict_stock.db.models import Prediction
    n = 0
    with session_scope(engine) as s:
        for c in cards:
            pid = None
            if c.strategy == "SWING":
                pid = s.scalar(select(Prediction.id).where(Prediction.model_id == c.model_id, Prediction.instrument_id == c.instrument_id, Prediction.as_of_date == pd.Timestamp(c.as_of).date()))
            s.execute(delete(Recommendation).where(Recommendation.strategy == c.strategy, Recommendation.instrument_id == c.instrument_id,
                                                   Recommendation.as_of_date == pd.Timestamp(c.as_of).date(), Recommendation.action == c.action))
            x, e = c.exits, c.entry
            target = x["target2"] if c.strategy == "SWING" else x["scenarios"]["bull"]["price"]
            stop = x["stop"] if c.strategy == "SWING" else x["bear_reference"]
            s.add(Recommendation(strategy=c.strategy, instrument_id=c.instrument_id, universe_id=uid, as_of_date=pd.Timestamp(c.as_of).date(), action=c.action, entry_price=e["reference"],
                                 target_price=target, stop_loss=stop, hold_days_min=cfg.market.settlement_days, hold_days_max=c.holding["max_sessions"],
                                 exit_conditions=c.to_dict()["exits"] | {"stop_is_hard": c.strategy == "SWING", "cancel": e.get("cancel_conditions")},
                                 rationale="\n".join(["LÝ DO: "] + c.rationale["reasons"] + ["RỦI RO: "] + c.rationale["risks"] + ["MẤT HIỆU LỰC: "] + c.rationale["invalidation"]),
                                 rationale_data={"contributions": c.rationale.get("facts") or [], "signal": c.signal}, confidence=c.confidence.get("p_display"), model_id=c.model_id,
                                 prediction_id=pid, status="open", run_id=run_id, valid_until=pd.Timestamp(c.valid_until).date(), card=c.to_dict(), card_text=c.text_vi()))
            n += 1
    return n
