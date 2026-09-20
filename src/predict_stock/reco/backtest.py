"""Backtest of the recommendation cards EXACTLY as they would have been issued.

For every session of the window: the SWING cards (top-ranked names that pass the rules) and, on the scheduled dates, the INVEST cards (B1, B2) are built from
the out-of-sample predictions of the stored walk-forward models, run through the aggregator (sleeve budgets, per-name cap over all sleeves), and each card
is translated into the engine order it states: price zone, order type, cancel condition, validity, stop, target 1 / 2, time-stop, size. One shared book of
cash; each sleeve is its own set of positions (a stock held in two sleeves is two positions, modelled as two price columns of the same stock)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from predict_stock.backtest.baselines import rebalance_sessions
from predict_stock.backtest.engine import EngineConfig, MarketData, Signal, SignalItem, run_backtest
from predict_stock.backtest.runner import Setup
from predict_stock.config import AppConfig
from predict_stock.invest import walkforward as IW
from predict_stock.reco import aggregate as AG
from predict_stock.reco import builders as B
from predict_stock.reco import sources as S
from predict_stock.reco.cards import Card

SLEEVE_CODE = {"swing": 1, "invest_b1": 2, "invest_b2": 3}
OFFSET = 1_000_000


def vid(sleeve: str, iid: int) -> int:
    return SLEEVE_CODE[sleeve] * OFFSET + int(iid)


def unvid(v: int) -> tuple[str, int]:
    code = {c: s for s, c in SLEEVE_CODE.items()}
    return code[int(v) // OFFSET], int(v) % OFFSET


def virtual_market(md: MarketData, sleeves: list[str]) -> MarketData:
    """One price column per (sleeve, stock): the sleeves share prices and cash but never a position."""
    def dup(df: pd.DataFrame) -> pd.DataFrame:
        parts = [df.set_axis([vid(s, c) for c in df.columns], axis=1) for s in sleeves]
        return pd.concat(parts, axis=1)
    return MarketData(dup(md.open), dup(md.high), dup(md.low), dup(md.close), dup(md.bands))


@dataclass
class CardPlan:
    cards: dict[str, Card] = field(default_factory=dict)                       # BUY cards that became orders (by card id)
    watch: list[Card] = field(default_factory=list)
    no_trade: list[Card] = field(default_factory=list)
    signals: dict[int, list[SignalItem]] = field(default_factory=dict)          # session index -> items (engine ids are virtual)
    invest_targets: dict[str, dict[int, dict[int, float]]] = field(default_factory=dict)   # sleeve -> session index -> {iid: target weight}
    stats: dict = field(default_factory=dict)
    universe_id: int | None = None


def _items_add(plan: CardPlan, i: int, item: SignalItem) -> None:
    plan.signals.setdefault(i, []).append(item)


def plan_invest(plan: CardPlan, cfg: AppConfig, setup: Setup, market: S.Market, symbols: S.Symbols, hist: S.InvestHistory, verdict: dict | None, first: int, last: int) -> None:
    iv, rc = cfg.invest, cfg.reco
    key = hist.key
    preset, sleeve = iv.presets[key], f"invest_{key}"
    budget = AG.sleeve_budgets(rc)[sleeve]
    ret_col = f"fwd_ret_{preset.horizon}"
    by_date = {d: g.sort_values("rank_in_universe") for d, g in hist.preds.groupby("trade_date")}
    sched = [i for i in rebalance_sessions(setup.calendar, preset.rebalance, first, last) if setup.calendar[i] in by_date]
    prev: dict[int, float] = {}
    plan.invest_targets[sleeve] = {}
    n_cards = n_rej = 0
    for n, i in enumerate(sched):
        rows = by_date[setup.calendar[i]]
        top = rows.head(iv.top_k)
        target = {int(r.instrument_id): budget / iv.top_k for r in top.itertuples()}
        nxt = sched[n + 1] if n + 1 < len(sched) else last + 1
        next_review = setup.calendar[min(nxt, len(setup.calendar) - 1)]
        cur = prev
        for j in range(preset.tranches):
            idx = i + j * preset.tranche_spacing
            if idx >= nxt or idx > last:
                break
            frac = (j + 1) / preset.tranches
            cur = {a: prev.get(a, 0.0) + (target.get(a, 0.0) - prev.get(a, 0.0)) * frac for a in set(prev) | set(target)}
            day = by_date.get(setup.calendar[idx])
            for iid, w in sorted(cur.items()):
                if w <= 1e-9:
                    _items_add(plan, idx, SignalItem(vid(sleeve, iid), 0.0, tag=f"rebalance:{sleeve}", group=sleeve))
                    continue
                rowset = day[day["instrument_id"] == iid] if day is not None else None
                if iid in target and w > prev.get(iid, 0.0) + 1e-12 and rowset is not None and len(rowset):
                    r = rowset.iloc[0]
                    model_id, mname, model = hist.models[int(r["fold"])]
                    ctx = market.ctx(idx, iid, symbols.at(iid, setup.calendar[idx]), None, {k: r.get(k) for k in ("mom_6m_csrank", "dd_252", "mom_6m", "vol_126", "sma_ratio_200")})
                    if ctx is None:
                        continue
                    pr = _invest_pred(r, model, model_id, mname)
                    pr.universe_id = plan.universe_id
                    card = B.build_invest_card(ctx, pr, key, preset, S.invest_similar(hist, setup.calendar[idx], iv.top_k, preset.horizon), None, verdict, rc, setup.rules, weight=w,
                                               next_review=next_review, top_k=iv.top_k, rs_floor=iv.thesis_rs_floor, tranche=(j + 1, preset.tranches, target.get(iid, 0.0)))
                    card.card_id = f"{card.strategy}:{card.symbol}:{setup.calendar[idx].date()}:t{j + 1}"
                    if card.action == "BUY":
                        plan.cards[card.card_id] = card
                        _items_add(plan, idx, _to_item(card, sleeve, iid, w))
                        n_cards += 1
                    else:                                                              # a rejected card is no order: the backtest buys only what a card asks for
                        plan.no_trade.append(card)
                        n_rej += 1
                else:
                    _items_add(plan, idx, SignalItem(vid(sleeve, iid), w, tag=f"rebalance:{sleeve}", group=sleeve))
            plan.invest_targets[sleeve][idx] = dict(cur)
        for r in rows.iloc[iv.top_k: iv.top_k + rc.invest.watch_ranks].itertuples():
            rr = rows[rows["instrument_id"] == r.instrument_id].iloc[0]
            model_id, mname, model = hist.models[int(rr["fold"])]
            ctx = market.ctx(i, int(r.instrument_id), symbols.at(int(r.instrument_id), setup.calendar[i]), None, {k: rr.get(k) for k in ("mom_6m_csrank", "dd_252", "mom_6m", "vol_126", "sma_ratio_200")})
            if ctx is not None:
                w = budget / iv.top_k
                wp = _invest_pred(rr, model, model_id, mname)
                wp.universe_id = plan.universe_id
                plan.watch.append(B.build_invest_card(ctx, wp, key, preset, None, None, verdict, rc, setup.rules, weight=w, next_review=next_review,
                                                     top_k=iv.top_k, rs_floor=iv.thesis_rs_floor, watch=True))
        prev = cur
    plan.stats[sleeve] = {"rebalances": len(sched), "buy_cards": n_cards, "rejected": n_rej}


def _invest_pred(r: pd.Series, model, model_id: int, mname: str) -> B.Pred:
    one = pd.DataFrame([r])
    contrib, names = model.contributions(one)
    inputs = model.inputs()
    order = np.argsort(-np.abs(contrib[0]), kind="stable")[:5]
    lst = [[names[j], (None if not np.isfinite(r.get(inputs[j], np.nan)) else float(r[inputs[j]])), float(contrib[0][j])] for j in order]
    return B.Pred(score=float(r["score"]), rank=int(r["rank_in_universe"]), n_universe=int(r["n_universe"]), model_id=model_id, model_name=mname, universe_id=None,
                  q10=float(r["q10"]), q50=float(r["q50"]), q90=float(r["q90"]), contributions=lst)


def _to_item(card: Card, sleeve: str, iid: int, weight: float) -> SignalItem:
    it = card.to_signal_item(weight=weight, group=sleeve)
    return replace(it, instrument_id=vid(sleeve, iid))


def plan_swing(plan: CardPlan, cfg: AppConfig, setup: Setup, market: S.Market, symbols: S.Symbols, hist: S.SwingHistory, evidence: S.Evidence, verdict: dict | None,
               first: int, last: int) -> None:
    rc = cfg.reco
    by_date = {d: g.sort_values("rank_in_universe").head(rc.swing.max_positions + 4) for d, g in hist.preds.groupby("trade_date")}
    counts = {"days": 0, "cards": 0, "watch": 0, "rejected": 0}
    reasons: dict[str, int] = {}
    for i in range(first, last + 1):
        d = setup.calendar[i]
        day = by_date.get(d)
        if day is None or day.empty:
            continue
        counts["days"] += 1
        ev = evidence.at(d)
        buys: list[Card] = []
        for k, (_, r) in enumerate(day.iterrows()):
            iid = int(r["instrument_id"])
            atr = float(r["atr_pct_14"]) * float(market._close[i, market.col[iid]])
            ctx = market.ctx(i, iid, symbols.at(iid, d), atr, {c: r.get(c) for c in S.CONTEXT_FEATURES})
            if ctx is None:
                continue
            pr = S.swing_pred(hist, r)
            pr.universe_id = plan.universe_id
            watch = k >= rc.swing.max_positions
            card = B.build_swing_card(ctx, pr, S.similar_for(hist, int(r["fold"]), pr.proba), ev, verdict, rc, setup.rules, watch=watch)
            if card.action == "NO_TRADE":
                counts["rejected"] += 1
                key = card.rejected.split(" =")[0].split(" (")[0][:60]
                reasons[key] = reasons.get(key, 0) + 1
                plan.no_trade.append(card)
            elif card.action == "WATCH":
                counts["watch"] += 1
                plan.watch.append(card)
            else:
                buys.append(card)
        holdings = {s: dict(t[max(k for k in t if k <= i)]) for s, t in plan.invest_targets.items() if t and min(t) <= i}
        agg = AG.aggregate(buys, rc, setup.rules, holdings=holdings)
        for c in agg.rejected:
            plan.no_trade.append(c)
            counts["rejected"] += 1
            reasons[c.rejected.split(" (")[0][:60]] = reasons.get(c.rejected.split(" (")[0][:60], 0) + 1
        for c in agg.accepted:
            if c.action == "BUY":
                plan.cards[c.card_id] = c
                _items_add(plan, i, _to_item(c, "swing", c.instrument_id, c.sizing["weight"]))
                counts["cards"] += 1
    plan.stats["swing"] = {**counts, "no_trade_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1]))}


def to_signals(plan: CardPlan) -> dict[int, Signal]:
    return {i: Signal(items, full_rebalance=False) for i, items in plan.signals.items()}


def run_book(setup: Setup, cfg: AppConfig, vmd: MarketData, signals: dict[int, Signal], first: int, last: int, mult: float, kill: bool = True):
    rc, bt = cfg.reco, cfg.backtest
    ec = EngineConfig(capital=rc.portfolio.capital, rebalance_threshold=bt.rebalance_threshold, tie=bt.tie, max_positions_by_group={"swing": rc.swing.max_positions},
                      kill_drawdown=rc.portfolio.kill_drawdown if (kill and rc.kill_switch) else None, kill_cooldown=rc.portfolio.kill_cooldown)
    return run_backtest(vmd, signals, setup.rules.scaled_costs(mult), ec, start=first, end=last + 1)


def build_plan(cfg: AppConfig, setup: Setup, market: S.Market, symbols: S.Symbols, swing: S.SwingHistory, evidence: S.Evidence, invests: dict[str, S.InvestHistory],
               verdicts: dict[str, dict | None], first: int, last: int, sleeves: tuple[str, ...] = ("swing", "invest_b1", "invest_b2"), universe_id: int | None = None) -> CardPlan:
    plan = CardPlan(universe_id=universe_id)
    for key, h in invests.items():
        if f"invest_{key}" in sleeves:
            plan_invest(plan, cfg, setup, market, symbols, h, verdicts.get(f"invest_{key}"), first, last)
    if "swing" in sleeves:
        plan_swing(plan, cfg, setup, market, symbols, swing, evidence, verdicts.get("swing"), first, last)
    return plan
