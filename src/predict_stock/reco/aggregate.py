"""The aggregator: combines the cards of the two sleeves (SWING, INVEST B1 / B2) into one portfolio view.

* capital is split between the sleeves (default 30% SWING / 70% INVEST, INVEST shared by B1 and B2), every weight is a share of TOTAL capital;
* positions are tracked per sleeve, so the same stock held in SWING and in INVEST is two positions with their own entry, stop and exit rules;
* limits: per-sleeve exposure, per-name total across sleeves, open risk of the SWING positions, a kill-switch on the portfolio drawdown;
* a card that would break a limit is scaled down to the room left (whole lots) when at least half of it fits, otherwise rejected with the reason."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from predict_stock.backtest.market import MarketRules
from predict_stock.config import RecoConfig
from predict_stock.reco.cards import Card, fmt_pct

SLEEVES = ("swing", "invest_b1", "invest_b2")


def sleeve_budgets(cfg: RecoConfig) -> dict[str, float]:
    """Share of total capital each sleeve may hold."""
    a, sp = cfg.portfolio.allocation, cfg.portfolio.invest_split
    return {"swing": a["swing"], "invest_b1": a["invest"] * sp.get("b1", 0.5), "invest_b2": a["invest"] * sp.get("b2", 0.5)}


def kill_state(equity: pd.Series, drawdown: float, cooldown: int) -> dict:
    """The kill-switch on an equity series, with the same rule as the backtest engine: a drawdown from the peak beyond ``drawdown`` stops all trading for
    ``cooldown`` sessions, after which the peak restarts from the then-current equity. Returns the state at the last point."""
    peak, killed, until, events = float(equity.iloc[0]), False, -1, []
    for i, v in enumerate(equity.to_numpy(float)):
        if killed and i >= until:
            killed, peak = False, v
        if not killed:
            peak = max(peak, v)
            if v / peak - 1.0 <= -drawdown:
                killed, until = True, i + cooldown
                events.append(str(equity.index[i].date()))
    last = equity.iloc[-1] / max(peak, 1e-12) - 1.0
    return {"killed": killed, "drawdown_from_peak": float(last), "events": events, "sessions_left": max(until - (len(equity) - 1), 0) if killed else 0}


def rescale(card: Card, weight: float, rules: MarketRules, cfg: RecoConfig, note: str) -> Card | None:
    """The card with a smaller weight (whole lots). None if less than one lot remains."""
    s = card.sizing
    price = s["sizing_price"]
    shares = rules.lots(weight * s["capital"] / price)
    if shares < rules.lot_size:
        return None
    s["shares"], s["lots"], s["weight"], s["value"] = int(shares), int(shares // rules.lot_size), shares * price / s["capital"], shares * price
    if s.get("risk_if_stop") is not None:
        s["risk_if_stop"] = shares * (card.exits["reference"] - card.exits["stop"])
        s["risk_pct_capital"] = s["risk_if_stop"] / s["capital"]
    if s.get("risk_if_bear_pct_capital") is not None:
        bear = card.exits["bear_reference"]
        s["risk_if_bear"] = shares * max(card.exits["reference"] - bear, 0)
        s["risk_if_bear_pct_capital"] = s["risk_if_bear"] / s["capital"]
    s.setdefault("caps_applied", []).append(note)
    return card


@dataclass
class Aggregate:
    accepted: list[Card] = field(default_factory=list)
    rejected: list[Card] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def aggregate(cards: list[Card], cfg: RecoConfig, rules: MarketRules, *, holdings: dict[str, dict[int, float]] | None = None, open_risk: float = 0.0,
              equity: pd.Series | None = None) -> Aggregate:
    """``cards``: candidate BUY cards (best first within each sleeve). ``holdings``: current sleeve -> {instrument_id: weight of total capital}.
    ``open_risk``: loss if every open SWING stop is hit, share of total capital. ``equity``: the portfolio's equity history for the kill-switch."""
    holdings = {k: dict(v) for k, v in (holdings or {}).items()}
    budgets = sleeve_budgets(cfg)
    used = {s: sum(holdings.get(s, {}).values()) for s in SLEEVES}
    by_name: dict[int, float] = {}
    for s, h in holdings.items():
        for iid, w in h.items():
            by_name[iid] = by_name.get(iid, 0.0) + w
    ks = kill_state(equity, cfg.portfolio.kill_drawdown, cfg.portfolio.kill_cooldown) if (equity is not None and len(equity) > 1 and cfg.kill_switch) else None
    out = Aggregate()
    risk = open_risk
    for card in cards:
        if card.action != "BUY":
            (out.rejected if card.action == "NO_TRADE" else out.accepted).append(card)
            continue
        reason = None
        if ks and ks["killed"]:
            reason = f"kill-switch đang hoạt động (drawdown danh mục {fmt_pct(ks['drawdown_from_peak'], 1)}, còn {ks['sessions_left']} phiên)"
        s = card.sleeve
        if reason is None:
            w, room = card.sizing["weight"], budgets[s] - used[s]
            if room < w - 1e-12:
                fits = room >= 0.5 * w and rescale(card, room, rules, cfg, f"hạn mức sleeve {s} ({fmt_pct(budgets[s], 0)} vốn) còn {fmt_pct(max(room, 0), 1)}") is not None
                reason = None if fits else f"hết hạn mức sleeve {s} ({fmt_pct(budgets[s], 0)} vốn)"
        if reason is None:
            w, room = card.sizing["weight"], cfg.portfolio.max_weight_name - by_name.get(card.instrument_id, 0.0)
            if room < w - 1e-12:
                fits = room >= 0.5 * w and rescale(card, room, rules, cfg, f"tổng tỷ trọng {card.symbol} trên mọi sleeve ≤ {fmt_pct(cfg.portfolio.max_weight_name, 0)} (còn {fmt_pct(max(room, 0), 1)})") is not None
                reason = None if fits else f"vượt trần tổng tỷ trọng {fmt_pct(cfg.portfolio.max_weight_name, 0)} của một mã trên mọi sleeve"
        if reason is None and card.sizing.get("risk_pct_capital") is not None and risk + card.sizing["risk_pct_capital"] > cfg.portfolio.max_open_risk + 1e-12:
            reason = f"rủi ro mở của các lệnh SWING sẽ vượt {fmt_pct(cfg.portfolio.max_open_risk, 1)} vốn nếu tất cả chạm stop"
        if reason is not None:
            card.action, card.rejected = "NO_TRADE", reason
            out.rejected.append(card)
            continue
        w = card.sizing["weight"]
        used[s] += w
        by_name[card.instrument_id] = by_name.get(card.instrument_id, 0.0) + w
        if card.sizing.get("risk_pct_capital") is not None:
            risk += card.sizing["risk_pct_capital"]
        out.accepted.append(card)
    out.summary = {"sleeve_budget": budgets, "sleeve_used": used, "sleeve_room": {s: budgets[s] - used[s] for s in SLEEVES}, "gross": sum(used.values()), "cash": 1 - sum(used.values()),
                   "open_swing_risk": risk, "kill_switch": ks,
                   "names_in_two_sleeves": sorted({c.symbol for c in out.accepted if sum(1 for d in out.accepted if d.symbol == c.symbol and d.action == "BUY") > 1})}
    return out
