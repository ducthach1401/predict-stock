"""The paper portfolio is REPLAYED, not simulated separately: the recommendations exactly as they were issued (frozen JSON) and the INVEST target book are turned into
engine orders and run through the very engine of the backtest over the real prices from the start of the paper record to the as-of date. So the paper portfolio follows
the same rules (T+2, lots, ticks, band, costs, no chasing, stops, targets, kill-switch) as everything validated before, and re-running a day gives the same state."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd
from sqlalchemy import Engine, select

from predict_stock.backtest.engine import BacktestResult, Signal, SignalItem
from predict_stock.backtest.runner import Setup
from predict_stock.config import AppConfig
from predict_stock.db.models import Recommendation
from predict_stock.db.session import session_scope
from predict_stock.paper import state as ST
from predict_stock.reco import backtest as BT
from predict_stock.reco.cards import Card

SLEEVES = ("swing", "invest_b1", "invest_b2")
STRATEGY_OF = {"swing": "SWING", "invest_b1": "INVEST_B1", "invest_b2": "INVEST_B2"}


@dataclass
class StoredCard:
    rec_id: int
    card: Card
    as_of: pd.Timestamp
    status: str


@dataclass
class Replay:
    res: BacktestResult
    cards: dict[str, StoredCard]          # card_id -> stored card
    first_idx: int
    last_idx: int
    as_of: pd.Timestamp
    signals: dict = field(default_factory=dict)


def load_cards(engine: Engine, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, StoredCard]:
    with session_scope(engine) as s:
        rows = list(s.scalars(select(Recommendation).where(Recommendation.action == "BUY", Recommendation.as_of_date >= start.date(), Recommendation.as_of_date <= end.date(),
                                                          Recommendation.card.is_not(None)).order_by(Recommendation.as_of_date, Recommendation.id)))
        return {r.card["card_id"]: StoredCard(r.id, Card.from_dict(r.card), pd.Timestamp(r.as_of_date), r.status) for r in rows}


def build_signals(engine: Engine, cfg: AppConfig, cards: dict[str, StoredCard], cal: pd.DatetimeIndex, end: pd.Timestamp) -> dict[int, Signal]:
    signals: dict[int, list[SignalItem]] = {}

    def add(d: pd.Timestamp, item: SignalItem) -> None:
        if d in cal:
            signals.setdefault(int(cal.get_loc(d)), []).append(item)

    for sc in cards.values():
        if sc.card.strategy == "SWING":
            it = sc.card.to_signal_item(weight=sc.card.sizing["weight"], group="swing")
            add(sc.as_of, replace(it, instrument_id=BT.vid("swing", sc.card.instrument_id)))
    by_key = {(sc.card.strategy, sc.card.instrument_id, sc.as_of): sc for sc in cards.values() if sc.card.strategy != "SWING"}
    for sleeve in ("invest_b1", "invest_b2"):
        prev: dict[int, float] = {}
        for row in ST.all_targets(engine, sleeve, end.date()):
            d = pd.Timestamp(row.as_of_date)
            cur = ST._key(row.current)
            for iid, w in cur.items():
                sc = by_key.get((STRATEGY_OF[sleeve], iid, d))
                if sc is not None:
                    it = replace(sc.card.to_signal_item(weight=w, group=sleeve), instrument_id=BT.vid(sleeve, iid))
                else:
                    it = SignalItem(BT.vid(sleeve, iid), w, tag=f"rebalance:{sleeve}", group=sleeve)
                add(d, it)
            for iid in set(prev) - set(cur):
                add(d, SignalItem(BT.vid(sleeve, iid), 0.0, tag=f"rebalance:{sleeve}", group=sleeve))
            prev = cur
    return {i: Signal(items, full_rebalance=False) for i, items in signals.items()}


def replay(engine: Engine, cfg: AppConfig, setup: Setup, as_of: pd.Timestamp, start: pd.Timestamp) -> Replay:
    cal = setup.calendar
    last = int(cal.get_loc(cal[cal <= as_of][-1]))
    first = int(cal.searchsorted(pd.Timestamp(start)))
    if first > last:
        raise ValueError(f"the paper record starts on {start.date()}, after the as-of date {as_of.date()}")
    cards = load_cards(engine, cal[first], cal[last])
    signals = build_signals(engine, cfg, cards, cal, cal[last])
    vmd = BT.virtual_market(setup.data, list(SLEEVES))
    res = BT.run_book(setup, cfg, vmd, signals, first, last, 1.0, kill=True)
    return Replay(res, cards, first, last, cal[last], signals)
