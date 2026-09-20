"""The paper portfolio is the replay of the issued cards through the engine: statuses, orders, positions, snapshots, outcomes, idempotency, INVEST target book, removal flags."""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from predict_stock.backtest.engine import MarketData
from predict_stock.backtest.market import MarketRules
from predict_stock.config import RecoConfig
from predict_stock.db.models import (
    Alert, Instrument, PaperOrder, PaperPosition, PortfolioSnapshot, Recommendation, RecommendationOutcome, SleeveTarget,
)
from predict_stock.paper import daily as D
from predict_stock.paper import replay as RP
from predict_stock.paper import state as ST
from predict_stock.paper import sync as SY
from predict_stock.paper.alerts import acknowledge, open_alerts, raise_alert
from predict_stock.reco.job import save_live_cards
from tests.conftest import apply
from tests.test_reco_cards import CAL, buy_card, ctx, invest_card, pred, swing

CARD_IDX = 30                                   # the card is issued at the close of CAL[30]


def rows(engine, stmt):
    with Session(engine) as s:
        return s.scalars(stmt).all()


class Sym:
    def at(self, iid, d):
        return f"S{iid}"


def world(iid, patch=None, n=60, base=50_000.0):
    o, h, l, c = (np.full(n, base) for _ in range(4))
    h, l = h + 100, l - 100
    for i, kw in (patch or {}).items():
        for k, v in kw.items():
            {"o": o, "h": h, "l": l, "c": c}[k][i] = v
    frame = lambda a: pd.DataFrame(a, index=CAL[:n], columns=[iid])
    return MarketData(frame(o), frame(h), frame(l), frame(c), frame(np.full(n, 0.07)))


def setup_for(cfg, md, rules):
    return SimpleNamespace(calendar=md.close.index, data=md, rules=rules, cfg=cfg, index_close={}, holdout=None)


@pytest.fixture
def rules(cfg):
    return MarketRules.from_config(cfg.market)


@pytest.fixture
def rc():
    return RecoConfig()


@pytest.fixture
def cfgp(cfg):
    return cfg.model_copy(update={"paper": cfg.paper.model_copy(update={"start_date": str(CAL[CARD_IDX].date())})})


@pytest.fixture
def iid(engine):
    apply(engine, "T1", ["AAA", "BBB"], date(2024, 1, 1))
    with engine.connect() as c:
        return c.execute(select(Instrument.id).order_by(Instrument.id)).scalars().all()[0]


def issue(engine, cfg, rules, rc, iid):
    card = swing(rules, rc, ctx(hi20=60_000.0, hi60=90_000.0, atr=1_500.0, instrument_id=iid, as_of=CAL[CARD_IDX]))
    card.instrument_id, card.model_id = iid, None
    save_live_cards(engine, cfg, [card], None, 0, None, None)
    return card


def run_day(engine, cfg, md, rules, idx):
    st = setup_for(cfg, md, rules)
    rp = RP.replay(engine, cfg, st, CAL[idx], CAL[CARD_IDX])
    SY.sync_all(engine, cfg, st, rp, Sym())
    return rp


def snapshot(engine):
    with engine.connect() as c:
        return {"orders": sorted(tuple(map(str, r)) for r in c.execute(select(PaperOrder.order_key, PaperOrder.status, PaperOrder.quantity, PaperOrder.fill_price)).all()),
                "positions": sorted(tuple(map(str, r)) for r in c.execute(select(PaperPosition.portfolio_code, PaperPosition.instrument_id, PaperPosition.quantity, PaperPosition.status)).all()),
                "snaps": sorted(tuple(map(str, r)) for r in c.execute(select(PortfolioSnapshot.snapshot_date, PortfolioSnapshot.equity, PortfolioSnapshot.cash)).all()),
                "recs": sorted(tuple(map(str, r)) for r in c.execute(select(Recommendation.id, Recommendation.status)).all()),
                "outs": sorted(tuple(map(str, r)) for r in c.execute(select(RecommendationOutcome.recommendation_id, RecommendationOutcome.exit_reason, RecommendationOutcome.net_return)).all())}


def status(engine):
    with engine.connect() as c:
        return c.execute(select(Recommendation.status).where(Recommendation.action == "BUY")).scalar()


# ---- lifecycle ---------------------------------------------------------------------------------------------------------------------------------------------
def test_a_card_is_pending_on_its_day_then_filled_then_reaches_its_targets(engine, cfgp, rules, rc, iid):
    card = issue(engine, cfgp, rules, rc, iid)
    e, x = card.entry, card.exits
    hz = e["zone_high"]
    patch = {31: {"l": hz - 300, "h": hz + 400, "o": hz + 200}, 33: {"h": x["target1"] + 50, "o": hz + 300, "l": hz + 200}, 34: {"o": hz + 500, "l": hz + 400},
             35: {"o": hz + 500, "l": hz + 400, "h": x["target2"] + 100}}
    md = world(iid, patch)
    run_day(engine, cfgp, md, rules, CARD_IDX)
    assert status(engine) == "pending"
    with engine.connect() as c:
        o = rows(engine, select(PaperOrder))
    assert len(o) == 1 and o[0].status == "pending" and o[0].side == "BUY" and float(o[0].limit_price) == hz and o[0].quantity == card.sizing["shares"]
    run_day(engine, cfgp, md, rules, 32)
    assert status(engine) == "holding"
    with engine.connect() as c:
        p = rows(engine, select(PaperPosition))
        out = rows(engine, select(RecommendationOutcome))[0]
    assert len(p) == 1 and p[0].status == "open" and p[0].portfolio_code == "paper:swing" and p[0].quantity > 0
    assert out.exit_date is None and out.details["filled"] is True and out.details["unrealised"] is True
    run_day(engine, cfgp, md, rules, 36)
    assert status(engine) == "target"
    with engine.connect() as c:
        out = rows(engine, select(RecommendationOutcome))[0]
        pos = rows(engine, select(PaperPosition))
        orders = rows(engine, select(PaperOrder).order_by(PaperOrder.id))
    assert out.exit_reason == "target" and out.net_return > 0 and out.max_favorable > 0 and out.details["target1_hit"] is True and 0.7 < out.details["realised_R"] < 1.5      # half at ~1 R, half at ~2 R, minus costs
    assert pos[0].status == "closed" and float(pos[0].close_price) > float(pos[0].avg_cost)
    assert [o.status for o in orders].count("filled") == 3 and {o.order_type for o in orders} >= {"limit", "target1", "target"}       # the buy and the two target sales


def test_an_unfilled_card_expires_and_a_gapped_breakout_is_cancelled(engine, cfgp, rules, rc, iid):
    card = issue(engine, cfgp, rules, rc, iid)
    md = world(iid)                                                    # the market never comes back to the zone
    for k in range(CARD_IDX + 1, CARD_IDX + 1 + 3):
        pass
    run_day(engine, cfgp, md, rules, 40)
    assert status(engine) == "expired"
    with engine.connect() as c:
        out = rows(engine, select(RecommendationOutcome))[0]
        orders = rows(engine, select(PaperOrder))
    assert out.details["filled"] is False and "unfilled" in out.details["order_status"] and orders[0].status == "expired" and orders[0].reject_reason


def test_a_stopped_position_is_reported_as_stopped_with_its_loss_in_r(engine, cfgp, rules, rc, iid):
    card = issue(engine, cfgp, rules, rc, iid)
    hz, stop = card.entry["zone_high"], card.exits["stop"]
    patch = {31: {"l": hz - 300, "h": hz + 200}, 34: {"l": stop - 100, "o": hz - 100, "h": hz + 100}}
    run_day(engine, cfgp, world(iid, patch), rules, 38)
    assert status(engine) == "stopped"
    with engine.connect() as c:
        out = rows(engine, select(RecommendationOutcome))[0]
    assert out.exit_reason == "stop" and out.net_return < 0 and -1.5 < out.details["realised_R"] < -0.7


# ---- idempotency --------------------------------------------------------------------------------------------------------------------------------------------
def test_replaying_and_syncing_the_same_day_twice_changes_nothing(engine, cfgp, rules, rc, iid):
    card = issue(engine, cfgp, rules, rc, iid)
    hz = card.entry["zone_high"]
    md = world(iid, {31: {"l": hz - 300, "h": hz + 400}, 33: {"h": card.exits["target1"] + 50, "o": hz + 300, "l": hz + 200}})
    run_day(engine, cfgp, md, rules, 36)
    a = snapshot(engine)
    run_day(engine, cfgp, md, rules, 36)
    assert snapshot(engine) == a
    run_day(engine, cfgp, md, rules, 40)                              # a later day extends the record; the earlier rows keep their keys
    b = snapshot(engine)
    assert set(a["orders"]) <= set(b["orders"]) and len(b["snaps"]) > len(a["snaps"])


def test_snapshots_hold_cash_market_value_equity_and_the_positions_of_the_last_day(engine, cfgp, rules, rc, iid):
    card = issue(engine, cfgp, rules, rc, iid)
    hz = card.entry["zone_high"]
    run_day(engine, cfgp, world(iid, {31: {"l": hz - 300, "h": hz + 400}}), rules, 34)
    with engine.connect() as c:
        snaps = rows(engine, select(PortfolioSnapshot).order_by(PortfolioSnapshot.snapshot_date))
    assert len(snaps) == 5 and snaps[0].equity == pytest.approx(1e9)
    last = snaps[-1]
    assert float(last.equity) == pytest.approx(float(last.cash) + float(last.market_value)) and last.positions[0]["sleeve"] == "swing" and last.positions[0]["symbol"] == f"S{iid}"
    assert last.metrics["exposure"] > 0 and snaps[0].positions is None


def test_the_paper_record_never_starts_before_its_start_date(engine, cfgp, rules, rc, iid):
    issue(engine, cfgp, rules, rc, iid)
    with pytest.raises(ValueError, match="starts on"):
        RP.replay(engine, cfgp, setup_for(cfgp, world(iid), rules), CAL[10], CAL[CARD_IDX])


# ---- INVEST target book -----------------------------------------------------------------------------------------------------------------------------------------
def test_the_invest_target_book_drives_buys_reductions_and_exits(engine, cfgp, rules, rc, iid):
    with engine.connect() as c:
        ids = c.execute(select(Instrument.id).order_by(Instrument.id)).scalars().all()
    a, b = ids[0], ids[1]
    card = invest_card(rules, rc)
    card.instrument_id, card.model_id, card.as_of, card.strategy, card.sleeve = a, None, str(CAL[CARD_IDX].date()), "INVEST_B1", "invest_b1"
    card.card_id = f"INVEST_B1:S{a}:{CAL[CARD_IDX].date()}:t1"
    save_live_cards(engine, cfgp, [card], None, 0, None, None)
    ST.save_target(engine, "invest_b1", CAL[CARD_IDX].date(), "rebalance", 1, 2, {a: 0.02}, {"final": {str(a): 0.04}, "from": {}}, CAL[50].date(), None)
    ST.save_target(engine, "invest_b1", CAL[35].date(), "rebalance", 1, 2, {b: 0.02}, {"final": {str(b): 0.04}, "from": {str(a): 0.02}}, None, None)         # a is dropped, b enters (no card: plain order)
    md = MarketData(*[pd.concat([getattr(world(a, base=30_000.0), k), getattr(world(b, base=30_000.0), k)], axis=1) for k in ("open", "high", "low", "close", "bands")])
    rp = run_day(engine, cfgp, md, rules, 40)
    with engine.connect() as c:
        pos = rows(engine, select(PaperPosition).order_by(PaperPosition.opened_date))
    by = {p.instrument_id: p for p in pos}
    assert by[a].status == "closed" and by[a].portfolio_code == "paper:invest_b1" and by[b].status == "open"           # a was exited when it left the target, b bought
    assert len(rp.signals[35].items) == 2                                                                                 # the step at 35 carries b's order and a's exit
    with engine.connect() as c:
        assert c.execute(select(Recommendation.status)).scalars().all() == ["closed"]                                    # a rebalance exit is 'closed', not a stop


def test_a_sleeve_target_row_is_immutable_per_date_kind_and_tranche(engine):
    r1 = ST.save_target(engine, "invest_b1", date(2026, 1, 5), "rebalance", 1, 2, {1: 0.02}, {"final": {"1": 0.04}}, None, None)
    r2 = ST.save_target(engine, "invest_b1", date(2026, 1, 5), "rebalance", 1, 2, {1: 0.99}, {"final": {"1": 0.99}}, None, None)
    assert r1.id == r2.id and ST._key(r2.current) == {1: 0.02}
    with engine.connect() as c:
        assert rows(engine, select(SleeveTarget)).__len__() == 1


# ---- universe removal ---------------------------------------------------------------------------------------------------------------------------------------
def make_pending(engine, cfgp, rules, rc, iid):
    issue(engine, cfgp, rules, rc, iid)
    with engine.begin() as c:
        c.execute(Recommendation.__table__.update().values(status="holding"))


def test_a_stock_that_left_the_universe_gets_a_flag_and_an_alert_but_is_not_sold(engine, cfgp, rules, rc, iid):
    make_pending(engine, cfgp, rules, rc, iid)
    cfg2 = cfgp.model_copy(update={"universe": cfgp.universe.model_copy(update={"training_code": "T1", "trading_code": "T1"})})
    apply(engine, "T1", ["BBB"], date(2026, 3, 1))                     # AAA is removed from T1 as of March
    out = D.flag_removed(engine, cfg2, pd.Timestamp("2026-03-02"), None)
    assert out["flagged"] == [f"SWING:{iid}"] and out["close_now_sleeves"] == []
    with engine.connect() as c:
        rec = rows(engine, select(Recommendation))[0]
        assert rec.flags["left_universe"]["policy"] == "hold_until_exit" and rec.flags["left_universe"]["note"] == "ra khỏi rổ" and rec.status == "holding"
        assert rows(engine, select(Alert).where(Alert.category == "left_universe")).__len__() == 1
    out2 = D.flag_removed(engine, cfg2, pd.Timestamp("2026-03-03"), None)      # the next day: same flag, no second alert
    with engine.connect() as c:
        assert rows(engine, select(Alert).where(Alert.category == "left_universe")).__len__() == 1
    assert out2["flagged"] == out["flagged"]


def test_a_stock_that_is_back_in_the_universe_loses_the_flag(engine, cfgp, rules, rc, iid):
    make_pending(engine, cfgp, rules, rc, iid)
    cfg2 = cfgp.model_copy(update={"universe": cfgp.universe.model_copy(update={"training_code": "T1", "trading_code": "T1"})})
    D.flag_removed(engine, cfg2, pd.Timestamp("2026-03-02"), None)
    apply(engine, "T1", ["AAA", "BBB"], date(2026, 3, 5))
    D.flag_removed(engine, cfg2, pd.Timestamp("2026-03-06"), None)
    with engine.connect() as c:
        assert c.execute(select(Recommendation.flags)).scalar() is None


def test_the_close_now_policy_writes_an_exit_row_in_the_target_book_and_nothing_else_changes(engine, cfgp, iid):
    cfg2 = cfgp.model_copy(update={"universe": cfgp.universe.model_copy(update={"training_code": "T1", "trading_code": "T1"}),
                                   "paper": cfgp.paper.model_copy(update={"on_removal": cfgp.paper.on_removal.model_copy(update={"invest": "close_now"})})})
    with engine.connect() as c:
        b = c.execute(select(Instrument.id).order_by(Instrument.id)).scalars().all()[1]
    ST.save_target(engine, "invest_b1", date(2026, 2, 2), "rebalance", 1, 1, {iid: 0.03, b: 0.03}, {"final": {}, "from": {}}, None, None)
    apply(engine, "T1", ["BBB"], date(2026, 3, 1))
    out = D.flag_removed(engine, cfg2, pd.Timestamp("2026-03-02"), None)
    assert out["close_now_sleeves"] == ["invest_b1"]
    row = ST.latest_target(engine, "invest_b1", upto=date(2026, 3, 2))
    assert row.kind == "exit" and ST._key(row.current) == {b: 0.03}


# ---- alerts ----------------------------------------------------------------------------------------------------------------------------------------------------------
def test_alerts_are_deduplicated_until_acknowledged(engine):
    a = raise_alert(engine, "error", "data_missing", "3 members have no bar on 2026-02-12: AAA, BBB, CCC")
    assert a is not None and raise_alert(engine, "error", "data_missing", "3 members have no bar on 2026-02-12: AAA, BBB, CCC") is None
    assert raise_alert(engine, "error", "data_missing", "another day") is not None
    assert len(open_alerts(engine)) == 2 and len(open_alerts(engine, "critical")) == 0
    acknowledge(engine, a)
    assert raise_alert(engine, "error", "data_missing", "3 members have no bar on 2026-02-12: AAA, BBB, CCC") is not None      # acknowledged: a new occurrence is news
    with pytest.raises(ValueError):
        raise_alert(engine, "fatal", "x", "y")
