"""The recommendation backtest plumbing: sleeve books, cards -> orders -> outcomes, calibration of the stated probability, the DB rows of the cards."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from predict_stock.backtest.engine import EngineConfig, MarketData, Signal, SignalItem, run_backtest
from predict_stock.backtest.market import MarketRules
from predict_stock.config import RecoConfig
from predict_stock.db.models import Recommendation
from predict_stock.reco import analyze as AN
from predict_stock.reco import backtest as BT
from predict_stock.reco.job import _book_pnl, next_schedule_date, save_live_cards
from tests.conftest import apply
from tests.test_reco_cards import CAL, FAILED, GOOD, SIMILAR, buy_card, ctx, pred, swing
from predict_stock.reco import builders as B


@pytest.fixture
def rules(cfg):
    return MarketRules.from_config(cfg.market)


@pytest.fixture
def rc():
    return RecoConfig()


def world(o, h, l, c, ids=(1,)):
    T = len(o)
    frame = lambda a: pd.DataFrame(np.asarray(a, float).reshape(T, -1) * np.ones((1, len(ids))), index=pd.bdate_range("2026-01-01", periods=T), columns=list(ids))
    return MarketData(frame(o), frame(h), frame(l), frame(c), frame([0.07] * T))


# ---- sleeve books ---------------------------------------------------------------------------------------------------------------------------------
def test_virtual_ids_round_trip_and_never_collide_across_sleeves():
    assert BT.unvid(BT.vid("swing", 17)) == ("swing", 17) and BT.unvid(BT.vid("invest_b2", 999_999)) == ("invest_b2", 999_999)
    ids = {BT.vid(s, i) for s in BT.SLEEVE_CODE for i in range(1, 60)}
    assert len(ids) == 3 * 59


def test_virtual_market_gives_each_sleeve_its_own_column_with_the_same_prices():
    md = world([20000, 20100, 20200], [20100, 20200, 20300], [19900, 20000, 20100], [20050, 20150, 20250], ids=(1, 2))
    v = BT.virtual_market(md, ["swing", "invest_b1"])
    assert list(v.close.columns) == [BT.vid("swing", 1), BT.vid("swing", 2), BT.vid("invest_b1", 1), BT.vid("invest_b1", 2)]
    assert (v.close[BT.vid("swing", 1)] == v.close[BT.vid("invest_b1", 1)]).all() and (v.bands.shape == v.close.shape)


def test_the_same_stock_in_two_sleeves_is_two_positions_with_their_own_exits(rules):
    md = world([20000] * 8, [20100] * 6 + [21500, 20100], [19900] * 8, [20000] * 8)
    v = BT.virtual_market(md, ["swing", "invest_b1"])
    items = [SignalItem(BT.vid("swing", 1), 0.02, target_price=21400, stop_price=19000, group="swing", tag="s"), SignalItem(BT.vid("invest_b1", 1), 0.02, group="invest_b1", tag="i")]
    r = run_backtest(v, {0: Signal(items, full_rebalance=False)}, rules.scaled_costs(0), EngineConfig(capital=1e9))
    sells = r.fills[r.fills.side == "sell"]
    assert len(r.fills[r.fills.side == "buy"]) == 2 and list(sells["tag"]) == ["s"]                 # only the SWING position hit its target; the INVEST one is still held
    assert r.stats["open_positions"] == 1 and BT.unvid(r.open_positions.iloc[0]["instrument_id"])[0] == "invest_b1"


def test_book_pnl_splits_the_result_by_sleeve_and_adds_up_to_the_equity_change(rules):
    o = [20000, 20000, 20000, 20000, 21000, 21000, 21000, 21000]
    md = world(o, [x + 100 for x in o], [x - 100 for x in o], o)
    v = BT.virtual_market(md, ["swing", "invest_b1"])
    items = [SignalItem(BT.vid("swing", 1), 0.03, group="swing"), SignalItem(BT.vid("invest_b1", 1), 0.05, group="invest_b1")]
    r = run_backtest(v, {0: Signal(items, full_rebalance=False), 5: Signal([SignalItem(BT.vid("swing", 1), 0.0)], full_rebalance=False)}, rules, EngineConfig(capital=1e9))
    pnl = _book_pnl(r, 1e9)
    assert set(pnl) == {"swing", "invest_b1"} and pnl["swing"] > 0 and pnl["invest_b1"] > pnl["swing"] * 0.5
    assert sum(pnl.values()) == pytest.approx(r.equity.iloc[-1] / 1e9 - 1, abs=1e-9)


# ---- cards -> orders -> outcomes --------------------------------------------------------------------------------------------------------------------
def card_world(rules, rc, path):
    card = swing(rules, rc, ctx(hi20=60_000.0, hi60=90_000.0, atr=1_500.0))
    e, x = card.entry, card.exits
    return card, e, x


def run_card(rules, card, o, h, l):
    md = world(o, h, l, [50_000] * len(o))
    item = card.to_signal_item()
    return run_backtest(md, {0: Signal([item], full_rebalance=False)}, rules.scaled_costs(0), EngineConfig(capital=1e9))


def test_a_card_that_reaches_both_targets_is_classified_and_its_realised_r_is_the_stated_one(rules, rc):
    card, e, x = card_world(rules, rc, None)
    hz = e["zone_high"]
    n = 10
    o = [50_000, hz, hz, hz, hz + 100, hz + 100, hz + 100, hz + 100, hz + 100, hz + 100]
    h = [50_100, hz + 100, hz + 100, x["target1"] + 10, x["target1"] + 10, x["target2"] + 10, x["target2"] + 10, x["target2"], x["target2"], x["target2"]]
    l = [49_900, hz - 400, hz, hz, hz + 50, hz + 50, hz + 50, hz + 50, hz + 50, hz + 50]
    r = run_card(rules, card, o, h, l)
    out = AN.swing_outcomes(r, {card.card_id: card}, {(pd.Timestamp(card.as_of), 1): 1.0})
    assert out["trades"] == 1 and out["outcome_n"] == {"target2": 1} and out["t1_touch_rate"] == 1.0 and out["orders"]["fill_rate"] == 1.0
    assert out["R"]["realised_R_mean"] > 1.0 and out["hold"]["actual_mean"] == r.round_trips.iloc[0]["sessions"]


def test_a_card_stopped_out_is_a_loss_of_about_one_r_when_the_stop_fills_at_its_level(rules, rc):
    card, e, x = card_world(rules, rc, None)
    hz = e["zone_high"]
    o = [50_000, hz, hz, hz, hz, hz, hz, hz]
    h = [50_100] + [hz + 100] * 7
    l = [49_900, hz - 400, hz, hz, x["stop"] - 10, x["stop"] - 10, hz, hz]
    r = run_card(rules, card, o, h, l)
    out = AN.swing_outcomes(r, {card.card_id: card}, {})
    assert out["outcome_n"] == {"stop": 1}
    risk_per_share = out["R"]["stated_rr_mean"] and (hz - x["stop"])
    assert -1.2 < out["R"]["realised_R_mean"] < -0.8                                            # about one R lost: the stop filled at its level, the risk is measured from the fill
    assert out["exit_fills"]["stop_fills"] == 1 and out["exit_fills"]["stop_fill_vs_stop_mean"] == pytest.approx(0.0, abs=1e-9)


def test_orders_that_never_fill_are_counted_as_expired_and_the_fill_rate_is_computed(rules, rc):
    card, e, x = card_world(rules, rc, None)
    o = [50_000] * 6
    r = run_card(rules, card, o, [50_100] * 6, [e["zone_high"] + 500] * 6)
    st = AN.order_stats(r, {card.card_id: card}, "SWING")
    assert st["orders_placed"] == 1 and st["unfilled_expired"] == 1 and st["filled"] == 0 and st["fill_rate"] == 0.0 and st["cards_not_ordered"] == 0


def test_a_card_that_never_became_an_order_is_counted_as_not_ordered(rules, rc):
    card, _, _ = card_world(rules, rc, None)
    md = world([50_000] * 4, [50_100] * 4, [49_900] * 4, [50_000] * 4)
    r = run_backtest(md, {}, rules.scaled_costs(0), EngineConfig(capital=1e9))
    st = AN.order_stats(r, {card.card_id: card}, "SWING")
    assert st["cards_issued"] == 1 and st["cards_not_ordered"] == 1


def test_probability_calibration_compares_the_stated_probability_with_the_realised_frequency(rules, rc):
    cards, labels = {}, {}
    rng = np.random.default_rng(0)
    for k in range(400):
        c = swing(rules, rc, ctx(hi20=60_000.0, hi60=90_000.0, as_of=CAL[k % 40 + 5]), p=pred(proba=0.5, proba_raw=0.5))
        c.card_id, c.instrument_id = f"c{k}", k
        cards[c.card_id] = c
        labels[(pd.Timestamp(c.as_of), k)] = 1.0 if rng.random() < 0.25 else -1.0        # the event really happens 25% of the time
    out = AN.probability_calibration(cards, labels, pd.DataFrame())
    assert out["n_events"] == 400 and out["realised_rate"] == pytest.approx(0.25, abs=0.06)
    assert out["p_model"]["mean_stated"] == pytest.approx(0.5) and out["p_model"]["mean_stated"] - out["realised_rate"] > 0.15          # said 50%, got ~25%: the gap is visible
    assert out["p_model"]["ece"] > 0.15 and sum(out["by_grade"].values()) == 400


def test_the_grade_and_display_policy_downgrade_a_probability_that_is_off_by_a_lot(rules, rc):
    """Says 58%, happened 40%: the evidence grade is not good enough for the model probability to be the headline number."""
    bad = {"n": 12_000, "auc": 0.52, "calibration_gap": 0.18}
    k = B.build_swing_card(ctx(), pred(proba=0.58), SIMILAR, bad, FAILED, rc, rules).confidence
    assert k["p_model"] == 0.58 and k["display_kind"] == "historical" and k["p_display"] == SIMILAR["win_rate"] and k["grade"] == "THẤP"


# ---- schedule dates ------------------------------------------------------------------------------------------------------------------------------
def test_the_next_review_date_is_the_first_weekday_of_the_next_period():
    assert next_schedule_date(pd.Timestamp("2026-09-18"), "monthly") == pd.Timestamp("2026-10-01")
    assert next_schedule_date(pd.Timestamp("2026-09-18"), "quarterly") == pd.Timestamp("2026-10-01")
    assert next_schedule_date(pd.Timestamp("2026-12-30"), "monthly") == pd.Timestamp("2027-01-01")
    assert next_schedule_date(pd.Timestamp("2026-11-30"), "quarterly") == pd.Timestamp("2027-01-01")
    assert next_schedule_date(pd.Timestamp("2026-08-31"), "monthly") == pd.Timestamp("2026-09-01")


# ---- database ---------------------------------------------------------------------------------------------------------------------------------------
def test_buy_and_watch_cards_are_stored_in_recommendations_idempotently_with_the_full_card(engine, cfg, rules, rc):
    apply(engine, "T1", ["AAA"], date(2024, 1, 1))
    with engine.connect() as conn:
        from predict_stock.db.models import Instrument
        iid = conn.execute(select(Instrument.id)).scalar()
    card = swing(rules, rc, ctx(hi20=60_000.0, hi60=90_000.0, instrument_id=iid))
    card.instrument_id = iid
    card.model_id = None
    assert save_live_cards(engine, cfg, [card], pd.DataFrame(), 0, None, None) == 1
    assert save_live_cards(engine, cfg, [card], pd.DataFrame(), 0, None, None) == 0                 # already stored: left exactly as issued
    with engine.connect() as conn:
        first_id = conn.execute(select(Recommendation.id)).scalar()
    assert save_live_cards(engine, cfg, [card], pd.DataFrame(), 0, None, None, overwrite=True) == 1   # explicit overwrite updates in place
    with engine.connect() as conn:
        rows = conn.execute(select(Recommendation)).all()
    assert len(rows) == 1 and rows[0].id == first_id
    r = rows[0]
    assert r.strategy == "SWING" and r.action == "BUY" and r.entry_price == card.exits["reference"] and r.target_price == card.exits["target2"] and r.stop_loss == card.exits["stop"]
    assert r.hold_days_min == 2 and r.hold_days_max == 10 and str(r.valid_until) == card.valid_until and r.status == "pending"
    assert r.card["exits"]["stop"] == card.exits["stop"] and "GIÁ VÀO LỆNH" in r.card_text and "LÝ DO" in r.rationale and "RỦI RO" in r.rationale and "MẤT HIỆU LỰC" in r.rationale
    assert r.exit_conditions["stop_is_hard"] is True and 0 <= r.confidence <= 1


def test_watch_cards_and_invest_cards_fit_the_recommendation_row_too(engine, cfg, rules, rc):
    from tests.test_reco_cards import invest_card
    apply(engine, "T1", ["AAA"], date(2024, 1, 1))
    with engine.connect() as conn:
        from predict_stock.db.models import Instrument
        iid = conn.execute(select(Instrument.id)).scalar()
    inv = invest_card(rules, rc)
    inv.instrument_id, inv.model_id = iid, None
    w = swing(rules, rc, ctx(hi20=60_000.0, hi60=90_000.0), watch=True)
    w.instrument_id, w.model_id = iid, None
    assert save_live_cards(engine, cfg, [inv, w], pd.DataFrame(), 0, None, None) == 2
    with engine.connect() as conn:
        rows = {(r.strategy, r.action): r for r in conn.execute(select(Recommendation)).all()}
    assert set(rows) == {("INVEST_B2", "BUY"), ("SWING", "WATCH")}
    r = rows[("INVEST_B2", "BUY")]
    assert r.target_price == inv.exits["scenarios"]["bull"]["price"] and r.stop_loss == inv.exits["bear_reference"] and r.exit_conditions["stop_is_hard"] is False and r.hold_days_max == 126


# ---- evidence for the display grade: past out-of-sample rows whose labels had ended --------------------------------------------------------------------
def evidence_frame(n_per_fold=600, informative=True, seed=0):
    from predict_stock.reco.sources import Evidence
    rng = np.random.default_rng(seed)
    rows = []
    for fold in range(3):
        dates = pd.bdate_range("2022-01-03", periods=n_per_fold) + pd.offsets.BDay(fold * n_per_fold)
        y = (rng.random(n_per_fold) < 0.3).astype(float)
        raw = np.clip(0.3 + (0.4 * (y - 0.3) if informative else 0) + 0.05 * rng.normal(size=n_per_fold), 0.01, 0.99)
        rows.append(pd.DataFrame({"trade_date": dates, "fold": fold, "proba_raw": raw, "proba": raw, "tb_label": np.where(y == 1, 1.0, -1.0), "tb_end": dates + pd.offsets.BDay(10)}))
    return Evidence(pd.concat(rows, ignore_index=True))


def test_evidence_uses_only_rows_whose_label_ended_before_the_month_starts():
    ev = evidence_frame()
    early = ev.at("2022-03-15")
    later = ev.at("2023-06-15")
    assert early["n"] < later["n"]
    start = pd.Timestamp("2022-03-01")
    assert early["n"] == int((ev.p["tb_end"] < start).sum())                 # a label still open on the first day of the month is not evidence yet


def test_evidence_auc_is_the_per_fold_average_and_separates_informative_from_random_probabilities():
    good, rnd = evidence_frame(informative=True).at("2030-01-15"), evidence_frame(informative=False).at("2030-01-15")
    assert good["auc"] > 0.9 and 0.4 < rnd["auc"] < 0.6 and good["folds_used"] == 3


def test_evidence_needs_enough_rows_per_fold_for_an_auc():
    ev = evidence_frame(n_per_fold=100)
    assert ev.at("2030-01-15")["auc"] is None and ev.at("2030-01-15")["n"] == 300


def test_evidence_is_cached_per_month_and_reports_the_realised_rate():
    ev = evidence_frame()
    a, b = ev.at("2023-05-03"), ev.at("2023-05-29")
    assert a is b and a["realised_rate"] == pytest.approx(0.3, abs=0.05)
