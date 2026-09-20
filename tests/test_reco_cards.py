"""Recommendation cards: tick rounding, entry zone inside the +-7% band, T+2, reward:risk, limit fill, mandatory sections, sizing, reasons, confidence, aggregator."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from predict_stock.backtest.engine import EngineConfig, MarketData, Signal, run_backtest
from predict_stock.backtest.market import MarketRules
from predict_stock.config import RecoConfig, load_config
from predict_stock.reco import aggregate as AG
from predict_stock.reco import builders as B
from predict_stock.reco.cards import Card, fmt_pct, fmt_price, session_after

CAL = pd.bdate_range("2026-01-01", periods=60)
RULES = MarketRules.from_config(load_config().market)


@pytest.fixture
def rules(cfg):
    return MarketRules.from_config(cfg.market)


@pytest.fixture
def rc():
    return RecoConfig()


def ctx(close=50_000.0, atr=1_500.0, hi20=None, hi60=None, band=0.07, **kw):
    floor, ceiling = RULES.round_up(close * (1 - band)), RULES.round_down(close * (1 + band))
    base = dict(symbol="AAA", instrument_id=1, as_of=CAL[30], calendar=CAL, close=close, atr=atr, floor=floor, ceiling=ceiling, hi20=hi20 or close * 1.06, hi60=hi60 or close * 1.06,
                sma50=close * 0.97, sma200=close * 0.9, rsi14=55.0, vol_spike=1.4, rs5=0.012, rs10=0.02, regime_off=False)
    base.update(kw)
    return B.Ctx(**base)


def pred(**kw):
    base = dict(score=0.61, rank=2, n_universe=48, model_id=7, model_name="swing_lgbm_final", universe_id=3, proba_raw=0.31, proba=0.33, q10=-0.05, q50=0.004, q90=0.06, hold_median=4.0,
                hold_p75=8.0, hold_n=1500, contributions=[["don_lo_dist_20", 0.21, 0.022], ["ret_5_csrank", 0.74, -0.007], ["gap_1", 0.009, 0.006]])
    base.update(kw)
    return B.Pred(**base)


SIMILAR = {"n": 900, "win_rate": 0.29, "t1_rate": 0.51, "stop_rate": 0.45, "timeout_rate": 0.26}
GOOD = {"n": 20000, "auc": 0.62, "calibration_gap": 0.01}
BAD = {"n": 20000, "auc": 0.51, "calibration_gap": 0.02}
FAILED = {"passed": False, "criteria": [{"name": "net Sharpe above every portfolio baseline", "ok": False}, {"name": "IC", "ok": True}]}


def swing(rules, rc, c=None, p=None, **kw):
    return B.build_swing_card(c or ctx(), p or pred(), SIMILAR, GOOD, FAILED, rc, rules, **kw)


# ---- tick rounding -----------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("close,atr", [(8_130.0, 230.0), (23_450.0, 610.0), (48_950.0, 1_240.0), (81_300.0, 2_030.0), (123_400.0, 3_100.0)])
def test_every_price_of_a_card_is_on_the_tick_grid_of_its_price_band(rules, rc, close, atr):
    card = swing(rules, rc, ctx(close=close, atr=atr, hi20=close * 1.10, hi60=close * 1.15))
    assert card.action == "BUY", card.rejected
    assert card.problems(rules) == []
    for v in (card.entry["zone_low"], card.entry["zone_high"], card.exits["stop"], card.exits["target1"], card.exits["target2"], card.entry["reference"]):
        t = rules.tick(v)
        assert abs(v / t - round(v / t)) < 1e-9, (v, t)


def test_pullback_zone_is_below_the_close_and_ordered(rules, rc):
    card = swing(rules, rc, ctx(hi20=60_000.0))                 # far from the 20-session high -> pullback
    e = card.entry
    assert e["style"] == "pullback" and e["order_type"] == "limit" and e["zone_low"] <= e["zone_high"] < 50_000
    assert e["zone_low"] == rules.round_down(50_000 - 0.60 * 1_500) and e["zone_high"] == rules.round_down(50_000 - 0.15 * 1_500)


def test_breakout_style_when_the_close_is_near_the_twenty_day_high(rules, rc):
    card = swing(rules, rc, ctx(hi20=50_400.0, hi60=60_000.0))
    e = card.entry
    assert e["style"] == "breakout" and e["order_type"] == "stop" and e["trigger"] == rules.round_up(50_400 + 0.10 * 1_500)
    assert e["zone_low"] == e["trigger"] and e["zone_high"] >= e["trigger"] and e["cancel_if_open_above"] == e["zone_high"]


def test_configured_ato_style(rules, rc):
    rc2 = rc.model_copy(update={"swing": rc.swing.model_copy(update={"entry_style": "ato"})})
    card = swing(rules, rc2, ctx(hi20=60_000.0, hi60=90_000.0))
    assert card.entry["order_type"] == "ato" and card.entry["zone_low"] == rules.round_down(49_500) and card.entry["zone_high"] == rules.round_down(50_500)


# ---- the entry zone stays inside the next session's band -------------------------------------------------------------------------------------
def test_the_zone_is_clamped_into_the_plus_minus_seven_percent_band(rules, rc):
    c = ctx(close=50_000.0, atr=1_500.0, hi20=60_000.0, hi60=90_000.0)
    c.floor = 49_500.0                                                        # the band leaves only 1% below the close: the raw zone (49,100 - 49,800) starts under it
    card = swing(rules, rc, c)
    assert card.action == "BUY" and card.entry["zone_low"] == 49_500.0 and card.entry["zone_high"] <= c.ceiling
    assert card.problems(rules) == []
    wide = ctx(close=50_000.0, atr=1_500.0, hi20=60_000.0, hi60=90_000.0)
    assert wide.floor == rules.round_up(50_000 * 0.93) and wide.ceiling == rules.round_down(50_000 * 1.07)
    assert swing(rules, rc, wide).entry["zone_low"] >= wide.floor


def test_a_zone_entirely_below_the_floor_is_not_issued(rules, rc):
    c = ctx(close=50_000.0, atr=12_000.0, hi20=90_000.0, hi60=90_000.0)       # 0.15 ATR below the close is already under the floor 46,500? (-1,800 -> 48,200, no) -> use larger
    c.floor = 49_900.0                                                        # a band that leaves almost no room below the close
    card = swing(rules, rc, c)
    assert card.action == "NO_TRADE" and "biên độ" in card.rejected


def test_a_breakout_trigger_above_the_ceiling_is_not_issued(rules, rc):
    c = ctx(hi20=50_450.0, hi60=90_000.0)                                     # close is within 2% of the 20-session high: breakout, trigger = 50,450 + 0.1 ATR = 50,600
    c.ceiling = 50_500.0
    card = swing(rules, rc, c)
    assert card.action == "NO_TRADE" and "trần" in card.rejected


# ---- reward : risk -----------------------------------------------------------------------------------------------------------------------------------
def test_reward_to_risk_is_computed_from_the_stated_levels(rules, rc):
    card = swing(rules, rc, ctx(hi20=60_000.0, hi60=90_000.0))
    x = card.exits
    assert x["rr_target2"] == pytest.approx((x["target2"] - x["reference"]) / (x["reference"] - x["stop"]))
    assert x["rr_target2"] >= 1.5 and x["rr_target1"] < x["rr_target2"]


def test_a_card_below_the_minimum_reward_to_risk_is_not_issued(rules, rc):
    card = swing(rules, rc, ctx(hi20=60_000.0, hi60=51_500.0), p=pred())        # resistance just above the entry caps target 2
    assert card.action == "NO_TRADE"
    assert "R:R" in card.rejected or "kháng cự" in card.rejected


def test_the_threshold_is_configurable(rules, rc):
    c = ctx(hi20=60_000.0, hi60=51_000.0)                                     # target 2 capped at 50,900: reward 1,450 against a risk of 1,500
    strict = swing(rules, rc, c)
    loose = swing(rules, rc.model_copy(update={"swing": rc.swing.model_copy(update={"min_rr": 0.5})}), c)
    assert strict.action == "NO_TRADE" and loose.action == "BUY" and loose.exits["rr_target2"] < 1.5


def test_the_resistance_cap_leaves_the_targets_just_below_the_resistance(rules, rc):
    c = ctx(hi20=60_000.0, hi60=51_500.0)
    card = swing(rules, rc.model_copy(update={"swing": rc.swing.model_copy(update={"min_rr": 1.0})}), c)
    assert card.action == "BUY" and card.exits["target2"] == rules.round_down(51_500 - rules.tick(51_500)) and card.exits["p_touch_target2"] is None
    assert any("kháng cự" in n for n in card.exits["notes"])


# ---- T+2 and dates ------------------------------------------------------------------------------------------------------------------------------------
def test_earliest_sale_is_two_sessions_after_the_first_possible_fill(rules, rc):
    card = swing(rules, rc, ctx(as_of=CAL[30]))
    assert card.holding["earliest_sell"] == str(CAL[33].date())            # signal day + 1 (fill) + 2 (T+2)


def test_dates_skip_holidays_present_in_the_calendar_and_extend_on_weekdays_after_it():
    cal = CAL.delete(31)                                                    # a holiday inside the calendar
    assert session_after(cal, cal[30], 2) == cal[32]
    last = cal[-1]
    assert session_after(cal, last, 3) == last + pd.offsets.BDay(3)          # beyond the known calendar: weekdays


def test_the_validity_ends_after_n_sessions(rules, rc):
    card = swing(rules, rc)
    assert card.valid_until == str(CAL[33].date())                          # validity_sessions = 3


# ---- limit fill, exactly as emitted ---------------------------------------------------------------------------------------------------------------
def make_world(o, h, l, c):
    T = len(o)
    frame = lambda a: pd.DataFrame(np.asarray(a, float).reshape(T, 1), index=pd.bdate_range("2026-01-01", periods=T), columns=[1])
    return MarketData(frame(o), frame(h), frame(l), frame(c), frame(np.full(T, 0.07)))


def test_the_card_becomes_the_engine_order_it_states_and_the_limit_fills_only_inside_the_zone(rules, rc):
    card = swing(rules, rc, ctx(hi20=60_000.0, hi60=90_000.0))
    e = card.entry
    item = card.to_signal_item()
    assert (item.order, item.limit_price, item.stop_price, item.target1_price, item.target_price, item.tag) == ("limit", e["zone_high"], card.exits["stop"], card.exits["target1"],
                                                                                                                  card.exits["target2"], card.card_id)
    n = 8
    hi_px, lo_px = e["zone_high"], e["zone_low"]
    # the market opens above the zone and never comes back: no fill
    up = make_world([50_000] * n, [51_000] * n, [hi_px + 500] * n, [50_000] * n)
    r = run_backtest(up, {0: Signal([item], full_rebalance=False)}, rules.scaled_costs(0), EngineConfig(capital=1e9))
    assert len(r.fills) == 0 and r.orders.iloc[0]["status"] == "unfilled"
    # it dips into the zone on the second session: filled at the limit, not below
    dip = make_world([50_000, 50_000, 50_000, 50_000, 50_000, 50_000, 50_000, 50_000], [51_000] * n, [50_000, 50_000, lo_px - 100, 50_000, 50_000, 50_000, 50_000, 50_000], [50_000] * n)
    r2 = run_backtest(dip, {0: Signal([item], full_rebalance=False)}, rules.scaled_costs(0), EngineConfig(capital=1e9))
    f = r2.fills.iloc[0]
    assert f["idx"] == 2 and f["price"] == hi_px and f["tag"] == card.card_id


def test_a_breakout_card_does_not_chase_a_gap_above_its_zone(rules, rc):
    card = swing(rules, rc, ctx(hi20=50_400.0, hi60=90_000.0))
    e = card.entry
    o = [50_000, e["zone_high"] + 1_000, e["zone_high"] + 1_000, e["zone_high"] + 1_000]
    d = make_world(o, [x + 300 for x in o], [x - 300 for x in o], [50_000] * 4)
    r = run_backtest(d, {0: Signal([card.to_signal_item()], full_rebalance=False)}, rules.scaled_costs(0), EngineConfig(capital=1e9))
    assert len(r.fills) == 0 and r.orders.iloc[0]["status"] == "cancelled"


# ---- mandatory sections ------------------------------------------------------------------------------------------------------------------------------
def test_a_complete_card_has_entry_target_rationale_and_holding_time(rules, rc):
    card = swing(rules, rc)
    assert card.problems(rules) == []
    for section in ("entry", "exits", "holding", "confidence", "rationale", "sizing", "validation"):
        assert getattr(card, section)


@pytest.mark.parametrize("section,key,expect", [("entry", "zone_high", "giá vào"), ("exits", "target2", "target"), ("holding", "earliest_sell", "thời gian nắm giữ"),
                                                 ("rationale", "risks", "lý do"), ("rationale", "invalidation", "lý do")])
def test_a_card_missing_any_mandatory_part_is_reported(rules, rc, section, key, expect):
    card = swing(rules, rc)
    getattr(card, section).pop(key)
    assert any(expect in p for p in card.problems(rules))


def test_prices_outside_the_session_band_or_off_the_tick_are_reported(rules, rc):
    card = swing(rules, rc)
    card.entry["zone_high"] = card.entry["session_ceiling"] + 500
    assert any("biên độ" in p for p in card.problems(rules))
    card = swing(rules, rc)
    card.exits["stop"] = card.exits["stop"] + 7
    assert any("bước giá" in p for p in card.problems(rules))


# ---- sizing --------------------------------------------------------------------------------------------------------------------------------------------
def test_the_size_is_set_by_the_risk_per_trade_in_whole_lots(rules, rc):
    card = swing(rules, rc, ctx(hi20=60_000.0, hi60=90_000.0, atr=800.0))
    s, x = card.sizing, card.exits
    assert s["shares"] % 100 == 0 and s["lots"] == s["shares"] // 100
    assert s["risk_if_stop"] == pytest.approx(s["shares"] * (x["reference"] - x["stop"]))
    assert s["risk_pct_capital"] <= rc.swing.risk_per_trade + 1e-9
    assert s["risk_pct_capital"] > 0.5 * rc.swing.risk_per_trade or s["caps_applied"]


def test_the_per_name_cap_limits_the_weight_and_says_so(rules, rc):
    card = swing(rules, rc, ctx(hi20=60_000.0, hi60=90_000.0, atr=400.0))         # tiny stop distance -> risk sizing would want a huge weight
    assert card.sizing["weight"] <= rc.swing.max_weight + 1e-9 and any("trần" in c for c in card.sizing["caps_applied"])


def test_capital_too_small_for_one_lot_rejects_the_card(rules, rc):
    small = rc.model_copy(update={"portfolio": rc.portfolio.model_copy(update={"capital": 1_000_000.0})})
    card = swing(rules, small, ctx(hi20=60_000.0, hi60=90_000.0))
    assert card.action == "NO_TRADE" and "lô" in card.rejected


# ---- confidence -------------------------------------------------------------------------------------------------------------------------------------------
def test_without_enough_past_evidence_the_probability_is_not_shown_as_the_headline_number(rules, rc):
    card = B.build_swing_card(ctx(), pred(), SIMILAR, {"n": 100, "auc": 0.7}, FAILED, rc, rules)
    k = card.confidence
    assert k["grade"] == "CHƯA ĐỦ BẰNG CHỨNG" and k["display_kind"] == "historical" and k["p_display"] == SIMILAR["win_rate"] and k["p_model"] == 0.33


def test_a_model_probability_with_no_discrimination_is_downgraded(rules, rc):
    k = B.build_swing_card(ctx(), pred(), SIMILAR, BAD, FAILED, rc, rules).confidence
    assert k["grade"] == "THẤP" and "AUC" in k["grade_reason"] and k["display_kind"] == "historical"


def test_a_probability_that_has_discriminated_and_is_calibrated_is_shown(rules, rc):
    k = B.build_swing_card(ctx(), pred(), SIMILAR, GOOD, FAILED, rc, rules).confidence
    assert k["grade"] == "KHÁ" and k["display_kind"] == "model" and k["p_display"] == 0.33


def test_a_miscalibrated_but_discriminating_model_is_medium_grade(rules, rc):
    k = B.build_swing_card(ctx(), pred(), SIMILAR, {"n": 9000, "auc": 0.6, "calibration_gap": 0.12}, FAILED, rc, rules).confidence
    assert k["grade"] == "TRUNG BÌNH"


def test_a_small_similar_group_is_flagged_with_its_n(rules, rc):
    card = B.build_swing_card(ctx(), pred(), {"n": 12, "win_rate": 0.5, "t1_rate": 0.6, "stop_rate": 0.3, "timeout_rate": 0.2}, BAD, FAILED, rc, rules)
    assert card.confidence["similar"]["small_n"] and any("n = 12" in r for r in card.rationale["risks"])
    assert "n nhỏ" in card.text_vi()


def test_the_score_percentile_is_from_the_rank(rules, rc):
    k = swing(rules, rc, p=pred(rank=1, n_universe=50)).confidence
    assert k["score_percentile"] == 1.0
    assert swing(rules, rc, p=pred(rank=50, n_universe=50)).confidence["score_percentile"] == 0.0


# ---- rationale: only what the system holds ------------------------------------------------------------------------------------------------------------
def test_reasons_cite_the_actual_shap_values_and_context(rules, rc):
    card = swing(rules, rc)
    txt = " ".join(card.rationale["reasons"])
    assert "hạng 2/48" in txt and "đáy 20 phiên" in txt and "0,022" in txt and "21,0%" in txt
    assert "RSI(14) = 55" in txt and "1,4 lần" in txt and "SMA200" in txt


def test_nothing_is_said_about_facts_that_are_not_in_the_data(rules, rc):
    c = ctx(rsi14=None, vol_spike=None, rs5=None, rs10=None, sma50=None, sma200=None, regime_off=None)
    card = swing(rules, rc, c)
    txt = " ".join(card.rationale["reasons"])
    assert "RSI" not in txt and "khối lượng" not in txt and "SMA" not in txt and "sức mạnh tương đối" not in txt.split("Bối cảnh")[-1]


def test_every_card_has_risks_and_invalidation_conditions(rules, rc):
    card = swing(rules, rc)
    assert len(card.rationale["risks"]) >= 1 and len(card.rationale["invalidation"]) >= 2
    assert any("T+2" in r for r in card.rationale["risks"])
    assert any(fmt_price(card.exits["stop"]) in t for t in card.rationale["invalidation"])


def test_a_failed_model_verdict_is_shown_on_the_card_and_in_the_risks(rules, rc):
    card = swing(rules, rc)
    assert card.validation["passed"] is False and card.validation["paper_trading_only"] and "CHƯA vượt baseline" in card.validation["model_status"]
    assert any("baseline" in r for r in card.rationale["risks"])
    assert "CHƯA vượt baseline" in card.text_vi() and "không phải cam kết" in card.text_vi()


def test_gating_on_the_verdict_turns_a_buy_into_a_watch(rules, rc):
    gated = rc.model_copy(update={"gate_on_verdict": True})
    assert swing(rules, gated).action == "WATCH" and swing(rules, rc).action == "BUY"
    assert swing(rules, rc, watch=True).action == "WATCH"


def test_the_vietnamese_text_has_all_six_parts(rules, rc):
    t = swing(rules, rc).text_vi()
    for part in ("1. GIÁ VÀO LỆNH", "2. TARGET & CẮT LỖ", "3. THỜI GIAN NẮM GIỮ", "4. ĐỘ TIN CẬY", "5. LÝ DO", "6. QUẢN TRỊ VỐN", "Rủi ro / lập luận ngược", "Tín hiệu mất hiệu lực"):
        assert part in t
    assert "MUA" in t and "R:R" in t


def test_a_rejected_card_says_why_and_carries_no_order(rules, rc):
    card = swing(rules, rc, ctx(hi20=60_000.0, hi60=51_500.0))
    assert card.action == "NO_TRADE" and card.rejected and "KHÔNG PHÁT HÀNH" in card.text_vi() and card.entry == {}


def test_json_round_trips_and_is_plain(rules, rc):
    import json
    d = json.loads(swing(rules, rc).to_json())
    assert d["strategy"] == "SWING" and d["entry"]["order_type"] in ("limit", "stop", "ato") and isinstance(d["exits"]["stop"], float)


# ---- INVEST -------------------------------------------------------------------------------------------------------------------------------------------
def invest_card(rules, rc, **kw):
    c = ctx(close=30_000.0, atr=700.0, hi252=36_000.0, feats={"mom_6m_csrank": 0.83, "dd_252": -0.12, "mom_6m": 0.18, "vol_126": 0.29})
    p = pred(model_name="invest_b2_factor_f03", contributions=[["mom_6m_csrank", 0.83, 0.03], ["-vol_126", 0.29, 0.02]], q10=-0.25, q50=0.06, q90=0.40)
    preset = rc_preset()
    return B.build_invest_card(c, p, "b2", preset, {"n": 40, "win_rate": 0.55}, None, FAILED, rc, rules, weight=0.035, next_review=CAL[50], top_k=10, rs_floor=0.4, **kw)


def rc_preset():
    from predict_stock.config import InvestConfig
    return InvestConfig().presets["b2"]


def test_an_invest_card_has_scenarios_thesis_break_conditions_and_no_hard_stop(rules, rc):
    card = invest_card(rules, rc)
    assert card.action == "BUY" and card.problems(rules) == []
    sc = card.exits["scenarios"]
    assert sc["bear"]["price"] < sc["base"]["price"] < sc["bull"]["price"] and card.exits["stop"] is None
    assert any("SMA200" in t for t in card.exits["thesis_break"]) and any("252" in t for t in card.exits["thesis_break"])
    assert card.holding["horizon_sessions"] == 126 and card.holding["next_review"] == str(CAL[50].date())
    assert [t["share"] for t in card.entry["tranches"]] == [0.5, 0.5]


def test_invest_scenario_prices_are_on_the_tick_and_the_text_says_no_hard_stop(rules, rc):
    card = invest_card(rules, rc)
    for k in ("bear", "base", "bull"):
        p = card.exits["scenarios"][k]["price"]
        assert abs(p / rules.tick(p) - round(p / rules.tick(p))) < 1e-9
    assert "không có stop cứng" in card.text_vi() and "KHÔNG phải lệnh stop" in card.text_vi()


def test_invest_weight_is_the_target_weight_in_whole_lots_within_the_cap(rules, rc):
    card = invest_card(rules, rc)
    assert card.sizing["shares"] % 100 == 0 and card.sizing["weight"] <= 0.035 + 1e-3 and card.sizing["risk_if_bear_pct_capital"] > 0
    capped = B.build_invest_card(ctx(close=30_000.0, hi252=36_000.0), pred(q10=-0.2, q50=0.05, q90=0.3, model_name="invest_b1_factor_f01"), "b1", rc_preset(), None, None, FAILED, rc, rules,
                                 weight=0.30, next_review=CAL[50], top_k=10, rs_floor=0.4)
    assert capped.sizing["weight"] <= rc.invest.max_weight + 1e-3 and capped.sizing["caps_applied"]


# ---- aggregator ----------------------------------------------------------------------------------------------------------------------------------------
def buy_card(rules, rc, iid, symbol, sleeve="swing", weight=0.05, price=50_000.0, risk=0.0075):
    c = swing(rules, rc, ctx(hi20=60_000.0, hi60=90_000.0, atr=1_500.0, symbol=symbol, instrument_id=iid))
    c = copy.deepcopy(c)
    c.instrument_id, c.symbol, c.sleeve = iid, symbol, sleeve
    s = c.sizing
    s["shares"] = int(weight * s["capital"] / s["sizing_price"] // 100 * 100)
    s["weight"] = s["shares"] * s["sizing_price"] / s["capital"]
    s["risk_pct_capital"] = risk
    return c


def test_sleeve_budgets_split_thirty_seventy_and_invest_is_shared(rc):
    b = AG.sleeve_budgets(rc)
    assert b == {"swing": 0.30, "invest_b1": pytest.approx(0.35), "invest_b2": pytest.approx(0.35)}


def test_a_card_that_does_not_fit_the_sleeve_is_scaled_to_the_room_or_rejected(rules, rc):
    cards = [buy_card(rules, rc, i, f"S{i}", weight=0.05) for i in range(1, 9)]
    out = AG.aggregate(cards, rc.model_copy(update={"portfolio": rc.portfolio.model_copy(update={"max_open_risk": 1.0})}), rules)
    assert len(out.accepted) >= 6 and out.summary["sleeve_used"]["swing"] <= 0.30 + 1e-9
    assert any("hạn mức sleeve" in c.rejected for c in out.rejected) or all(c.sizing["weight"] > 0 for c in out.accepted)
    assert sum(c.sizing["weight"] for c in out.accepted) <= 0.30 + 1e-9


def test_the_same_stock_in_two_sleeves_is_two_positions_and_the_total_is_capped(rules, rc):
    h = {"invest_b1": {1: 0.10}, "invest_b2": {1: 0.03}}
    c = buy_card(rules, rc, 1, "AAA", "swing", weight=0.035)                  # 2% of room left: at least half of the card fits, so it is scaled, not rejected
    out = AG.aggregate([c], rc, rules, holdings=h)
    assert len(out.accepted) == 1 and out.accepted[0].sizing["weight"] <= 0.15 - 0.13 + 1e-3        # total across sleeves stays at 15%
    assert any("tổng tỷ trọng" in x for x in out.accepted[0].sizing["caps_applied"])
    assert out.summary["sleeve_used"]["invest_b1"] == 0.10 and out.summary["sleeve_used"]["invest_b2"] == 0.03      # sleeves stay separate


def test_a_name_already_at_its_total_cap_gets_no_new_swing_card(rules, rc):
    out = AG.aggregate([buy_card(rules, rc, 1, "AAA")], rc, rules, holdings={"invest_b1": {1: 0.10}, "invest_b2": {1: 0.05}})
    assert not out.accepted and "trần tổng tỷ trọng" in out.rejected[0].rejected


def test_open_risk_of_the_swing_sleeve_is_limited(rules, rc):
    cards = [buy_card(rules, rc, i, f"S{i}", weight=0.03, risk=0.0075) for i in range(1, 6)]
    out = AG.aggregate(cards, rc, rules, open_risk=0.03)                                     # 3% already at risk, limit 5%: room for two more 0.75% cards
    assert len(out.accepted) == 2 and "rủi ro mở" in out.rejected[0].rejected


def test_kill_switch_state_follows_the_drawdown_rule():
    eq = pd.Series([100, 110, 105, 90, 86, 89, 90], index=pd.bdate_range("2026-01-01", periods=7), dtype=float)
    ks = AG.kill_state(eq, 0.20, 3)
    assert ks["killed"] and ks["drawdown_from_peak"] == pytest.approx(90 / 110 - 1) and ks["sessions_left"] == 1 and ks["events"] == ["2026-01-07"]
    assert not AG.kill_state(eq, 0.30, 3)["killed"]


def test_a_killed_portfolio_issues_no_buy_cards(rules, rc):
    eq = pd.Series([100, 110, 105, 90, 86], index=pd.bdate_range("2026-01-01", periods=5), dtype=float)
    out = AG.aggregate([buy_card(rules, rc, 1, "AAA")], rc, rules, equity=eq)
    assert not out.accepted and "kill-switch" in out.rejected[0].rejected and out.summary["kill_switch"]["killed"]


def test_watch_cards_pass_through_and_no_trade_cards_stay_rejected(rules, rc):
    w = swing(rules, rc, watch=True)
    nt = swing(rules, rc, ctx(hi20=60_000.0, hi60=51_500.0))
    out = AG.aggregate([w, nt], rc, rules)
    assert out.accepted == [w] and out.rejected == [nt]
