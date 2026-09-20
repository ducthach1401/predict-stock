"""ACCEPTANCE: the simulator under Vietnamese rules. Every scenario is small enough to check by hand.

Prices sit on the tick grid (tick 50 between 10,000 and 49,950) so that rounding does not obscure the rule under test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict_stock.backtest.engine import EngineConfig, MarketData, Signal, SignalItem, run_backtest
from predict_stock.backtest.market import MarketRules

NaN = np.nan


@pytest.fixture
def rules(cfg):
    return MarketRules.from_config(cfg.market)


def make_data(o, h=None, l=None, c=None, band=0.07):
    """One instrument (id 1) unless the arrays are 2-D. h/l/c default to the open."""
    o = np.asarray(o, float)
    o = o.reshape(len(o), -1)
    T, N = o.shape
    arr = lambda a: o if a is None else np.asarray(a, float).reshape(T, N)
    cal = pd.bdate_range("2024-01-01", periods=T)
    frame = lambda a: pd.DataFrame(a, index=cal, columns=list(range(1, N + 1)))
    return MarketData(frame(o), frame(arr(h)), frame(arr(l)), frame(arr(c)), frame(np.full((T, N), band)))


def buy(w, **kw):
    return Signal([SignalItem(1, w, **kw)])


def sell_all():
    return Signal([SignalItem(1, 0.0)])


def go(data, sigs, rules, cost=0, capital=1e9, **cfg):
    return run_backtest(data, sigs, rules.scaled_costs(cost), EngineConfig(capital=capital, **cfg))


# ---- timing: decided at the close of t, filled at the open of t+1 --------------------------------------------------------------
def test_a_signal_fills_at_the_next_open_never_the_same_session(rules):
    d = make_data(o=[20000, 20500, 21000, 21500, 22000], c=[20250, 20750, 21250, 21750, 22250])
    r = go(d, {0: buy(0.1)}, rules)
    f = r.fills.iloc[0]
    assert f["idx"] == 1 and f["price"] == 20500                       # open of session 1, not the open of session 0 (20,000)
    assert r.fills["idx"].min() >= 1


def test_no_signal_no_trade_and_a_decision_on_the_last_session_is_ignored(rules):
    d = make_data(o=[20000] * 5)
    assert len(go(d, {}, rules).fills) == 0
    r = go(d, {4: buy(0.1)}, rules)
    assert len(r.fills) == 0 and r.equity.iloc[-1] == 1e9
    assert len(go(d, {3: buy(0.1)}, rules, ).fills) == 1


def test_future_prices_cannot_change_the_past(rules):
    """Equity up to session k is identical whatever happens after k (the simulator only reads the future to fill)."""
    o = np.array([20000, 20500, 21000, 21500, 22000, 22500, 23000, 23500], float)
    o2 = o.copy(); o2[5:] *= 0.5
    sigs = {0: buy(0.2), 2: buy(0.3)}
    a, b = go(make_data(o), sigs, rules, cost=1), go(make_data(o2), sigs, rules, cost=1)
    assert a.equity.iloc[:5].equals(b.equity.iloc[:5])


def test_the_run_is_deterministic(rules):
    rng = np.random.default_rng(0)
    o = 20000 + 50 * rng.integers(-20, 20, 60).cumsum()
    o = np.maximum(o, 10050).astype(float)
    sigs = {i: buy(0.1 * (1 + i % 3)) for i in range(0, 50, 7)}
    a, b = go(make_data(o), sigs, rules, cost=1), go(make_data(o), sigs, rules, cost=1)
    assert a.equity.equals(b.equity) and a.fills.equals(b.fills)


# ---- fees, tax, slippage, lots --------------------------------------------------------------------------------------------------
def test_buy_cash_flow_includes_the_fee_and_whole_lots(rules):
    d = make_data(o=[20000] * 6)
    free = go(d, {0: buy(0.02)}, rules, cost=0)
    assert free.fills.iloc[0]["qty"] == 1000 and free.fills.iloc[0]["fee"] == 0                    # frictionless: 2e7 / 20,000
    fee_only = run_backtest(d, {0: buy(0.02)}, rules.with_costs(slippage=0), EngineConfig(capital=1e9))
    f = fee_only.fills.iloc[0]
    assert f["qty"] == 900 and f["price"] == 20000 and f["qty"] % 100 == 0                        # 2e7 / (20,000 x 1.0015) = 998 shares -> 9 lots
    assert f["fee"] == pytest.approx(900 * 20000 * 0.0015)
    assert fee_only.cash.iloc[-1] == pytest.approx(1e9 - 900 * 20000 - 27_000)


def test_a_full_round_trip_pays_fee_both_ways_tax_on_the_sell_and_slippage(rules):
    d = make_data(o=[20000] * 8)
    r = go(d, {0: buy(0.02), 3: sell_all()}, rules, cost=1)
    b, s = r.fills.iloc[0], r.fills.iloc[1]
    assert (b["side"], b["price"], s["side"], s["price"]) == ("buy", 20050, "sell", 19950)      # 0.1% against you, tick-rounded against you
    assert b["qty"] == s["qty"] == 900                                                             # 2e7 / (20,050 x 1.0015) = 996 shares -> 900
    assert b["fee"] == pytest.approx(900 * 20050 * 0.0015)
    assert s["fee"] == pytest.approx(900 * 19950 * 0.0015) and s["tax"] == pytest.approx(900 * 19950 * 0.001)
    assert b["tax"] == 0                                                                           # tax on sells only
    expected = 1e9 - 900 * 20050 - b["fee"] + 900 * 19950 - s["fee"] - s["tax"]
    assert r.equity.iloc[-1] == pytest.approx(expected)
    (trip,) = r.round_trips.to_dict("records")
    assert trip["pnl"] == pytest.approx(expected - 1e9) and trip["net_return"] < trip["gross_return"] < 0
    assert r.stats["open_positions"] == 0


def test_a_target_smaller_than_one_lot_is_not_traded(rules):
    d = make_data(o=[20000] * 5)
    r = go(d, {0: buy(0.001)}, rules, capital=1e9)                          # 1e6 buys 49 shares
    assert len(r.fills) == 0 and r.orders.iloc[0]["status"] == "unfilled" and r.orders.iloc[0]["reason"] == "lot_too_small"
    assert r.stats["blocked"]["lot_too_small"] == 1


def test_buys_are_limited_by_cash_and_sells_fund_buys_of_the_same_session(rules):
    d = make_data(o=[[20000, 20000]] * 8)
    both = Signal([SignalItem(1, 0.8), SignalItem(2, 0.8)])                   # 160% of the money
    r = run_backtest(d, {0: both}, rules.scaled_costs(0), EngineConfig(capital=1e9))
    q = r.fills.groupby("instrument_id")["qty"].sum()
    assert q[1] == 40000 and 0 < q[2] < 40000 and r.cash.iloc[-1] >= 0     # the second buy gets what is left, whole lots
    assert r.cash.iloc[-1] < 20000 * 100
    # rotate: sell 1, buy 2 in the same session; the proceeds are available
    rot = {0: Signal([SignalItem(1, 0.9)]), 3: Signal([SignalItem(2, 0.9)])}
    r2 = run_backtest(d, rot, rules.scaled_costs(0), EngineConfig(capital=1e9))
    fs = r2.fills
    assert fs[(fs.idx == 4) & (fs.side == "sell")].qty.iloc[0] == 45000 and fs[(fs.idx == 4) & (fs.side == "buy")].qty.iloc[0] == 45000


def test_long_only_negative_weights_are_rejected(rules):
    with pytest.raises(ValueError, match="long-only"):
        go(make_data(o=[20000] * 4), {0: buy(-0.1)}, rules)


# ---- T+2 ---------------------------------------------------------------------------------------------------------------------------
def test_shares_cannot_be_sold_before_t_plus_2(rules):
    d = make_data(o=[20000] * 8)
    r = go(d, {0: buy(0.1), 1: sell_all()}, rules)                 # bought at session 1; asks to sell at the close of session 1
    fs = r.fills
    assert fs[fs.side == "buy"].idx.iloc[0] == 1 and fs[fs.side == "sell"].idx.iloc[0] == 3          # session 2 is too early (1 session held)
    assert r.stats["blocked"]["t_plus_settlement"] == 1


def test_settlement_days_is_configurable(rules):
    d = make_data(o=[20000] * 9)
    sigs = {0: buy(0.1), 1: sell_all()}
    from dataclasses import replace
    for days, first_sell in ((0, 2), (1, 2), (2, 3), (3, 4)):
        r = run_backtest(d, sigs, replace(rules.scaled_costs(0), settlement_days=days), EngineConfig())
        assert r.fills[r.fills.side == "sell"].idx.iloc[0] == first_sell, days


def test_a_full_rebalance_cannot_sell_early_either_and_sells_only_what_is_sellable(rules):
    d = make_data(o=[20000] * 10)
    sigs = {0: buy(0.1), 2: Signal([SignalItem(1, 0.2)]), 3: sell_all()}                # top up at session 3, then ask to sell everything
    r = go(d, sigs, rules)
    fs = r.fills
    sells = fs[fs.side == "sell"]
    assert sells.qty.sum() == fs[fs.side == "buy"].qty.sum()                            # everything is eventually sold
    assert sells.idx.min() == 4 and sells.qty.iloc[0] == 5000                           # session 4: the first lot (bought at 1) is sellable, the top-up (bought at 3) is not
    assert sells.idx.max() == 5 and len(sells) == 2


def test_stops_do_not_fire_during_the_settlement_period(rules):
    o = [20000, 20000, 18000, 18000, 18000, 18000]
    l = [19900, 19900, 17900, 17900, 17900, 17900]
    r = go(make_data(o, h=[20100, 20100, 18100, 18100, 18100, 18100], l=l, c=[20000, 20000, 18000, 18000, 18000, 18000]),
           {0: buy(0.1, stop_pct=0.05)}, rules)
    s = r.fills[r.fills.side == "sell"].iloc[0]
    assert s["idx"] == 3 and s["price"] == 18000 and s["reason"] == "stop_gap"          # the crash happened at 2; the first chance to sell is 3, at the open


# ---- stop / target ------------------------------------------------------------------------------------------------------------------
def bar_world(day3_low, day3_high, day3_open=20000):
    o = [20000, 20000, 20000, day3_open, 20000, 20000]
    h = [20100, 20100, 20100, day3_high, 20100, 20100]
    l = [19900, 19900, 19900, day3_low, 19900, 19900]
    return make_data(o, h=h, l=l, c=[20000, 20000, 20000, 20000, 20000, 20000])


def test_stop_is_hit_inside_the_session_at_the_stop_price(rules):
    r = go(bar_world(18900, 20100), {0: buy(0.1, stop_pct=0.05, target_pct=0.05)}, rules)
    s = r.fills[r.fills.side == "sell"].iloc[0]
    assert (s["idx"], s["price"], s["reason"]) == (3, 19000, "stop")                     # 20,000 x 0.95


def test_target_is_hit_inside_the_session_at_the_target_price(rules):
    r = go(bar_world(19900, 21100), {0: buy(0.1, stop_pct=0.05, target_pct=0.05)}, rules)
    s = r.fills[r.fills.side == "sell"].iloc[0]
    assert (s["idx"], s["price"], s["reason"]) == (3, 21000, "target")


def test_when_both_levels_are_inside_one_bar_the_stop_is_taken_first(rules):
    d = bar_world(18900, 21100)
    r = go(d, {0: buy(0.1, stop_pct=0.05, target_pct=0.05)}, rules)
    s = r.fills[r.fills.side == "sell"].iloc[0]
    assert (s["price"], s["reason"]) == (19000, "stop")                                  # the default: the pessimistic order
    r2 = go(d, {0: buy(0.1, stop_pct=0.05, target_pct=0.05)}, rules, tie="target_first")
    assert r2.fills[r2.fills.side == "sell"].iloc[0]["reason"] == "target"


def test_a_gap_through_a_level_exits_at_the_open_not_at_the_level(rules):
    r = go(bar_world(18650, 18900, day3_open=18700), {0: buy(0.1, stop_pct=0.05)}, rules)           # stop 19,000; opens at 18,700 (-6.5%, inside the band)
    s = r.fills[r.fills.side == "sell"].iloc[0]
    assert (s["price"], s["reason"]) == (18700, "stop_gap")                                       # not 19,000
    r = go(bar_world(21300, 21400, day3_open=21350), {0: buy(0.1, target_pct=0.05)}, rules)         # target 21,000; opens at 21,350 (+6.75%)
    s = r.fills[r.fills.side == "sell"].iloc[0]
    assert (s["price"], s["reason"]) == (21350, "target_gap")


def test_a_session_locked_at_the_floor_cannot_be_exited_by_a_stop(rules):
    o = [20000, 20000, 20000, 18600, 18600, 19000]
    r = go(make_data(o, h=o, l=o, c=o), {0: buy(0.1, stop_pct=0.05)}, rules)                          # session 3: open = high = low = the floor 18,600
    s = r.fills[r.fills.side == "sell"].iloc[0]
    assert r.stats["blocked"]["limit_down_locked"] == 1                                            # session 3: nobody can sell at the floor
    assert s["idx"] == 4 and s["price"] == 18600 and s["reason"] == "stop_gap"                     # session 4's reference is 18,600, so it is no longer at the floor


def test_stop_exit_pays_slippage_and_never_prints_below_the_low(rules):
    r = go(bar_world(18900, 20100), {0: buy(0.1, stop_pct=0.05)}, rules, cost=1)
    b = r.fills[r.fills.side == "buy"].iloc[0]
    stop = rules.round_down(b["price"] * 0.95)
    s = r.fills[r.fills.side == "sell"].iloc[0]
    assert s["price"] == max(rules.sell_price(stop), 18900) and s["price"] <= stop and s["price"] >= 18900
    r = go(bar_world(19000, 20100), {0: buy(0.1, stop_pct=0.05)}, rules, cost=1)          # the low is above the slipped price: cannot print below the low
    assert r.fills[r.fills.side == "sell"].iloc[0]["price"] >= 19000


def test_target_exit_is_a_limit_and_pays_no_slippage(rules):
    r = go(bar_world(19900, 21100), {0: buy(0.1, target_pct=0.05)}, rules, cost=1)
    b = r.fills[r.fills.side == "buy"].iloc[0]
    assert r.fills[r.fills.side == "sell"].iloc[0]["price"] == rules.round_up(b["price"] * 1.05)


def test_time_stop_exits_at_the_open_after_the_holding_period(rules):
    d = make_data(o=[20000, 20000, 20100, 20200, 20300, 20400, 20500, 20600, 20700])
    r = go(d, {0: buy(0.1, max_hold=5)}, rules)
    s = r.fills[r.fills.side == "sell"].iloc[0]
    assert (s["idx"], s["reason"], s["price"]) == (6, "time", 20500)                     # bought at 1 -> exits at the open of 1 + 5


def test_a_time_stop_shorter_than_settlement_waits(rules):
    d = make_data(o=[20000] * 8)
    r = go(d, {0: buy(0.1, max_hold=1)}, rules)
    assert r.fills[r.fills.side == "sell"].iloc[0]["idx"] == 3


# ---- price band, locked limits -----------------------------------------------------------------------------------------------------------
def test_a_buy_at_the_ceiling_cannot_fill(rules):
    d = make_data(o=[20000, 21400, 21400, 21400], c=[20000, 21400, 21400, 21400])          # 20,000 + 7% = 21,400 = the ceiling
    r = go(d, {0: buy(0.1)}, rules)
    assert len(r.fills) == 0 and r.stats["blocked"]["limit_up_locked"] == 1 and r.orders.iloc[0]["reason"] == "limit_up_locked"


def test_a_buy_just_below_the_ceiling_fills_but_never_above_it(rules):
    d = make_data(o=[20000, 21350, 21350, 21350], c=[20000, 21350, 21350, 21350])
    r = go(d, {0: buy(0.1)}, rules, cost=1)
    assert r.fills.iloc[0]["price"] == 21400                                                 # 21,350 x 1.001 -> 21,400: capped at the ceiling


def test_a_wider_band_allows_what_a_narrow_one_forbids(rules):
    d7 = make_data(o=[20000, 21400, 21400, 21400], band=0.07)
    d10 = make_data(o=[20000, 21400, 21400, 21400], band=0.10)                               # ceiling 22,000: 21,400 is not locked
    assert len(go(d7, {0: buy(0.1)}, rules).fills) == 0 and len(go(d10, {0: buy(0.1)}, rules).fills) == 1


def test_a_sell_at_the_floor_waits_until_it_can_trade(rules):
    o = [20000, 20000, 20000, 20000, 18600, 19000, 19000]
    d = make_data(o, h=o, l=o, c=[20000, 20000, 20000, 20000, 18600, 19000, 19000])          # 20,000 - 7% = 18,600 = the floor at session 4
    r = go(d, {0: buy(0.1), 3: sell_all()}, rules)
    s = r.fills[r.fills.side == "sell"].iloc[0]
    assert s["idx"] == 5 and s["price"] == 19000 and r.stats["blocked"]["limit_down_locked"] == 1


def test_the_band_is_measured_from_the_last_traded_close_after_a_suspension(rules):
    """Session 1 closes at 20,000, then two suspended sessions; the reference on resumption is still 20,000 (ceiling 21,400)."""
    def world(resume_open):
        o = [20000, 20000, NaN, NaN, resume_open, resume_open, resume_open]
        return make_data(o, h=o, l=o, c=o)
    locked = go(world(21400), {3: buy(0.1)}, rules)                                                   # decided during the suspension
    assert len(locked.fills) == 0 and locked.stats["blocked"]["limit_up_locked"] == 1
    open_ = go(world(21350), {3: buy(0.1)}, rules)
    assert len(open_.fills) == 1 and open_.fills.iloc[0]["idx"] == 4


# ---- suspension ------------------------------------------------------------------------------------------------------------------------------
def test_a_suspended_session_blocks_a_buy_and_valuation_uses_the_last_close(rules):
    o = [20000, 20000, NaN, 20000, 20000]
    d = make_data(o, h=o, l=o, c=o)
    r = go(d, {1: buy(0.1)}, rules)                                                            # decided at 1, executes at 2: no bar
    assert len(r.fills) == 0 and r.stats["blocked"]["suspended_or_no_open"] == 1
    r = go(d, {0: buy(0.1)}, rules)                                                            # bought at 1
    assert r.equity.iloc[2] == r.equity.iloc[1]                                                # marked at the last close during the suspension


def test_a_sell_persists_through_a_suspension_and_executes_when_trading_resumes(rules):
    o = [20000, 20000, 20000, NaN, NaN, 20500, 20500]
    d = make_data(o, h=o, l=o, c=o)
    r = go(d, {0: buy(0.1), 2: sell_all()}, rules)
    s = r.fills[r.fills.side == "sell"].iloc[0]
    assert s["idx"] == 5 and s["price"] == 20500 and r.stats["blocked"]["suspended_or_no_open"] == 2


def test_an_unrepairable_open_blocks_market_fills_that_day(rules):
    o = [20000, NaN, 20000, 20000, 20000]
    c = [20000, 20000, 20000, 20000, 20000]
    r = go(make_data(o, h=[20100] * 5, l=[19900] * 5, c=c), {0: buy(0.1)}, rules)             # the bar exists but its open was declared unreliable
    assert len(r.fills) == 0 and r.orders.iloc[0]["reason"] == "suspended_or_no_open"


# ---- limit orders ----------------------------------------------------------------------------------------------------------------------------
def limit_world(lows, opens=None):
    n = len(lows)
    o = opens or [20000] * n
    return make_data(o, h=[max(x, 20100) for x in o], l=lows, c=[20000] * n)


def test_a_limit_buy_fills_only_if_the_low_reaches_the_limit(rules):
    d = limit_world([19900, 19900, 19900, 19900, 19900])                                    # limit = 20,000 x (1 - 1%) = 19,800: never reached
    r = go(d, {0: buy(0.1, order="limit", limit_offset=0.01)}, rules)
    assert len(r.fills) == 0 and r.orders.iloc[0]["status"] == "unfilled" and r.orders.iloc[0]["reason"] == "limit_not_reached"
    d = limit_world([19900, 19700, 19900, 19900, 19900])
    r = go(d, {0: buy(0.1, order="limit", limit_offset=0.01)}, rules)
    f = r.fills.iloc[0]
    assert (f["idx"], f["price"], f["kind"]) == (1, 19800, "limit")                          # the limit price, no slippage


def test_a_limit_buy_fills_at_the_open_when_the_open_is_better(rules):
    d = limit_world([19700, 19700, 19700, 19700], opens=[20000, 19750, 19750, 19750])
    r = go(d, {0: buy(0.1, order="limit", limit_offset=0.01)}, rules, cost=1)
    assert r.fills.iloc[0]["price"] == 19750


def test_a_limit_order_can_stay_valid_for_several_sessions_and_the_fill_rate_is_recorded(rules):
    d = limit_world([19900, 19900, 19700, 19900, 19900, 19900])
    one = go(d, {0: buy(0.1, order="limit", limit_offset=0.01, valid_sessions=1)}, rules)          # valid for session 1 only; the low touches at session 2
    assert len(one.fills) == 0 and one.orders.iloc[0]["status"] == "unfilled"
    two = go(d, {0: buy(0.1, order="limit", limit_offset=0.01, valid_sessions=2)}, rules)          # valid for sessions 1 and 2
    assert two.fills.iloc[0]["idx"] == 2 and two.orders.iloc[0]["status"] == "filled"
    both = go(d, {0: buy(0.05), 1: Signal([SignalItem(1, 0.1, order="limit", limit_offset=0.05)])}, rules)   # one market fill, one limit that never fills
    lim = both.orders[both.orders["kind"] == "limit"]
    assert len(lim) == 1 and lim.iloc[0]["status"] == "unfilled"
    rate = (both.orders[both.orders.kind.isin(["limit"])].status == "filled").mean()
    assert rate == 0.0


def test_a_new_decision_replaces_a_pending_order(rules):
    d = limit_world([19900] * 6)
    r = go(d, {0: buy(0.1, order="limit", limit_offset=0.5, valid_sessions=4), 1: buy(0.2)}, rules)
    st = r.orders.set_index("order_id")["status"]
    assert "superseded" in st.tolist() and r.fills.iloc[0]["idx"] == 2 and r.fills.iloc[0]["kind"] == "open"


# ---- rebalancing ---------------------------------------------------------------------------------------------------------------------------------
def test_a_small_change_below_the_threshold_is_not_traded(rules):
    d = make_data(o=[20000] * 10)
    sigs = {0: buy(0.10), 3: Signal([SignalItem(1, 0.105)])}                               # the target moves 5%: +0.5 points of equity
    assert len(go(d, sigs, rules, rebalance_threshold=0.10).fills) == 1                    # 5% of the target < 10%: no trade
    assert len(go(d, sigs, rules, rebalance_threshold=0.01).fills) == 2
    sigs = {0: buy(0.10), 3: sell_all()}
    assert len(go(d, sigs, rules, rebalance_threshold=5.0).fills) == 2                     # an exit is never blocked by the threshold


def test_the_threshold_is_relative_to_the_target_so_small_weights_are_not_frozen(rules):
    """A 2% position drifting to half its target must be topped up even though the gap is only 1% of equity."""
    o = np.array([[20000.0, 20000.0]] * 3 + [[20000.0, 10000.0]] * 8)                       # instrument 2 halves in price after being bought
    d = make_data(o)
    sigs = {0: Signal([SignalItem(1, 0.02), SignalItem(2, 0.02)]), 5: Signal([SignalItem(1, 0.02), SignalItem(2, 0.02)])}
    r = run_backtest(d, sigs, rules.scaled_costs(0), EngineConfig(capital=1e9, rebalance_threshold=0.2))
    second_buys = r.fills[(r.fills.idx == 6) & (r.fills.side == "buy")]
    assert list(second_buys.instrument_id) == [2]                                          # instrument 2 is 50% under target -> bought; 1 is on target -> untouched


def test_reducing_a_weight_sells_whole_lots(rules):
    d = make_data(o=[20000] * 10)
    r = go(d, {0: buy(0.2), 4: Signal([SignalItem(1, 0.1)])}, rules)
    s = r.fills[r.fills.side == "sell"].iloc[0]
    assert s["qty"] % 100 == 0 and s["qty"] == 5000 and r.fills[r.fills.side == "buy"].qty.iloc[0] == 10000


def test_full_rebalance_sells_what_is_not_listed_but_partial_signals_leave_it(rules):
    d = make_data(o=[[20000, 20000]] * 10)
    a = {0: Signal([SignalItem(1, 0.2)]), 3: Signal([SignalItem(2, 0.2)], full_rebalance=True)}
    r = run_backtest(d, a, rules.scaled_costs(0), EngineConfig())
    assert (r.fills[(r.fills.instrument_id == 1) & (r.fills.side == "sell")].idx == 4).all()
    b = {0: Signal([SignalItem(1, 0.2)]), 3: Signal([SignalItem(2, 0.2)], full_rebalance=False)}
    r = run_backtest(d, b, rules.scaled_costs(0), EngineConfig())
    assert len(r.fills[(r.fills.instrument_id == 1) & (r.fills.side == "sell")]) == 0


# ---- accounting -------------------------------------------------------------------------------------------------------------------------------------
def test_equity_is_cash_plus_marked_positions_and_costs_reduce_it(rules):
    rng = np.random.default_rng(3)
    o = np.maximum(20000 + 50 * rng.integers(-10, 10, 80).cumsum(), 10050).astype(float)
    sigs = {i: buy(0.3 if (i // 10) % 2 == 0 else 0.0) for i in range(0, 70, 10)}
    net = run_backtest(make_data(o), sigs, rules, EngineConfig())
    gross = run_backtest(make_data(o), sigs, rules.scaled_costs(0), EngineConfig())
    assert net.equity.iloc[-1] < gross.equity.iloc[-1]                                        # costs cost something
    paid = net.fills["fee"].sum() + net.fills["tax"].sum()
    assert paid > 0 and gross.fills["fee"].sum() == 0 and gross.fills["tax"].sum() == 0
    d = make_data(o)
    for res in (net, gross):
        held = (res.equity - res.cash) / res.equity
        assert np.allclose(held, res.exposure) and (res.cash >= -1e-6).all()
    closed = net.round_trips
    if len(net.open_positions) == 0 and len(closed):
        assert net.equity.iloc[-1] == pytest.approx(1e9 + closed["pnl"].sum())                   # money in = money out, to the dong


def test_a_window_can_start_mid_history_using_the_previous_close_as_reference(rules):
    d = make_data(o=[20000] * 5 + [21400, 21400, 21400], c=[20000] * 5 + [21400, 21400, 21400])
    full = go(d, {4: buy(0.1)}, rules)
    part = run_backtest(d, {4: buy(0.1)}, rules.scaled_costs(0), EngineConfig(), start=3)
    assert len(full.fills) == 0 and len(part.fills) == 0 and part.equity.index[0] == d.close.index[3]      # both see the locked ceiling at session 5
