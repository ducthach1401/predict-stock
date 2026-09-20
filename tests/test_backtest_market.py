"""Market rules: tick size, rounding, price band, fees, tax, slippage, lots, band inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict_stock.backtest.market import MarketRules, infer_bands


@pytest.fixture
def rules(cfg):
    return MarketRules.from_config(cfg.market)


def test_rules_come_from_the_config(rules, cfg):
    m = cfg.market
    assert (rules.lot_size, rules.fee_rate, rules.sell_tax_rate, rules.slippage_rate, rules.settlement_days) == (
        m.lot_size, m.fee_rate, m.sell_tax_rate, m.slippage_rate, m.settlement_days)
    assert rules.bands == (0.07, 0.10, 0.15) and rules.lot_size == 100


@pytest.mark.parametrize("price,tick", [(1, 10), (5000, 10), (9990, 10), (10000, 50), (10050, 50), (49950, 50), (50000, 100), (50100, 100), (250000, 100)])
def test_tick_size_by_price_level(rules, price, tick):
    assert rules.tick(price) == tick
    assert rules.tick(np.array([price]))[0] == tick


@pytest.mark.parametrize("price,down,up", [
    (10000, 10000, 10000), (10001, 10000, 10050), (10024, 10000, 10050), (10025, 10000, 10050), (49990, 49950, 50000),
    (50001, 50000, 50100), (9995, 9990, 10000), (20049.99, 20000, 20050), (10700.000000000002, 10700, 10700)])   # last: float noise must not push a price a tick
def test_rounding_to_the_tick(rules, price, down, up):
    assert rules.round_down(price) == down and rules.round_up(price) == up


def test_rounding_works_on_arrays(rules):
    p = np.array([10001.0, 49990.0, 50001.0, 9995.0])
    assert rules.round_down(p).tolist() == [10000, 49950, 50000, 9990] and rules.round_up(p).tolist() == [10050, 50000, 50100, 10000]


@pytest.mark.parametrize("prev,band,ceil,floor", [
    (10000, 0.07, 10700, 9300),          # 10,700 / 9,300 sit on the tick grid
    (27300, 0.07, 29200, 25400),         # 29,211 rounds DOWN to 29,200; 25,389 rounds UP to 25,400
    (9900, 0.07, 10550, 9210),           # 10,593 -> 10,550 (tick 50); 9,207 -> 9,210 (tick 10)
    (20000, 0.10, 22000, 18000), (20000, 0.15, 23000, 17000), (100000, 0.07, 107000, 93000)])
def test_price_band_ceiling_and_floor(rules, prev, band, ceil, floor):
    assert rules.ceiling(prev, band) == ceil and rules.floor(prev, band) == floor
    assert floor <= prev <= ceil                                             # the band never excludes the reference


def test_band_is_never_wider_than_the_rule(rules):
    for prev in (9990, 10010, 27300, 49950, 51230, 88800):
        assert rules.ceiling(prev, 0.07) <= prev * 1.07 + 1e-6 and rules.floor(prev, 0.07) >= prev * 0.93 - 1e-6


def test_fees_tax_and_slippage(rules):
    assert rules.buy_fee(20_000_000) == pytest.approx(30_000)                          # 0.15%
    fee, tax = rules.sell_costs(20_000_000)
    assert fee == pytest.approx(30_000) and tax == pytest.approx(20_000)               # 0.15% + 0.1% tax on SELLS only
    assert rules.buy_price(20000) == 20050 and rules.sell_price(20000) == 19950        # 0.1% against you, rounded to the tick, in your disfavour
    assert rules.buy_price(60000) == 60100 and rules.sell_price(60000) == 59900        # tick 100 at this level
    assert rules.buy_price(10000) >= 10010 and rules.sell_price(10000) <= 9990


def test_round_trip_cost_on_a_flat_price(rules):
    """Buy then sell at the same reference price loses fee x2 + tax + 2 x slippage (before tick effects)."""
    buy, sell = rules.buy_price(20000), rules.sell_price(20000)
    qty = 1000
    cash = -(qty * buy) - rules.buy_fee(qty * buy)
    fee, tax = rules.sell_costs(qty * sell)
    cash += qty * sell - fee - tax
    assert cash == pytest.approx(-(qty * 20000) * (2 * 0.0015 + 0.001 + 2 * 0.0025), rel=0.03)   # 0.25% = one tick on 20,000, applied twice
    assert cash < 0


def test_lots(rules):
    assert [rules.lots(q) for q in (0, 99, 100, 199, 250, 1000)] == [0, 0, 100, 100, 200, 1000]
    assert rules.buy_quantity(2_000_000, 20000) == 0                                       # 2m buys 99.85 shares: not a lot
    assert rules.buy_quantity(3_000_000, 20000) == 100
    assert rules.buy_quantity(20_000_000, 20000) == 900                                    # 998 shares fit; whole lots only: 900
    assert rules.buy_quantity(20_030_000, 20000) == 1000                                   # exactly enough for price x qty x (1 + fee)
    assert rules.buy_quantity(20_029_999, 20000) == 900
    assert rules.buy_quantity(0, 20000) == 0 and rules.buy_quantity(1e9, 0) == 0


def test_scaled_and_overridden_costs(rules):
    free = rules.scaled_costs(0)
    assert (free.fee_rate, free.sell_tax_rate, free.slippage_rate) == (0, 0, 0) and free.lot_size == rules.lot_size and free.settlement_days == 2
    dbl = rules.scaled_costs(2)
    assert dbl.fee_rate == pytest.approx(0.003) and dbl.sell_tax_rate == pytest.approx(0.002) and dbl.slippage_rate == pytest.approx(0.002)
    assert rules.with_costs(fee=0.001).fee_rate == 0.001 and rules.with_costs(fee=0.001).sell_tax_rate == rules.sell_tax_rate


# ---- band inference ----------------------------------------------------------------------------------------------------------
def closes(returns, start=100.0):
    idx = pd.bdate_range("2020-01-01", periods=len(returns) + 1)
    return pd.DataFrame({1: start * np.r_[1, np.cumprod(1 + np.asarray(returns))]}, index=idx)


@pytest.mark.parametrize("move,band", [(0.03, 0.07), (0.069, 0.07), (0.074, 0.07), (0.076, 0.10), (0.099, 0.10), (0.104, 0.10), (0.106, 0.15), (0.14, 0.15)])
def test_band_is_the_smallest_allowed_band_containing_the_largest_recent_move(rules, move, band):
    c = closes([0.001] * 20 + [move] + [0.001] * 10)
    assert infer_bands(c, rules).iloc[-1, 0] == band


def test_the_band_uses_information_up_to_yesterday_only(rules):
    c = closes([0.001] * 20 + [0.095] + [0.001] * 5)
    b = infer_bands(c, rules)[1]
    move_day = 21
    assert b.iloc[move_day] == 0.07 and b.iloc[move_day + 1] == 0.10                     # the day of the move still had the old band
    assert b.iloc[0] == 0.07                                                                # no history: the narrowest band


def test_the_band_forgets_a_move_after_the_window(rules):
    c = closes([0.001] * 5 + [0.095] + [0.001] * 60)
    b = infer_bands(c, rules, window=20)[1]
    assert b.iloc[10] == 0.10 and b.iloc[40] == 0.07


def test_a_known_exchange_overrides_the_inference(rules):
    c = closes([0.001] * 10 + [0.095] + [0.001] * 5)
    ex = pd.DataFrame(np.nan, index=c.index, columns=[1])
    ex.iloc[:8, 0] = 0.15
    b = infer_bands(c, rules, exchange_band=ex)[1]
    assert (b.iloc[:8] == 0.15).all() and b.iloc[14] == 0.10
