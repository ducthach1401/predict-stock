"""Hits the real DNSE endpoint. Excluded by default; run with:  pytest -m live"""
from datetime import date

import pytest

from predict_stock.data.dnse_client import DnseClient, DnseInvalidSymbol

pytestmark = pytest.mark.live


def test_live_stock_bars(cfg):
    bars = DnseClient(cfg.dnse).fetch_daily("VCB", "stock", date(2025, 1, 2), date(2025, 1, 15))
    assert 5 <= len(bars) <= 10
    assert bars[0].trade_date == date(2025, 1, 2)
    assert all(b.trade_date.weekday() < 5 for b in bars)
    assert all(10_000 < b.close < 300_000 for b in bars)          # VND, not thousand VND
    assert all(b.low <= min(b.open, b.close) and b.high >= max(b.open, b.close) for b in bars)


def test_live_index_and_invalid_symbol(cfg):
    (first, *_), = [DnseClient(cfg.dnse).fetch_daily("VNINDEX", "index", date(2025, 1, 2), date(2025, 1, 3))]
    assert 800 < first.close < 3000                               # points
    with pytest.raises(DnseInvalidSymbol):
        DnseClient(cfg.dnse).fetch_daily("ZZZZ", "stock", date(2025, 1, 2), date(2025, 1, 3))
