from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
import responses
from responses import matchers

from predict_stock.data.dnse_client import DnseClient, DnseError, DnseInvalidSymbol

URL = "https://api.dnse.com.vn/chart-api/v2/ohlcs/stock"
IDX = "https://api.dnse.com.vn/chart-api/v2/ohlcs/index"


def ts(y, m, d, h=2):  # h is the UTC hour: 02:00 UTC == 09:00 ICT
    return int(datetime(y, m, d, h, tzinfo=timezone.utc).timestamp())


def client(cfg, sleeps=None):
    sleeps = sleeps if sleeps is not None else []
    return DnseClient(cfg.dnse, sleep=sleeps.append, monotonic=lambda: 0.0), sleeps


def payload(rows):
    return {
        "t": [r[0] for r in rows], "o": [r[1] for r in rows], "h": [r[2] for r in rows],
        "l": [r[3] for r in rows], "c": [r[4] for r in rows], "v": [r[5] for r in rows], "nextTime": 0,
    }


@responses.activate
def test_stock_prices_converted_exactly_to_vnd(cfg):
    responses.get(URL, json=payload([(ts(2025, 1, 2), 60.33, 60.91, 60.26, 60.52, 1630500)]))
    (bar,) = client(cfg)[0].fetch_daily("VCB", "stock", date(2025, 1, 1), date(2025, 1, 31))
    assert (bar.open, bar.high, bar.low, bar.close) == (Decimal("60330"), Decimal("60910"), Decimal("60260"), Decimal("60520"))
    assert bar.trade_date == date(2025, 1, 2) and bar.volume == 1630500


@responses.activate
def test_index_values_are_not_scaled(cfg):
    responses.get(IDX, json=payload([(ts(2025, 1, 2), 1200.55, 1210.1, 1195.0, 1205.3, 500_000_000)]))
    (bar,) = client(cfg)[0].fetch_daily("VNINDEX", "index", date(2025, 1, 1), date(2025, 1, 31))
    assert bar.close == Decimal("1205.3")


@responses.activate
def test_bar_date_uses_vietnam_timezone(cfg):
    # 17:30 UTC on Jan 1 is already Jan 2 in Vietnam (UTC+7)
    responses.get(URL, json=payload([(ts(2025, 1, 1, 17), 1, 1, 1, 1, 1)]))
    (bar,) = client(cfg)[0].fetch_daily("X", "stock", date(2025, 1, 1), date(2025, 1, 5))
    assert bar.trade_date == date(2025, 1, 2)


@responses.activate
def test_split_session_rows_are_merged_and_reported(cfg):
    """Real rows returned by DNSE for VCB on 2022-12-27 (two rows, same date)."""
    responses.get(URL, json=payload([
        (ts(2022, 12, 26), 44.16, 45.06, 44, 44, 1203600),
        (ts(2022, 12, 27, 0), 44.72, 44.72, 44.27, 44.55, 136100),   # 00:00 UTC fragment
        (ts(2022, 12, 27, 2), 44.55, 44.89, 44, 44.11, 1058000),
    ]))
    c, _ = client(cfg)
    bars = c.fetch_daily("VCB", "stock", date(2022, 12, 20), date(2022, 12, 30))
    assert [b.trade_date for b in bars] == [date(2022, 12, 26), date(2022, 12, 27)]
    m = bars[1]
    assert (m.open, m.high, m.low, m.close, m.volume) == (
        Decimal("44720"), Decimal("44890"), Decimal("44000"), Decimal("44110"), 1194100)
    assert any("merged 2 same-date rows for 2022-12-27" in w for w in c.warnings)


@responses.activate
def test_unsorted_response_is_sorted(cfg):
    responses.get(URL, json=payload([(ts(2025, 1, 3), 2, 2, 2, 2, 2), (ts(2025, 1, 2), 1, 1, 1, 1, 1)]))
    bars = client(cfg)[0].fetch_daily("X", "stock", date(2025, 1, 1), date(2025, 1, 5))
    assert [b.trade_date.day for b in bars] == [2, 3]


@responses.activate
def test_request_range_is_inclusive_in_vietnam_time(cfg):
    responses.get(URL, json={}, match=[matchers.query_param_matcher({
        "symbol": "VCB", "resolution": "1D",
        "from": str(ts(2025, 1, 1, 17) - 0),   # 2025-01-02 00:00 ICT == 2025-01-01 17:00 UTC
        "to": str(ts(2025, 1, 10, 16) + 3599), # 2025-01-10 23:59:59 ICT
    })])
    assert client(cfg)[0].fetch_daily("VCB", "stock", date(2025, 1, 2), date(2025, 1, 10)) == []


@responses.activate
def test_empty_payload_returns_no_bars(cfg):
    responses.get(URL, json={"t": [], "o": [], "h": [], "l": [], "c": [], "v": []})
    assert client(cfg)[0].fetch_daily("X", "stock", date(2025, 1, 1), date(2025, 1, 5)) == []


@responses.activate
def test_invalid_symbol_raises(cfg):
    responses.get(URL, status=400, json={"status": 400, "code": "BAD_REQUEST", "message": "invalid symbol"})
    with pytest.raises(DnseInvalidSymbol):
        client(cfg)[0].fetch_daily("ZZZZ", "stock", date(2025, 1, 1), date(2025, 1, 5))
    assert len(responses.calls) == 1  # not retried


@responses.activate
def test_retries_5xx_with_backoff_then_succeeds(cfg):
    responses.get(URL, status=503)
    responses.get(URL, status=429)
    responses.get(URL, json=payload([(ts(2025, 1, 2), 1, 1, 1, 1, 1)]))
    c, sleeps = client(cfg)
    assert len(c.fetch_daily("X", "stock", date(2025, 1, 1), date(2025, 1, 5))) == 1
    backoffs = [s for s in sleeps if s >= cfg.dnse.backoff_base_s]
    assert backoffs == [cfg.dnse.backoff_base_s, cfg.dnse.backoff_base_s * 2]


@responses.activate
def test_gives_up_after_max_retries(cfg):
    responses.get(URL, status=500)
    with pytest.raises(DnseError, match="gave up"):
        client(cfg)[0].fetch_daily("X", "stock", date(2025, 1, 1), date(2025, 1, 5))
    assert len(responses.calls) == cfg.dnse.max_retries + 1


@responses.activate
def test_non_retryable_4xx_raises_immediately(cfg):
    responses.get(URL, status=404, body="nope")
    with pytest.raises(DnseError, match="HTTP 404"):
        client(cfg)[0].fetch_daily("X", "stock", date(2025, 1, 1), date(2025, 1, 5))
    assert len(responses.calls) == 1


@responses.activate
def test_malformed_payloads_are_rejected(cfg):
    responses.get(URL, json={"t": [ts(2025, 1, 2)], "o": [1], "h": [1]})
    with pytest.raises(DnseError, match="missing fields"):
        client(cfg)[0].fetch_daily("X", "stock", date(2025, 1, 1), date(2025, 1, 5))
    responses.replace(responses.GET, URL, json={**payload([(ts(2025, 1, 2), 1, 1, 1, 1, 1)]), "v": []})
    with pytest.raises(DnseError, match="different lengths"):
        client(cfg)[0].fetch_daily("X", "stock", date(2025, 1, 1), date(2025, 1, 5))


@responses.activate
def test_next_time_is_surfaced_not_ignored(cfg):
    responses.get(URL, json={**payload([(ts(2025, 1, 2), 1, 1, 1, 1, 1)]), "nextTime": 1735000000})
    c, _ = client(cfg)
    c.fetch_daily("X", "stock", date(2025, 1, 1), date(2025, 1, 5))
    assert any("nextTime" in w for w in c.warnings)


def test_requests_are_throttled(cfg):
    now = [0.0]
    sleeps: list[float] = []
    c = DnseClient(cfg.dnse, sleep=sleeps.append, monotonic=lambda: now[0])
    with responses.RequestsMock() as rs:
        rs.get(URL, json={})
        rs.get(URL, json={})
        c.fetch_daily("X", "stock", date(2025, 1, 1), date(2025, 1, 5))
        now[0] = 0.05  # only 50 ms since the previous request
        c.fetch_daily("X", "stock", date(2025, 1, 1), date(2025, 1, 5))
    assert sleeps == [pytest.approx(cfg.dnse.min_interval_s - 0.05)]


def test_bad_range_rejected(cfg):
    with pytest.raises(ValueError):
        client(cfg)[0].fetch_daily("X", "stock", date(2025, 2, 1), date(2025, 1, 1))


# ---- intraday (1H) -------------------------------------------------------------------------------------
def test_intraday_bars_are_utc_scaled_and_sorted(cfg):
    day = datetime(2026, 9, 18)
    ts = lambda h: int(datetime(2026, 9, 18, h, tzinfo=timezone.utc).timestamp())        # 02:00 UTC = 09:00 ICT
    with responses.RequestsMock() as rs:
        rs.get(URL, json=payload([(ts(3), 60.4, 60.9, 60.3, 60.5, 200), (ts(2), 60.3, 60.6, 60.2, 60.4, 100)]),
               match=[matchers.query_param_matcher({"symbol": "VCB", "resolution": "1H",
                                                     "from": str(int(datetime(2026, 9, 17, 17, tzinfo=timezone.utc).timestamp())),
                                                     "to": str(int(datetime(2026, 9, 18, 16, 59, 59, tzinfo=timezone.utc).timestamp()))})])
        bars = client(cfg)[0].fetch_intraday("VCB", "stock", date(2026, 9, 18), date(2026, 9, 18))
    assert [b.bar_time for b in bars] == [datetime(2026, 9, 18, 2), datetime(2026, 9, 18, 3)]        # naive UTC, ascending
    assert (bars[0].open, bars[0].close, bars[0].volume) == (Decimal("60300"), Decimal("60400"), 100)


@responses.activate
def test_intraday_repeated_timestamp_keeps_last_and_warns(cfg):
    t = int(datetime(2026, 9, 18, 2, tzinfo=timezone.utc).timestamp())
    responses.get(URL, json=payload([(t, 1, 1, 1, 1, 1), (t, 2, 2, 2, 2, 2)]))
    c, _ = client(cfg)
    (bar,) = c.fetch_intraday("X", "stock", date(2026, 9, 18), date(2026, 9, 18))
    assert bar.close == Decimal("2000") and any("repeated 1H timestamp" in w for w in c.warnings)


@responses.activate
def test_intraday_drops_bars_outside_the_requested_days_and_handles_empty(cfg):
    inside = int(datetime(2026, 9, 18, 2, tzinfo=timezone.utc).timestamp())
    outside = int(datetime(2026, 9, 25, 2, tzinfo=timezone.utc).timestamp())
    responses.get(URL, json=payload([(inside, 1, 1, 1, 1, 1), (outside, 1, 1, 1, 1, 1)]))
    assert len(client(cfg)[0].fetch_intraday("X", "stock", date(2026, 9, 18), date(2026, 9, 18))) == 1
    responses.replace(responses.GET, URL, json={"t": []})
    assert client(cfg)[0].fetch_intraday("X", "stock", date(2026, 9, 18), date(2026, 9, 18)) == []


@responses.activate
def test_intraday_malformed_and_invalid_symbol(cfg):
    responses.get(URL, json={"t": [1], "o": [1]})
    with pytest.raises(DnseError, match="malformed"):
        client(cfg)[0].fetch_intraday("X", "stock", date(2026, 9, 18), date(2026, 9, 18))
    responses.replace(responses.GET, URL, status=400, json={"message": "invalid symbol"})
    with pytest.raises(DnseInvalidSymbol):
        client(cfg)[0].fetch_intraday("ZZ", "stock", date(2026, 9, 18), date(2026, 9, 18))
    with pytest.raises(ValueError):
        client(cfg)[0].fetch_intraday("X", "stock", date(2026, 9, 19), date(2026, 9, 18))
