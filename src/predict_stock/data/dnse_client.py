"""DNSE public market-data client (daily OHLCV).

Every assumption about the endpoint lives in this file (principle 1). They were
verified by probing on 2026-09-20 and are recorded in docs/DNSE_API_NOTES.md.
The endpoint is UNDOCUMENTED by DNSE, so treat all of these as assumptions:

* GET {base_url}/{stock|index}?symbol=&resolution=1D&from=<epoch s>&to=<epoch s>
  needs no authentication.
* Response: parallel arrays ``t,o,h,l,c,v`` (+ ``nextTime``). ``t`` is an epoch
  in seconds; a daily bar is stamped 09:00 Asia/Ho_Chi_Minh of its trade date.
* Stock prices are in thousand VND (converted to VND here, exactly, via Decimal);
  index values are points; ``v`` is a share count.
* Stock history appears back-adjusted for corporate actions (no unexplained
  gaps over 2018-2026 on 8 large caps). Adjustment can change past values at
  any time, which is why ingestion re-checks an overlap window.
* Invalid symbol -> HTTP 400 ``{"code": "BAD_REQUEST", "message": "invalid symbol"}``.
* Real rate limit unknown; we throttle client-side and retry 429/5xx with backoff.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timezone
from decimal import Decimal
from typing import Callable, Literal
from zoneinfo import ZoneInfo

import requests

from predict_stock.config import DnseConfig

log = logging.getLogger(__name__)

Kind = Literal["stock", "index"]
SOURCE = "dnse-chart-api-v2"
_RETRY_STATUS = {429, 500, 502, 503, 504}


class DnseError(RuntimeError):
    pass


class DnseInvalidSymbol(DnseError):
    pass


@dataclass(frozen=True)
class DailyBar:
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


class DnseClient:
    def __init__(
        self,
        cfg: DnseConfig,
        session: requests.Session | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self._session = session or requests.Session()
        self._session.headers["User-Agent"] = cfg.user_agent
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_request_at: float | None = None
        self._tz = ZoneInfo(cfg.bar_timezone)
        self.warnings: list[str] = []

    # ---- public -----------------------------------------------------------
    def fetch_daily(self, symbol: str, kind: Kind, start: date, end: date) -> list[DailyBar]:
        """Daily bars with start <= trade_date <= end (inclusive), ascending, one per date."""
        if end < start:
            raise ValueError(f"end {end} before start {start}")
        payload = self._get_json(
            f"{self.cfg.base_url}/{kind}",
            {
                "symbol": symbol,
                "resolution": self.cfg.resolution,
                "from": self._epoch(start, dtime(0, 0)),
                "to": self._epoch(end, dtime(23, 59, 59)),
            },
            symbol,
        )
        bars = self._parse(payload, symbol, kind)
        return [b for b in bars if start <= b.trade_date <= end]

    # ---- internals --------------------------------------------------------
    def _epoch(self, d: date, t: dtime) -> int:
        return int(datetime.combine(d, t, tzinfo=self._tz).timestamp())

    def _throttle(self) -> None:
        if self._last_request_at is not None:
            wait = self.cfg.min_interval_s - (self._monotonic() - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = self._monotonic()

    def _get_json(self, url: str, params: dict, symbol: str) -> dict:
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            self._throttle()
            try:
                resp = self._session.get(url, params=params, timeout=self.cfg.timeout_s)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_err = exc
            else:
                if resp.status_code == 200:
                    # parse_float=Decimal keeps prices exact (60.33 -> Decimal("60.33")).
                    return json.loads(resp.text, parse_float=Decimal)
                if resp.status_code == 400 and "invalid symbol" in resp.text.lower():
                    raise DnseInvalidSymbol(symbol)
                if resp.status_code not in _RETRY_STATUS:
                    raise DnseError(f"HTTP {resp.status_code} for {symbol}: {resp.text[:200]}")
                last_err = DnseError(f"HTTP {resp.status_code}")
            if attempt < self.cfg.max_retries:
                delay = self.cfg.backoff_base_s * (2**attempt)
                log.warning("DNSE %s attempt %d failed (%s); retry in %.1fs", symbol, attempt + 1, last_err, delay)
                self._sleep(delay)
        raise DnseError(f"gave up on {symbol} after {self.cfg.max_retries + 1} attempts: {last_err}")

    def _parse(self, payload: dict, symbol: str, kind: Kind) -> list[DailyBar]:
        keys = ("t", "o", "h", "l", "c", "v")
        if not payload or not payload.get("t"):
            return []  # no data in the requested range
        missing = [k for k in keys if k not in payload]
        if missing:
            raise DnseError(f"{symbol}: response missing fields {missing}")
        n = len(payload["t"])
        if any(len(payload[k]) != n for k in keys):
            raise DnseError(f"{symbol}: response arrays have different lengths")
        next_time = payload.get("nextTime")
        if next_time not in (None, 0):
            # Semantics of nextTime are undocumented; surface it rather than guess.
            msg = f"{symbol}: nextTime={next_time} (possible truncated response)"
            log.warning(msg)
            self.warnings.append(msg)

        mult = Decimal(self.cfg.price_multiplier_stock if kind == "stock" else self.cfg.price_multiplier_index)
        by_date: dict[date, list[DailyBar]] = {}
        for i in sorted(range(n), key=lambda k: payload["t"][k]):
            d = datetime.fromtimestamp(int(payload["t"][i]), tz=timezone.utc).astimezone(self._tz).date()
            by_date.setdefault(d, []).append(
                DailyBar(
                    trade_date=d,
                    open=Decimal(payload["o"][i]) * mult,
                    high=Decimal(payload["h"][i]) * mult,
                    low=Decimal(payload["l"][i]) * mult,
                    close=Decimal(payload["c"][i]) * mult,
                    volume=int(payload["v"][i]),
                )
            )
        bars = []
        for d, parts in sorted(by_date.items()):
            if len(parts) > 1:
                bars.append(self._merge_fragments(symbol, d, parts))
            else:
                bars.append(parts[0])
        return bars

    def _merge_fragments(self, symbol: str, d: date, parts: list[DailyBar]) -> DailyBar:
        """Observed quirk: on 2022-12-27 every stock came back as two rows for the same
        date (one stamped 00:00 UTC with small volume, one 02:00 UTC), where the second
        row's open equals the first row's close. We treat them as fragments of one
        session and combine them. This is an INFERENCE from the data, not confirmed
        with DNSE or the exchange, so every merge is reported in ``warnings``."""
        merged = DailyBar(
            trade_date=d,
            open=parts[0].open,
            high=max(p.high for p in parts),
            low=min(p.low for p in parts),
            close=parts[-1].close,
            volume=sum(p.volume for p in parts),
        )
        msg = f"{symbol}: merged {len(parts)} same-date rows for {d} (inferred split session)"
        log.warning(msg)
        self.warnings.append(msg)
        return merged
