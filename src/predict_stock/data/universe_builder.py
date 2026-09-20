"""Build a fixed "large & liquid" universe from a candidate list.

DNSE's public endpoint exposes no market cap or share count, so we cannot rank by
market capitalisation. Instead candidates are ranked by the MEDIAN DAILY TRADED
VALUE (close x volume, VND) over the last ``window`` sessions, a liquidity proxy
for size. The candidate list itself is supplied by a human and is NOT verified data.

Known bias: ranking uses *today's* liquidity but the resulting membership is applied
back to 2018, so backtests on it are optimistic (survivorship / look-ahead in the
universe choice).
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from predict_stock.data.dnse_client import DailyBar, DnseClient, DnseError, DnseInvalidSymbol

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    symbol: str
    first_date: date
    last_date: date
    n_bars: int
    median_value_vnd: Decimal  # median of close*volume over the last `window` bars


def read_candidates(path: str | Path) -> list[str]:
    seen: dict[str, None] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip().upper()
        if s:
            seen.setdefault(s)
    return list(seen)


def summarize(symbol: str, bars: list[DailyBar], window: int) -> Candidate | None:
    """None when there are fewer than ``window`` bars (not enough history to rank)."""
    if len(bars) < window:
        return None
    values = [b.close * b.volume for b in bars[-window:]]
    return Candidate(symbol, bars[0].trade_date, bars[-1].trade_date, len(bars), statistics.median(values))


def rank(cands: list[Candidate]) -> list[Candidate]:
    return sorted(cands, key=lambda c: (-c.median_value_vnd, c.symbol))


def build(
    client: DnseClient,
    symbols: list[str],
    *,
    start: date,
    end: date,
    window: int,
    top_n: int,
    max_stale_days: int = 10,
) -> tuple[list[Candidate], list[Candidate], dict[str, str]]:
    """Returns (selected, all_ranked, excluded {symbol: reason})."""
    ranked_in: list[Candidate] = []
    excluded: dict[str, str] = {}
    for sym in symbols:
        try:
            bars = client.fetch_daily(sym, "stock", start, end)
        except DnseInvalidSymbol:
            excluded[sym] = "invalid symbol (unknown to DNSE)"
            continue
        except DnseError as exc:
            excluded[sym] = f"fetch failed: {exc}"
            continue
        c = summarize(sym, bars, window)
        if c is None:
            excluded[sym] = f"only {len(bars)} bars (< window {window})"
        elif (end - c.last_date) > timedelta(days=max_stale_days):
            excluded[sym] = f"stale: last bar {c.last_date} (delisted or suspended?)"
        else:
            ranked_in.append(c)
    ranked = rank(ranked_in)
    return ranked[:top_n], ranked, excluded


def write_snapshot_csv(path: str | Path, selected: list[Candidate], start: date, note: str) -> None:
    """Snapshot for `universe apply`. ``valid_from`` = max(start, first bar): a backfill
    that only applies when seeding an empty universe, so no symbol is a member before
    DNSE has any data for it."""
    lines = [f"# {note}", "symbol,valid_from"]
    for c in sorted(selected, key=lambda c: c.symbol):
        lines.append(f"{c.symbol},{max(start, c.first_date).isoformat()}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
