"""Idempotent daily-bar ingestion.

* Only rows that are new or whose values changed are written, so re-running a
  job on unchanged data touches nothing.
* DNSE serves back-adjusted prices, so appending only new bars could splice
  pre- and post-adjustment prices. Incremental runs therefore re-fetch an
  overlap window; if any stored close disagrees with the fresh one (beyond
  ``adjust_rel_tolerance``) the whole symbol is re-fetched and rewritten.
* Today's bar is accepted only after close + buffer (no partial bars).
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import Engine, delete, func, select, text
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.orm import Session

from predict_stock.config import AppConfig
from predict_stock.data.dnse_client import SOURCE, DailyBar, DnseClient, DnseError, Kind
from predict_stock.db.models import OhlcvDaily, TradingDay
from predict_stock.db.repo import ensure_instruments
from predict_stock.db.session import session_scope
from predict_stock.runs import tracked_run
from predict_stock.universe import get_symbols_between

log = logging.getLogger(__name__)
_CHUNK = 1000


@dataclass
class SymbolResult:
    symbol: str
    kind: str
    mode: str = "noop"  # noop | full | incremental | full-after-drift
    fetched: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    drift: bool = False
    first_date: str | None = None
    last_date: str | None = None


def is_final(bar_date: date, now_local: datetime, cfg: AppConfig) -> bool:
    """A bar is final once its session is over (plus a buffer for today's)."""
    today = now_local.date()
    if bar_date < today:
        return True
    if bar_date > today:
        return False
    cutoff = datetime.combine(today, cfg.ingest.close_time(), tzinfo=now_local.tzinfo) + timedelta(
        minutes=cfg.ingest.final_bar_buffer_minutes
    )
    return now_local >= cutoff


def _differs(row: OhlcvDaily, bar: DailyBar) -> bool:
    return (
        Decimal(row.open) != bar.open
        or Decimal(row.high) != bar.high
        or Decimal(row.low) != bar.low
        or Decimal(row.close) != bar.close
        or row.volume != bar.volume
    )


def _drifted(existing: dict[date, OhlcvDaily], bars: list[DailyBar], tol: float) -> bool:
    for b in bars:
        row = existing.get(b.trade_date)
        if row is None:
            continue
        old = Decimal(row.close)
        if old == 0 or abs(b.close - old) / old > Decimal(str(tol)):
            return True
    return False


def _existing(session: Session, symbol: str, since: date | None) -> dict[date, OhlcvDaily]:
    stmt = select(OhlcvDaily).where(OhlcvDaily.symbol == symbol)
    if since is not None:
        stmt = stmt.where(OhlcvDaily.trade_date >= since)
    return {r.trade_date: r for r in session.scalars(stmt)}


def ingest_symbol(
    session: Session,
    client: DnseClient,
    symbol: str,
    kind: Kind,
    start: date,
    end: date,
    cfg: AppConfig,
    run_id: int | None,
    now: datetime,
    *,
    refetch: bool = False,
) -> SymbolResult:
    now_local = now.astimezone(ZoneInfo(cfg.dnse.bar_timezone))
    end = min(end, now_local.date())
    res = SymbolResult(symbol=symbol, kind=kind)
    ensure_instruments(session, [symbol], kind)

    stored_last = session.scalar(select(func.max(OhlcvDaily.trade_date)).where(OhlcvDaily.symbol == symbol))
    if refetch or stored_last is None:
        res.mode, fetch_from = "full", start
        existing = _existing(session, symbol, None)
    else:
        res.mode = "incremental"
        fetch_from = max(start, stored_last - timedelta(days=cfg.ingest.overlap_calendar_days))
        existing = _existing(session, symbol, fetch_from)

    def fetch(frm: date) -> list[DailyBar]:
        return [b for b in client.fetch_daily(symbol, kind, frm, end) if is_final(b.trade_date, now_local, cfg)]

    bars = fetch(fetch_from)
    if res.mode == "incremental" and _drifted(existing, bars, cfg.ingest.adjust_rel_tolerance):
        log.warning("%s: stored closes disagree with DNSE (re-adjustment?) -> full re-fetch", symbol)
        res.drift, res.mode = True, "full-after-drift"
        bars = fetch(start)
        existing = _existing(session, symbol, None)

    res.fetched = len(bars)
    to_write = []
    for b in bars:
        row = existing.get(b.trade_date)
        if row is None:
            res.inserted += 1
        elif _differs(row, b):
            res.updated += 1
        else:
            res.unchanged += 1
            continue
        to_write.append(
            {
                "symbol": symbol,
                "trade_date": b.trade_date,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "source": SOURCE,
                "fetched_at": now.astimezone(timezone.utc).replace(tzinfo=None),
                "run_id": run_id,
            }
        )
    for i in range(0, len(to_write), _CHUNK):
        stmt = mysql_insert(OhlcvDaily).values(to_write[i : i + _CHUNK])
        session.execute(
            stmt.on_duplicate_key_update(
                **{c: stmt.inserted[c] for c in ("open", "high", "low", "close", "volume", "source", "fetched_at", "run_id")}
            )
        )
    if bars:
        res.first_date, res.last_date = bars[0].trade_date.isoformat(), bars[-1].trade_date.isoformat()
    return res


CALENDAR_SOURCE = "stock-consensus"


def compute_trading_days(bars: pd.DataFrame, min_fraction: float) -> set[date]:
    """``bars``: columns (symbol, trade_date) for stocks only. A date is a trading day
    when at least ``min_fraction`` of the *active* stocks (first <= date <= last stored
    bar) have a bar on it."""
    if bars.empty:
        return set()
    span = bars.groupby("symbol")["trade_date"].agg(["min", "max"])
    present = bars.groupby("trade_date")["symbol"].nunique()
    days = set()
    for d, n in present.items():
        active = int(((span["min"] <= d) & (span["max"] >= d)).sum())
        if n >= max(1.0, min_fraction * active):
            days.add(d)
    return days


def sync_trading_days(session: Session, min_fraction: float) -> dict[str, int]:
    """Recompute the calendar from stock bars and make ``trading_days`` match it exactly
    (it is a derived table). Returns {"added": n, "removed": m, "total": t}."""
    rows = session.execute(
        text(
            "SELECT o.symbol, o.trade_date FROM ohlcv_daily o "
            "JOIN instruments i ON i.symbol = o.symbol WHERE i.kind = 'stock'"
        )
    ).all()
    want = compute_trading_days(pd.DataFrame(rows, columns=["symbol", "trade_date"]), min_fraction)
    have = set(session.scalars(select(TradingDay.trade_date)))
    stale = have - want
    for i in range(0, len(stale), _CHUNK):
        session.execute(delete(TradingDay).where(TradingDay.trade_date.in_(list(stale)[i : i + _CHUNK])))
    fresh = sorted(want - have)
    for i in range(0, len(fresh), _CHUNK):
        session.execute(
            mysql_insert(TradingDay).values([{"trade_date": d, "source": CALENDAR_SOURCE} for d in fresh[i : i + _CHUNK]])
        )
    return {"added": len(fresh), "removed": len(stale), "total": len(want)}


def ingest_universe(
    engine: Engine,
    client: DnseClient,
    cfg: AppConfig,
    universe_code: str,
    start: date,
    end: date,
    *,
    refetch: bool = False,
    now: datetime | None = None,
) -> dict:
    """Ingest every symbol that was a member of ``universe_code`` during [start, end]
    plus the configured benchmark indices. One transaction per symbol."""
    now = now or datetime.now(timezone.utc)
    params = {"universe": universe_code, "start": start.isoformat(), "end": end.isoformat(), "refetch": refetch}
    with tracked_run(engine, "ingest_ohlcv", cfg, params) as (run_id, stats):
        with session_scope(engine) as s:
            members = get_symbols_between(s, universe_code, start, end)
        if not members:
            raise DnseError(f"universe {universe_code!r} has no members in {start}..{end}; load membership first")
        targets: list[tuple[str, Kind]] = [(m, "stock") for m in members]
        targets += [(b, "index") for b in cfg.ingest.benchmark_symbols if b not in members]

        results, failures = [], {}
        for symbol, kind in targets:
            try:
                with session_scope(engine) as s:
                    r = ingest_symbol(s, client, symbol, kind, start, end, cfg, run_id, now, refetch=refetch)
                results.append(asdict(r))
                log.info("%s [%s] %s: +%d ~%d =%d", symbol, kind, r.mode, r.inserted, r.updated, r.unchanged)
            except DnseError as exc:
                failures[symbol] = f"{type(exc).__name__}: {exc}"
                log.error("%s failed: %s", symbol, exc)

        with session_scope(engine) as s:
            stats["trading_days"] = sync_trading_days(s, cfg.ingest.calendar_min_stock_fraction)
        stats.update(
            symbols=results,
            failures=failures,
            totals={k: sum(r[k] for r in results) for k in ("fetched", "inserted", "updated", "unchanged")},
            drifted=[r["symbol"] for r in results if r["drift"]],
            client_warnings=client.warnings,
        )
        if failures:
            raise DnseError(f"{len(failures)} symbol(s) failed: {failures}")
    return stats
