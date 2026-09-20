"""Idempotent daily-bar ingestion into ``price_bar``.

* Only rows that are new or whose values changed are written, so re-running a job on
  unchanged data touches nothing.
* DNSE serves back-adjusted prices, so appending only new bars could splice pre- and
  post-adjustment prices. Incremental runs therefore re-fetch an overlap window; if any
  stored close disagrees with the fresh one (beyond ``adjust_rel_tolerance``) the whole
  instrument is re-fetched and rewritten. Every overwritten row is first copied to
  ``price_bar_revisions``, so earlier versions of the series are never lost.
* Today's bar is accepted only after close + buffer (no partial bars).
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import Engine, delete, func, select, text
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.orm import Session

from predict_stock.config import AppConfig
from predict_stock.data.dnse_client import SOURCE, DailyBar, DnseClient, DnseError, IntradayBar, Kind
from predict_stock.db.models import DataIngestRun, PriceBar, PriceBarIntraday, PriceBarRevision, TradingCalendar
from predict_stock.db.repo import get_or_create_instrument
from predict_stock.db.session import session_scope
from predict_stock.runs import tracked_run
from predict_stock.universe import Member, get_instruments_between

log = logging.getLogger(__name__)
_CHUNK = 1000
_UPDATE_COLS = ("open", "high", "low", "close", "volume", "price_basis", "source", "fetched_at", "run_id")


@dataclass(frozen=True)
class Target:
    instrument_id: int
    symbol: str
    kind: Kind


@dataclass
class SymbolResult:
    instrument_id: int
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
    resolution: str = "1D"


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


def _differs(row: PriceBar, bar: DailyBar) -> bool:
    return (
        Decimal(row.open) != bar.open
        or Decimal(row.high) != bar.high
        or Decimal(row.low) != bar.low
        or Decimal(row.close) != bar.close
        or row.volume != bar.volume
    )


def _drifted(existing: dict[date, PriceBar], bars: list[DailyBar], tol: float) -> bool:
    for b in bars:
        row = existing.get(b.trade_date)
        if row is None:
            continue
        old = Decimal(row.close)
        if old == 0 or abs(b.close - old) / old > Decimal(str(tol)):
            return True
    return False


def _existing(session: Session, instrument_id: int, since: date | None) -> dict[date, PriceBar]:
    stmt = select(PriceBar).where(PriceBar.instrument_id == instrument_id)
    if since is not None:
        stmt = stmt.where(PriceBar.trade_date >= since)
    return {r.trade_date: r for r in session.scalars(stmt)}


def ingest_symbol(
    session: Session,
    client: DnseClient,
    target: Target,
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
    iid, symbol, kind = target.instrument_id, target.symbol, target.kind
    res = SymbolResult(instrument_id=iid, symbol=symbol, kind=kind)

    stored_last = session.scalar(select(func.max(PriceBar.trade_date)).where(PriceBar.instrument_id == iid))
    if refetch or stored_last is None:
        res.mode, fetch_from = "full", start
        existing = _existing(session, iid, None)
    else:
        res.mode = "incremental"
        fetch_from = max(start, stored_last - timedelta(days=cfg.ingest.overlap_calendar_days))
        existing = _existing(session, iid, fetch_from)

    def fetch(frm: date) -> list[DailyBar]:
        return [b for b in client.fetch_daily(symbol, kind, frm, end) if is_final(b.trade_date, now_local, cfg)]

    bars = fetch(fetch_from)
    if res.mode == "incremental" and _drifted(existing, bars, cfg.ingest.adjust_rel_tolerance):
        log.warning("%s: stored closes disagree with DNSE (re-adjustment?) -> full re-fetch", symbol)
        res.drift, res.mode = True, "full-after-drift"
        bars = fetch(start)
        existing = _existing(session, iid, None)

    basis = session.scalar(
        select(PriceBar.price_basis).where(PriceBar.instrument_id == iid).order_by(PriceBar.trade_date.desc()).limit(1)
    ) or "vendor_adjusted"  # an instrument an operator marked 'raw' keeps receiving raw bars
    res.fetched = len(bars)
    to_write, revisions = [], []
    fetched_at = now.astimezone(timezone.utc).replace(tzinfo=None)
    for b in bars:
        row = existing.get(b.trade_date)
        if row is None:
            res.inserted += 1
        elif _differs(row, b):
            res.updated += 1
            revisions.append({
                "instrument_id": iid, "trade_date": row.trade_date, "open": row.open, "high": row.high, "low": row.low,
                "close": row.close, "volume": row.volume, "price_basis": row.price_basis, "source": row.source,
                "fetched_at": row.fetched_at, "superseded_by_run_id": run_id,
            })
        else:
            res.unchanged += 1
            continue
        to_write.append({
            "instrument_id": iid, "trade_date": b.trade_date, "open": b.open, "high": b.high, "low": b.low,
            "close": b.close, "volume": b.volume, "price_basis": basis, "source": SOURCE,
            "fetched_at": fetched_at, "run_id": run_id,
        })
    for i in range(0, len(revisions), _CHUNK):  # keep the old values BEFORE overwriting them
        session.execute(PriceBarRevision.__table__.insert().values(revisions[i : i + _CHUNK]))
    for i in range(0, len(to_write), _CHUNK):
        stmt = mysql_insert(PriceBar).values(to_write[i : i + _CHUNK])
        session.execute(stmt.on_duplicate_key_update(**{c: stmt.inserted[c] for c in _UPDATE_COLS}))
    if bars:
        res.first_date, res.last_date = bars[0].trade_date.isoformat(), bars[-1].trade_date.isoformat()
    return res


def ingest_intraday_symbol(
    session: Session, client: DnseClient, target: Target, end: date, cfg: AppConfig, run_id: int | None, now: datetime,
) -> SymbolResult:
    """Optional 1H bars (config ingest.intraday.enabled). Incremental with an overlap window; only new or
    changed bars are written. Today's bars are dropped until the session is final. No revision history."""
    icfg = cfg.ingest.intraday
    iid = target.instrument_id
    now_local = now.astimezone(ZoneInfo(cfg.dnse.bar_timezone))
    end = min(end, now_local.date())
    tz = ZoneInfo(cfg.dnse.bar_timezone)
    res = SymbolResult(instrument_id=iid, symbol=target.symbol, kind=target.kind, resolution=icfg.resolution)
    last = session.scalar(select(func.max(PriceBarIntraday.bar_time)).where(
        PriceBarIntraday.instrument_id == iid, PriceBarIntraday.resolution == icfg.resolution))
    floor = date.fromisoformat(icfg.history_start)
    if last is None:
        res.mode, start = "full", floor
    else:
        res.mode = "incremental"
        last_day = last.replace(tzinfo=timezone.utc).astimezone(tz).date()
        start = max(floor, last_day - timedelta(days=icfg.overlap_calendar_days))
    bars = [b for b in client.fetch_intraday(target.symbol, target.kind, start, end, icfg.resolution)
            if is_final(b.bar_time.replace(tzinfo=timezone.utc).astimezone(tz).date(), now_local, cfg)]
    existing = {
        r.bar_time: r for r in session.scalars(select(PriceBarIntraday).where(
            PriceBarIntraday.instrument_id == iid, PriceBarIntraday.resolution == icfg.resolution,
            PriceBarIntraday.bar_time >= datetime.combine(start, datetime.min.time(), tzinfo=tz).astimezone(timezone.utc).replace(tzinfo=None)))
    }
    fetched_at = now.astimezone(timezone.utc).replace(tzinfo=None)
    to_write = []
    res.fetched = len(bars)
    for b in bars:
        row = existing.get(b.bar_time)
        if row is None:
            res.inserted += 1
        elif (Decimal(row.open), Decimal(row.high), Decimal(row.low), Decimal(row.close), row.volume) != (b.open, b.high, b.low, b.close, b.volume):
            res.updated += 1
        else:
            res.unchanged += 1
            continue
        to_write.append({"instrument_id": iid, "resolution": icfg.resolution, "bar_time": b.bar_time, "open": b.open,
                         "high": b.high, "low": b.low, "close": b.close, "volume": b.volume,
                         "price_basis": "vendor_adjusted", "source": SOURCE, "fetched_at": fetched_at, "run_id": run_id})
    for i in range(0, len(to_write), _CHUNK):
        stmt = mysql_insert(PriceBarIntraday).values(to_write[i : i + _CHUNK])
        session.execute(stmt.on_duplicate_key_update(**{c: stmt.inserted[c] for c in _UPDATE_COLS}))
    if bars:
        res.first_date, res.last_date = bars[0].bar_time.date().isoformat(), bars[-1].bar_time.date().isoformat()
    return res


def compute_trading_days(bars: pd.DataFrame, min_fraction: float, grace_days: int = 30) -> set[date]:
    """``bars``: columns (symbol, trade_date), stocks only; ``symbol`` is any per-instrument key.
    A date is a trading day when at least ``min_fraction`` of the *active* stocks have a bar on it. A stock is
    active from its first bar until ``grace_days`` after its last bar, so late or suspended stocks still count
    in the denominator while a lone stray bar (a stock whose data runs ahead of the others) cannot create a day."""
    if bars.empty:
        return set()
    span = bars.groupby("symbol")["trade_date"].agg(["min", "max"])
    first = pd.to_datetime(span["min"]).to_numpy()
    last = (pd.to_datetime(span["max"]) + pd.Timedelta(days=grace_days)).to_numpy()
    present = bars.groupby("trade_date")["symbol"].nunique()
    days = set()
    for d, n in present.items():
        ts = np.datetime64(pd.Timestamp(d))
        active = int(((first <= ts) & (last >= ts)).sum())
        if n >= max(1.0, min_fraction * active):
            days.add(d)
    return days


def sync_trading_days(session: Session, calendar_code: str, min_fraction: float, grace_days: int = 30) -> dict[str, int]:
    """Recompute the calendar from stock bars and make ``trading_calendar`` match it exactly
    (it is a derived table). Returns {"added": n, "removed": m, "total": t}."""
    rows = session.execute(
        text(
            "SELECT b.instrument_id, b.trade_date FROM price_bar b "
            "JOIN instruments i ON i.id = b.instrument_id WHERE i.kind = 'stock'"
        )
    ).all()
    want = compute_trading_days(pd.DataFrame(rows, columns=["symbol", "trade_date"]), min_fraction, grace_days)
    have = set(session.scalars(select(TradingCalendar.trade_date).where(TradingCalendar.calendar_code == calendar_code)))
    stale = sorted(have - want)
    for i in range(0, len(stale), _CHUNK):
        session.execute(
            delete(TradingCalendar).where(
                TradingCalendar.calendar_code == calendar_code, TradingCalendar.trade_date.in_(stale[i : i + _CHUNK])
            )
        )
    fresh = sorted(want - have)
    for i in range(0, len(fresh), _CHUNK):
        session.execute(
            mysql_insert(TradingCalendar).values(
                [{"calendar_code": calendar_code, "trade_date": d, "source": "stock-consensus"} for d in fresh[i : i + _CHUNK]]
            )
        )
    return {"added": len(fresh), "removed": len(stale), "total": len(want)}


def resolve_targets(session: Session, cfg: AppConfig, universe_codes: list[str], start: date, end: date) -> list[Target]:
    """Every instrument that was a member of any of ``universe_codes`` during [start, end],
    plus the configured benchmark indices."""
    targets: dict[int, Target] = {}
    for code in universe_codes:
        for m in get_instruments_between(session, code, start, end):
            targets.setdefault(m.instrument_id, Target(m.instrument_id, m.symbol, "stock"))
    if not targets:
        raise DnseError(f"universes {universe_codes} have no members in {start}..{end}; run `universe apply` first")
    for sym in cfg.ingest.benchmark_symbols:
        iid = get_or_create_instrument(session, sym, "index", floor=cfg.universe.floor())
        targets.setdefault(iid, Target(iid, sym, "index"))
    return sorted(targets.values(), key=lambda t: t.symbol)


def _run_row(run_id: int, t: Target, r: SymbolResult | None, *, start: date, end: date, params: dict, ms: int,
             warnings: list[str], status: str = "ok", error: str | None = None, resolution: str = "1D") -> DataIngestRun:
    return DataIngestRun(
        run_id=run_id, instrument_id=t.instrument_id, source=SOURCE, resolution=resolution,
        mode=r.mode if r else "failed", status=status, error=error, params=params, requested_start=start, requested_end=end,
        fetched=r.fetched if r else 0, inserted=r.inserted if r else 0, updated=r.updated if r else 0,
        unchanged=r.unchanged if r else 0, drift=r.drift if r else False,
        first_date=date.fromisoformat(r.first_date) if r and r.first_date else None,
        last_date=date.fromisoformat(r.last_date) if r and r.last_date else None,
        duration_ms=ms, warnings=warnings or None)


def ingest_targets(
    engine: Engine,
    client: DnseClient,
    cfg: AppConfig,
    targets: list[Target],
    start: date,
    end: date,
    run_id: int,
    now: datetime,
    *,
    refetch: bool = False,
    intraday: bool | None = None,
) -> tuple[list[dict], dict[str, str]]:
    """Ingest each target in its own transaction; every outcome (ok or failed) gets a data_ingest_runs row
    with parameters and duration. Returns (results, failures)."""
    intraday = cfg.ingest.intraday.enabled if intraday is None else intraday
    end_eff = min(end, now.astimezone(ZoneInfo(cfg.dnse.bar_timezone)).date())
    params = {"start": start.isoformat(), "end": end.isoformat(), "refetch": refetch}
    results: list[dict] = []
    failures: dict[str, str] = {}
    for t in targets:
        jobs = [("1D", lambda s_, t=t: ingest_symbol(s_, client, t, start, end, cfg, run_id, now, refetch=refetch), params)]
        if intraday and t.kind == "stock":
            icfg = cfg.ingest.intraday
            jobs.append((icfg.resolution, lambda s_, t=t: ingest_intraday_symbol(s_, client, t, end, cfg, run_id, now),
                         {**params, "start": icfg.history_start, "resolution": icfg.resolution}))
        for resolution, fn, jparams in jobs:
            w0, t0 = len(client.warnings), time.monotonic()
            try:
                with session_scope(engine) as s:
                    r = fn(s)
                    ms = int((time.monotonic() - t0) * 1000)
                    s.add(_run_row(run_id, t, r, start=start, end=end_eff, params=jparams, ms=ms,
                                   warnings=client.warnings[w0:], resolution=resolution))
                results.append(asdict(r))
                log.info("%s [%s %s] %s: +%d ~%d =%d in %d ms", t.symbol, t.kind, resolution, r.mode, r.inserted, r.updated, r.unchanged, ms)
            except DnseError as exc:
                ms = int((time.monotonic() - t0) * 1000)
                key = t.symbol if resolution == "1D" else f"{t.symbol}@{resolution}"
                failures[key] = f"{type(exc).__name__}: {exc}"
                log.error("%s failed: %s", key, exc)
                with session_scope(engine) as s:
                    s.add(_run_row(run_id, t, None, start=start, end=end_eff, params=jparams, ms=ms,
                                   warnings=client.warnings[w0:], status="failed", error=failures[key], resolution=resolution))
    return results, failures


def ingest_universe(
    engine: Engine,
    client: DnseClient,
    cfg: AppConfig,
    universe_codes: list[str],
    start: date,
    end: date,
    *,
    refetch: bool = False,
    now: datetime | None = None,
    intraday: bool | None = None,
) -> dict:
    """Ingest [start, end] for every instrument in ``universe_codes`` (+ benchmarks).
    One transaction per instrument; failures are recorded and isolated."""
    now = now or datetime.now(timezone.utc)
    params = {"universes": universe_codes, "start": start.isoformat(), "end": end.isoformat(), "refetch": refetch,
              "intraday": cfg.ingest.intraday.enabled if intraday is None else intraday}
    with tracked_run(engine, "ingest_ohlcv", cfg, params) as (run_id, stats):
        with session_scope(engine) as s:
            targets = resolve_targets(s, cfg, universe_codes, start, end)
        results, failures = ingest_targets(engine, client, cfg, targets, start, end, run_id, now, refetch=refetch, intraday=intraday)
        with session_scope(engine) as s:
            stats["trading_days"] = sync_trading_days(s, cfg.ingest.calendar_code, cfg.ingest.calendar_min_stock_fraction, cfg.ingest.calendar_grace_days)
        daily = [r for r in results if r["resolution"] == "1D"]
        stats.update(
            symbols=results,
            failures=failures,
            totals={k: sum(r[k] for r in daily) for k in ("fetched", "inserted", "updated", "unchanged")},
            totals_intraday={k: sum(r[k] for r in results if r["resolution"] != "1D") for k in ("fetched", "inserted", "updated", "unchanged")},
            drifted=[r["symbol"] for r in daily if r["drift"]],
            client_warnings=client.warnings,
        )
        if failures:
            raise DnseError(f"{len(failures)} symbol(s) failed: {failures}")
    return stats
