"""Backfill of instruments that have no price history yet, and the minimum-history gate.

A new member of a training universe is detected by having no bars at all; its full history is
fetched (`ingest_targets`, mode 'full') and timed. An instrument with fewer sessions than
``ingest.min_sessions`` is *blocked*: its data is stored, but it is not ready for training
(`ready_members`), the run row says so with the reason, and an alert is raised. Readiness is
derived from the bars every time, so an instrument unblocks itself once enough sessions exist.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from predict_stock.config import AppConfig
from predict_stock.data.dnse_client import DnseClient
from predict_stock.data.dnse_client import DnseError
from predict_stock.data.ingest import Target, ingest_targets, resolve_targets, sync_trading_days
from predict_stock.db.models import Alert, DataIngestRun, PriceBar
from predict_stock.db.session import session_scope
from predict_stock.runs import tracked_run
from predict_stock.universe import Member, training_members

log = logging.getLogger(__name__)
ALERT_CATEGORY = "insufficient_history"


def sessions_available(session: Session, instrument_id: int, as_of: date) -> tuple[int, date | None]:
    """(number of daily sessions stored up to and including ``as_of``, first session date)."""
    n, first = session.execute(
        select(func.count(), func.min(PriceBar.trade_date)).where(PriceBar.instrument_id == instrument_id, PriceBar.trade_date <= as_of)
    ).one()
    return n, first


def block_reason(session: Session, cfg: AppConfig, target_id: int, symbol: str, as_of: date) -> str | None:
    n, first = sessions_available(session, target_id, as_of)
    if n >= cfg.ingest.min_sessions:
        return None
    since = f"first bar {first}" if first else "no bars"
    return f"{symbol}: only {n} sessions up to {as_of} ({since}); ingest.min_sessions is {cfg.ingest.min_sessions}"


def readiness(session: Session, cfg: AppConfig, members: list[Member], as_of: date) -> dict[str, str]:
    """{symbol: reason} for members that do not have enough history."""
    out = {}
    for m in members:
        reason = block_reason(session, cfg, m.instrument_id, m.symbol, as_of)
        if reason:
            out[m.symbol] = reason
    return out


def ready_members(session: Session, cfg: AppConfig, as_of: date) -> list[Member]:
    """Training members with at least ``min_sessions`` sessions up to ``as_of``."""
    members = training_members(session, cfg, as_of)
    blocked = readiness(session, cfg, members, as_of)
    return [m for m in members if m.symbol not in blocked]


def sync_readiness_alerts(session: Session, cfg: AppConfig, members: list[Member], as_of: date, run_id: int | None) -> dict[str, str]:
    """One open alert per blocked instrument (updated in place, never duplicated); alerts of instruments
    that became ready are acknowledged. Returns the blocked map."""
    blocked = readiness(session, cfg, members, as_of)
    by_symbol = {m.symbol: m.instrument_id for m in members}
    open_alerts = {a.instrument_id: a for a in session.scalars(
        select(Alert).where(Alert.category == ALERT_CATEGORY, Alert.acknowledged_at.is_(None)))}
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for sym, reason in blocked.items():
        iid = by_symbol[sym]
        if iid in open_alerts:
            if open_alerts[iid].message != reason:  # untouched when nothing changed
                open_alerts[iid].message, open_alerts[iid].run_id = reason, run_id
        else:
            session.add(Alert(severity="warn", category=ALERT_CATEGORY, message=reason, instrument_id=iid, run_id=run_id))
    ready_ids = {m.instrument_id for m in members if m.symbol not in blocked}
    for iid, alert in open_alerts.items():
        if iid in ready_ids:
            alert.acknowledged_at = now
    return blocked


def find_new_targets(session: Session, cfg: AppConfig, universe_codes: list[str], start: date, end: date) -> list[Target]:
    """Instruments of the universes (and benchmarks) that have no bars at all."""
    targets = resolve_targets(session, cfg, universe_codes, start, end)
    have = set(session.scalars(select(PriceBar.instrument_id).where(PriceBar.instrument_id.in_([t.instrument_id for t in targets])).distinct()))
    return [t for t in targets if t.instrument_id not in have]


def backfill(
    engine: Engine,
    client: DnseClient,
    cfg: AppConfig,
    universe_codes: list[str],
    start: date,
    end: date,
    *,
    now: datetime | None = None,
    as_of: date | None = None,
) -> dict:
    """Fetch the full history of every instrument that has none, time it, and block (with the reason) those
    that end up with fewer than ``min_sessions`` sessions. Returns a report; raises nothing for a blocked
    instrument (that is data, not an error) but re-raises fetch failures after recording them."""
    now = now or datetime.now(timezone.utc)
    as_of = as_of or end
    params = {"universes": universe_codes, "start": start.isoformat(), "end": end.isoformat(), "min_sessions": cfg.ingest.min_sessions}
    with tracked_run(engine, "backfill", cfg, params) as (run_id, stats):
        with session_scope(engine) as s:
            targets = find_new_targets(s, cfg, universe_codes, start, end)
        _, failures = ingest_targets(engine, client, cfg, targets, start, end, run_id, now, intraday=False)

        blocked: dict[str, str] = {}
        with session_scope(engine) as s:
            for t in targets:
                if t.kind != "stock" or t.symbol in {k.split("@")[0] for k in failures}:
                    continue
                reason = block_reason(s, cfg, t.instrument_id, t.symbol, as_of)
                if reason:
                    blocked[t.symbol] = reason
                    row = s.scalars(select(DataIngestRun).where(DataIngestRun.run_id == run_id, DataIngestRun.instrument_id == t.instrument_id,
                                                                 DataIngestRun.resolution == "1D")).one()
                    row.status, row.error = "blocked", reason
            members = training_members(s, cfg, as_of)
            sync_readiness_alerts(s, cfg, members, as_of, run_id)
            sync_trading_days(s, cfg.ingest.calendar_code, cfg.ingest.calendar_min_stock_fraction, cfg.ingest.calendar_grace_days)
        symbol_of = {t.instrument_id: t.symbol for t in targets}
        with session_scope(engine) as s:
            timings = {symbol_of[d.instrument_id]: d.duration_ms
                       for d in s.scalars(select(DataIngestRun).where(DataIngestRun.run_id == run_id, DataIngestRun.resolution == "1D"))}
        stats.update(new_instruments=[t.symbol for t in targets], blocked=blocked, failures=failures, duration_ms=timings,
                     client_warnings=client.warnings)
        if failures:
            raise DnseError(f"{len(failures)} instrument(s) failed: {failures}")
    return stats
