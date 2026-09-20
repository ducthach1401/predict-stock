"""Alerts (table `alerts`): missing data, API errors, failed jobs, universe changes, kill-switch. An identical unacknowledged alert is not written twice."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import Engine, select

from predict_stock.db.models import Alert
from predict_stock.db.session import session_scope

SEVERITIES = ("info", "warn", "error", "critical")


def raise_alert(engine: Engine, severity: str, category: str, message: str, *, instrument_id: int | None = None, run_id: int | None = None, details: dict | None = None,
                dedupe_hours: float = 36.0) -> int | None:
    """Writes an alert unless the same (category, message, instrument) is already open (not acknowledged) within ``dedupe_hours``. Returns the id, or None if suppressed."""
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}")
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=dedupe_hours)
    with session_scope(engine) as s:
        q = select(Alert.id).where(Alert.category == category, Alert.message == message[:65000], Alert.acknowledged_at.is_(None), Alert.created_at >= since)
        q = q.where(Alert.instrument_id == instrument_id) if instrument_id is not None else q.where(Alert.instrument_id.is_(None))
        if s.scalar(q) is not None:
            return None
        a = Alert(severity=severity, category=category, message=message[:65000], instrument_id=instrument_id, run_id=run_id, details=details)
        s.add(a)
        s.flush()
        return a.id


def open_alerts(engine: Engine, min_severity: str = "info") -> list[Alert]:
    rank = {s: i for i, s in enumerate(SEVERITIES)}
    with session_scope(engine) as s:
        rows = list(s.scalars(select(Alert).where(Alert.acknowledged_at.is_(None)).order_by(Alert.id.desc())))
        for r in rows:
            s.expunge(r)
    return [r for r in rows if rank.get(r.severity, 0) >= rank[min_severity]]


def acknowledge(engine: Engine, alert_id: int) -> None:
    with session_scope(engine) as s:
        a = s.get(Alert, alert_id)
        if a is not None and a.acknowledged_at is None:
            a.acknowledged_at = datetime.now(timezone.utc).replace(tzinfo=None)
