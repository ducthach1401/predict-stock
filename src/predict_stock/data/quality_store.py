"""Persisting data-quality findings and matching them against known issues.

* `data_quality_findings` holds one row per (instrument, check, date). Re-running the checks updates rows
  in place (no duplicates); a problem that no longer occurs becomes 'resolved'.
* A finding is 'explained' only when a known issue covers it (same check, the instrument if the issue names
  one, date inside [date_from, date_to]); otherwise it is 'open'. Known issues live in
  `data_quality_known_issues`, loaded from a CSV so the explanation and its evidence are reviewable.
* `unexplained_errors` is what decides whether the data may be used: any open error blocks.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import func, select, update
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.orm import Session

from predict_stock.data.quality import SEVERITY
from predict_stock.db.models import DataQualityFinding, DataQualityKnownIssue
from predict_stock.db.repo import find_symbol_row

KNOWN_ISSUE_COLUMNS = ("key", "check", "symbol", "date_from", "date_to", "treatment", "explanation", "evidence")


class KnownIssueError(ValueError):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def load_known_issues(session: Session, path: str | Path) -> int:
    """Idempotently upsert known issues from a CSV (columns: key, check, symbol, date_from, date_to,
    treatment, explanation, evidence). ``symbol`` empty = any instrument; dates empty = unbounded."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(line for line in fh if line.strip() and not line.lstrip().startswith("#"))
        missing = [c for c in KNOWN_ISSUE_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise KnownIssueError(f"{path}: missing columns {missing}")
        rows = list(reader)
    values = []
    for n, r in enumerate(rows, start=2):
        get = lambda k: (r.get(k) or "").strip() or None
        if get("check") not in SEVERITY:
            raise KnownIssueError(f"{path}:{n}: unknown check {get('check')!r}")
        if not (get("key") and get("treatment") and get("explanation")):
            raise KnownIssueError(f"{path}:{n}: key, treatment and explanation are required")
        try:
            d_from = date.fromisoformat(get("date_from")) if get("date_from") else None
            d_to = date.fromisoformat(get("date_to")) if get("date_to") else None
        except ValueError as exc:
            raise KnownIssueError(f"{path}:{n}: bad date ({exc})") from exc
        if d_from and d_to and d_to < d_from:
            raise KnownIssueError(f"{path}:{n}: date_to before date_from")
        iid = None
        if get("symbol"):
            row = find_symbol_row(session, get("symbol").upper())
            if row is None:
                raise KnownIssueError(f"{path}:{n}: unknown symbol {get('symbol')!r}")
            iid = row.instrument_id
        values.append({"issue_key": get("key"), "check_name": get("check"), "instrument_id": iid, "date_from": d_from,
                       "date_to": d_to, "treatment": get("treatment"), "explanation": get("explanation"),
                       "evidence": get("evidence"), "source": path.name})
    if len({v["issue_key"] for v in values}) != len(values):
        raise KnownIssueError(f"{path}: duplicate keys")
    if values:
        stmt = mysql_insert(DataQualityKnownIssue).values(values)
        session.execute(stmt.on_duplicate_key_update(**{c: stmt.inserted[c] for c in (
            "check_name", "instrument_id", "date_from", "date_to", "treatment", "explanation", "evidence", "source")}))
    return len(values)


@dataclass(frozen=True)
class _Known:
    id: int
    check: str
    instrument_id: int | None
    date_from: date | None
    date_to: date | None

    def covers(self, check: str, iid: int, d: date | None) -> bool:
        if check != self.check or (self.instrument_id is not None and iid != self.instrument_id):
            return False
        if self.date_from is None and self.date_to is None:
            return True
        if d is None:
            return False
        return (self.date_from is None or d >= self.date_from) and (self.date_to is None or d <= self.date_to)


def match_known_issue(known: list[_Known], check: str, iid: int, d: date | None) -> int | None:
    for k in known:
        if k.covers(check, iid, d):
            return k.id
    return None


def store_findings(session: Session, issues: pd.DataFrame, instrument_ids: list[int], run_id: int | None) -> dict[str, int]:
    """Upsert the current findings for ``instrument_ids`` and resolve the ones that disappeared.
    Returns counts of {open, explained, resolved, new}."""
    now = _utcnow()
    known = [_Known(k.id, k.check_name, k.instrument_id, k.date_from, k.date_to)
             for k in session.scalars(select(DataQualityKnownIssue))]
    existing = {(f.instrument_id, f.check_name, f.subject_key): f for f in session.scalars(
        select(DataQualityFinding).where(DataQualityFinding.instrument_id.in_(instrument_ids)))}
    seen: set[tuple] = set()
    inserts, new = [], 0
    for r in issues.itertuples():
        d = None if r.trade_date is None or pd.isna(r.trade_date) else r.trade_date
        key = (int(r.instrument_id), r.check, d.isoformat() if d else "series")
        if key in seen:
            continue
        seen.add(key)
        kid = match_known_issue(known, r.check, int(r.instrument_id), d)
        status = "explained" if kid else "open"
        f = existing.get(key)
        if f is None:
            new += 1
            inserts.append({"instrument_id": key[0], "check_name": key[1], "severity": r.severity, "trade_date": d,
                            "subject_key": key[2], "detail": r.detail, "status": status, "known_issue_id": kid,
                            "first_seen_run_id": run_id, "last_seen_run_id": run_id, "first_seen_at": now, "last_seen_at": now})
        else:
            f.severity, f.detail, f.status, f.known_issue_id = r.severity, r.detail, status, kid
            f.last_seen_run_id, f.last_seen_at, f.resolved_at = run_id, now, None
    if inserts:
        session.execute(DataQualityFinding.__table__.insert().values(inserts))
    resolved = 0
    for key, f in existing.items():
        if key not in seen and f.status != "resolved":
            f.status, f.resolved_at = "resolved", now
            resolved += 1
    session.flush()
    counts = dict(session.execute(
        select(DataQualityFinding.status, func.count()).where(DataQualityFinding.instrument_id.in_(instrument_ids))
        .group_by(DataQualityFinding.status)).all())
    return {"open": counts.get("open", 0), "explained": counts.get("explained", 0), "resolved": counts.get("resolved", 0), "new": new}


def unexplained_errors(session: Session, instrument_ids: list[int] | None = None) -> list[DataQualityFinding]:
    """Open findings of severity 'error': what still blocks the data from being trusted."""
    stmt = select(DataQualityFinding).where(DataQualityFinding.status == "open", DataQualityFinding.severity == "error")
    if instrument_ids is not None:
        stmt = stmt.where(DataQualityFinding.instrument_id.in_(instrument_ids))
    return list(session.scalars(stmt.order_by(DataQualityFinding.instrument_id, DataQualityFinding.trade_date)))
