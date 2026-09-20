"""Point-in-time universe membership (principle 5).

Nothing here knows about "VN30" or the number 30: a universe is just a code with
dated membership rows. A symbol is a member on day d iff
``effective_from <= d < effective_to`` (``effective_to`` empty = still a member).

DNSE exposes no constituent list, so membership is loaded from a CSV you supply
(see data/universe/README.md).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sqlalchemy import or_, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.orm import Session

from predict_stock.db.models import UniverseMembership
from predict_stock.db.repo import ensure_instruments

REQUIRED_COLUMNS = ("universe_code", "symbol", "effective_from", "effective_to")


class MembershipError(ValueError):
    pass


def get_members(session: Session, universe_code: str, as_of: date) -> list[str]:
    """Members of ``universe_code`` on ``as_of`` (sorted)."""
    stmt = (
        select(UniverseMembership.symbol)
        .where(
            UniverseMembership.universe_code == universe_code,
            UniverseMembership.effective_from <= as_of,
            or_(UniverseMembership.effective_to.is_(None), UniverseMembership.effective_to > as_of),
        )
        .distinct()
        .order_by(UniverseMembership.symbol)
    )
    return list(session.scalars(stmt))


def get_symbols_between(session: Session, universe_code: str, start: date, end: date) -> list[str]:
    """Every symbol that was a member on at least one day in [start, end]."""
    stmt = (
        select(UniverseMembership.symbol)
        .where(
            UniverseMembership.universe_code == universe_code,
            UniverseMembership.effective_from <= end,
            or_(UniverseMembership.effective_to.is_(None), UniverseMembership.effective_to > start),
        )
        .distinct()
        .order_by(UniverseMembership.symbol)
    )
    return list(session.scalars(stmt))


@dataclass(frozen=True)
class _Row:
    universe_code: str
    symbol: str
    effective_from: date
    effective_to: date | None


def _parse_csv(path: Path) -> list[_Row]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(line for line in fh if line.strip() and not line.lstrip().startswith("#"))
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise MembershipError(f"{path}: missing columns {missing}; expected {list(REQUIRED_COLUMNS)}")
        rows: list[_Row] = []
        for n, rec in enumerate(reader, start=2):
            try:
                eff_from = date.fromisoformat(rec["effective_from"].strip())
                eff_to = date.fromisoformat(rec["effective_to"].strip()) if rec["effective_to"].strip() else None
            except ValueError as exc:
                raise MembershipError(f"{path}:{n}: bad date ({exc})") from exc
            code, symbol = rec["universe_code"].strip(), rec["symbol"].strip().upper()
            if not code or not symbol:
                raise MembershipError(f"{path}:{n}: empty universe_code/symbol")
            if eff_to is not None and eff_to <= eff_from:
                raise MembershipError(f"{path}:{n}: effective_to {eff_to} must be after effective_from {eff_from}")
            rows.append(_Row(code, symbol, eff_from, eff_to))
    if not rows:
        raise MembershipError(f"{path}: no data rows")
    return rows


def _check_no_overlap(rows: list[_Row]) -> None:
    by_key: dict[tuple[str, str], list[_Row]] = defaultdict(list)
    for r in rows:
        by_key[(r.universe_code, r.symbol)].append(r)
    for (code, symbol), items in by_key.items():
        items.sort(key=lambda r: r.effective_from)
        for prev, nxt in zip(items, items[1:]):
            if prev.effective_to is None or prev.effective_to > nxt.effective_from:
                raise MembershipError(
                    f"{code}/{symbol}: interval starting {prev.effective_from} "
                    f"(to {prev.effective_to}) overlaps interval starting {nxt.effective_from}"
                )


def load_membership_csv(session: Session, path: str | Path, source: str | None = None) -> int:
    """Validate and idempotently upsert a membership CSV. Returns rows processed.

    Re-loading the same file changes nothing. A row with the same
    (universe_code, symbol, effective_from) but a different effective_to updates
    it (used to close an open interval). Overlaps — within the file or against
    rows already in the DB — are rejected before anything is written.
    """
    path = Path(path)
    new_rows = _parse_csv(path)
    keys = {(r.universe_code, r.symbol) for r in new_rows}

    # Merge with existing rows for the same (universe, symbol); file rows win on same key.
    existing = session.execute(
        select(
            UniverseMembership.universe_code,
            UniverseMembership.symbol,
            UniverseMembership.effective_from,
            UniverseMembership.effective_to,
        ).where(UniverseMembership.universe_code.in_({k[0] for k in keys}))
    ).all()
    merged: dict[tuple[str, str, date], _Row] = {
        (e.universe_code, e.symbol, e.effective_from): _Row(*e) for e in existing if (e.universe_code, e.symbol) in keys
    }
    for r in new_rows:
        merged[(r.universe_code, r.symbol, r.effective_from)] = r
    _check_no_overlap(list(merged.values()))

    ensure_instruments(session, {r.symbol for r in new_rows}, "stock")
    src = source or path.name
    values = [
        {
            "universe_code": r.universe_code,
            "symbol": r.symbol,
            "effective_from": r.effective_from,
            "effective_to": r.effective_to,
            "source": src,
        }
        for r in new_rows
    ]
    stmt = mysql_insert(UniverseMembership).values(values)
    session.execute(stmt.on_duplicate_key_update(effective_to=stmt.inserted.effective_to, source=stmt.inserted.source))
    return len(values)
