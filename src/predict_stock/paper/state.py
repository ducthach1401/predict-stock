"""Small DB helpers of the paper trader: the INVEST target book, recommendation lookups, flags."""
from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import Engine, select

from predict_stock.db.models import Recommendation, SleeveTarget
from predict_stock.db.session import session_scope


def _key(d: dict | None) -> dict[int, float]:
    return {int(k): float(v) for k, v in (d or {}).items()}


def latest_target(engine: Engine, sleeve: str, before: date | None = None, upto: date | None = None) -> SleeveTarget | None:
    """The newest target-book row of a sleeve strictly before ``before`` (or up to and including ``upto``)."""
    with session_scope(engine) as s:
        q = select(SleeveTarget).where(SleeveTarget.sleeve == sleeve)
        if before is not None:
            q = q.where(SleeveTarget.as_of_date < before)
        if upto is not None:
            q = q.where(SleeveTarget.as_of_date <= upto)
        row = s.scalars(q.order_by(SleeveTarget.as_of_date.desc(), SleeveTarget.id.desc())).first()
        if row is not None:
            s.expunge(row)
        return row


def get_target(engine: Engine, sleeve: str, d: date, kind: str, n: int = 1) -> SleeveTarget | None:
    with session_scope(engine) as s:
        row = s.scalars(select(SleeveTarget).where(SleeveTarget.sleeve == sleeve, SleeveTarget.as_of_date == d, SleeveTarget.kind == kind, SleeveTarget.tranche_n == n)).first()
        if row is not None:
            s.expunge(row)
        return row


def save_target(engine: Engine, sleeve: str, d: date, kind: str, n: int, total: int, current: dict, target: dict | None, next_review: date | None, run_id: int | None) -> SleeveTarget:
    """Immutable per (sleeve, date, kind, tranche): a second call returns the stored row unchanged (the record of what was decided that day)."""
    with session_scope(engine) as s:
        row = s.scalars(select(SleeveTarget).where(SleeveTarget.sleeve == sleeve, SleeveTarget.as_of_date == d, SleeveTarget.kind == kind, SleeveTarget.tranche_n == n)).first()
        if row is None:
            row = SleeveTarget(sleeve=sleeve, as_of_date=d, kind=kind, tranche_n=n, tranches_total=total, current={str(k): v for k, v in current.items()},
                               target=target, next_review=next_review, run_id=run_id)
            s.add(row)
            s.flush()
        s.expunge(row)
        return row


def all_targets(engine: Engine, sleeve: str, upto: date) -> list[SleeveTarget]:
    with session_scope(engine) as s:
        rows = list(s.scalars(select(SleeveTarget).where(SleeveTarget.sleeve == sleeve, SleeveTarget.as_of_date <= upto).order_by(SleeveTarget.as_of_date, SleeveTarget.id)))
        for r in rows:
            s.expunge(r)
        return rows


def open_recommendations(engine: Engine, since: date | None = None) -> list[Recommendation]:
    """BUY recommendations still alive (pending / holding), optionally only those issued on or after ``since`` (the start of the paper record)."""
    with session_scope(engine) as s:
        q = select(Recommendation).where(Recommendation.status.in_(("open", "pending", "holding")), Recommendation.action == "BUY")
        if since is not None:
            q = q.where(Recommendation.as_of_date >= since)
        rows = list(s.scalars(q))
        for r in rows:
            s.expunge(r)
        return rows


def set_flag(engine: Engine, rec_id: int, key: str, value) -> bool:
    """True if the flag was new or changed."""
    with session_scope(engine) as s:
        r = s.get(Recommendation, rec_id)
        flags = dict(r.flags or {})
        if flags.get(key) == value:
            return False
        flags[key] = value
        r.flags = flags
        return True


def clear_flag(engine: Engine, rec_id: int, key: str) -> None:
    with session_scope(engine) as s:
        r = s.get(Recommendation, rec_id)
        flags = dict(r.flags or {})
        if key in flags:
            del flags[key]
            r.flags = flags or None


def members_on(engine: Engine, code: str, d: date) -> dict[int, str]:
    """instrument_id -> ticker of the universe members on ``d``."""
    from predict_stock.universe import get_member_records
    with session_scope(engine) as s:
        return {m.instrument_id: m.symbol for m in get_member_records(s, code, d)}
