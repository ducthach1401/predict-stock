"""Instrument life-cycle events: rename (đổi mã), exchange move (chuyển sàn),
suspension (tạm dừng) and delisting (hủy niêm yết).

Each event closes the row that was open on ``effective_date`` and opens the next one
([valid_from, valid_to) intervals), so any past date can be answered point-in-time.
"""
from __future__ import annotations

from datetime import date

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from predict_stock.db.models import InstrumentStatusHistory as StatusRow, InstrumentSymbolHistory as SymbolRow
from predict_stock.db.repo import find_symbol_row, open_symbol_row

EXCHANGES = ("HOSE", "HNX", "UPCOM")
STATUSES = ("active", "suspended", "delisted")


class InstrumentError(ValueError):
    pass


def rename_symbol(session: Session, old: str, new: str, effective_date: date, *, exchange: str | None = None) -> int:
    """Old ticker stops being valid on ``effective_date``; the same instrument continues as ``new``."""
    old_row = find_symbol_row(session, old, effective_date)
    if old_row is None or old_row.valid_to is not None:
        raise InstrumentError(f"rename: {old!r} is not an open symbol on {effective_date}")
    if effective_date <= old_row.valid_from:
        raise InstrumentError(f"rename: effective date {effective_date} must be after {old!r} became valid ({old_row.valid_from})")
    clash = session.scalars(
        select(SymbolRow).where(
            SymbolRow.symbol == new,
            SymbolRow.instrument_id != old_row.instrument_id,
            or_(SymbolRow.valid_to.is_(None), SymbolRow.valid_to > effective_date),
        )
    ).first()
    if clash is not None:
        raise InstrumentError(f"rename: {new!r} is already used by instrument {clash.instrument_id} from {effective_date}")
    old_row.valid_to = effective_date
    session.add(SymbolRow(instrument_id=old_row.instrument_id, symbol=new, exchange=exchange or old_row.exchange, valid_from=effective_date))
    session.flush()
    return old_row.instrument_id


def change_exchange(session: Session, instrument_id: int, exchange: str, effective_date: date) -> str | None:
    """Returns 'exchange_set' (unknown -> known, in place), 'exchange_change' (history row rolled) or None (no-op)."""
    if exchange not in EXCHANGES:
        raise InstrumentError(f"unknown exchange {exchange!r}; expected one of {EXCHANGES}")
    row = open_symbol_row(session, instrument_id, effective_date)
    if row is None or row.valid_to is not None:
        raise InstrumentError(f"instrument {instrument_id} has no open symbol row on {effective_date}")
    if row.exchange == exchange:
        return None
    if row.exchange is None:
        row.exchange = exchange
        session.flush()
        return "exchange_set"
    if effective_date <= row.valid_from:
        raise InstrumentError(f"exchange change on {effective_date} must be after the current row started ({row.valid_from})")
    row.valid_to = effective_date
    session.add(SymbolRow(instrument_id=instrument_id, symbol=row.symbol, exchange=exchange, valid_from=effective_date))
    session.flush()
    return "exchange_change"


def _open_status(session: Session, instrument_id: int) -> StatusRow | None:
    return session.scalars(
        select(StatusRow).where(StatusRow.instrument_id == instrument_id, StatusRow.valid_to.is_(None))
    ).first()


def status_at(session: Session, instrument_id: int, as_of: date) -> str:
    row = session.scalars(
        select(StatusRow).where(
            StatusRow.instrument_id == instrument_id,
            StatusRow.valid_from <= as_of,
            or_(StatusRow.valid_to.is_(None), as_of < StatusRow.valid_to),
        )
    ).first()
    return row.status if row else "active"


def set_status(session: Session, instrument_id: int, status: str, effective_date: date, note: str | None = None) -> bool:
    """active | suspended | delisted. Returns True if anything changed.
    Delisting is terminal and also closes the open symbol row."""
    if status not in STATUSES:
        raise InstrumentError(f"unknown status {status!r}; expected one of {STATUSES}")
    cur = _open_status(session, instrument_id)
    if cur is not None and cur.status == "delisted":
        if status == "delisted":
            return False
        raise InstrumentError(f"instrument {instrument_id} was delisted on {cur.valid_from}; it cannot become {status}")
    if status == "active":
        if cur is None:
            return False
        if effective_date <= cur.valid_from:
            raise InstrumentError(f"resume on {effective_date} must be after the suspension started ({cur.valid_from})")
        cur.valid_to = effective_date
        session.flush()
        return True
    if status == "suspended":
        if cur is not None:
            return False
        session.add(StatusRow(instrument_id=instrument_id, status="suspended", valid_from=effective_date, note=note))
        session.flush()
        return True
    # delisted
    if cur is not None:  # a suspension ends when the delisting takes effect
        if effective_date < cur.valid_from:
            raise InstrumentError(f"delisting {effective_date} precedes the open suspension ({cur.valid_from})")
        if effective_date == cur.valid_from:
            cur.status, cur.note = "delisted", note
            _close_symbol(session, instrument_id, effective_date)
            session.flush()
            return True
        cur.valid_to = effective_date
    session.add(StatusRow(instrument_id=instrument_id, status="delisted", valid_from=effective_date, note=note))
    _close_symbol(session, instrument_id, effective_date)
    session.flush()
    return True


def _close_symbol(session: Session, instrument_id: int, effective_date: date) -> None:
    row = open_symbol_row(session, instrument_id)
    if row is not None:
        if effective_date <= row.valid_from:
            raise InstrumentError(f"delisting {effective_date} must be after the symbol became valid ({row.valid_from})")
        row.valid_to = effective_date
