"""Small lookups shared by ingestion, universes and instrument maintenance."""
from __future__ import annotations

from datetime import date

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from predict_stock.db.models import Instrument, InstrumentSymbolHistory as SymbolRow


def find_symbol_row(session: Session, symbol: str, as_of: date | None = None) -> SymbolRow | None:
    """The history row for ``symbol`` valid on ``as_of`` (half-open interval); with
    ``as_of=None`` the most recent row for that symbol, whatever its dates."""
    stmt = select(SymbolRow).where(SymbolRow.symbol == symbol)
    if as_of is not None:
        stmt = stmt.where(SymbolRow.valid_from <= as_of, or_(SymbolRow.valid_to.is_(None), as_of < SymbolRow.valid_to))
    return session.scalars(stmt.order_by(SymbolRow.valid_from.desc()).limit(1)).first()


def open_symbol_row(session: Session, instrument_id: int, as_of: date | None = None) -> SymbolRow | None:
    """The symbol row of an instrument valid on ``as_of``; ``None`` as_of = the currently open row."""
    stmt = select(SymbolRow).where(SymbolRow.instrument_id == instrument_id)
    if as_of is None:
        stmt = stmt.where(SymbolRow.valid_to.is_(None))
    else:
        stmt = stmt.where(SymbolRow.valid_from <= as_of, or_(SymbolRow.valid_to.is_(None), as_of < SymbolRow.valid_to))
    return session.scalars(stmt.order_by(SymbolRow.valid_from.desc()).limit(1)).first()


def latest_symbol_row(session: Session, instrument_id: int, as_of: date | None = None) -> SymbolRow | None:
    """Latest row with valid_from <= as_of (even if it has since been closed, e.g. delisted)."""
    stmt = select(SymbolRow).where(SymbolRow.instrument_id == instrument_id)
    if as_of is not None:
        stmt = stmt.where(SymbolRow.valid_from <= as_of)
    return session.scalars(stmt.order_by(SymbolRow.valid_from.desc()).limit(1)).first()


def create_instrument(
    session: Session, symbol: str, kind: str, *, valid_from: date, exchange: str | None = None, name: str | None = None
) -> int:
    inst = Instrument(kind=kind, name=name)
    session.add(inst)
    session.flush()
    session.add(SymbolRow(instrument_id=inst.id, symbol=symbol, exchange=exchange, valid_from=valid_from))
    session.flush()
    return inst.id


def get_or_create_instrument(session: Session, symbol: str, kind: str, *, floor: date) -> int:
    """Instrument currently (or most recently) known under ``symbol``; created if unseen."""
    row = find_symbol_row(session, symbol)
    if row is not None:
        return row.instrument_id
    return create_instrument(session, symbol, kind, valid_from=floor)
