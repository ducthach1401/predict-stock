from __future__ import annotations

from typing import Iterable

from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.orm import Session

from predict_stock.db.models import Instrument


def ensure_instruments(session: Session, symbols: Iterable[str], kind: str) -> None:
    """Idempotently register symbols (existing rows are left untouched)."""
    rows = [{"symbol": s, "kind": kind} for s in sorted(set(symbols))]
    if not rows:
        return
    stmt = mysql_insert(Instrument).values(rows)
    session.execute(stmt.on_duplicate_key_update(symbol=stmt.inserted.symbol))
