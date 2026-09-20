"""sync_universe: apply a membership snapshot (CSV) to a universe as of an effective date.

DNSE offers no constituent list (verified in the API probe), so the only source is a CSV
you supply. The file is a *snapshot*: the members you want on ``--effective-date``. The job
diffs it against the current open memberships, closes rows that left, opens rows that
joined, and records every change in ``universe_change_log``.

CSV columns (only ``symbol`` is required; ``#`` lines and blanks are ignored):

    symbol           ticker on the effective date
    previous_symbol  rename (đổi mã): the ticker the instrument had before the effective date
    exchange         HOSE | HNX | UPCOM; a different value than recorded = exchange move (chuyển sàn)
    weight           optional index weight (fraction)
    status           active (default) | suspended (tạm dừng) | delisted (hủy niêm yết)
    valid_from       backfill: membership start for a NEW member, only allowed into an empty universe
    note             free text

Rules: applying the same file again changes nothing; the effective date may not precede the
latest change already recorded for the universe (history is not rewritten); a member absent
from the file is removed; ``delisted`` closes the membership and the ticker and is terminal;
``suspended`` keeps membership but the instrument stops being tradable until a later
snapshot lists it as active.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from predict_stock.db.models import (
    InstrumentSymbolHistory as SymbolRow,
    Universe,
    UniverseChangeLog,
    UniverseMembership as Membership,
)
from predict_stock.db.repo import create_instrument, find_symbol_row, latest_symbol_row, open_symbol_row
from predict_stock.instruments import EXCHANGES, STATUSES, InstrumentError, change_exchange, rename_symbol, set_status, status_at


class SnapshotError(ValueError):
    pass


@dataclass(frozen=True)
class SnapshotRow:
    symbol: str
    previous_symbol: str | None = None
    exchange: str | None = None
    weight: Decimal | None = None
    status: str | None = None
    valid_from: date | None = None
    note: str | None = None


@dataclass
class PlanItem:
    action: str
    symbol: str | None
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        d = ", ".join(f"{k}={v}" for k, v in self.detail.items())
        return f"{self.action:<16} {self.symbol or '-':<8} {d}"


def read_snapshot_csv(path: str | Path) -> list[SnapshotRow]:
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(line for line in fh if line.strip() and not line.lstrip().startswith("#"))
        if "symbol" not in (reader.fieldnames or []):
            raise SnapshotError(f"{path}: missing required column 'symbol'")
        rows, seen = [], set()
        for n, rec in enumerate(reader, start=2):
            get = lambda k: (rec.get(k) or "").strip() or None
            symbol = (get("symbol") or "").upper()
            if not symbol:
                raise SnapshotError(f"{path}:{n}: empty symbol")
            if symbol in seen:
                raise SnapshotError(f"{path}:{n}: duplicate symbol {symbol}")
            seen.add(symbol)
            exchange, status = get("exchange"), get("status")
            if exchange and exchange.upper() not in EXCHANGES:
                raise SnapshotError(f"{path}:{n}: unknown exchange {exchange!r}; expected one of {EXCHANGES}")
            if status and status.lower() not in STATUSES:
                raise SnapshotError(f"{path}:{n}: unknown status {status!r}; expected one of {STATUSES}")
            try:
                weight = Decimal(get("weight")) if get("weight") else None
                valid_from = date.fromisoformat(get("valid_from")) if get("valid_from") else None
            except (InvalidOperation, ValueError) as exc:
                raise SnapshotError(f"{path}:{n}: bad weight/valid_from ({exc})") from exc
            if weight is not None and not (Decimal(0) <= weight <= Decimal(1)):
                raise SnapshotError(f"{path}:{n}: weight {weight} outside [0, 1]")
            prev = (get("previous_symbol") or "").upper() or None
            if prev and prev == symbol:
                raise SnapshotError(f"{path}:{n}: previous_symbol equals symbol")
            rows.append(SnapshotRow(symbol, prev, exchange.upper() if exchange else None, weight,
                                    status.lower() if status else None, valid_from, get("note")))
    if not rows:
        raise SnapshotError(f"{path}: no data rows")
    return rows


def _latest_event(session: Session, universe_id: int) -> date | None:
    dates = [
        session.scalar(select(func.max(Membership.valid_from)).where(Membership.universe_id == universe_id)),
        session.scalar(select(func.max(Membership.valid_to)).where(Membership.universe_id == universe_id)),
        session.scalar(select(func.max(UniverseChangeLog.effective_date)).where(UniverseChangeLog.universe_id == universe_id)),
    ]
    dates = [d for d in dates if d is not None]
    return max(dates) if dates else None


def _same_weight(a: Decimal | None, b: Decimal | None) -> bool:
    return (a is None and b is None) or (a is not None and b is not None and Decimal(a) == Decimal(b))


def apply_snapshot(
    session: Session,
    universe_code: str,
    rows: list[SnapshotRow],
    effective_date: date,
    *,
    source: str,
    floor: date,
    run_id: int | None = None,
    create: bool = False,
    name: str | None = None,
    kind: str = "stock",
) -> list[PlanItem]:
    """Apply the snapshot inside the caller's transaction and return what changed
    (empty list = already up to date). The caller commits, or rolls back for a dry run."""
    items: list[PlanItem] = []
    uni = session.scalars(select(Universe).where(Universe.code == universe_code)).first()
    if uni is None:
        if not create:
            raise SnapshotError(f"unknown universe {universe_code!r}; pass --create to create it")
        uni = Universe(code=universe_code, name=name or universe_code)
        session.add(uni)
        session.flush()
        items.append(PlanItem("create_universe", None, {"code": universe_code}))

    last_event = _latest_event(session, uni.id)
    if last_event is not None and effective_date < last_event:
        raise SnapshotError(
            f"effective date {effective_date} precedes the latest recorded change ({last_event}) of {universe_code}; "
            "history is not rewritten"
        )
    is_empty = last_event is None
    open_rows = {
        m.instrument_id: m for m in session.scalars(select(Membership).where(Membership.universe_id == uni.id, Membership.valid_to.is_(None)))
    }
    handled: set[int] = set()

    def emit(action, symbol, inst_id, old=None, new=None, **detail):
        items.append(PlanItem(action, symbol, {**detail, **({"new": new} if new else {})}))
        session.add(UniverseChangeLog(
            universe_id=uni.id, instrument_id=inst_id, symbol=symbol, action=action, effective_date=effective_date,
            old_value=old, new_value=new, source=source, run_id=run_id))

    for row in rows:
        try:
            inst_id, is_new = _resolve_instrument(session, row, effective_date, floor, kind, emit)
        except InstrumentError as exc:
            raise SnapshotError(f"{row.symbol}: {exc}") from exc
        handled.add(inst_id)

        srow = open_symbol_row(session, inst_id, effective_date)
        if row.exchange and srow is not None and not is_new:
            try:
                ev = change_exchange(session, inst_id, row.exchange, effective_date)
            except InstrumentError as exc:
                raise SnapshotError(f"{row.symbol}: {exc}") from exc
            if ev:
                emit(ev, row.symbol, inst_id, old={"exchange": srow.exchange} if ev == "exchange_change" else None,
                     new={"exchange": row.exchange})

        want = row.status or "active"
        cur = status_at(session, inst_id, effective_date)
        try:
            if want == "delisted":
                if set_status(session, inst_id, "delisted", effective_date, row.note):
                    emit("delist", row.symbol, inst_id, old={"status": cur}, new={"status": "delisted"})
            elif want == "suspended":
                if set_status(session, inst_id, "suspended", effective_date, row.note):
                    emit("suspend", row.symbol, inst_id, old={"status": cur}, new={"status": "suspended"})
            elif cur == "suspended":
                set_status(session, inst_id, "active", effective_date)
                emit("resume", row.symbol, inst_id, old={"status": "suspended"}, new={"status": "active"})
            elif cur == "delisted":
                raise SnapshotError(f"{row.symbol}: instrument was delisted; it cannot be listed as active")
        except InstrumentError as exc:
            raise SnapshotError(f"{row.symbol}: {exc}") from exc

        m = open_rows.get(inst_id)
        if want == "delisted":
            if m is not None:
                _close(m, effective_date, row.symbol)
                emit("remove", row.symbol, inst_id, old=_val(m), reason="delisted")
                open_rows.pop(inst_id)
            continue
        if m is None:
            vf = row.valid_from or effective_date
            if vf > effective_date:
                raise SnapshotError(f"{row.symbol}: valid_from {vf} is after the effective date {effective_date}")
            if vf != effective_date and not is_empty:
                raise SnapshotError(f"{row.symbol}: valid_from backfill is only allowed into an empty universe")
            session.add(Membership(universe_id=uni.id, instrument_id=inst_id, valid_from=vf, weight=row.weight,
                                   source=source, note=row.note))
            emit("add", row.symbol, inst_id, new={"valid_from": vf.isoformat(), "weight": _w(row.weight)})
        else:
            if not _same_weight(m.weight, row.weight):
                old = _val(m)
                if m.valid_from == effective_date:
                    m.weight = row.weight
                else:
                    _close(m, effective_date, row.symbol)
                    session.add(Membership(universe_id=uni.id, instrument_id=inst_id, valid_from=effective_date,
                                           weight=row.weight, source=source, note=row.note or m.note))
                emit("weight_change", row.symbol, inst_id, old=old, new={"weight": _w(row.weight)})
            if row.note is not None and row.note != m.note:
                m.note = row.note  # annotation only: no history event

    for inst_id, m in open_rows.items():
        if inst_id in handled:
            continue
        symbol = latest_symbol_row(session, inst_id, effective_date)
        _close(m, effective_date, symbol.symbol if symbol else str(inst_id))
        emit("remove", symbol.symbol if symbol else None, inst_id, old=_val(m), reason="absent from snapshot")
    session.flush()
    return items


def _w(w: Decimal | None) -> str | None:
    return None if w is None else str(w)


def _val(m: Membership) -> dict:
    return {"valid_from": m.valid_from.isoformat(), "weight": _w(m.weight)}


def _close(m: Membership, effective_date: date, symbol: str) -> None:
    if effective_date <= m.valid_from:
        raise SnapshotError(
            f"{symbol}: cannot end membership on {effective_date}; it started {m.valid_from}. Use a later effective date."
        )
    m.valid_to = effective_date


def _resolve_instrument(session, row: SnapshotRow, eff: date, floor: date, kind: str, emit) -> tuple[int, bool]:
    """Returns (instrument_id, created_now). Handles renames and re-applied renames."""
    current = find_symbol_row(session, row.symbol, eff)
    if row.previous_symbol:
        if current is not None and session.scalars(
            select(SymbolRow).where(SymbolRow.instrument_id == current.instrument_id,
                                    SymbolRow.symbol == row.previous_symbol, SymbolRow.valid_to == eff)
        ).first():
            return current.instrument_id, False  # rename already applied
        prev = find_symbol_row(session, row.previous_symbol, eff)
        if prev is None:
            raise SnapshotError(f"{row.symbol}: previous_symbol {row.previous_symbol!r} is not a known symbol on {eff}")
        inst_id = rename_symbol(session, row.previous_symbol, row.symbol, eff, exchange=row.exchange)
        emit("rename", row.symbol, inst_id, old={"symbol": row.previous_symbol}, new={"symbol": row.symbol})
        return inst_id, False
    if current is not None:
        return current.instrument_id, False
    ended = find_symbol_row(session, row.symbol)  # most recent row for the ticker, e.g. after delisting
    if ended is not None:
        if row.status == "delisted":
            return ended.instrument_id, False
        raise SnapshotError(
            f"{row.symbol}: ticker belonged to instrument {ended.instrument_id} until {ended.valid_to}; "
            "re-using a ended ticker needs a rename/new-instrument decision, not an automatic guess"
        )
    inst_id = create_instrument(session, row.symbol, kind, valid_from=floor, exchange=row.exchange)
    return inst_id, True
