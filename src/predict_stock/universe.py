"""Point-in-time universe queries (principle 5).

Nothing here knows about "VN30", "VN100" or a universe size: a universe is a row in
``universes`` plus dated membership rows. Half-open intervals throughout:

    member on d  <=>  valid_from <= d AND (valid_to IS NULL OR d < valid_to)

The symbol returned for a member is the ticker the instrument had on that date, so a
rename or exchange move never changes who was a member.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from sqlalchemy import exists, or_, select
from sqlalchemy.orm import Session, aliased

from predict_stock.config import AppConfig
from predict_stock.db.models import (
    InstrumentStatusHistory as StatusRow,
    InstrumentSymbolHistory as SymbolRow,
    Universe,
    UniverseMembership as Membership,
)
from predict_stock.db.repo import latest_symbol_row


@dataclass(frozen=True)
class Member:
    instrument_id: int
    symbol: str
    exchange: str | None
    weight: Decimal | None


def get_universe(session: Session, code: str) -> Universe | None:
    return session.scalars(select(Universe).where(Universe.code == code)).first()


def get_member_records(session: Session, universe_code: str, as_of: date, *, tradable_only: bool = False) -> list[Member]:
    """Members of ``universe_code`` on ``as_of``, ordered by ticker.
    ``tradable_only`` drops instruments suspended or delisted on that date."""
    stmt = (
        select(Membership.instrument_id, SymbolRow.symbol, SymbolRow.exchange, Membership.weight)
        .join(Universe, Universe.id == Membership.universe_id)
        .join(
            SymbolRow,
            (SymbolRow.instrument_id == Membership.instrument_id)
            & (SymbolRow.valid_from <= as_of)
            & or_(SymbolRow.valid_to.is_(None), as_of < SymbolRow.valid_to),
        )
        .where(
            Universe.code == universe_code,
            Membership.valid_from <= as_of,
            or_(Membership.valid_to.is_(None), as_of < Membership.valid_to),
        )
        .order_by(SymbolRow.symbol)
    )
    if tradable_only:
        stmt = stmt.where(
            ~exists().where(
                StatusRow.instrument_id == Membership.instrument_id,
                StatusRow.valid_from <= as_of,
                or_(StatusRow.valid_to.is_(None), as_of < StatusRow.valid_to),
            )
        )
    return [Member(*row) for row in session.execute(stmt)]


def get_members(session: Session, universe_code: str, as_of: date, *, tradable_only: bool = False) -> list[str]:
    """Tickers of ``universe_code`` on ``as_of`` (sorted)."""
    return [m.symbol for m in get_member_records(session, universe_code, as_of, tradable_only=tradable_only)]


def get_instruments_between(session: Session, universe_code: str, start: date, end: date) -> list[Member]:
    """Every instrument that was a member on at least one day of [start, end], labelled with
    its ticker as of ``end`` (or its last ticker if it was delisted before then)."""
    ids = list(
        session.scalars(
            select(Membership.instrument_id)
            .join(Universe, Universe.id == Membership.universe_id)
            .where(
                Universe.code == universe_code,
                Membership.valid_from <= end,
                or_(Membership.valid_to.is_(None), Membership.valid_to > start),
            )
            .distinct()
        )
    )
    out = []
    for iid in ids:
        row = latest_symbol_row(session, iid, end) or latest_symbol_row(session, iid)
        out.append(Member(iid, row.symbol, row.exchange, None))
    return sorted(out, key=lambda m: m.symbol)


def training_members(session: Session, cfg: AppConfig, as_of: date) -> list[Member]:
    """Instruments the models may learn from on ``as_of`` (configurable, normally the wider set)."""
    return get_member_records(session, cfg.universe.training_code, as_of)


def trading_members(session: Session, cfg: AppConfig, as_of: date) -> list[Member]:
    """Instruments allowed to receive a recommendation on ``as_of``: members of the trading
    universe that are not suspended or delisted."""
    return get_member_records(session, cfg.universe.trading_code, as_of, tradable_only=True)


def trading_outside_training(session: Session, cfg: AppConfig, as_of: date) -> list[str]:
    """Tradable instruments the model was never trained on (should be empty)."""
    train = {m.instrument_id for m in training_members(session, cfg, as_of)}
    return [m.symbol for m in trading_members(session, cfg, as_of) if m.instrument_id not in train]


# ---- integrity ------------------------------------------------------------------------

def _overlaps(intervals: list[tuple[date, date | None, str]]) -> list[str]:
    problems = []
    items = sorted(intervals, key=lambda t: t[0])
    for a, b in zip(items, items[1:]):
        if a[1] is None or a[1] > b[0]:
            problems.append(f"{a[2]} [{a[0]}, {a[1] or 'open'}) overlaps {b[2]} [{b[0]}, {b[1] or 'open'})")
    return problems


def check_integrity(session: Session) -> list[str]:
    """Structural problems the database cannot enforce by itself (MySQL has no exclusion
    constraints). Empty list = healthy."""
    problems: list[str] = []
    by_inst: dict[int, list] = defaultdict(list)
    by_symbol: dict[str, list] = defaultdict(list)
    for r in session.scalars(select(SymbolRow)):
        by_inst[r.instrument_id].append((r.valid_from, r.valid_to, f"instrument {r.instrument_id} {r.symbol}"))
        by_symbol[r.symbol].append((r.valid_from, r.valid_to, f"{r.symbol} (instrument {r.instrument_id})"))
    for iid, iv in by_inst.items():
        problems += [f"symbol history: {p}" for p in _overlaps(iv)]
    for sym, iv in by_symbol.items():
        problems += [f"ticker used by two instruments: {p}" for p in _overlaps(iv)]

    st: dict[int, list] = defaultdict(list)
    for r in session.scalars(select(StatusRow)):
        st[r.instrument_id].append((r.valid_from, r.valid_to, f"instrument {r.instrument_id} {r.status}"))
    for iv in st.values():
        problems += [f"status history: {p}" for p in _overlaps(iv)]

    U = aliased(Universe)
    mem: dict[tuple[str, int], list] = defaultdict(list)
    for m, code in session.execute(select(Membership, U.code).join(U, U.id == Membership.universe_id)):
        mem[(code, m.instrument_id)].append((m.valid_from, m.valid_to, f"{code}/instrument {m.instrument_id}"))
        covered = any(f <= m.valid_from and (t is None or m.valid_from < t) for f, t, _ in by_inst.get(m.instrument_id, []))
        if not covered:
            problems.append(f"membership {code}/instrument {m.instrument_id} starts {m.valid_from} but no symbol row covers that date")
    for iv in mem.values():
        problems += [f"membership: {p}" for p in _overlaps(iv)]
    return problems
