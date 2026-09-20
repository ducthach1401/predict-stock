"""Instrument life-cycle events in isolation."""
from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select

from predict_stock.db.models import InstrumentSymbolHistory as SymbolRow
from predict_stock.db.repo import create_instrument, find_symbol_row, open_symbol_row
from predict_stock.db.session import session_scope
from predict_stock.instruments import InstrumentError, change_exchange, rename_symbol, set_status, status_at

D = date


def make(s, symbol="AAA", exchange=None, since=D(2000, 1, 1)):
    return create_instrument(s, symbol, "stock", valid_from=since, exchange=exchange)


def test_rename_closes_old_row_and_opens_new_one(engine):
    with session_scope(engine) as s:
        iid = make(s, exchange="HOSE")
        assert rename_symbol(s, "AAA", "AAX", D(2024, 1, 1)) == iid
        assert find_symbol_row(s, "AAA", D(2023, 12, 31)).instrument_id == iid
        assert find_symbol_row(s, "AAA", D(2024, 1, 1)) is None                   # old ticker ends on the effective date
        new = find_symbol_row(s, "AAX", D(2024, 1, 1))
        assert new.instrument_id == iid and new.exchange == "HOSE"               # exchange carried over
        assert find_symbol_row(s, "AAX", D(2023, 12, 31)) is None
        # chain of renames stays on one instrument
        rename_symbol(s, "AAX", "AAY", D(2025, 1, 1))
        assert [r.symbol for r in s.scalars(select(SymbolRow).order_by(SymbolRow.valid_from))] == ["AAA", "AAX", "AAY"]


@pytest.mark.parametrize("old,new,eff,msg", [
    ("NOPE", "X", D(2024, 1, 1), "not an open symbol"),
    ("AAA", "X", D(2000, 1, 1), "must be after"),
    ("AAA", "BBB", D(2024, 1, 1), "already used"),
])
def test_rename_errors(engine, old, new, eff, msg):
    with session_scope(engine) as s:
        make(s, "AAA"); make(s, "BBB")
        with pytest.raises(InstrumentError, match=msg):
            rename_symbol(s, old, new, eff)


def test_a_freed_ticker_can_be_reused_after_the_old_owner_left(engine):
    with session_scope(engine) as s:
        a = make(s, "AAA")
        rename_symbol(s, "AAA", "AAX", D(2020, 1, 1))
        b = make(s, "AAA", since=D(2021, 1, 1))                                   # a different company takes the ticker later
        assert find_symbol_row(s, "AAA", D(2019, 1, 1)).instrument_id == a
        assert find_symbol_row(s, "AAA", D(2022, 1, 1)).instrument_id == b


def test_change_exchange(engine):
    with session_scope(engine) as s:
        iid = make(s)
        assert change_exchange(s, iid, "HNX", D(2019, 1, 1)) == "exchange_set"     # unknown -> known: in place
        assert change_exchange(s, iid, "HNX", D(2020, 1, 1)) is None               # same value: no-op
        assert change_exchange(s, iid, "HOSE", D(2021, 3, 1)) == "exchange_change"
        assert open_symbol_row(s, iid, D(2021, 2, 28)).exchange == "HNX"
        assert open_symbol_row(s, iid, D(2021, 3, 1)).exchange == "HOSE"
        with pytest.raises(InstrumentError, match="unknown exchange"):
            change_exchange(s, iid, "NYSE", D(2022, 1, 1))
        with pytest.raises(InstrumentError, match="must be after"):
            change_exchange(s, iid, "UPCOM", D(2021, 3, 1))


def test_status_transitions(engine):
    with session_scope(engine) as s:
        iid = make(s)
        assert status_at(s, iid, D(2024, 1, 1)) == "active"
        assert set_status(s, iid, "active", D(2024, 1, 1)) is False                # nothing to resume
        assert set_status(s, iid, "suspended", D(2024, 2, 1)) is True
        assert set_status(s, iid, "suspended", D(2024, 2, 5)) is False              # already suspended
        assert status_at(s, iid, D(2024, 1, 31)) == "active" and status_at(s, iid, D(2024, 2, 1)) == "suspended"
        assert set_status(s, iid, "active", D(2024, 3, 1)) is True
        assert status_at(s, iid, D(2024, 3, 1)) == "active"
        assert set_status(s, iid, "suspended", D(2024, 4, 1)) is True               # can be suspended again
        with pytest.raises(InstrumentError, match="unknown status"):
            set_status(s, iid, "halted", D(2024, 5, 1))


def test_delisting_is_terminal_and_closes_the_ticker(engine):
    with session_scope(engine) as s:
        iid = make(s)
        set_status(s, iid, "suspended", D(2024, 2, 1))
        assert set_status(s, iid, "delisted", D(2024, 6, 1)) is True               # ends the suspension too
        assert status_at(s, iid, D(2024, 5, 31)) == "suspended" and status_at(s, iid, D(2024, 6, 1)) == "delisted"
        assert open_symbol_row(s, iid) is None and find_symbol_row(s, "AAA", D(2024, 6, 1)) is None
        assert set_status(s, iid, "delisted", D(2024, 7, 1)) is False               # idempotent
        for st in ("active", "suspended"):
            with pytest.raises(InstrumentError, match="cannot become"):
                set_status(s, iid, st, D(2025, 1, 1))
