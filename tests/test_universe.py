"""Point-in-time queries: members added/removed mid-period, renames, exchange moves,
suspension and delisting, training vs trading universes."""
from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import text

from conftest import apply
from predict_stock import universe as uni
from predict_stock.db.session import session_scope
from predict_stock.instruments import rename_symbol

D = date


def members(engine, code, d, **kw):
    with session_scope(engine) as s:
        return uni.get_members(s, code, d, **kw)


def test_members_added_and_removed_mid_period(engine):
    apply(engine, "U1", ["AAA", "BBB"], D(2023, 1, 1))
    apply(engine, "U1", ["BBB", "CCC"], D(2024, 7, 1))       # AAA leaves, CCC joins
    assert members(engine, "U1", D(2022, 12, 31)) == []
    assert members(engine, "U1", D(2023, 1, 1)) == ["AAA", "BBB"]              # valid_from is inclusive
    assert members(engine, "U1", D(2024, 6, 30)) == ["AAA", "BBB"]
    assert members(engine, "U1", D(2024, 7, 1)) == ["BBB", "CCC"]              # valid_to is exclusive
    assert members(engine, "U1", D(2030, 1, 1)) == ["BBB", "CCC"]
    assert members(engine, "OTHER", D(2024, 1, 1)) == []


def test_symbol_can_leave_and_rejoin(engine):
    apply(engine, "U1", ["AAA", "BBB"], D(2020, 1, 1))
    apply(engine, "U1", ["BBB"], D(2021, 1, 1))
    apply(engine, "U1", ["AAA", "BBB"], D(2022, 1, 1))
    assert members(engine, "U1", D(2020, 6, 1)) == ["AAA", "BBB"]
    assert members(engine, "U1", D(2021, 6, 1)) == ["BBB"]
    assert members(engine, "U1", D(2022, 6, 1)) == ["AAA", "BBB"]


def test_rename_keeps_membership_and_identity(engine):
    apply(engine, "U1", ["AAA", "BBB"], D(2023, 1, 1))
    apply(engine, "U1", ["BBB", ("AAX", {"previous_symbol": "AAA"})], D(2024, 1, 1))
    with session_scope(engine) as s:
        before = {m.symbol: m.instrument_id for m in uni.get_member_records(s, "U1", D(2023, 6, 1))}
        after = {m.symbol: m.instrument_id for m in uni.get_member_records(s, "U1", D(2024, 1, 1))}
        assert sorted(before) == ["AAA", "BBB"] and sorted(after) == ["AAX", "BBB"]
        assert before["AAA"] == after["AAX"]                                     # same instrument
        n = s.execute(text("SELECT COUNT(*) FROM universe_membership")).scalar()
        assert n == 2                                                             # no add/remove was needed
        assert uni.check_integrity(s) == []
    assert members(engine, "U1", D(2023, 12, 31)) == ["AAA", "BBB"]              # before the rename: old ticker


def test_exchange_move_is_point_in_time(engine):
    apply(engine, "U1", [("AAA", {"exchange": "HNX"})], D(2019, 1, 1))
    apply(engine, "U1", [("AAA", {"exchange": "HOSE"})], D(2021, 3, 1))
    with session_scope(engine) as s:
        ex = lambda d: uni.get_member_records(s, "U1", d)[0].exchange
        assert ex(D(2019, 1, 1)) == "HNX" and ex(D(2021, 2, 28)) == "HNX"
        assert ex(D(2021, 3, 1)) == "HOSE" and ex(D(2026, 1, 1)) == "HOSE"
        assert s.execute(text("SELECT COUNT(*) FROM universe_membership")).scalar() == 1   # still one continuous membership
        assert uni.check_integrity(s) == []


def test_suspension_keeps_membership_but_not_tradability(engine):
    apply(engine, "U1", ["AAA", "BBB"], D(2024, 1, 1))
    apply(engine, "U1", ["AAA", ("BBB", {"status": "suspended"})], D(2024, 5, 1))
    apply(engine, "U1", ["AAA", "BBB"], D(2024, 6, 1))                          # resumed
    assert members(engine, "U1", D(2024, 5, 15)) == ["AAA", "BBB"]              # still a member
    assert members(engine, "U1", D(2024, 5, 15), tradable_only=True) == ["AAA"]
    assert members(engine, "U1", D(2024, 4, 30), tradable_only=True) == ["AAA", "BBB"]
    assert members(engine, "U1", D(2024, 6, 1), tradable_only=True) == ["AAA", "BBB"]   # suspension is half-open too


def test_delisting_closes_membership_and_ticker(engine):
    apply(engine, "U1", ["AAA", "BBB"], D(2023, 1, 1))
    apply(engine, "U1", ["AAA", ("BBB", {"status": "delisted"})], D(2024, 9, 1))
    assert members(engine, "U1", D(2024, 8, 31)) == ["AAA", "BBB"]
    assert members(engine, "U1", D(2024, 9, 1)) == ["AAA"]
    with session_scope(engine) as s:
        # it was a member during the period, so data ingestion must still cover it
        got = {m.symbol for m in uni.get_instruments_between(s, "U1", D(2023, 6, 1), D(2026, 1, 1))}
        assert got == {"AAA", "BBB"}
        assert {m.symbol for m in uni.get_instruments_between(s, "U1", D(2024, 10, 1), D(2026, 1, 1))} == {"AAA"}
        assert uni.check_integrity(s) == []


def test_instruments_between_uses_symbol_valid_at_end(engine):
    apply(engine, "U1", ["AAA"], D(2023, 1, 1))
    with session_scope(engine) as s:
        rename_symbol(s, "AAA", "AAX", D(2024, 1, 1))
        assert [m.symbol for m in uni.get_instruments_between(s, "U1", D(2023, 1, 1), D(2023, 12, 31))] == ["AAA"]
        assert [m.symbol for m in uni.get_instruments_between(s, "U1", D(2023, 1, 1), D(2024, 6, 1))] == ["AAX"]


def test_universe_size_is_not_assumed(engine):
    apply(engine, "BIG", [f"S{i:03d}" for i in range(100)], D(2024, 1, 1))
    assert len(members(engine, "BIG", D(2024, 6, 1))) == 100


def test_weights_are_point_in_time(engine):
    apply(engine, "U1", [("AAA", {"weight": "0.6"}), ("BBB", {"weight": "0.4"})], D(2024, 1, 1))
    apply(engine, "U1", [("AAA", {"weight": "0.5"}), ("BBB", {"weight": "0.5"})], D(2024, 7, 1))
    with session_scope(engine) as s:
        w = lambda d: {m.symbol: float(m.weight) for m in uni.get_member_records(s, "U1", d)}
        assert w(D(2024, 3, 1)) == {"AAA": 0.6, "BBB": 0.4}
        assert w(D(2024, 7, 1)) == {"AAA": 0.5, "BBB": 0.5}


def test_training_wider_than_trading(engine, cfg):
    cfg2 = cfg.model_copy(update={"universe": cfg.universe.model_copy(update={"training_code": "TRAIN", "trading_code": "TRADE"})})
    apply(engine, "TRAIN", [f"S{i}" for i in range(8)], D(2024, 1, 1))
    apply(engine, "TRADE", ["S1", "S2", "S3"], D(2024, 1, 1))
    apply(engine, "TRADE", ["S1", ("S2", {"status": "suspended"}), "S3"], D(2024, 3, 1))
    with session_scope(engine) as s:
        assert len(uni.training_members(s, cfg2, D(2024, 2, 1))) == 8
        assert [m.symbol for m in uni.trading_members(s, cfg2, D(2024, 2, 1))] == ["S1", "S2", "S3"]
        assert [m.symbol for m in uni.trading_members(s, cfg2, D(2024, 3, 1))] == ["S1", "S3"]   # suspended: no recommendations
        assert uni.trading_outside_training(s, cfg2, D(2024, 2, 1)) == []


def test_trading_outside_training_is_reported(engine, cfg):
    cfg2 = cfg.model_copy(update={"universe": cfg.universe.model_copy(update={"training_code": "TRAIN", "trading_code": "TRADE"})})
    apply(engine, "TRAIN", ["S1", "S2"], D(2024, 1, 1))
    apply(engine, "TRADE", ["S1", "S9"], D(2024, 1, 1))
    with session_scope(engine) as s:
        assert uni.trading_outside_training(s, cfg2, D(2024, 2, 1)) == ["S9"]


def test_integrity_check_detects_overlaps_and_uncovered_membership(engine):
    apply(engine, "U1", ["AAA"], D(2024, 1, 1))
    with engine.begin() as c:   # DML the database itself cannot forbid (no exclusion constraints in MySQL)
        c.execute(text("INSERT INTO universe_membership (universe_id, instrument_id, valid_from, valid_to, source) "
                       "SELECT universe_id, instrument_id, '2024-06-01', '2024-09-01', 'bad' FROM universe_membership LIMIT 1"))
        c.execute(text("INSERT INTO universe_membership (universe_id, instrument_id, valid_from, valid_to, source) "
                       "SELECT universe_id, instrument_id, '1990-01-01', '1991-01-01', 'bad' FROM universe_membership LIMIT 1"))
    with session_scope(engine) as s:
        problems = uni.check_integrity(s)
    assert any("membership:" in p and "overlaps" in p for p in problems)
    assert any("no symbol row covers" in p for p in problems)
