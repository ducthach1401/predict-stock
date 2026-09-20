"""sync_universe / `universe apply`: diffing, idempotency, dry run, change log, CSV validation."""
from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import func, select

from conftest import FLOOR, apply, rows
from predict_stock import universe as uni
from predict_stock.db.models import Instrument, JobRun, Universe, UniverseChangeLog, UniverseMembership
from predict_stock.db.session import session_scope
from predict_stock.universe_sync import SnapshotError, apply_snapshot, read_snapshot_csv

D = date
HEADER = "symbol,previous_symbol,exchange,weight,status,valid_from,note\n"


def count(engine, model):
    with session_scope(engine) as s:
        return s.scalar(select(func.count()).select_from(model))


def log(engine):
    with session_scope(engine) as s:
        return [(r.action, r.symbol, r.effective_date) for r in s.scalars(select(UniverseChangeLog).order_by(UniverseChangeLog.id))]


def write(tmp_path, body, header=HEADER, name="s.csv"):
    p = tmp_path / name
    p.write_text(header + body, encoding="utf-8")
    return p


# ---- idempotency ---------------------------------------------------------------------
def test_reapplying_the_same_snapshot_changes_nothing(engine):
    first = apply(engine, "U1", ["AAA", "BBB"], D(2024, 1, 1))
    assert [i.action for i in first] == ["create_universe", "add", "add"]
    n_mem, n_log, n_inst = count(engine, UniverseMembership), count(engine, UniverseChangeLog), count(engine, Instrument)
    assert apply(engine, "U1", ["AAA", "BBB"], D(2024, 1, 1)) == []                 # same file, same date
    assert apply(engine, "U1", ["BBB", "AAA"], D(2025, 6, 1)) == []                 # same set, later date
    assert (count(engine, UniverseMembership), count(engine, UniverseChangeLog), count(engine, Instrument)) == (n_mem, n_log, n_inst)


def test_idempotent_with_every_kind_of_row(engine):
    snap = [("AAA", {"exchange": "HNX", "weight": "0.5", "note": "n"}), ("BBB", {"weight": "0.5"})]
    apply(engine, "U1", snap, D(2024, 1, 1))
    later = [("AAA", {"exchange": "HOSE", "weight": "0.5", "note": "n"}),      # exchange move
             ("NEW", {"previous_symbol": "BBB", "weight": "0.5", "status": "suspended"})]   # rename + suspension
    apply(engine, "U1", later, D(2024, 8, 1))
    before = (count(engine, UniverseMembership), count(engine, UniverseChangeLog))
    assert apply(engine, "U1", later, D(2024, 8, 1)) == []                          # renames/exchange/status re-applied: no-op
    assert (count(engine, UniverseMembership), count(engine, UniverseChangeLog)) == before


# ---- dry run ---------------------------------------------------------------------------
def test_dry_run_reports_but_writes_nothing(engine):
    apply(engine, "U1", ["AAA", "BBB"], D(2024, 1, 1))
    snapshot = (count(engine, UniverseMembership), count(engine, UniverseChangeLog), count(engine, Instrument))
    with session_scope(engine) as s:
        items = apply_snapshot(s, "U1", rows("BBB", "CCC"), D(2024, 7, 1), source="t", floor=FLOOR)
        assert sorted((i.action, i.symbol) for i in items) == [("add", "CCC"), ("remove", "AAA")]
        s.rollback()                                                                # what --dry-run does
    assert (count(engine, UniverseMembership), count(engine, UniverseChangeLog), count(engine, Instrument)) == snapshot
    with session_scope(engine) as s:
        assert uni.get_members(s, "U1", D(2024, 8, 1)) == ["AAA", "BBB"]


# ---- change log ------------------------------------------------------------------------
def test_change_log_records_each_change_with_provenance(engine):
    apply(engine, "U1", [("AAA", {"weight": "0.5"}), ("BBB", {"weight": "0.5"})], D(2024, 1, 1))
    apply(engine, "U1", [("BBB", {"weight": "0.7"}), ("CCC", {"weight": "0.3"})], D(2024, 7, 1))
    assert log(engine) == [
        ("add", "AAA", D(2024, 1, 1)), ("add", "BBB", D(2024, 1, 1)),
        ("weight_change", "BBB", D(2024, 7, 1)), ("add", "CCC", D(2024, 7, 1)), ("remove", "AAA", D(2024, 7, 1))]
    with session_scope(engine) as s:
        rem = s.scalars(select(UniverseChangeLog).where(UniverseChangeLog.action == "remove")).one()
        assert rem.old_value["valid_from"] == "2024-01-01" and rem.source == "test"
        wc = s.scalars(select(UniverseChangeLog).where(UniverseChangeLog.action == "weight_change")).one()
        assert wc.old_value["weight"].startswith("0.5") and wc.new_value["weight"].startswith("0.7")


def test_rename_exchange_status_are_logged(engine):
    apply(engine, "U1", [("AAA", {"exchange": "HNX"}), "BBB", "CCC"], D(2024, 1, 1))
    apply(engine, "U1", [("AAA", {"exchange": "HOSE"}), ("BBX", {"previous_symbol": "BBB"}), ("CCC", {"status": "suspended"})], D(2024, 4, 1))
    apply(engine, "U1", [("AAA", {"exchange": "HOSE"}), "BBX", ("CCC", {"status": "delisted"})], D(2024, 9, 1))
    actions = [(a, s) for a, s, d in log(engine) if d > D(2024, 1, 1)]
    assert actions == [("exchange_change", "AAA"), ("rename", "BBX"), ("suspend", "CCC"), ("delist", "CCC"), ("remove", "CCC")]


def test_job_run_id_is_stored_in_the_log(engine, cfg):
    from predict_stock.runs import tracked_run
    with tracked_run(engine, "sync_universe", cfg, {}) as (run_id, stats):
        with session_scope(engine) as s:
            apply_snapshot(s, "U1", rows("AAA"), D(2024, 1, 1), source="f", floor=FLOOR, create=True, run_id=run_id)
    with session_scope(engine) as s:
        assert {r.run_id for r in s.scalars(select(UniverseChangeLog))} == {run_id}
        assert s.get(JobRun, run_id).status == "success"


# ---- a new universe is data, not code -----------------------------------------------------
def test_new_universe_from_a_csv_needs_no_code_change(engine, tmp_path):
    body = "".join(f"S{i:03d},,{'HOSE' if i % 2 else 'HNX'},0.01,,,\n" for i in range(100))
    f = write(tmp_path, body)
    with session_scope(engine) as s:                                                # creation is explicit
        with pytest.raises(SnapshotError, match="--create"):
            apply_snapshot(s, "VN100", read_snapshot_csv(f), D(2025, 1, 1), source="f", floor=FLOOR)
    with session_scope(engine) as s:
        items = apply_snapshot(s, "VN100", read_snapshot_csv(f), D(2025, 1, 1), source="f", floor=FLOOR, create=True, name="VN100 (data only)")
        assert sum(i.action == "add" for i in items) == 100
    with session_scope(engine) as s:
        assert len(uni.get_members(s, "VN100", D(2025, 6, 1))) == 100
        assert s.scalars(select(Universe).where(Universe.code == "VN100")).one().name == "VN100 (data only)"
        assert uni.check_integrity(s) == []


def test_universes_do_not_interfere(engine):
    apply(engine, "A", ["AAA", "BBB"], D(2024, 1, 1))
    apply(engine, "B", ["BBB", "CCC"], D(2024, 1, 1))
    apply(engine, "B", ["CCC"], D(2024, 6, 1))
    with session_scope(engine) as s:
        assert uni.get_members(s, "A", D(2024, 7, 1)) == ["AAA", "BBB"]
        assert uni.get_members(s, "B", D(2024, 7, 1)) == ["CCC"]
        assert count(engine, Instrument) == 3                                       # BBB is one instrument shared by both


# ---- guards ------------------------------------------------------------------------------
def test_history_cannot_be_rewritten(engine):
    apply(engine, "U1", ["AAA"], D(2024, 6, 1))
    with pytest.raises(SnapshotError, match="precedes the latest recorded change"):
        apply(engine, "U1", ["AAA", "BBB"], D(2024, 1, 1))


def test_same_day_reapply_with_new_weight_updates_in_place(engine):
    apply(engine, "U1", [("AAA", {"weight": "0.4"})], D(2024, 1, 1))
    apply(engine, "U1", [("AAA", {"weight": "0.6"})], D(2024, 1, 1))
    with session_scope(engine) as s:
        m = s.scalars(select(UniverseMembership)).all()
        assert len(m) == 1 and float(m[0].weight) == 0.6


def test_cannot_remove_on_the_day_it_joined(engine):
    apply(engine, "U1", ["AAA", "BBB"], D(2024, 1, 1))
    with pytest.raises(SnapshotError, match="cannot end membership"):
        apply(engine, "U1", ["AAA"], D(2024, 1, 1))


def test_backfill_valid_from_only_seeds_an_empty_universe(engine):
    seed = [("AAA", {"valid_from": D(2018, 1, 2)}), ("BBB", {"valid_from": D(2020, 12, 23)})]
    apply(engine, "U1", seed, D(2026, 9, 18))
    with session_scope(engine) as s:
        assert uni.get_members(s, "U1", D(2019, 1, 1)) == ["AAA"]
        assert uni.get_members(s, "U1", D(2021, 1, 1)) == ["AAA", "BBB"]
    assert apply(engine, "U1", seed, D(2026, 9, 18)) == []                          # re-applying the seed file is a no-op
    with pytest.raises(SnapshotError, match="only allowed into an empty universe"):
        apply(engine, "U1", ["AAA", "BBB", ("CCC", {"valid_from": D(2019, 1, 1)})], D(2027, 1, 1))


def test_valid_from_after_effective_date_is_rejected(engine):
    with pytest.raises(SnapshotError, match="after the effective date"):
        apply(engine, "U1", [("AAA", {"valid_from": D(2030, 1, 1)})], D(2024, 1, 1))


def test_rename_errors(engine):
    apply(engine, "U1", ["AAA", "BBB"], D(2024, 1, 1))
    with pytest.raises(SnapshotError, match="not a known symbol"):
        apply(engine, "U1", ["AAA", ("XXX", {"previous_symbol": "NOPE"})], D(2025, 1, 1))
    with pytest.raises(SnapshotError, match="already used"):                        # AAA is still another instrument's ticker
        apply(engine, "U1", [("AAA", {"previous_symbol": "BBB"})], D(2025, 1, 1))


def test_delisted_instrument_cannot_come_back_silently(engine):
    apply(engine, "U1", ["AAA", "BBB"], D(2024, 1, 1))
    apply(engine, "U1", ["AAA", ("BBB", {"status": "delisted"})], D(2024, 6, 1))
    assert apply(engine, "U1", ["AAA", ("BBB", {"status": "delisted"})], D(2024, 6, 1)) == []   # delisting is idempotent
    with pytest.raises(SnapshotError, match="re-using a ended ticker"):
        apply(engine, "U1", ["AAA", "BBB"], D(2025, 1, 1))


def test_suspended_then_active_again_is_a_resume(engine):
    apply(engine, "U1", ["AAA"], D(2024, 1, 1))
    apply(engine, "U1", [("AAA", {"status": "suspended"})], D(2024, 3, 1))
    plan = apply(engine, "U1", ["AAA"], D(2024, 4, 1))
    assert [i.action for i in plan] == ["resume"]


# ---- CSV parsing -------------------------------------------------------------------------
def test_csv_full_row_and_defaults(tmp_path):
    f = write(tmp_path, "# provenance\nvcb,,hose,0.25,,2018-01-02,big bank\nfpt\n", header="# header note\n" + HEADER)
    a, b = read_snapshot_csv(f)
    assert (a.symbol, a.exchange, str(a.weight), a.valid_from, a.note) == ("VCB", "HOSE", "0.25", D(2018, 1, 2), "big bank")
    assert (b.symbol, b.exchange, b.weight, b.status, b.previous_symbol) == ("FPT", None, None, None, None)


def test_csv_only_symbol_column_is_required(tmp_path):
    assert [r.symbol for r in read_snapshot_csv(write(tmp_path, "AAA\nBBB\n", header="symbol\n"))] == ["AAA", "BBB"]


@pytest.mark.parametrize("body,msg", [
    ("AAA\nAAA\n", "duplicate symbol"),
    ("AAA,,NYSE\n", "unknown exchange"),
    ("AAA,,,,frozen\n", "unknown status"),
    ("AAA,,,1.5\n", "outside"),
    ("AAA,,,abc\n", "bad weight"),
    ("AAA,,,,,31-12-2024\n", "bad weight/valid_from"),
    (",,\n", "empty symbol"),
    ("AAA,AAA\n", "previous_symbol equals symbol"),
])
def test_csv_validation_errors(tmp_path, body, msg):
    with pytest.raises(SnapshotError, match=msg):
        read_snapshot_csv(write(tmp_path, body))


def test_csv_missing_column_and_empty_file(tmp_path):
    with pytest.raises(SnapshotError, match="missing required column"):
        read_snapshot_csv(write(tmp_path, "AAA\n", header="ticker\n"))
    with pytest.raises(SnapshotError, match="no data rows"):
        read_snapshot_csv(write(tmp_path, "", header="symbol\n"))
