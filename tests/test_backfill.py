"""Backfill of instruments without history, the minimum-sessions gate, and the auto-backfill after `universe apply`."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from sqlalchemy import func, select

import predict_stock.cli as cli
from conftest import FakeClient, apply, make_bars
from predict_stock.config import load_config
from predict_stock.data.backfill import (
    backfill, find_new_targets, ready_members, readiness, sessions_available, sync_readiness_alerts,
)
from predict_stock.data.dnse_client import DnseError
from predict_stock.data.ingest import ingest_universe
from predict_stock.db.models import Alert, DataIngestRun, JobRun, PriceBar
from predict_stock.db.repo import find_symbol_row
from predict_stock.db.session import session_scope
from predict_stock.universe import training_members

START, END = date(2026, 1, 5), date(2026, 9, 18)


@pytest.fixture
def cfg20(cfg):
    """min_sessions = 20, no benchmarks, training = trading = U1."""
    return cfg.model_copy(update={
        "ingest": cfg.ingest.model_copy(update={"benchmark_symbols": [], "min_sessions": 20}),
        "universe": cfg.universe.model_copy(update={"training_code": "U1", "trading_code": "U1"}),
    })


def run(engine, cfg20, series, now, codes=("U1",)):
    return backfill(engine, FakeClient(series), cfg20, list(codes), START, END, now=now)


def test_only_instruments_without_bars_are_backfilled(engine, cfg20, after_close):
    apply(engine, "U1", ["OLD", "NEW"], date(2020, 1, 1))
    ingest_universe(engine, FakeClient({"OLD": make_bars(START, 30), "NEW": make_bars(START, 30)}), cfg20.model_copy(update={
        "ingest": cfg20.ingest}), ["U1"], START, END, now=after_close)
    apply(engine, "U1", ["OLD", "NEW", "LATER"], date(2026, 6, 1))            # a member joins later
    with session_scope(engine) as s:
        assert [t.symbol for t in find_new_targets(s, cfg20, ["U1"], START, END)] == ["LATER"]
    client = FakeClient({"OLD": make_bars(START, 30), "NEW": make_bars(START, 30), "LATER": make_bars(START, 30)})
    stats = backfill(engine, client, cfg20, ["U1"], START, END, now=after_close)
    assert stats["new_instruments"] == ["LATER"] and stats["blocked"] == {}
    assert [c[0] for c in client.calls] == ["LATER"]                          # nothing else was fetched
    with session_scope(engine) as s:
        assert sessions_available(s, find_symbol_row(s, "LATER").instrument_id, END)[0] == 30


def test_backfill_measures_and_records_duration(engine, cfg20, after_close):
    apply(engine, "U1", ["A"], date(2020, 1, 1))
    stats = run(engine, cfg20, {"A": make_bars(START, 30)}, after_close)
    assert set(stats["duration_ms"]) == {"A"} and isinstance(stats["duration_ms"]["A"], int) and stats["duration_ms"]["A"] >= 0
    with session_scope(engine) as s:
        row = s.scalars(select(DataIngestRun)).one()
        assert row.duration_ms == stats["duration_ms"]["A"] and row.mode == "full" and row.status == "ok" and row.inserted == 30
        job = s.scalars(select(JobRun).where(JobRun.job_name == "backfill")).one()
        assert job.status == "success" and job.stats["new_instruments"] == ["A"]


def test_too_short_history_is_blocked_with_a_reason(engine, cfg20, after_close):
    apply(engine, "U1", ["LONG", "SHORT"], date(2020, 1, 1))
    stats = run(engine, cfg20, {"LONG": make_bars(START, 30), "SHORT": make_bars(START, 8)}, after_close)
    assert set(stats["blocked"]) == {"SHORT"} and "only 8 sessions" in stats["blocked"]["SHORT"] and "min_sessions is 20" in stats["blocked"]["SHORT"]
    with session_scope(engine) as s:
        rows = {r.instrument_id: r for r in s.scalars(select(DataIngestRun))}
        blocked = rows[find_symbol_row(s, "SHORT").instrument_id]
        assert blocked.status == "blocked" and "only 8 sessions" in blocked.error and blocked.inserted == 8     # data is kept
        assert rows[find_symbol_row(s, "LONG").instrument_id].status == "ok"
        alert = s.scalars(select(Alert)).one()
        assert alert.category == "insufficient_history" and alert.severity == "warn" and "only 8 sessions" in alert.message and alert.acknowledged_at is None
        assert [m.symbol for m in ready_members(s, cfg20, END)] == ["LONG"]                 # the gate for training
        assert len(training_members(s, cfg20, END)) == 2


def test_blocking_is_point_in_time(engine, cfg20, after_close):
    apply(engine, "U1", ["A"], date(2020, 1, 1))
    run(engine, cfg20, {"A": make_bars(START, 30)}, after_close)
    days = [b.trade_date for b in make_bars(START, 30)]
    with session_scope(engine) as s:
        members = training_members(s, cfg20, END)
        assert readiness(s, cfg20, members, days[-1]) == {}
        early = readiness(s, cfg20, members, days[9])                                       # only 10 sessions existed on day 10
        assert "only 10 sessions" in early["A"]


def test_alert_is_not_duplicated_and_clears_when_history_grows(engine, cfg20, after_close):
    apply(engine, "U1", ["SHORT"], date(2020, 1, 1))
    series = {"SHORT": make_bars(START, 8)}
    run(engine, cfg20, series, after_close)
    with session_scope(engine) as s:
        members = training_members(s, cfg20, END)
        sync_readiness_alerts(s, cfg20, members, END, None)
        sync_readiness_alerts(s, cfg20, members, END, None)
        assert s.scalar(select(func.count()).select_from(Alert)) == 1                      # one open alert, updated in place
    series["SHORT"] = make_bars(START, 30)                                                  # more history arrives
    ingest_universe(engine, FakeClient(series), cfg20, ["U1"], START, END, now=after_close)
    with session_scope(engine) as s:
        members = training_members(s, cfg20, END)
        assert sync_readiness_alerts(s, cfg20, members, END, None) == {}
        assert s.scalars(select(Alert)).one().acknowledged_at is not None                   # acknowledged automatically
        assert [m.symbol for m in ready_members(s, cfg20, END)] == ["SHORT"]


def test_rerunning_backfill_finds_nothing_new(engine, cfg20, after_close):
    apply(engine, "U1", ["A", "B"], date(2020, 1, 1))
    series = {"A": make_bars(START, 30), "B": make_bars(START, 8)}
    run(engine, cfg20, series, after_close)
    with session_scope(engine) as s:
        n = s.scalar(select(func.count()).select_from(PriceBar))
    client = FakeClient(series)
    stats = backfill(engine, client, cfg20, ["U1"], START, END, now=after_close)
    assert stats["new_instruments"] == [] and client.calls == []
    with session_scope(engine) as s:
        assert s.scalar(select(func.count()).select_from(PriceBar)) == n
        assert s.scalar(select(func.count()).select_from(Alert)) == 1                       # B's alert still exists, not duplicated


def test_index_benchmarks_are_backfilled_but_never_blocked(engine, cfg20, after_close):
    cfg_b = cfg20.model_copy(update={"ingest": cfg20.ingest.model_copy(update={"benchmark_symbols": ["IDX"]})})
    apply(engine, "U1", ["A"], date(2020, 1, 1))
    stats = backfill(engine, FakeClient({"A": make_bars(START, 30), "IDX": make_bars(START, 5, base=1200, step=1)}), cfg_b, ["U1"], START, END, now=after_close)
    assert set(stats["new_instruments"]) == {"A", "IDX"} and stats["blocked"] == {}


def test_fetch_failure_is_recorded_and_others_are_kept(engine, cfg20, after_close):
    apply(engine, "U1", ["A", "GONE"], date(2020, 1, 1))
    with pytest.raises(DnseError, match="GONE"):
        run(engine, cfg20, {"A": make_bars(START, 30)}, after_close)
    with session_scope(engine) as s:
        assert s.scalar(select(func.count()).select_from(PriceBar)) == 30
        rows = {r.status: r for r in s.scalars(select(DataIngestRun))}
        assert set(rows) == {"ok", "failed"} and "GONE" in rows["failed"].error
        assert s.scalars(select(JobRun).where(JobRun.job_name == "backfill")).one().status == "failed"


# ---- automatic backfill after `universe apply` (CLI wiring, against the test database) ---------------------
@pytest.fixture
def cli_env(engine, monkeypatch):
    monkeypatch.setenv("MYSQL_DATABASE", "predict_stock_test")             # the CLI builds its engine from the environment
    return engine


def cli_run(monkeypatch, series, tmp_path, argv, min_sessions=20):
    created = {}

    def fake_client(_cfg):
        created["client"] = FakeClient(series)
        return created["client"]

    monkeypatch.setattr(cli, "DnseClient", fake_client)
    real = load_config()
    cfg = real.model_copy(update={"ingest": real.ingest.model_copy(update={"benchmark_symbols": [], "min_sessions": min_sessions, "history_start": "2026-01-05"})})
    monkeypatch.setattr(cli, "load_config", lambda _p=None: cfg)
    return cli.main(argv), created.get("client")


def snapshot_csv(tmp_path, *symbols):
    p = tmp_path / "s.csv"
    p.write_text("symbol\n" + "".join(f"{s}\n" for s in symbols))
    return str(p)


def test_universe_apply_backfills_new_members_automatically(cli_env, monkeypatch, tmp_path, capsys):
    series = {"AAA": make_bars(START, 30), "BBB": make_bars(START, 8)}
    code, client = cli_run(monkeypatch, series, tmp_path, ["universe", "apply", "--universe", "U1", "--file", snapshot_csv(tmp_path, "AAA", "BBB"),
                                                            "--effective-date", "2026-09-01", "--create"])
    out = capsys.readouterr().out
    assert code == 0 and "backfilling 2 new member(s)" in out and "blocked" in out and "BBB" in out
    with session_scope(cli_env) as s:
        assert s.scalar(select(func.count()).select_from(PriceBar)) == 38
    # a second apply of the same file adds nobody, so nothing is fetched
    code, client = cli_run(monkeypatch, series, tmp_path, ["universe", "apply", "--universe", "U1", "--file", snapshot_csv(tmp_path, "AAA", "BBB"),
                                                            "--effective-date", "2026-09-01"])
    assert code == 0 and client is None


def test_no_backfill_flag_and_dry_run_do_not_fetch(cli_env, monkeypatch, tmp_path):
    series = {"AAA": make_bars(START, 30)}
    for extra in (["--no-backfill"], ["--dry-run"]):
        code, client = cli_run(monkeypatch, series, tmp_path, ["universe", "apply", "--universe", "U1", "--file", snapshot_csv(tmp_path, "AAA"),
                                                                "--effective-date", "2026-09-01", "--create", *extra])
        assert code == 0 and client is None
    with session_scope(cli_env) as s:
        assert s.scalar(select(func.count()).select_from(PriceBar)) == 0
