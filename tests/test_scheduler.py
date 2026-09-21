"""The container's scheduler: which jobs are due when (same times as the cron entries), catch-up after downtime, retries, state."""
from __future__ import annotations

import subprocess
from datetime import datetime

import pytest

from predict_stock.paper import scheduler as SCH


def at(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm)


MON, SAT, SUN = (2026, 9, 21), (2026, 9, 26), (2026, 9, 27)


def test_cron_day_of_week_specs_become_python_weekdays():
    assert SCH.weekdays("1-5") == {0, 1, 2, 3, 4}
    assert SCH.weekdays("0") == {6} == SCH.weekdays("7")
    assert SCH.weekdays("1,3,5") == {0, 2, 4} and SCH.weekdays("*") == set(range(7))


def test_the_daily_job_is_due_from_the_configured_time_on_weekdays_only(cfg):
    assert SCH.due(cfg, {}, at(*MON, 16, 29)) == []
    assert SCH.due(cfg, {}, at(*MON, 16, 30)) == ["daily"]
    assert SCH.due(cfg, {}, at(*MON, 17, 30)) == ["daily", "backup"]
    assert SCH.due(cfg, {}, at(*SAT, 16, 45)) == []                                      # no daily job on Saturday
    assert SCH.due(cfg, {}, at(*SAT, 17, 45)) == ["backup"]                              # the backup runs every day
    assert SCH.due(cfg, {}, at(*SUN, 3, 5)) == ["restore_test"]


def test_a_container_that_was_down_catches_up_and_a_done_job_does_not_run_twice(cfg):
    late = at(*MON, 21, 0)
    assert SCH.due(cfg, {}, late) == ["daily", "backup"]
    done = {"daily": {"date": "2026-09-21", "status": "ok", "tries": 1, "last": "2026-09-21T21:00:00"}}
    assert SCH.due(cfg, done, late) == ["backup"]
    assert SCH.due(cfg, done, at(2026, 9, 22, 16, 31)) == ["daily"]                      # tomorrow it is due again


def test_a_failed_job_is_retried_after_the_pause_and_only_a_few_times(cfg):
    failed = lambda n, last: {"daily": {"date": "2026-09-21", "status": "failed", "tries": n, "last": last}}
    assert SCH.due(cfg, failed(1, "2026-09-21T16:30:00"), at(*MON, 16, 45)) == []        # too soon
    assert SCH.due(cfg, failed(1, "2026-09-21T16:30:00"), at(*MON, 17, 0)) == ["daily"]
    assert SCH.due(cfg, failed(SCH.MAX_TRIES, "2026-09-21T16:30:00"), at(*MON, 20, 0)) == ["backup"]   # the daily job is given up for today; the backup is a different job


def test_tick_runs_what_is_due_records_the_outcome_and_logs_the_output(cfg, tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, cwd, stdout, stderr):
        calls.append(cmd[-2:])
        stdout.write("job output\n")
        return subprocess.CompletedProcess(cmd, 1 if cmd[-1] == "run" else 0)            # the daily job fails, the rest succeed
    monkeypatch.setattr(SCH.subprocess, "run", fake_run)
    now = datetime(2026, 9, 21, 17, 45, tzinfo=SCH.TZ)
    assert SCH.tick(cfg, now, root=tmp_path) == {"daily": 1, "backup": 0}
    state = SCH.load_state(tmp_path)
    assert state["daily"]["status"] == "failed" and state["daily"]["tries"] == 1 and state["backup"]["status"] == "ok"
    assert "job output" in (tmp_path / "logs" / "paper_daily.log").read_text() and "daily (try 1)" in (tmp_path / "logs" / "paper_daily.log").read_text()
    assert SCH.tick(cfg, now, root=tmp_path) == {}                                       # nothing more right away: the retry pause and the finished backup
    assert SCH.tick(cfg, datetime(2026, 9, 21, 18, 30, tzinfo=SCH.TZ), root=tmp_path) == {"daily": 1}
    assert SCH.load_state(tmp_path)["daily"]["tries"] == 2
