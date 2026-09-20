"""`make data` = the whole pipeline. Running it twice in a row must not create duplicate rows."""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest
from sqlalchemy import text

import predict_stock.cli as cli
from conftest import FakeClient, apply, make_bars
from predict_stock.config import load_config
from predict_stock.pipeline import run_data_pipeline
from predict_stock.db.session import session_scope

START, END = date(2026, 1, 5), date(2026, 9, 18)
DATA_TABLES = ["instruments", "instrument_symbol_history", "universes", "universe_membership", "price_bar", "price_bar_revisions",
               "price_bar_intraday", "trading_calendar", "corporate_actions", "adjustment_factors", "data_quality_findings",
               "data_quality_known_issues", "alerts"]
HEADER = "key,check,symbol,date_from,date_to,treatment,explanation,evidence\n"


def cfg_small(cfg, **ingest):
    return cfg.model_copy(update={"ingest": cfg.ingest.model_copy(update={"benchmark_symbols": [], "min_sessions": 20, **ingest}),
                                  "universe": cfg.universe.model_copy(update={"training_code": "U1", "trading_code": "U1"})})


def series(defect=True):
    bad = make_bars(START, 60)
    if defect:
        bad[20] = replace(bad[20], open=bad[20].high + 80)                       # an OHLC error
        bad[30] = replace(bad[30], volume=0)                                      # a warning
    return {"BAD": bad, "OK": make_bars(START, 60, base=30000), "SHORT": make_bars(START, 8, base=9000)}


def table_state(engine):
    """Full content of every data table (excluding the job/run bookkeeping)."""
    out = {}
    with engine.connect() as c:
        for t in DATA_TABLES:
            # findings carry "last seen" bookkeeping (run id / time) that moves on every run by design; the
            # finding itself (what, where, status) must not
            cols = "*" if t != "data_quality_findings" else (
                "id, instrument_id, check_name, severity, trade_date, subject_key, detail, status, known_issue_id, "
                "first_seen_run_id, first_seen_at, resolved_at")
            rows = c.execute(text(f"SELECT {cols} FROM {t}")).all()
            out[t] = sorted(tuple(map(str, r)) for r in rows)
    return out


def pipeline(engine, cfg, now, data, tmp_path, known=None):
    kpath = tmp_path / "known.csv"
    if known is not None:
        kpath.write_text(HEADER + known)
    return run_data_pipeline(engine, FakeClient(data), cfg, ["U1"], START, END, now=now, report_path=tmp_path / "dq.md",
                             known_issues_path=kpath)


KNOWN = 'k1,ohlc_inconsistent,BAD,,,keep_flagged,"planted test defect, root cause known",planted\n'


def test_data_pipeline_twice_creates_no_duplicates(engine, cfg, after_close, tmp_path):
    c = cfg_small(cfg)
    apply(engine, "U1", ["BAD", "OK", "SHORT"], date(2020, 1, 1))
    first = pipeline(engine, c, after_close, series(), tmp_path, KNOWN)
    assert first["ok"] is True and first["errors"] == []
    assert first["backfill"]["new_instruments"] == ["BAD", "OK", "SHORT"] and set(first["backfill"]["blocked"]) == {"SHORT"}
    state1, report1 = table_state(engine), (tmp_path / "dq.md").read_text()
    assert len(state1["price_bar"]) == 60 + 60 + 8

    second = pipeline(engine, c, after_close, series(), tmp_path, KNOWN)
    assert second["ok"] is True and second["backfill"]["new_instruments"] == []              # nothing new to backfill
    assert second["ingest"]["totals"]["inserted"] == 0 and second["ingest"]["totals"]["updated"] == 0
    assert table_state(engine) == state1                                                       # every data table is identical
    assert (tmp_path / "dq.md").read_text() == report1                                         # ...and so is the report
    with engine.connect() as cn:
        assert cn.execute(text("SELECT COUNT(*) FROM (SELECT instrument_id, trade_date FROM price_bar GROUP BY 1, 2 HAVING COUNT(*) > 1) d")).scalar() == 0
        assert cn.execute(text("SELECT COUNT(*) FROM (SELECT instrument_id, check_name, subject_key FROM data_quality_findings GROUP BY 1, 2, 3 HAVING COUNT(*) > 1) d")).scalar() == 0


def test_pipeline_reports_an_unexplained_error(engine, cfg, after_close, tmp_path):
    c = cfg_small(cfg)
    apply(engine, "U1", ["BAD", "OK", "SHORT"], date(2020, 1, 1))
    res = pipeline(engine, c, after_close, series(), tmp_path, known=None)                    # no known issues at all
    assert res["ok"] is False
    assert any("ohlc_inconsistent" in e and "BAD" in e for e in res["quality"]["unexplained_errors"])
    assert "## Verdict: **FAIL**" in (tmp_path / "dq.md").read_text()


def test_pipeline_continues_and_reports_when_a_fetch_fails(engine, cfg, after_close, tmp_path):
    c = cfg_small(cfg)
    apply(engine, "U1", ["BAD", "OK", "GONE"], date(2020, 1, 1))
    data = series()
    del data["SHORT"]
    res = pipeline(engine, c, after_close, data, tmp_path, KNOWN)
    assert res["ok"] is False and any(e.startswith("backfill") and "GONE" in e for e in res["errors"])
    assert (tmp_path / "dq.md").exists()                                                       # the report is still produced
    with engine.connect() as cn:
        assert cn.execute(text("SELECT COUNT(*) FROM price_bar")).scalar() == 120             # BAD and OK were kept


def test_pipeline_adds_a_new_member_on_the_next_run(engine, cfg, after_close, tmp_path):
    c = cfg_small(cfg)
    apply(engine, "U1", ["BAD", "OK"], date(2020, 1, 1))
    pipeline(engine, c, after_close, series(), tmp_path, KNOWN)
    apply(engine, "U1", ["BAD", "OK", "SHORT"], date(2026, 9, 1))
    res = pipeline(engine, c, after_close, series(), tmp_path, KNOWN)
    assert res["backfill"]["new_instruments"] == ["SHORT"] and "SHORT" in res["backfill"]["blocked"]
    assert "SHORT" in (tmp_path / "dq.md").read_text()


# ---- the CLI entry point behind `make data` -----------------------------------------------------------------
def test_cli_data_exit_codes(engine, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MYSQL_DATABASE", "predict_stock_test")
    real = load_config()
    cfg = cfg_small(real, history_start="2026-01-05")
    cfg = cfg.model_copy(update={"quality": cfg.quality.model_copy(update={"report_path": str(tmp_path / "r.md"), "known_issues_path": str(tmp_path / "k.csv")})})
    monkeypatch.setattr(cli, "load_config", lambda _p=None: cfg)
    monkeypatch.setattr(cli, "DnseClient", lambda _c: FakeClient(series()))
    apply(engine, "U1", ["BAD", "OK", "SHORT"], date(2020, 1, 1))
    assert cli.main(["data"]) == 1                                                             # an unexplained error -> exit code 1
    assert "unexplained errors: 1" in capsys.readouterr().out
    (tmp_path / "k.csv").write_text(HEADER + KNOWN)
    assert cli.main(["data"]) == 0                                                             # explained -> success
    out = capsys.readouterr().out
    assert "unexplained errors: 0" in out and (tmp_path / "r.md").exists()
    assert cli.main(["data"]) == 0                                                             # and again: still success, nothing to do
