"""Findings persistence, known-issue explanations, the unexplained-error gate and the Markdown report."""
from __future__ import annotations

import os
import time
from dataclasses import replace
from datetime import date

import pytest
from sqlalchemy import func, select, text

from conftest import FakeClient, apply, make_bars
from predict_stock.data.ingest import ingest_universe
from predict_stock.data.quality import run_quality_checks
from predict_stock.data.quality_report import build_report, write_report
from predict_stock.data.quality_store import (
    KnownIssueError, load_known_issues, store_findings, unexplained_errors,
)
from predict_stock.db.models import DataQualityFinding, DataQualityKnownIssue
from predict_stock.db.repo import find_symbol_row
from predict_stock.db.session import session_scope

START, END = date(2026, 1, 5), date(2026, 9, 18)
DAYS = [b.trade_date for b in make_bars(START, 60)]
HEADER = "key,check,symbol,date_from,date_to,treatment,explanation,evidence\n"


def setup_data(engine, cfg, now, defects=None):
    """Two stocks; BAD gets the given defects (fn(bars) -> bars)."""
    cfg0 = cfg.model_copy(update={"ingest": cfg.ingest.model_copy(update={"benchmark_symbols": []})})
    bad = make_bars(START, 60)
    if defects:
        bad = defects(bad)
    apply(engine, "U1", ["BAD", "OK"], date(2020, 1, 1))
    ingest_universe(engine, FakeClient({"BAD": bad, "OK": make_bars(START, 60, base=30000)}), cfg0, ["U1"], START, END, now=now)
    with session_scope(engine) as s:
        return cfg0, [(find_symbol_row(s, x).instrument_id, x) for x in ("BAD", "OK")]


def defect_open(bars):
    x = bars[20]
    bars[20] = replace(x, open=x.high + 80)                     # ohlc_inconsistent on DAYS[20]
    y = bars[30]
    bars[30] = replace(y, volume=0)                             # zero_volume warning on DAYS[30]
    return bars


def refresh(engine, cfg0, pairs, run_id=None):
    with session_scope(engine) as s:
        issues = run_quality_checks(s, pairs, cfg0)
        return store_findings(s, issues, [i for i, _ in pairs], run_id)


def write_issues(tmp_path, body, name="k.csv"):
    p = tmp_path / name
    p.write_text(HEADER + body, encoding="utf-8")
    return p


def statuses(engine):
    with session_scope(engine) as s:
        return {(f.check_name, f.subject_key): (f.status, f.severity) for f in s.scalars(select(DataQualityFinding))}


# ---- persistence ------------------------------------------------------------------------------
def test_findings_are_stored_and_rerun_is_idempotent(engine, cfg, after_close):
    cfg0, pairs = setup_data(engine, cfg, after_close, defect_open)
    first = refresh(engine, cfg0, pairs)
    assert first["new"] == 2 and first["open"] == 2
    st = statuses(engine)
    assert st[("ohlc_inconsistent", DAYS[20].isoformat())] == ("open", "error")
    assert st[("zero_volume", DAYS[30].isoformat())] == ("open", "warn")
    again = refresh(engine, cfg0, pairs)
    assert again["new"] == 0 and again["open"] == 2
    with session_scope(engine) as s:
        assert s.scalar(select(func.count()).select_from(DataQualityFinding)) == 2                      # no duplicates
        f = s.scalars(select(DataQualityFinding).where(DataQualityFinding.check_name == "zero_volume")).one()
        assert f.first_seen_at <= f.last_seen_at and f.trade_date == DAYS[30] and f.subject_key == DAYS[30].isoformat()


def test_a_fixed_problem_becomes_resolved_and_can_reopen(engine, cfg, after_close):
    cfg0, pairs = setup_data(engine, cfg, after_close, defect_open)
    refresh(engine, cfg0, pairs)
    bad_id = pairs[0][0]
    with engine.begin() as c:                                    # the vendor corrects the bar
        c.execute(text("UPDATE price_bar SET open = low WHERE instrument_id = :i AND trade_date = :d"), {"i": bad_id, "d": DAYS[20]})
    out = refresh(engine, cfg0, pairs)
    assert out["resolved"] == 1 and out["open"] == 1
    assert statuses(engine)[("ohlc_inconsistent", DAYS[20].isoformat())][0] == "resolved"
    with engine.begin() as c:                                    # ...and it breaks again
        c.execute(text("UPDATE price_bar SET open = high + 90 WHERE instrument_id = :i AND trade_date = :d"), {"i": bad_id, "d": DAYS[20]})
    refresh(engine, cfg0, pairs)
    with session_scope(engine) as s:
        f = s.scalars(select(DataQualityFinding).where(DataQualityFinding.check_name == "ohlc_inconsistent")).one()
        assert f.status == "open" and f.resolved_at is None


# ---- known issues -----------------------------------------------------------------------------------
def test_known_issue_explains_only_what_it_covers(engine, cfg, after_close, tmp_path):
    def two_errors(b):
        b = defect_open(b)
        b[50] = replace(b[50], open=b[50].high + 90)             # a second, later defect
        return b
    cfg0, pairs = setup_data(engine, cfg, after_close, two_errors)
    with session_scope(engine) as s:
        load_known_issues(s, write_issues(tmp_path, f'k1,ohlc_inconsistent,BAD,{DAYS[15]},{DAYS[25]},keep_flagged,"characterised",evidence\n'))
    refresh(engine, cfg0, pairs)
    st = statuses(engine)
    assert st[("ohlc_inconsistent", DAYS[20].isoformat())][0] == "explained"
    assert st[("ohlc_inconsistent", DAYS[50].isoformat())][0] == "open"           # outside the range: still unexplained
    with session_scope(engine) as s:
        left = unexplained_errors(s)
        assert [(f.trade_date, f.check_name) for f in left] == [(DAYS[50], "ohlc_inconsistent")]
        assert s.scalars(select(DataQualityFinding).where(DataQualityFinding.status == "explained")).one().known_issue_id is not None


def test_known_issue_date_range_is_inclusive_and_symbol_specific(engine, cfg, after_close, tmp_path):
    cfg0, pairs = setup_data(engine, cfg, after_close, defect_open)
    with session_scope(engine) as s:
        load_known_issues(s, write_issues(tmp_path, f'wrong-symbol,ohlc_inconsistent,OK,,,keep,"x",\n'))
    refresh(engine, cfg0, pairs)
    assert statuses(engine)[("ohlc_inconsistent", DAYS[20].isoformat())][0] == "open"
    for lo, hi, expect in [(DAYS[20], DAYS[20], "explained"), (DAYS[21], DAYS[30], "open"), (DAYS[1], DAYS[19], "open"), ("", "", "explained")]:
        with session_scope(engine) as s:                          # same key each time: the loader upserts
            load_known_issues(s, write_issues(tmp_path, f'r,ohlc_inconsistent,BAD,{lo},{hi},keep,"x",\n'))
        refresh(engine, cfg0, pairs)
        assert statuses(engine)[("ohlc_inconsistent", DAYS[20].isoformat())][0] == expect, (lo, hi)


def test_known_issue_for_any_instrument_and_check_must_match(engine, cfg, after_close, tmp_path):
    cfg0, pairs = setup_data(engine, cfg, after_close, defect_open)
    with session_scope(engine) as s:
        load_known_issues(s, write_issues(tmp_path, 'any,ohlc_inconsistent,,,,keep,"all instruments",\nother-check,zero_volume,,,,keep,"x",\n'))
    refresh(engine, cfg0, pairs)
    st = statuses(engine)
    assert st[("ohlc_inconsistent", DAYS[20].isoformat())][0] == "explained" and st[("zero_volume", DAYS[30].isoformat())][0] == "explained"
    with session_scope(engine) as s:
        assert unexplained_errors(s) == []


def test_known_issue_loading_is_idempotent_and_updates(engine, tmp_path):
    apply(engine, "U1", ["BAD"], date(2020, 1, 1))
    f = write_issues(tmp_path, 'k1,ohlc_inconsistent,BAD,,,keep,"first",\n')
    with session_scope(engine) as s:
        assert load_known_issues(s, f) == 1
    with session_scope(engine) as s:
        load_known_issues(s, f)
        load_known_issues(s, write_issues(tmp_path, 'k1,ohlc_inconsistent,BAD,,,exclude_bar,"second",\n', "k2.csv"))
        rows = s.scalars(select(DataQualityKnownIssue)).all()
        assert len(rows) == 1 and rows[0].explanation == "second" and rows[0].treatment == "exclude_bar"


@pytest.mark.parametrize("body,msg", [
    ('k,not_a_check,,,,keep,"x",\n', "unknown check"),
    ('k,zero_volume,NOPE,,,keep,"x",\n', "unknown symbol"),
    ('k,zero_volume,,2026-13-01,,keep,"x",\n', "bad date"),
    ('k,zero_volume,,2026-02-01,2026-01-01,keep,"x",\n', "date_to before"),
    (',zero_volume,,,,keep,"x",\n', "required"),
    ('k,zero_volume,,,,,"x",\n', "required"),
    ('k,zero_volume,,,,keep,,\n', "required"),
    ('k,zero_volume,,,,keep,"x",\nk,zero_volume,,,,keep,"y",\n', "duplicate keys"),
])
def test_known_issue_validation(engine, tmp_path, body, msg):
    apply(engine, "U1", ["BAD"], date(2020, 1, 1))
    with session_scope(engine) as s:
        with pytest.raises(KnownIssueError, match=msg):
            load_known_issues(s, write_issues(tmp_path, body))


def test_known_issue_file_needs_all_columns(engine, tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("key,check\nk,zero_volume\n")
    with session_scope(engine) as s:
        with pytest.raises(KnownIssueError, match="missing columns"):
            load_known_issues(s, p)


# ---- report -------------------------------------------------------------------------------------------
def report(engine, cfg0, pairs, blocked=None):
    with session_scope(engine) as s:
        return build_report(s, cfg0, pairs, date(2026, 9, 18), ["U1"], blocked)


def test_report_fails_while_an_error_is_unexplained_and_passes_once_explained(engine, cfg, after_close, tmp_path):
    cfg0, pairs = setup_data(engine, cfg, after_close, defect_open)
    refresh(engine, cfg0, pairs)
    text_ = report(engine, cfg0, pairs)
    assert "## Verdict: **FAIL**" in text_ and "Unexplained errors: **1**" in text_ and "ohlc_inconsistent" in text_
    assert "open outside [low, high]" in text_ and "zero_volume" in text_
    with session_scope(engine) as s:
        load_known_issues(s, write_issues(tmp_path, 'k1,ohlc_inconsistent,BAD,,,keep_flagged,"vendor defect, root cause unknown","measured on 1 bar"\n'))
    refresh(engine, cfg0, pairs)
    text_ = report(engine, cfg0, pairs)
    assert "## Verdict: **PASS**" in text_ and "Unexplained errors: **0**" in text_
    assert "k1" in text_ and "vendor defect, root cause unknown" in text_ and "*Evidence:* measured on 1 bar" in text_


def test_report_is_deterministic_and_the_file_is_only_rewritten_on_change(engine, cfg, after_close, tmp_path):
    cfg0, pairs = setup_data(engine, cfg, after_close, defect_open)
    refresh(engine, cfg0, pairs)
    a, b = report(engine, cfg0, pairs), report(engine, cfg0, pairs)
    assert a == b
    path = tmp_path / "sub" / "dq.md"
    write_report(path, a)
    os.utime(path, (1, 1))
    write_report(path, b)
    assert path.stat().st_mtime == 1                                     # identical content: file untouched
    write_report(path, a + "x")
    assert path.stat().st_mtime > 1


def test_report_lists_blocked_instruments_and_calendar(engine, cfg, after_close):
    cfg0, pairs = setup_data(engine, cfg, after_close)
    refresh(engine, cfg0, pairs)
    text_ = report(engine, cfg0, pairs, {"NEW1": "NEW1: only 10 sessions up to 2026-09-18"})
    assert "instruments blocked for insufficient history: 1" in text_ and "only 10 sessions" in text_
    assert "## Trading calendar" in text_ and "## Price adjustment" in text_ and "## History coverage" in text_
    assert "## Verdict: **PASS**" in text_
