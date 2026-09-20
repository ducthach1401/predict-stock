from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from conftest import FakeClient, make_bars
from predict_stock.config import PROJECT_ROOT
from predict_stock.data.ingest import ingest_universe
from predict_stock.data.quality import run_quality_checks, summarize
from predict_stock.db.session import make_engine, session_scope
from predict_stock.universe import load_membership_csv

START, END = date(2026, 8, 3), date(2026, 9, 18)


def build(engine, cfg, tmp_path, now, series):
    cfg0 = cfg.model_copy(update={"ingest": cfg.ingest.model_copy(update={"benchmark_symbols": []})})
    p = tmp_path / "m.csv"
    p.write_text("universe_code,symbol,effective_from,effective_to\n" + "".join(f"U1,{s},2020-01-01,\n" for s in series))
    with session_scope(engine) as s:
        load_membership_csv(s, p)
    ingest_universe(engine, FakeClient(series), cfg0, "U1", START, END, now=now)
    return cfg0


def test_quality_checks_flag_each_planted_defect(engine, cfg, tmp_path, after_close):
    a, b, c = make_bars(START, 20), make_bars(START, 20, base=20000), make_bars(START, 20, base=30000)
    a[5] = replace(a[5], high=a[5].low - 1)                                        # high < low
    b[7] = replace(b[7], close=b[6].close * Decimal("1.25"), high=b[6].close * Decimal("1.30"))   # +25% jump
    c[3] = replace(c[3], volume=0)                                                 # zero volume
    missing_day = c[10].trade_date
    del c[10]                                                                      # C skips a day A and B traded
    stale_last = b[-1].trade_date
    b = b[:-2]                                                                     # B stops 2 days early
    cfg0 = build(engine, cfg, tmp_path, after_close, {"A": a, "B": b, "C": c})

    with session_scope(engine) as s:
        issues = run_quality_checks(s, ["A", "B", "C"], cfg0)
    found = {(r.symbol, r.check, r.trade_date) for r in issues.itertuples()}
    assert ("A", "ohlc_inconsistent", make_bars(START, 20)[5].trade_date) in found
    assert ("B", "big_move", make_bars(START, 20)[7].trade_date) in found
    assert ("C", "zero_volume", make_bars(START, 20)[3].trade_date) in found
    assert ("C", "missing_trading_day", missing_day) in found
    assert any(r.symbol == "B" and r.check == "stale_series" for r in issues.itertuples())
    assert not issues[(issues.symbol == "A") & (issues.check != "ohlc_inconsistent")].size    # A has only its planted defect
    assert set(summarize(issues)["check"]) >= {"ohlc_inconsistent", "big_move", "zero_volume", "missing_trading_day", "stale_series"}


def test_clean_data_reports_no_issues(engine, cfg, tmp_path, after_close):
    cfg0 = build(engine, cfg, tmp_path, after_close, {"A": make_bars(START, 20), "B": make_bars(START, 20, base=9000)})
    with session_scope(engine) as s:
        assert run_quality_checks(s, ["A", "B"], cfg0).empty


def test_app_role_has_dml_but_no_ddl(engine):
    with engine.begin() as c:
        c.execute(text("INSERT INTO instruments (symbol, kind) VALUES ('T1', 'stock')"))
        c.execute(text("UPDATE instruments SET kind = 'index' WHERE symbol = 'T1'"))
        c.execute(text("DELETE FROM instruments WHERE symbol = 'T1'"))
    for stmt in ("CREATE TABLE zz_nope (i INT)", "DROP TABLE instruments", "ALTER TABLE instruments ADD COLUMN zz INT"):
        with pytest.raises(DBAPIError, match="denied"):
            with engine.begin() as c:
                c.execute(text(stmt))


def test_models_match_migrations(_schema):
    """`alembic check` fails if the ORM models drift from the migration history."""
    r = subprocess.run(
        [sys.executable, "-m", "alembic", "-x", "db=test", "check"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_database_is_utf8mb4(_schema):
    eng = make_engine("migrator", test=True)
    with eng.connect() as c:
        charset = c.execute(text("SELECT @@character_set_database")).scalar()
        engine_names = {r[0] for r in c.execute(text(
            "SELECT DISTINCT engine FROM information_schema.tables WHERE table_schema = DATABASE()"))}
    eng.dispose()
    assert charset == "utf8mb4" and engine_names == {"InnoDB"}
