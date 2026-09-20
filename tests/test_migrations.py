"""Migration 0001 <-> 0002 with real rows in the old schema. Leaves the test DB at head."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest
from sqlalchemy import text

from predict_stock.config import PROJECT_ROOT
from predict_stock.db.session import make_engine


def alembic(*args: str) -> None:
    r = subprocess.run([sys.executable, "-m", "alembic", "-x", "db=test", *args],
                       cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"alembic {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}"


def load_legacy_fixture(c):
    snap = json.dumps({"seed": 42, "market": {"fee_rate": 0.0015}})
    c.execute(text("INSERT INTO instruments (symbol, kind) VALUES ('AAA','stock'),('BBB','stock'),('IDX','index')"))
    c.execute(text("INSERT INTO job_runs (job_name,status,seed,config_snapshot) VALUES "
                   "('ingest_ohlcv','success',42,:s),('ingest_ohlcv','success',42,:s),('ingest_ohlcv','failed',7,:t)"),
              {"s": snap, "t": json.dumps({"seed": 7})})
    c.execute(text("INSERT INTO ohlcv_daily VALUES "
                   "('AAA','2026-01-02',10,11,9,10.5,1000,'src','2026-02-01 00:00:00',1),"
                   "('AAA','2026-01-05',10.5,11,10,10.7,2000,'src','2026-02-01 00:00:00',1),"
                   "('BBB','2026-01-02',20,21,19,20.5,3000,'src','2026-02-01 00:00:00',2),"
                   "('IDX','2026-01-02',1200,1210,1190,1205,0,'src','2026-02-01 00:00:00',2)"))
    c.execute(text("INSERT INTO trading_days VALUES ('2026-01-02','stock-consensus'),('2026-01-05','stock-consensus')"))
    c.execute(text("INSERT INTO universe_membership (universe_code,symbol,effective_from,effective_to,source) VALUES "
                   "('U1','AAA','2020-01-01',NULL,'f.csv'),('U1','BBB','2020-01-01','2025-06-01','f.csv')"))


def test_migration_roundtrip_preserves_data(_schema):
    eng = make_engine("migrator", test=True)
    try:
        alembic("downgrade", "base")
        alembic("upgrade", "0001")
        with eng.begin() as c:
            load_legacy_fixture(c)

        alembic("upgrade", "head")
        with eng.connect() as c:
            q = lambda s: c.execute(text(s)).all()
            assert q("select h.symbol, i.kind, h.exchange from instruments i join instrument_symbol_history h "
                     "on h.instrument_id = i.id order by h.symbol") == [("AAA", "stock", None), ("BBB", "stock", None), ("IDX", "index", None)]
            bars = q("select h.symbol, b.trade_date, b.close, b.volume, b.price_basis, b.run_id from price_bar b "
                     "join instrument_symbol_history h on h.instrument_id = b.instrument_id order by 1, 2")
            assert [(b[0], str(b[1]), float(b[2]), b[3], b[4], b[5]) for b in bars] == [
                ("AAA", "2026-01-02", 10.5, 1000, "vendor_adjusted", 1), ("AAA", "2026-01-05", 10.7, 2000, "vendor_adjusted", 1),
                ("BBB", "2026-01-02", 20.5, 3000, "vendor_adjusted", 2), ("IDX", "2026-01-02", 1205.0, 0, "vendor_adjusted", 2)]
            assert q("select count(*) from trading_calendar where calendar_code = 'VN_CONSENSUS'")[0][0] == 2
            assert [(r[0], str(r[1]), str(r[2]) if r[2] else None) for r in q(
                "select h.symbol, m.valid_from, m.valid_to from universe_membership m join instrument_symbol_history h "
                "on h.instrument_id = m.instrument_id order by 1")] == [("AAA", "2020-01-01", None), ("BBB", "2020-01-01", "2025-06-01")]
            assert sorted(r[0] for r in q("select action from universe_change_log")) == ["add", "add", "remove"]
            # three runs, two distinct configs -> deduplicated snapshots
            assert q("select count(*) from config_snapshots")[0][0] == 2
            assert len({r[0] for r in q("select config_snapshot_id from job_runs")}) == 2
            assert q("select count(*) from job_runs where config_snapshot_id is null")[0][0] == 0
        alembic("check")

        alembic("downgrade", "0001")
        with eng.connect() as c:
            q = lambda s: c.execute(text(s)).all()
            assert q("select symbol, kind from instruments order by 1") == [("AAA", "stock"), ("BBB", "stock"), ("IDX", "index")]
            assert q("select count(*) from ohlcv_daily")[0][0] == 4
            assert float(q("select symbol, close from ohlcv_daily where symbol = 'AAA' order by trade_date")[1][1]) == pytest.approx(10.7)
            assert [(r[0], str(r[1]), str(r[2]) if r[2] else None) for r in q(
                "select symbol, effective_from, effective_to from universe_membership order by 1")] == [
                ("AAA", "2020-01-01", None), ("BBB", "2020-01-01", "2025-06-01")]
            assert q("select count(*) from trading_days")[0][0] == 2
            cfgs = [json.loads(r[0]) if isinstance(r[0], str) else r[0] for r in q("select config_snapshot from job_runs order by id")]
            assert [c["seed"] for c in cfgs] == [42, 42, 7]
            tables = {r[0] for r in q("show tables")}
            assert "price_bar" not in tables and "universes" not in tables and "config_snapshots" not in tables
    finally:
        alembic("upgrade", "head")
        eng.dispose()
    alembic("check")
