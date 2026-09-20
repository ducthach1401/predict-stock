"""Schema-level guarantees: types, defaults, constraints, views, privileges, migrations."""
from __future__ import annotations

import subprocess
import sys
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from predict_stock.config import PROJECT_ROOT
from predict_stock.db.repo import create_instrument
from predict_stock.db.session import make_engine, session_scope

GROUPS = {
    "A": ["instruments", "instrument_symbol_history", "instrument_status_history", "universes", "universe_membership", "universe_change_log"],
    "B": ["price_bar", "price_bar_revisions", "price_bar_intraday", "corporate_actions", "adjustment_factors", "trading_calendar", "data_ingest_runs",
          "data_quality_findings", "data_quality_known_issues"],
    "C": ["feature_sets", "label_specs", "datasets"],
    "D": ["models", "model_metrics", "experiments", "predictions", "monitoring_metrics"],
    "E": ["recommendations", "recommendation_outcomes", "paper_orders", "paper_positions", "portfolio_snapshots"],
    "F": ["config_snapshots", "job_runs", "alerts"],
}


@pytest.fixture
def admin(_schema):
    eng = make_engine("migrator", test=True)
    yield eng
    eng.dispose()


def info(admin, sql, **params):
    with admin.connect() as c:
        return c.execute(text(sql), params).all()


def test_all_table_groups_exist(admin):
    have = {r[0] for r in info(admin, "SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE()")}
    missing = [t for tables in GROUPS.values() for t in tables if t not in have]
    assert not missing, missing
    assert "v_price_adjusted" in have


def test_no_binary_columns_anywhere(admin):
    """Model artifacts and datasets live on disk; the DB stores only path + sha256."""
    bad = info(admin, "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = DATABASE() "
                      "AND data_type IN ('blob','tinyblob','mediumblob','longblob','binary','varbinary')")
    assert bad == []
    cols = lambda t: {r[0] for r in info(admin, "SELECT column_name FROM information_schema.columns WHERE table_schema = DATABASE() AND table_name = :t", t=t)}
    assert {"artifact_path", "artifact_sha256"} <= cols("models")
    assert {"path", "sha256"} <= cols("datasets")
    assert info(admin, "SELECT character_maximum_length FROM information_schema.columns WHERE table_schema = DATABASE() "
                       "AND table_name = 'models' AND column_name = 'artifact_sha256'")[0][0] == 64


def test_prices_are_decimal_and_dates_are_date(admin):
    price_cols = info(admin, "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = DATABASE() "
                             "AND (column_name IN ('open','high','low','close','entry_price','target_price','stop_loss','fill_price','avg_cost','limit_price','exit_price','close_price','cash_per_share') "
                             "OR column_name LIKE '%\\_price')")
    assert price_cols and all(t == "decimal" for _, _, t in price_cols), [r for r in price_cols if r[2] != "decimal"]
    date_cols = info(admin, "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = DATABASE() "
                            "AND (column_name IN ('trade_date','valid_from','valid_to','as_of_date','ex_date','effective_date','metric_date','snapshot_date','placed_date'))")
    assert date_cols and all(t == "date" for _, _, t in date_cols), [r for r in date_cols if r[2] != "date"]


def test_timestamps_default_to_utc(admin):
    defaults = info(admin, "SELECT table_name, column_default FROM information_schema.columns WHERE table_schema = DATABASE() "
                           "AND column_name IN ('created_at', 'started_at', 'superseded_at')")
    assert defaults and all("utc_timestamp" in (d or "").lower() for _, d in defaults), [r for r in defaults if "utc_timestamp" not in (r[1] or "").lower()]
    assert info(admin, "SELECT @@global.time_zone")[0][0] in ("+00:00", "UTC")


def test_database_is_utf8mb4_innodb_and_stores_vietnamese(admin, engine):
    assert info(admin, "SELECT @@character_set_database")[0][0] == "utf8mb4"
    assert {r[0] for r in info(admin, "SELECT DISTINCT engine FROM information_schema.tables WHERE table_schema = DATABASE() AND table_type = 'BASE TABLE'")} == {"InnoDB"}
    name = "Rổ vốn hóa lớn — Đầu tư dài hạn"
    with engine.begin() as c:
        c.execute(text("INSERT INTO universes (code, name) VALUES ('VN', :n)"), {"n": name})
        assert c.execute(text("SELECT name FROM universes WHERE code = 'VN'")).scalar() == name


def test_interval_constraints_reject_empty_or_reversed_ranges(engine):
    with session_scope(engine) as s:
        iid = create_instrument(s, "AAA", "stock", valid_from=date(2000, 1, 1))
        s.execute(text("INSERT INTO universes (code) VALUES ('U')"))
    for valid_from, valid_to in [("2024-05-01", "2024-05-01"), ("2024-05-01", "2024-01-01")]:
        with pytest.raises(DBAPIError, match="ck_membership_interval"):
            with engine.begin() as c:
                c.execute(text("INSERT INTO universe_membership (universe_id, instrument_id, valid_from, valid_to, source) "
                               "SELECT id, :i, :f, :t, 's' FROM universes"), {"i": iid, "f": valid_from, "t": valid_to})
    with pytest.raises(DBAPIError, match="ck_symbol_hist_interval"):
        with engine.begin() as c:
            c.execute(text("INSERT INTO instrument_symbol_history (instrument_id, symbol, valid_from, valid_to) VALUES (:i, 'X', '2024-05-01', '2024-04-01')"), {"i": iid})
    with pytest.raises(DBAPIError, match="ck_status_hist_status"):
        with engine.begin() as c:
            c.execute(text("INSERT INTO instrument_status_history (instrument_id, status, valid_from) VALUES (:i, 'halted', '2024-05-01')"), {"i": iid})


def test_ticker_and_membership_uniqueness(engine):
    with session_scope(engine) as s:
        iid = create_instrument(s, "AAA", "stock", valid_from=date(2000, 1, 1))
    with pytest.raises(DBAPIError, match="uq_symbol_hist_instrument_from"):
        with engine.begin() as c:
            c.execute(text("INSERT INTO instrument_symbol_history (instrument_id, symbol, valid_from) VALUES (:i, 'AAA2', '2000-01-01')"), {"i": iid})


RECO = dict(strategy="SWING", d="2026-09-18", entry=50000, target=55000, stop=47000, hmin=3, hmax=15,
            exit='{"time_stop_days": 15}', why="momentum + volume breakout")


def insert_reco(c, iid, **over):
    p = {**RECO, **over, "i": iid}
    c.execute(text("INSERT INTO recommendations (strategy, instrument_id, as_of_date, action, entry_price, target_price, stop_loss, "
                   "hold_days_min, hold_days_max, exit_conditions, rationale) VALUES (:strategy, :i, :d, 'BUY', :entry, :target, :stop, :hmin, :hmax, :exit, :why)"), p)


def test_recommendation_cannot_be_written_without_mandatory_fields(engine):
    with session_scope(engine) as s:
        iid = create_instrument(s, "AAA", "stock", valid_from=date(2000, 1, 1))
    with engine.begin() as c:
        insert_reco(c, iid)                                                          # a complete one is accepted
    for over, msg in [({"entry": None}, "entry_price"), ({"target": None}, "target_price"), ({"stop": None}, "stop_loss"),
                      ({"hmin": None}, "hold_days_min"), ({"hmax": None}, "hold_days_max"), ({"exit": None}, "exit_conditions"),
                      ({"why": None}, "rationale")]:
        with pytest.raises(DBAPIError, match=msg):
            with engine.begin() as c:
                insert_reco(c, iid, d="2026-09-19", **over)
    for over, name in [({"entry": 0}, "ck_reco_prices"), ({"stop": -1}, "ck_reco_prices"), ({"hmin": 10, "hmax": 5}, "ck_reco_horizon")]:
        with pytest.raises(DBAPIError, match=name):
            with engine.begin() as c:
                insert_reco(c, iid, d="2026-09-19", **over)
    with pytest.raises(DBAPIError, match="uq_recommendation"):                        # idempotent per (strategy, instrument, day, action)
        with engine.begin() as c:
            insert_reco(c, iid)


def test_adjusted_price_view_uses_the_active_factor_version(engine):
    with session_scope(engine) as s:
        iid = create_instrument(s, "AAA", "stock", valid_from=date(2000, 1, 1))
        vend = create_instrument(s, "VEN", "stock", valid_from=date(2000, 1, 1))
    bar = "INSERT INTO price_bar (instrument_id, trade_date, open, high, low, close, volume, price_basis, source, fetched_at) VALUES (:i, :d, 100, 110, 90, 100, 1000, :b, 's', '2026-01-01 00:00:00')"
    fac = "INSERT INTO adjustment_factors (instrument_id, version, effective_date, factor, source) VALUES (:i, :v, :d, :f, 't')"
    with engine.begin() as c:
        for d in ("2026-01-02", "2026-01-04", "2026-01-05"):
            c.execute(text(bar), {"i": iid, "d": d, "b": "raw"})
        c.execute(text(bar), {"i": vend, "d": "2026-01-02", "b": "vendor_adjusted"})
        c.execute(text(fac), {"i": iid, "v": 1, "d": "2026-01-05", "f": "0.5"})
        c.execute(text(fac), {"i": vend, "v": 1, "d": "2026-01-05", "f": "0.5"})

    def closes(i):
        with engine.connect() as c:
            return {str(d): (Decimal(p), Decimal(f)) for d, p, f in c.execute(text(
                "SELECT trade_date, close, adj_factor FROM v_price_adjusted WHERE instrument_id = :i"), {"i": i})}

    v1 = closes(iid)
    assert v1["2026-01-02"][0] == Decimal("50") and v1["2026-01-04"][0] == Decimal("50")      # bars before the ex-date are scaled
    assert v1["2026-01-05"][0] == Decimal("100")                                                # the ex-date itself is not
    with engine.begin() as c:                                                                   # version 2 supersedes version 1
        c.execute(text(fac), {"i": iid, "v": 2, "d": "2026-01-05", "f": "0.5"})
        c.execute(text(fac), {"i": iid, "v": 2, "d": "2026-01-03", "f": "0.8"})
    v2 = closes(iid)
    assert v2["2026-01-02"][0] == Decimal("40") and v2["2026-01-04"][0] == Decimal("50") and v2["2026-01-05"][0] == Decimal("100")
    assert closes(vend)["2026-01-02"][0] == Decimal("100")                                      # vendor-adjusted bars are never re-adjusted
    with engine.connect() as c:                                                                 # version 1 is still there (history)
        assert c.execute(text("SELECT COUNT(*) FROM adjustment_factors WHERE instrument_id = :i"), {"i": iid}).scalar() == 3


def test_app_role_has_dml_but_no_ddl(engine):
    with engine.begin() as c:
        c.execute(text("INSERT INTO instruments (kind) VALUES ('stock')"))
        c.execute(text("UPDATE instruments SET kind = 'index' ORDER BY id LIMIT 1"))
        c.execute(text("DELETE FROM instruments"))
    for stmt in ("CREATE TABLE zz_nope (i INT)", "DROP TABLE instruments", "ALTER TABLE instruments ADD COLUMN zz INT",
                 "DROP VIEW v_price_adjusted", "CREATE VIEW zz_v AS SELECT 1"):
        with pytest.raises(DBAPIError, match="denied"):
            with engine.begin() as c:
                c.execute(text(stmt))


def test_models_match_migrations(_schema):
    """`alembic check` fails if the ORM models drift from the migration history."""
    r = subprocess.run([sys.executable, "-m", "alembic", "-x", "db=test", "check"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
