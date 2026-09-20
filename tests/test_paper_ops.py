"""Operations: run modes (no live mode), schedule, backup retention, dump / restore test, report files, notifications, no order-placing code."""
from __future__ import annotations

import gzip
import os
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select

from predict_stock.config import PROJECT_ROOT
from predict_stock.db.models import Alert
from predict_stock.paper import backup as BK
from predict_stock.paper import check_mode
from predict_stock.paper import report as RPT
from predict_stock.paper.schedule import crontab
from tests.conftest import apply


# ---- modes --------------------------------------------------------------------------------------------------------------------------------------------------
def test_only_backtest_and_paper_modes_exist():
    assert check_mode("paper") == "paper" and check_mode("backtest") == "backtest"
    for bad in ("live", "real", "prod", ""):
        with pytest.raises(ValueError, match="no live-trading mode"):
            check_mode(bad)


def test_the_daily_job_refuses_any_mode_but_paper(engine, cfg):
    from predict_stock.paper.daily import run_daily
    with pytest.raises(ValueError, match="only runs in mode 'paper'"):
        run_daily(engine, cfg, None, date(2026, 1, 5), mode="backtest")
    with pytest.raises(ValueError, match="no live-trading mode"):
        run_daily(engine, cfg, None, date(2026, 1, 5), mode="live")


def test_the_cli_refuses_live_mode():
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "predict_stock", "paper", "run", "--mode", "live"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "no live-trading mode" in r.stderr


def test_no_source_file_can_place_a_real_order():
    """There is no broker client, no order endpoint and no credentials for one anywhere in the code base."""
    pat = re.compile(r"place_?order|submit_?order|send_?order|create_?order|broker_?(client|api|key|token)|/orders?\b", re.I)
    hits = [f"{f.relative_to(PROJECT_ROOT)}:{n}: {line.strip()[:100]}" for f in (PROJECT_ROOT / "src").rglob("*.py")
            for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1) if pat.search(line)]
    assert hits == [], hits


# ---- schedule --------------------------------------------------------------------------------------------------------------------------------------------------
def test_crontab_runs_the_daily_job_after_the_close_single_flight_with_backup_and_restore_test(cfg):
    t = crontab(cfg, python="/py")
    assert "CRON_TZ=Asia/Ho_Chi_Minh" in t and "30 16 * * 1-5" in t and "paper run" in t and t.count("flock -n") == 3
    assert "paper backup --prune" in t and "paper restore-test" in t and "live" not in t.lower().replace("flock", "")


# ---- backup retention -------------------------------------------------------------------------------------------------------------------------------------------
def fake_dumps(d: Path, stamps):
    for st in stamps:
        (d / f"predict_stock_{st}.sql.gz").write_bytes(b"x")
        (d / f"predict_stock_{st}.sql.gz.sha256").write_text("x")


def test_retention_keeps_the_latest_dump_of_each_recent_day_and_the_oldest_of_each_recent_week(tmp_path):
    days = [f"202601{d:02d}_0230{s:02d}" for d in range(1, 31) for s in (0, 5)]          # two dumps a day for 30 days
    fake_dumps(tmp_path, days)
    removed = BK.prune(tmp_path, daily_keep=7, weekly_keep=2)
    left = {f.name.split("_", 2)[2].removesuffix(".sql.gz") for _, f in BK.list_dumps(tmp_path)}
    daily = {f"202601{d:02d}_023005" for d in range(24, 31)}                             # the LATEST dump of each of the newest 7 days
    weekly = {"20260119_023000", "20260112_023000"}                                      # the OLDEST dump of each of the 2 newest older ISO weeks
    assert left == daily | weekly and len(removed) == 60 - 9
    assert sorted(p.name for p in tmp_path.glob("*.sha256")) == sorted(f"predict_stock_{x}.sql.gz.sha256" for x in left)      # checksums go with their dumps


def test_retention_with_fewer_dumps_than_the_limits_removes_nothing(tmp_path):
    fake_dumps(tmp_path, ["20260105_023000", "20260106_023000"])
    assert BK.prune(tmp_path, 14, 8) == [] and len(BK.list_dumps(tmp_path)) == 2


def test_only_files_that_look_like_dumps_are_touched(tmp_path):
    fake_dumps(tmp_path, ["20260105_023000"] * 1)
    (tmp_path / "notes.txt").write_text("keep me")
    (tmp_path / "other_db_20200101_000000.sql.gz").write_bytes(b"x")
    BK.prune(tmp_path, 0, 0, database="predict_stock")
    assert (tmp_path / "notes.txt").exists() and (tmp_path / "other_db_20200101_000000.sql.gz").exists() and not list(tmp_path.glob("predict_stock_*"))


# ---- dump / restore on the test database ---------------------------------------------------------------------------------------------------------------------
needs_admin = pytest.mark.skipif(not (os.environ.get("MYSQL_ROOT_PASSWORD") or (PROJECT_ROOT / ".env").exists() and "MYSQL_ROOT_PASSWORD" in (PROJECT_ROOT / ".env").read_text()),
                                 reason="the restore test needs the admin account (MYSQL_ROOT_PASSWORD)")


@needs_admin
def test_a_dump_restores_into_a_scratch_database_with_identical_tables(engine, cfg, tmp_path):
    apply(engine, "T1", ["AAA", "BBB", "CCC"], date(2024, 1, 1))
    test_db = os.environ.get("MYSQL_TEST_DATABASE", "predict_stock_test")
    c = cfg.model_copy(update={"backup": cfg.backup.model_copy(update={"dir": str(tmp_path), "restore_database": "predict_stock_restore_pytest"})})
    path = BK.dump(c, database=test_db, out_dir=tmp_path)
    assert path.exists() and BK.verify(path) and path.name.startswith(test_db)
    res = BK.restore_test(c, path, source_database=test_db, fresh=False)
    assert res["ok"] and res["tables"] >= 30 and res["rows"] > 0 and res["mismatches"] == []
    import pymysql
    conn = BK._admin(BK._env())
    with conn.cursor() as cur:
        cur.execute("SHOW DATABASES LIKE 'predict_stock_restore_pytest'")
        assert cur.fetchall() == ()                                     # the scratch database is dropped afterwards
    conn.close()


@needs_admin
def test_the_restore_test_notices_when_the_live_database_moved_on_and_when_a_dump_is_damaged(engine, cfg, tmp_path):
    apply(engine, "T1", ["AAA"], date(2024, 1, 1))
    test_db = os.environ.get("MYSQL_TEST_DATABASE", "predict_stock_test")
    c = cfg.model_copy(update={"backup": cfg.backup.model_copy(update={"dir": str(tmp_path), "restore_database": "predict_stock_restore_pytest"})})
    path = BK.dump(c, database=test_db, out_dir=tmp_path)
    from predict_stock.paper.alerts import raise_alert
    raise_alert(engine, "info", "x", "written after the dump")
    res = BK.restore_test(c, path, source_database=test_db, fresh=False)
    assert not res["ok"] and [m["table"] for m in res["mismatches"]] == ["alerts"]
    with gzip.open(path, "wb") as f:
        f.write(b"-- tampered")
    with pytest.raises(BK.BackupError, match="checksum"):
        BK.restore_test(c, path, source_database=test_db, fresh=False)


def test_the_restore_database_can_never_be_a_real_database(cfg, tmp_path):
    bad = cfg.model_copy(update={"backup": cfg.backup.model_copy(update={"restore_database": os.environ.get("MYSQL_DATABASE", "predict_stock")})})
    with pytest.raises(BK.BackupError, match="scratch"):
        BK.restore_test(bad, tmp_path / "x.sql.gz", fresh=False)


# ---- report and notifications ----------------------------------------------------------------------------------------------------------------------------------
def collected():
    return {"as_of": "2026-09-18", "mode": "paper", "results": {"ingest": {"ok": True, "seconds": 3.2, "result": {}}, "cards": {"ok": False, "seconds": 1.0, "error": "boom"}},
            "cards": [{"strategy": "SWING", "symbol": "FPT", "action": "BUY", "status": "pending", "entry": 64600.0, "target": 67700.0, "stop_or_bear": 63000.0, "valid_until": "2026-09-23",
                       "confidence": 0.41, "flags": None, "rejected": None, "zone": [64200.0, 64900.0]}],
            "status_counts": {"pending": 1}, "positions": [{"portfolio": "paper:swing", "symbol": "FPT", "quantity": 700, "avg_cost": 64600.0, "opened": "2026-09-18"}],
            "orders_today": [], "pending_orders": [{"portfolio": "paper:swing", "symbol": "FPT", "side": "BUY", "type": "limit", "qty": 700, "limit": 64900.0}],
            "equity": {"date": "2026-09-18", "equity": 1.01e9, "cash": 9e8, "market_value": 1.1e8, "metrics": {"exposure": 0.11, "drawdown": -0.01}, "since_start": 0.01},
            "equity_curve": [{"date": "2026-09-18", "equity": 1.01e9, "cash": 9e8}], "closed": [], "alerts": [{"id": 1, "severity": "error", "category": "data_missing", "message": "2 members have no bar", "created": "x"}]}


def test_report_files_md_html_and_csv_are_written(cfg, tmp_path):
    c = cfg.model_copy(update={"paper": cfg.paper.model_copy(update={"report_dir": str(tmp_path)})})
    files = RPT.write_files(c, collected())
    assert {"md", "html", "csv:cards", "csv:positions", "csv:equity_curve"} <= set(files)
    md = Path(files["md"]).read_text(encoding="utf-8")
    assert "Paper trading report — 2026-09-18" in md and "no real order is ever placed" in md and "**FAILED**" in md and "FPT" in md and "data_missing" in md
    h = Path(files["html"]).read_text(encoding="utf-8")
    assert h.startswith("<!doctype html>") and "<table>" in h and "<h1>" in h and "<strong>FAILED</strong>" in h
    df = pd.read_csv(files["csv:cards"])
    assert list(df["symbol"]) == ["FPT"] and float(df["target"].iloc[0]) == 67700.0


def test_notifications_go_only_where_enabled_and_a_failure_becomes_an_alert(engine, cfg, tmp_path):
    c = cfg.model_copy(update={"paper": cfg.paper.model_copy(update={"notify": cfg.paper.notify.model_copy(update={"telegram": True, "email": True})})})
    sent = []

    def bad_telegram(text):
        raise RuntimeError("network down")
    out = RPT.notify(engine, c, collected(), {}, telegram=bad_telegram, email=lambda s, t, a: sent.append((s, t)))
    assert out["sent"] == ["email"] and "FAILED steps: cards" in sent[0][1] and "1 BUY" in sent[0][1]
    with engine.connect() as conn:
        assert conn.execute(select(Alert.category).where(Alert.category == "notify_failed")).scalars().all() == ["notify_failed"]
    off = RPT.notify(engine, cfg, collected(), {}, telegram=lambda t: sent.append("tg"), email=lambda *a: sent.append("mail"))
    assert off["sent"] == [] and "tg" not in sent                                # both off by default


def test_telegram_and_email_need_their_credentials_from_the_environment():
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        RPT.send_telegram("x", env={})
    with pytest.raises(RuntimeError, match="SMTP_HOST"):
        RPT.send_email("s", "t", None, env={})
