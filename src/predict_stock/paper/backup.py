"""Backups: a compressed mysqldump per run (consistent snapshot, no table locks), a sha256 next to it, retention (the newest N daily dumps + the oldest dump of each of the
last M weeks), and a RESTORE TEST that loads the newest dump into a scratch database and compares every table with the live one. Credentials come from the environment (.env);
the dump uses the schema-owner account, the scratch database needs the admin account (MYSQL_ROOT_PASSWORD) and is dropped afterwards."""
from __future__ import annotations

import gzip
import hashlib
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pymysql
from dotenv import load_dotenv

from predict_stock.config import PROJECT_ROOT, AppConfig

STAMP = re.compile(r"^(?P<db>.+)_(?P<ts>\d{8}_\d{6})\.sql(?:\.gz)?$")


class BackupError(RuntimeError):
    pass


def _env() -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    return dict(os.environ)


def _conn_args(env: dict, database: str | None, admin: bool = False) -> tuple[list[str], dict]:
    """(mysql client arguments, environment): the password goes in MYSQL_PWD, never on the command line."""
    user = "root" if admin else env["MYSQL_MIGRATOR_USER"]
    pw = env["MYSQL_ROOT_PASSWORD"] if admin else env["MYSQL_MIGRATOR_PASSWORD"]
    args = [f"--host={env.get('MYSQL_HOST', '127.0.0.1')}", f"--port={env.get('MYSQL_PORT', '3307')}", f"--user={user}", "--protocol=TCP"]
    return args, {**env, "MYSQL_PWD": pw}


def backup_dir(cfg: AppConfig) -> Path:
    d = PROJECT_ROOT / cfg.backup.dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def dump(cfg: AppConfig, *, database: str | None = None, out_dir: Path | None = None, now: datetime | None = None) -> Path:
    """One consistent, gzip-compressed dump of the database (schema, data, routines, triggers). Returns the file; a `.sha256` file is written next to it."""
    env = _env()
    db = database or env["MYSQL_DATABASE"]
    out_dir = out_dir or backup_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{db}_{ts}.sql{'.gz' if cfg.backup.compress else ''}"
    args, cenv = _conn_args(env, db)
    cmd = ["mysqldump", *args, "--single-transaction", "--routines", "--triggers", "--no-tablespaces", "--set-gtid-purged=OFF", "--default-character-set=utf8mb4", db]
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=cenv) as proc, (gzip.open(tmp, "wb") if cfg.backup.compress else open(tmp, "wb")) as f:
            shutil.copyfileobj(proc.stdout, f)
            err = proc.stderr.read().decode(errors="replace")
        if proc.returncode != 0:
            raise BackupError(f"mysqldump failed ({proc.returncode}): {err.strip()[-500:]}")
        tmp.rename(path)                                                       # a dump is either complete or absent
    finally:
        if tmp.exists():
            tmp.unlink()
    (path.with_suffix(path.suffix + ".sha256")).write_text(f"{sha256_file(path)}  {path.name}\n")
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path) -> bool:
    side = path.with_suffix(path.suffix + ".sha256")
    return side.exists() and side.read_text().split()[0] == sha256_file(path)


def list_dumps(out_dir: Path, database: str | None = None) -> list[tuple[datetime, Path]]:
    rows = []
    for f in out_dir.glob("*.sql*"):
        m = STAMP.match(f.name)
        if m and not f.name.endswith((".sha256", ".part")) and (database is None or m["db"] == database):
            rows.append((datetime.strptime(m["ts"], "%Y%m%d_%H%M%S"), f))
    return sorted(rows)


def prune(out_dir: Path, daily_keep: int, weekly_keep: int, database: str | None = None) -> list[Path]:
    """Retention: keep the LATEST dump of each of the newest ``daily_keep`` days, plus the OLDEST dump of each of the newest ``weekly_keep`` ISO weeks (older than those days);
    everything else is deleted together with its checksum. Returns the deleted files."""
    dumps = list_dumps(out_dir, database)
    by_day: dict = {}
    for ts, f in dumps:
        by_day[ts.date()] = (ts, f)                                            # sorted ascending: the last one of a day wins
    days = sorted(by_day, reverse=True)
    keep = {by_day[d][1] for d in days[:daily_keep]}
    weeks: dict = {}
    for ts, f in dumps:
        if by_day[ts.date()][1] in keep:
            continue
        weeks.setdefault(ts.isocalendar()[:2], f)                              # the oldest of the week (ascending order)
    for wk in sorted(weeks, reverse=True)[:weekly_keep]:
        keep.add(weeks[wk])
    removed = []
    for ts, f in dumps:
        if f not in keep:
            f.unlink()
            side = f.with_suffix(f.suffix + ".sha256")
            if side.exists():
                side.unlink()
            removed.append(f)
    return removed


def _admin(env: dict, database: str | None = None):
    if not env.get("MYSQL_ROOT_PASSWORD"):
        raise BackupError("MYSQL_ROOT_PASSWORD is not set: the restore test needs the admin account to create a scratch database")
    return pymysql.connect(host=env.get("MYSQL_HOST", "127.0.0.1"), port=int(env.get("MYSQL_PORT", "3307")), user="root", password=env["MYSQL_ROOT_PASSWORD"], database=database, charset="utf8mb4")


def table_fingerprints(conn, database: str) -> dict[str, tuple[int, int | None]]:
    """{table: (row count, CHECKSUM TABLE value)} of a database."""
    out = {}
    with conn.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema=%s AND table_type='BASE TABLE' ORDER BY table_name", (database,))
        tables = [r[0] for r in cur.fetchall()]
        for t in tables:
            cur.execute(f"SELECT COUNT(*) FROM `{database}`.`{t}`")
            n = cur.fetchone()[0]
            cur.execute(f"CHECKSUM TABLE `{database}`.`{t}`")
            out[t] = (n, cur.fetchone()[1])
    return out


def restore_test(cfg: AppConfig, dump_path: Path | None = None, *, source_database: str | None = None, fresh: bool = True) -> dict:
    """Restore a dump into a scratch database and compare every table (row count and CHECKSUM TABLE) with the source database. The scratch database is dropped afterwards.
    Returns {ok, tables, mismatches, seconds}. By default it takes a fresh dump first; with ``fresh=False`` it restores the newest existing dump, and a table written since then (an alert, a job run) legitimately differs."""
    import time
    env = _env()
    src_db = source_database or env["MYSQL_DATABASE"]
    scratch = cfg.backup.restore_database
    if scratch in (src_db, env.get("MYSQL_DATABASE"), env.get("MYSQL_TEST_DATABASE")):
        raise BackupError(f"the restore database {scratch!r} must be a scratch database, not {src_db!r} or the test database")
    if dump_path is None and fresh:
        dump_path = dump(cfg, database=src_db)                                 # a dump taken now: nothing written since can make the comparison differ
    path = dump_path or (list_dumps(backup_dir(cfg), src_db) or [(None, None)])[-1][1]
    if path is None:
        raise BackupError("no dump to restore: run `paper backup` first")
    if not verify(path):
        raise BackupError(f"{path.name}: checksum does not match: the dump is damaged")
    t0 = time.time()
    admin = _admin(env)
    try:
        with admin.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS `{scratch}`")
            cur.execute(f"CREATE DATABASE `{scratch}` CHARACTER SET utf8mb4")
        args, cenv = _conn_args(env, scratch, admin=True)
        opener = gzip.open if path.suffix == ".gz" else open
        with subprocess.Popen(["mysql", *args, "--default-character-set=utf8mb4", scratch], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=cenv) as proc:
            with opener(path, "rb") as f:
                try:
                    shutil.copyfileobj(f, proc.stdin)                          # decompressed on the fly: mysql reads plain SQL
                except BrokenPipeError:
                    pass                                                       # mysql stopped early: its stderr says why
            proc.stdin.close()
            err = proc.stderr.read().decode(errors="replace")
            rc = proc.wait()
        if rc != 0:
            raise BackupError(f"restore failed: {err[-500:]}")
        src, dst = table_fingerprints(admin, src_db), table_fingerprints(admin, scratch)
        mism = [{"table": t, "source": src.get(t), "restored": dst.get(t)} for t in sorted(set(src) | set(dst)) if src.get(t) != dst.get(t)]
        return {"ok": not mism and bool(src), "dump": path.name, "tables": len(src), "rows": sum(n for n, _ in src.values()), "mismatches": mism, "seconds": round(time.time() - t0, 1)}
    finally:
        try:
            with admin.cursor() as cur:
                cur.execute(f"DROP DATABASE IF EXISTS `{scratch}`")
        finally:
            admin.close()


def prune_datasets(engine, cfg: AppConfig, keep: int = 5) -> list[str]:
    """Every daily run may create a new dataset version (the data changed). Delete the Parquet files of old versions that no model was trained on (the manifest rows stay,
    marked pruned); files used by a model are never removed."""
    from sqlalchemy import select
    from predict_stock.db.models import Dataset, Model
    from predict_stock.db.session import session_scope
    removed = []
    with session_scope(engine) as s:
        used = {i for i in s.scalars(select(Model.dataset_id)) if i is not None}
        by_name: dict[str, list] = {}
        for d in s.scalars(select(Dataset).order_by(Dataset.name, Dataset.version.desc())):
            by_name.setdefault(d.name, []).append(d)
        for rows in by_name.values():
            for d in rows[keep:]:
                p = PROJECT_ROOT / d.path if not Path(d.path).is_absolute() else Path(d.path)
                if d.id in used or not p.exists():
                    continue
                p.unlink()
                m = dict(d.manifest or {})
                m["file_pruned"] = True
                d.manifest = m
                removed.append(str(p))
    return removed
