"""The scheduler of the container: the same three jobs as `paper crontab`, without cron.

    daily job      `paper run`           at `paper.schedule_time` (Asia/Ho_Chi_Minh) on `paper.schedule_days`
    backup         `paper backup --prune` one hour later, every day
    restore test   `paper restore-test`   Sundays 03:00

A job is due once per day when its time has passed and it has not succeeded today. That is also the catch-up rule: a container that was down at 16:30 runs the daily job when it comes
back (every step is idempotent). A failed job is retried after `retry_minutes`, at most `max_tries` times a day (the vendor may publish late; the alerts say what happened).
State (last success / tries per job) lives in `logs/scheduler_state.json`; a heartbeat file lets the container healthcheck see that the loop is alive."""
from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from predict_stock.config import PROJECT_ROOT, AppConfig

TZ = ZoneInfo("Asia/Ho_Chi_Minh")
MAX_TRIES = 4
RETRY_MINUTES = 30


def weekdays(spec: str) -> set[int]:
    """cron day-of-week ('1-5', '1,3,5', '*'; 0 or 7 = Sunday) -> Python weekdays (Monday = 0)."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if part == "*":
            out |= set(range(7))
        elif "-" in part:
            a, b = (int(x) for x in part.split("-"))
            out |= {(d - 1) % 7 for d in range(a, b + 1)}
        else:
            out.add((int(part) - 1) % 7)
    return out


def jobs(cfg: AppConfig) -> dict[str, dict]:
    hh, mm = (int(x) for x in cfg.paper.schedule_time.split(":"))
    backup = (datetime(2000, 1, 1, hh, mm) + timedelta(hours=1)).time()
    return {"daily": {"args": ["paper", "run"], "at": datetime(2000, 1, 1, hh, mm).time(), "days": weekdays(cfg.paper.schedule_days), "log": "paper_daily.log"},
            "backup": {"args": ["paper", "backup", "--prune"], "at": backup, "days": set(range(7)), "log": "paper_backup.log"},
            "restore_test": {"args": ["paper", "restore-test"], "at": datetime(2000, 1, 1, 3, 0).time(), "days": {6}, "log": "paper_restore_test.log"}}


def due(cfg: AppConfig, state: dict, now: datetime, *, max_tries: int = MAX_TRIES, retry_minutes: int = RETRY_MINUTES) -> list[str]:
    """Names of the jobs that should run now."""
    today = now.date().isoformat()
    out = []
    for name, j in jobs(cfg).items():
        if now.weekday() not in j["days"] or now.time() < j["at"]:
            continue
        s = state.get(name) or {}
        if s.get("date") != today:
            out.append(name)
        elif s.get("status") == "failed" and s.get("tries", 0) < max_tries and now - datetime.fromisoformat(s["last"]) >= timedelta(minutes=retry_minutes):
            out.append(name)
    return out


def _state_path(root: Path) -> Path:
    return root / "logs" / "scheduler_state.json"


def load_state(root: Path = PROJECT_ROOT) -> dict:
    try:
        return json.loads(_state_path(root).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state: dict, root: Path = PROJECT_ROOT) -> None:
    p = _state_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(p)


def run_job(cfg: AppConfig, name: str, state: dict, now: datetime, root: Path = PROJECT_ROOT) -> int:
    """Run one job as a child process (its output goes to logs/<job>.log); update the state. Only one of them at a time, also across containers sharing the logs directory."""
    j = jobs(cfg)[name]
    (root / "logs").mkdir(parents=True, exist_ok=True)
    today = now.date().isoformat()
    prev = state.get(name) or {}
    tries = prev.get("tries", 0) + 1 if prev.get("date") == today else 1
    with open(root / "logs" / ".jobs.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open(root / "logs" / j["log"], "a") as log:
            log.write(f"\n===== {now:%Y-%m-%d %H:%M:%S %Z} {name} (try {tries}) =====\n")
            log.flush()
            code = subprocess.run([sys.executable, "-m", "predict_stock", *j["args"]], cwd=root, stdout=log, stderr=subprocess.STDOUT).returncode
    state[name] = {"date": today, "status": "ok" if code == 0 else "failed", "tries": tries, "last": now.replace(tzinfo=None).isoformat(timespec="seconds"), "exit_code": code}
    save_state(state, root)
    return code


def tick(cfg: AppConfig, now: datetime | None = None, root: Path = PROJECT_ROOT) -> dict[str, int]:
    """Run whatever is due at ``now`` (local time). Returns {job: exit code}."""
    now = now or datetime.now(TZ)
    state = load_state(root)
    out = {}
    for name in due(cfg, state, now.replace(tzinfo=None)):
        print(f"{now:%Y-%m-%d %H:%M} running {name}", flush=True)
        out[name] = run_job(cfg, name, state, now.replace(tzinfo=None), root)
        print(f"{datetime.now(TZ):%Y-%m-%d %H:%M} {name} {'ok' if out[name] == 0 else 'FAILED (see logs/' + jobs(cfg)[name]['log'] + ')'}", flush=True)
    return out


def wait_for_database(timeout: int = 300) -> None:
    from sqlalchemy import text
    from predict_stock.db.session import make_engine
    end, last = time.time() + timeout, None
    while time.time() < end:
        try:
            with make_engine("app").connect() as c:
                c.execute(text("SELECT 1"))
            return
        except Exception as exc:                                              # noqa: BLE001
            last = exc
            time.sleep(3)
    raise RuntimeError(f"database not reachable after {timeout}s: {last}")


def heartbeat(root: Path = PROJECT_ROOT) -> None:
    p = root / "logs" / ".scheduler_heartbeat"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(datetime.now(TZ).isoformat(timespec="seconds"))


def run_forever(cfg: AppConfig, poll_seconds: int = 30, root: Path = PROJECT_ROOT) -> None:
    wait_for_database()
    print(f"scheduler up: daily job {cfg.paper.schedule_time} days {cfg.paper.schedule_days} (Asia/Ho_Chi_Minh), backup an hour later, restore test Sundays 03:00; paper mode only", flush=True)
    while True:
        heartbeat(root)
        try:
            tick(cfg, root=root)
        except Exception as exc:                                              # noqa: BLE001 - the loop must survive a failing job
            print(f"scheduler error: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(poll_seconds)
