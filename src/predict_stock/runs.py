"""Job-run tracking: config snapshot, seed and git commit for every run (principle 6)."""
from __future__ import annotations

import subprocess
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from predict_stock.config import PROJECT_ROOT, AppConfig
from predict_stock.db.models import JobRun


def git_state() -> tuple[str | None, bool | None]:
    """(HEAD commit, dirty?) — (None, None) when git or a commit is unavailable."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@contextmanager
def tracked_run(engine: Engine, job_name: str, cfg: AppConfig, params: dict | None = None) -> Iterator[tuple[int, dict]]:
    """Yield (run_id, stats). ``stats`` is persisted on exit; failures are recorded
    in their own transaction so they survive the caller's rollback."""
    commit, dirty = git_state()
    with Session(engine) as s:
        run = JobRun(
            job_name=job_name,
            status="running",
            git_commit=commit,
            git_dirty=dirty,
            seed=cfg.seed,
            config_snapshot=cfg.snapshot(),
            params=params,
        )
        s.add(run)
        s.commit()
        run_id = run.id
    stats: dict = {}
    try:
        yield run_id, stats
    except BaseException:
        _finish(engine, run_id, "failed", stats, traceback.format_exc()[-4000:])
        raise
    else:
        _finish(engine, run_id, "success", stats, None)


def _finish(engine: Engine, run_id: int, status: str, stats: dict, error: str | None) -> None:
    with Session(engine) as s:
        run = s.get(JobRun, run_id)
        run.status, run.stats, run.error, run.finished_at = status, stats, error, _utcnow()
        s.commit()
