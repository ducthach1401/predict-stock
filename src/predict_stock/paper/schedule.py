"""Scheduling: cron. The jobs are idempotent and single-flight (`flock`), so an overlapping or repeated invocation is harmless. Nothing is installed automatically."""
from __future__ import annotations

from predict_stock.config import PROJECT_ROOT, AppConfig


def crontab(cfg: AppConfig, python: str | None = None) -> str:
    py = python or str(PROJECT_ROOT / ".venv" / "bin" / "python")
    hh, mm = cfg.paper.schedule_time.split(":")
    root = PROJECT_ROOT
    lock = "/tmp/predict_stock_paper.lock"
    return "\n".join([
        "# predict-stock: paper trading (paper mode only: no real order is ever placed)",
        "CRON_TZ=Asia/Ho_Chi_Minh",
        f"{int(mm)} {int(hh)} * * {cfg.paper.schedule_days} cd {root} && /usr/bin/flock -n {lock} {py} -m predict_stock paper run >> {root}/logs/paper_daily.log 2>&1",
        f"30 {(int(hh) + 1) % 24} * * * cd {root} && /usr/bin/flock -n {lock} {py} -m predict_stock paper backup --prune >> {root}/logs/paper_backup.log 2>&1",
        f"0 3 * * 0 cd {root} && /usr/bin/flock -n {lock} {py} -m predict_stock paper restore-test >> {root}/logs/paper_restore_test.log 2>&1",
        "",
    ])
