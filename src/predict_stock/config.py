"""Typed application config (YAML) and DB URLs (environment).

Secrets never live in YAML: DB credentials are read from the environment /
.env only. ``AppConfig.snapshot()`` is what gets stored with every job run.
"""
from __future__ import annotations

import os
from datetime import time
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.engine import URL

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MarketConfig(_Strict):
    long_only: bool = True
    lot_size: int = 100
    settlement_days: int = 2
    fee_rate: float = 0.0015
    sell_tax_rate: float = 0.001
    slippage_rate: float = 0.001
    price_limit_pct: dict[str, float] = Field(
        default_factory=lambda: {"HOSE": 0.07, "HNX": 0.10, "UPCOM": 0.15}
    )
    default_exchange: str = "HOSE"
    tick_size_vnd: list[tuple[float, float]] = Field(
        default_factory=lambda: [(0, 10), (10000, 50), (50000, 100)]
    )

    def price_limit(self, exchange: str | None = None) -> float:
        return self.price_limit_pct[exchange or self.default_exchange]


class DnseConfig(_Strict):
    base_url: str
    timeout_s: float = 30
    min_interval_s: float = 0.2
    max_retries: int = 4
    backoff_base_s: float = 1.0
    user_agent: str = "predict-stock research client"
    price_multiplier_stock: int = 1000
    price_multiplier_index: int = 1
    resolution: str = "1D"
    bar_timezone: str = "Asia/Ho_Chi_Minh"


class IngestConfig(_Strict):
    history_start: str = "2018-01-01"
    overlap_calendar_days: int = 21
    adjust_rel_tolerance: float = 0.0005
    session_close_time: str = "15:00"
    final_bar_buffer_minutes: int = 15
    calendar_min_stock_fraction: float = 0.5
    benchmark_symbols: list[str] = Field(default_factory=lambda: ["VNINDEX"])

    def close_time(self) -> time:
        hh, mm = self.session_close_time.split(":")
        return time(int(hh), int(mm))


class QualityConfig(_Strict):
    big_move_tolerance: float = 0.01


class AppConfig(_Strict):
    seed: int = 42
    market: MarketConfig = Field(default_factory=MarketConfig)
    dnse: DnseConfig
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)

    def snapshot(self) -> dict:
        """JSON-serialisable copy for job_runs.config_snapshot (contains no secrets)."""
        return self.model_dump(mode="json")


def load_config(path: str | os.PathLike | None = None) -> AppConfig:
    load_dotenv(PROJECT_ROOT / ".env")
    cfg_path = Path(path or os.environ.get("PREDICT_CONFIG") or DEFAULT_CONFIG_PATH)
    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return AppConfig.model_validate(raw)


# ---- database URLs ---------------------------------------------------------

Role = Literal["app", "migrator"]


def db_url(role: Role = "app", *, test: bool = False) -> URL:
    """Build a SQLAlchemy URL from environment variables (loads .env if present)."""
    load_dotenv(PROJECT_ROOT / ".env")
    prefix = "MYSQL_APP" if role == "app" else "MYSQL_MIGRATOR"
    try:
        user = os.environ[f"{prefix}_USER"]
        password = os.environ[f"{prefix}_PASSWORD"]
        database = os.environ["MYSQL_TEST_DATABASE" if test else "MYSQL_DATABASE"]
    except KeyError as exc:  # fail loudly, never fall back to defaults with secrets
        raise RuntimeError(f"missing environment variable {exc.args[0]} (see .env.example)") from exc
    return URL.create(
        "mysql+pymysql",
        username=user,
        password=password,
        host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
        port=int(os.environ.get("MYSQL_PORT", "3307")),
        database=database,
        query={"charset": "utf8mb4"},
    )
