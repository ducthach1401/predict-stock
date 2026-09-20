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
    band_inference_window: int = 252  # sessions of return history used to infer the daily price band when the exchange is unknown
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


class UniverseConfig(_Strict):
    """Which universe plays which role. Codes are data: any universe loaded via
    `universe apply` can be named here. Training is normally wider than trading."""

    training_code: str  # the model learns from these instruments
    trading_code: str  # only these may receive recommendations
    history_floor: str = "2000-01-01"  # symbol history of instruments with unknown listing date starts here

    def floor(self):
        from datetime import date
        return date.fromisoformat(self.history_floor)


class IntradayConfig(_Strict):
    """Optional intraday bars. DNSE served 1H bars only from 2023-09-21 when probed."""

    enabled: bool = False
    resolution: str = "1H"
    history_start: str = "2023-09-01"
    overlap_calendar_days: int = 5


class IngestConfig(_Strict):
    history_start: str = "2018-01-01"
    overlap_calendar_days: int = 21
    adjust_rel_tolerance: float = 0.0005
    session_close_time: str = "15:00"
    final_bar_buffer_minutes: int = 15
    min_sessions: int = 500  # daily sessions an instrument needs before it is usable for training
    auto_backfill: bool = True  # `universe apply` backfills newly added instruments
    intraday: IntradayConfig = Field(default_factory=IntradayConfig)
    calendar_code: str = "VN_CONSENSUS"
    calendar_min_stock_fraction: float = 0.5
    calendar_grace_days: int = 30  # a stock stays 'active' this long after its last bar
    benchmark_symbols: list[str] = Field(default_factory=lambda: ["VNINDEX"])

    def close_time(self) -> time:
        hh, mm = self.session_close_time.split(":")
        return time(int(hh), int(mm))


class QualityConfig(_Strict):
    big_move_tolerance: float = 0.01
    return_outlier_z: float = 8.0  # robust z (median/MAD) of daily log returns
    volume_spike_ratio: float = 30.0  # volume / median of the previous `volume_spike_window` sessions
    volume_spike_window: int = 60
    price_range_vnd: tuple[float, float] = (500.0, 5_000_000.0)  # plausible stock price, VND
    unit_jump_ratio: float = 100.0  # day-over-day ratio beyond this (either way) = probable unit change
    report_path: str = "docs/DATA_QUALITY.md"
    known_issues_path: str = "data/quality_known_issues.csv"


class AdjustmentConfig(_Strict):
    """Detection of unadjusted corporate actions. Candidates are only ever reported."""

    gap_tolerance: float = 0.01  # open/prev_close gap beyond (price band + this) = candidate
    detect_max_band: bool = True  # unknown exchange history: use the widest configured band


class FundamentalsConfig(_Strict):
    enabled: bool = False  # DNSE's public endpoint has no fundamentals; strategies must run on prices alone


class FeaturesConfig(_Strict):
    definitions_path: str = "config/feature_sets.yaml"
    benchmark_symbol: str | None = "VN30"  # relative strength / downside beta reference
    benchmark_fallback_symbol: str | None = None  # e.g. VNINDEX: used only before the primary exists (VN30 starts 2020-05-11)
    benchmark_ffill_limit: int = 5  # sessions a benchmark value may be carried forward over its own gaps
    fundamentals: FundamentalsConfig = Field(default_factory=FundamentalsConfig)


class DatasetConfig(_Strict):
    dir: str = "artifacts/datasets"
    audit_lookahead: bool = True  # prove, on every build, that features do not use data after t
    audit_cuts: int = 3
    require_ready: bool = True  # skip instruments blocked for insufficient history (ingest.min_sessions) at the end date


class BacktestConfig(_Strict):
    capital: float = 1_000_000_000.0  # VND
    start: str = "2019-03-01"  # first session of the evaluation (the INVEST features need ~274 sessions of warm-up)
    oos_months: int = 12  # the last months are held out and evaluated once
    top_k: int = 10
    max_weight: float | None = 0.15  # cap per instrument
    rebalance_threshold: float = 0.20  # skip a position change smaller than this fraction of the TARGET position value
    tie: str = "stop_first"  # both barriers in one bar
    risk_free_annual: float = 0.0
    cost_multipliers: list[float] = Field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0, 3.0])  # of fee / tax / slippage
    regime_window: int = 126  # sessions per market-condition window
    regime_threshold: float = 0.10  # benchmark move separating up / sideways / down
    wf_train_sessions: int = 500
    wf_test_sessions: int = 250
    wf_embargo_sessions: int = 10
    benchmark_symbols: list[str] = Field(default_factory=lambda: ["VNINDEX", "VN30"])
    report_path: str = "docs/BASELINES.md"
    image_dir: str = "docs/img"
    artifacts_dir: str = "artifacts/backtests"


class AppConfig(_Strict):
    seed: int = 42
    market: MarketConfig = Field(default_factory=MarketConfig)
    dnse: DnseConfig
    universe: UniverseConfig
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    adjustments: AdjustmentConfig = Field(default_factory=AdjustmentConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    datasets: DatasetConfig = Field(default_factory=DatasetConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

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
