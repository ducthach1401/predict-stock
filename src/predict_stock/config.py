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


class SwingFolds(_Strict):
    scheme: str = "expanding"  # expanding | rolling
    train_min_sessions: int = 500  # first test window starts after this many sessions of data (+ embargo)
    train_window_sessions: int = 750  # rolling scheme only
    test_sessions: int = 125
    step_sessions: int = 125
    embargo_sessions: int = 10  # >= the triple-barrier horizon, so purging by label end is (almost) a no-op for the barrier label
    val_sessions: int = 125  # last sessions of each training window: early stopping, calibration, holding-time table


class SwingOptuna(_Strict):
    trials: int = 25  # hard cap: every trial (also the failed ones) is recorded in `experiments`
    seed: int = 42
    timeout_s: int | None = 1200


class SwingStrategy(_Strict):
    top_k: int = 10  # PRIMARY strategy: top-K by the rank score, equal weight, rebalanced weekly (like-for-like with the baselines)
    rebalance: str = "weekly"
    barrier_target_mult: float = 2.0  # SECONDARY strategy: barrier trades, exits by ATR multiples (same as the triple-barrier label)
    barrier_stop_mult: float = 1.0
    barrier_max_hold: int = 10
    min_prob_multiple: float = 1.0  # enter only if the calibrated probability >= this multiple of the training base rate
    max_positions: int = 10


class SwingDecision(_Strict):
    """PRE-REGISTERED (written to `experiments` before the first model is trained). The held-out period is opened only on PASS."""

    min_ic_tstat: float = 2.0  # mean daily rank IC vs the forward-return rank, overlap-adjusted t-statistic
    min_positive_fold_share: float = 0.6  # share of walk-forward test windows with net Sharpe > 0
    stress_cost_multiple: float = 2.0  # net Sharpe must stay > 0 at this multiple of the configured costs


class SwingConfig(_Strict):
    feature_set: str = "swing:1"
    label_spec: str = "swing:1"
    rank_target: str = "fwd_rank_5"
    return_target: str = "fwd_ret_5"
    event_label: str = "tb_label"  # event = the target is touched before the stop (tb_label == 1)
    hold_column: str = "tb_time"
    horizon: int = 10
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    n_estimators: int = 600
    early_stopping_rounds: int = 50
    calibration: str = "isotonic"  # isotonic | platt
    isotonic_min_bin: int = 150  # isotonic is fitted on equal-count bins of at least this many validation rows (raw points let a few lucky rows reach probability 1)
    holding_buckets: int = 10
    seed: int = 42
    num_threads: int = 4
    lgbm: dict = Field(default_factory=lambda: {"learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 200, "feature_fraction": 0.8,
                                                 "bagging_fraction": 0.8, "bagging_freq": 5, "lambda_l2": 10.0})
    folds: SwingFolds = Field(default_factory=SwingFolds)
    optuna: SwingOptuna = Field(default_factory=SwingOptuna)
    strategy: SwingStrategy = Field(default_factory=SwingStrategy)
    decision: SwingDecision = Field(default_factory=SwingDecision)
    sensitivity_k: list[int] = Field(default_factory=lambda: [5, 10, 20])
    sensitivity_min_prob: list[float] = Field(default_factory=lambda: [0.0, 1.0, 1.25, 1.5])
    sensitivity_atr: list[tuple[float, float]] = Field(default_factory=lambda: [(2.0, 1.0), (3.0, 1.5), (1.5, 1.0), (4.0, 2.0)])  # (target, stop)
    artifacts_dir: str = "artifacts/models"
    report_path: str = "docs/SWING.md"


class InvestPreset(_Strict):
    label: str
    horizon: int  # sessions: forward-return label horizon AND the embargo between training and test windows
    rebalance: str  # weekly | monthly | quarterly
    tranches: int = 2  # a rebalance is carried out in this many equal steps
    tranche_spacing: int = 5  # sessions between the steps
    drawdown_break: float = 0.15  # thesis-break condition: drawdown from the 252-session high beyond this


class InvestFolds(_Strict):
    train_min_sessions: int = 500
    test_sessions: int = 125
    step_sessions: int = 125
    val_sessions: int = 125  # last sessions of each training window (early stopping / hyper-parameter grid); its labels are purged like any other


class InvestRegime(_Strict):
    symbol: str = "VN30"
    fallback: str = "VNINDEX"  # used where the symbol has no 200-session average yet (VN30 exists only from 2020-05)
    sma_window: int = 200
    equity_share: float = 0.5  # share of the target stock weights kept when the index is below its average; the rest stays in cash


class InvestDca(_Strict):
    monthly_contribution: float = 10_000_000.0  # VND, invested on the first session of each month
    initial: float = 0.0


class InvestBootstrap(_Strict):
    resamples: int = 2000
    block: int = 21  # mean block length of the stationary bootstrap (sessions)
    level: float = 0.90  # two-sided confidence level
    seed: int = 20260920


class InvestDecision(_Strict):
    """PRE-REGISTERED (written to `experiments` before any INVEST model is trained). Judged for the candidate with the best net Sharpe of each preset."""

    reality_check_p: float = 0.10  # White's reality check over the candidates: p-value that the best one has no edge over equal-weight (Sharpe difference)
    min_ic_tstat: float = 2.0
    min_rolling_share: float = 0.60  # share of rolling 1-year windows in which the strategy's net return beats equal-weight
    stress_cost_multiple: float = 2.0


class InvestConfig(_Strict):
    feature_set: str = "invest:2"
    label_spec: str = "invest:1"
    presets: dict[str, InvestPreset] = Field(default_factory=lambda: {
        "b1": InvestPreset(label="INVEST B1 (1-3 months)", horizon=63, rebalance="monthly", tranches=2, tranche_spacing=5, drawdown_break=0.15),
        "b2": InvestPreset(label="INVEST B2 (6-12+ months)", horizon=126, rebalance="quarterly", tranches=2, tranche_spacing=10, drawdown_break=0.25)})
    candidates: list[str] = Field(default_factory=lambda: ["factor", "ridge", "elasticnet", "lgbm"])
    # the rule-based composite, fixed BEFORE any result was seen: (cross-sectional-rank column, sign). No parameter is fitted.
    factors: list[tuple[str, int]] = Field(default_factory=lambda: [("mom_6m_csrank", 1), ("mom_12m_csrank", 1), ("vol_126_csrank", -1),
                                                                      ("sma_ratio_200_csrank", 1), ("dd_252_csrank", 1)])
    ridge_alphas: list[float] = Field(default_factory=lambda: [1.0, 10.0, 100.0, 1000.0, 10000.0])
    enet_alphas: list[float] = Field(default_factory=lambda: [0.0005, 0.002, 0.01, 0.05])
    enet_l1_ratios: list[float] = Field(default_factory=lambda: [0.2, 0.5, 0.9])
    lgbm: dict = Field(default_factory=lambda: {"learning_rate": 0.03, "num_leaves": 4, "max_depth": 2, "min_child_samples": 400, "lambda_l2": 50.0,
                                                 "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 5})
    n_estimators: int = 400
    early_stopping_rounds: int = 40
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    seed: int = 42
    num_threads: int = 4
    folds: InvestFolds = Field(default_factory=InvestFolds)
    top_k: int = 10
    weighting: str = "equal"  # equal | inverse_vol | risk_parity | min_variance | hrp
    max_weight: float = 0.15
    cov_window: int = 252  # trailing sessions for the covariance-based weightings (point-in-time)
    regime: InvestRegime = Field(default_factory=InvestRegime)
    dca: InvestDca = Field(default_factory=InvestDca)
    bootstrap: InvestBootstrap = Field(default_factory=InvestBootstrap)
    decision: InvestDecision = Field(default_factory=InvestDecision)
    rolling_windows: list[int] = Field(default_factory=lambda: [252, 756])
    thesis_rs_floor: float = 0.40  # thesis-break: cross-sectional rank of 6-month momentum (relative strength) below this
    sensitivity_k: list[int] = Field(default_factory=lambda: [8, 10, 15])
    sensitivity_weighting: list[str] = Field(default_factory=lambda: ["inverse_vol", "risk_parity", "min_variance", "hrp"])
    sensitivity_tranches: list[int] = Field(default_factory=lambda: [1, 3])
    fundamental_prefixes: list[str] = Field(default_factory=lambda: ["fund_"])  # feature columns starting with these get their own ablation report
    artifacts_dir: str = "artifacts/models"
    report_dir: str = "docs"


class RecoSwing(_Strict):
    """SWING card rules. Every number here is a rule stated BEFORE the recommendations were backtested; none was fitted to the backtest."""

    max_positions: int = 6  # at most this many SWING positions / new cards per day (the sleeve is 30% of capital, 5% per name)
    entry_style: str = "auto"  # auto | pullback | breakout | ato   (auto: breakout when the close is within `breakout_within_pct` of the 20-session high, else pullback)
    breakout_within_pct: float = 0.02
    breakout_buffer_atr: float = 0.10  # breakout trigger = 20-session high + this many ATR (rounded up to the tick)
    breakout_zone_atr: float = 0.30  # do not chase above trigger + this many ATR
    pullback_atr: tuple[float, float] = (0.60, 0.15)  # limit zone = close - (0.60 .. 0.15) ATR
    ato_zone_pct: tuple[float, float] = (-0.01, 0.01)  # ATO only if the open is within these percent of the previous close
    stop_atr: float = 1.0  # stop = reference entry - stop_atr x ATR
    target1_atr: float = 1.0
    target2_atr: float = 2.0
    use_resistance: bool = True  # cap the targets just under the nearest resistance (the highest high of the last `resistance_lookback` sessions) when it lies above the entry
    resistance_lookback: int = 60
    t1_fraction: float = 0.5  # share of the position sold at target 1
    breakeven_after_t1: bool = True  # after target 1 the stop moves to the average entry price (from the next session)
    max_hold: int = 10  # time-stop, sessions
    expected_hold_cap: int = 10
    min_rr: float = 1.5  # a card whose reward:risk (to target 2) is below this is NOT issued
    validity_sessions: int = 3
    risk_per_trade: float = 0.0075  # share of total capital lost if the stop is hit (0.5% - 1% is the brief's range)
    max_weight: float = 0.05  # per name, share of total capital


class RecoInvest(_Strict):
    entry_zone_pct: tuple[float, float] = (-0.02, 0.005)  # buy limit zone around the close; the order is a limit at the zone's high
    validity_sessions: int = 5
    max_weight: float = 0.10  # per name per preset, share of total capital
    watch_ranks: int = 5  # names ranked just below the top-K get a WATCH card
    time_review_multiple: float = 1.0  # the maximum holding time = horizon x this


class RecoPortfolio(_Strict):
    capital: float = 1_000_000_000.0
    allocation: dict[str, float] = Field(default_factory=lambda: {"swing": 0.30, "invest": 0.70})
    invest_split: dict[str, float] = Field(default_factory=lambda: {"b1": 0.5, "b2": 0.5})  # of the INVEST share
    max_weight_name: float = 0.15  # a name in total over all sleeves
    max_open_risk: float = 0.05  # sum over open SWING cards of the loss if every stop is hit, share of total capital
    kill_drawdown: float = 0.20  # portfolio drawdown from its peak that stops all trading
    kill_cooldown: int = 21  # sessions flat after a kill-switch before trading resumes


class RecoConfig(_Strict):
    swing: RecoSwing = Field(default_factory=RecoSwing)
    invest: RecoInvest = Field(default_factory=RecoInvest)
    portfolio: RecoPortfolio = Field(default_factory=RecoPortfolio)
    card_model: dict[str, str] = Field(default_factory=lambda: {"b1": "factor", "b2": "factor"})  # the INVEST candidate behind the cards (the pre-declared, nothing-fitted one)
    similar_min_n: int = 30  # a similar-signal group with fewer past signals than this is flagged
    evidence_min_rows: int = 3000  # past out-of-sample rows needed before the display grade can be anything but "not enough evidence"
    evidence_auc_ok: float = 0.55  # below this AUC of past out-of-sample predictions the model probability is not shown as a point estimate
    gate_on_verdict: bool = False  # True: a sleeve whose model failed the pre-registered criteria only issues WATCH cards (paper trading: False)
    kill_switch: bool = True
    report_path: str = "docs/RECOMMENDATIONS.md"
    artifacts_dir: str = "artifacts/reco"


class PaperRemoval(_Strict):
    """What happens to a position when its stock leaves the universe. Never an urgent sale unless the config says `close_now`."""

    swing: str = "hold_until_exit"  # hold_until_exit (target / stop / time-stop) | close_now
    invest: str = "next_rebalance"  # next_rebalance | close_now


class PaperNotify(_Strict):
    telegram: bool = False  # needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment
    email: bool = False  # needs SMTP_HOST, SMTP_USER, SMTP_PASSWORD, SMTP_FROM, SMTP_TO in the environment
    only_on_alerts_or_cards: bool = False


class PaperConfig(_Strict):
    portfolio_code: str = "paper"
    start_date: str | None = None  # first session whose cards belong to the paper record; set by `paper init` (never in the past of the data cut-off)
    universe_snapshot: str | None = None  # CSV kept by the user (symbol[, exchange, weight, ...]); applied every run, a no-op when nothing changed
    on_removal: PaperRemoval = Field(default_factory=PaperRemoval)
    stop_on_quality_error: bool = True  # unexplained data errors: no new cards that day (open positions are still managed)
    stale_days: int = 4  # the newest bar older than this many calendar days is an alert
    min_share_with_bar: float = 0.90  # members with a bar on the as-of date; below this no cards are issued
    pending_min_sessions: int = 5  # a recommendation older than validity + this many sessions is closed out of the pending list
    report_dir: str = "reports/paper"
    report_formats: list[str] = Field(default_factory=lambda: ["md", "html", "csv"])
    notify: PaperNotify = Field(default_factory=PaperNotify)
    schedule_time: str = "16:30"  # local time (Asia/Ho_Chi_Minh), after the close and after the vendor publishes the day's bars
    schedule_days: str = "1-5"


class BackupConfig(_Strict):
    dir: str = "backups"
    daily_keep: int = 14  # newest dumps kept
    weekly_keep: int = 8  # plus the oldest dump of each of the last N weeks
    restore_database: str = "predict_stock_restore"  # scratch database for the restore test (created and dropped with the admin credentials)
    compress: bool = True


class RunConfig(_Strict):
    mode: str = "paper"  # backtest | paper. There is no live-trading mode: no code in this project places a real order.


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
    swing: SwingConfig = Field(default_factory=SwingConfig)
    invest: InvestConfig = Field(default_factory=InvestConfig)
    reco: RecoConfig = Field(default_factory=RecoConfig)
    paper: PaperConfig = Field(default_factory=PaperConfig)
    backup: BackupConfig = Field(default_factory=BackupConfig)
    run: RunConfig = Field(default_factory=RunConfig)

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
