"""Runs the baselines on the development period: data setup, simulation before/after costs, cost sensitivity, sub-periods, folds.

The held-out final period is NEVER touched here: every run stops at the last session before it (`assert_development_only`),
and revealing it goes through ``walkforward.reveal_holdout`` (once)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from predict_stock.backtest import metrics as M
from predict_stock.backtest.baselines import BASELINES, PORTFOLIO_KEYS, BaselineSpec, make_signals
from predict_stock.backtest.engine import EngineConfig, MarketData, run_backtest
from predict_stock.backtest.market import MarketRules, infer_bands
from predict_stock.backtest.walkforward import Holdout, assert_development_only, make_folds, make_holdout, purged_count
from predict_stock.config import AppConfig
from predict_stock.db.models import InstrumentSymbolHistory as SymbolRow
from predict_stock.db.models import TradingCalendar
from predict_stock.db.repo import find_symbol_row
from predict_stock.db.session import session_scope
from predict_stock.features.data import _bars, load_panel
from predict_stock.features.dataset import build_dataset, read_dataset
from predict_stock.features.registry import spec_hash

BACKTEST_VERSION = 2  # bump when simulation semantics change (part of the config hash). v2: rebalance threshold relative to the target position
DATASETS = {"swing": (("swing", 1), ("swing", 1)), "invest": (("invest", 2), ("invest", 1))}   # feature set / label spec per dataset


@dataclass
class Setup:
    cfg: AppConfig
    rules: MarketRules
    data: MarketData
    calendar: pd.DatetimeIndex
    frames: dict[str, pd.DataFrame]
    index_close: dict[str, pd.Series]
    start_idx: int
    dev_end_idx: int
    holdout: Holdout
    dataset_hashes: dict[str, str] = field(default_factory=dict)
    dataset_ids: dict[str, int] = field(default_factory=dict)
    universe: str = ""
    panel_info: dict = field(default_factory=dict)

    @property
    def dev_sessions(self) -> pd.DatetimeIndex:
        return self.calendar[self.start_idx: self.dev_end_idx + 1]


def exchange_bands(session: Session, close: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame | None:
    """Band from the KNOWN exchange of each instrument on each date (symbol history); None if no exchange is known anywhere."""
    rows = session.execute(select(SymbolRow.instrument_id, SymbolRow.exchange, SymbolRow.valid_from, SymbolRow.valid_to)
                           .where(SymbolRow.instrument_id.in_(list(close.columns)), SymbolRow.exchange.is_not(None))).all()
    if not rows:
        return None
    out = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for iid, ex, vf, vt in rows:
        sel = close.index >= pd.Timestamp(vf)
        if vt is not None:
            sel &= close.index < pd.Timestamp(vt)
        out.loc[sel, iid] = cfg.market.price_limit_pct[ex]
    return out


def load_setup(engine: Engine, cfg: AppConfig, universe: str | None = None) -> Setup:
    universe = universe or cfg.universe.training_code
    bt = cfg.backtest
    today, hist = date.today(), date.fromisoformat(cfg.ingest.history_start)
    frames, hashes, ids_by_name = {}, {}, {}
    for name, (fs, ls) in DATASETS.items():                          # built (or reused) through the Phase 3 job, so they are reproducible
        res = build_dataset(engine, cfg, universe_code=universe, feature_set=fs, label_spec=ls, start=hist, end=today)
        frames[name] = read_dataset(res.path, verify_sha256=res.sha256)
        hashes[name] = res.content_hash
        ids_by_name[name] = res.dataset_id
    ids = sorted(set(frames["swing"]["instrument_id"]) | set(frames["invest"]["instrument_id"]))
    rules = MarketRules.from_config(cfg.market)
    with session_scope(engine) as s:
        last_day = s.scalar(select(func.max(TradingCalendar.trade_date)).where(TradingCalendar.calendar_code == cfg.ingest.calendar_code))
        panel = load_panel(s, calendar_code=cfg.ingest.calendar_code, instrument_ids=ids, load_start=hist, cutoff=last_day)
        cal = panel.calendar
        index_close = {}
        for sym in bt.benchmark_symbols:
            row = find_symbol_row(s, sym)
            if row is not None:
                b = _bars(s, [row.instrument_id], hist, last_day)
                index_close[sym] = b.set_index("trade_date")["close"].reindex(cal).ffill(limit=cfg.features.benchmark_ffill_limit)
        ex_band = exchange_bands(s, panel.close, cfg)
    bands = infer_bands(panel.close, rules, cfg.market.band_inference_window, ex_band)
    data = MarketData(panel.open, panel.high, panel.low, panel.close, bands)
    holdout = make_holdout(cal, bt.oos_months)
    dev_end_idx = int(cal.get_loc(holdout.start)) - 1
    start_idx = int(cal.searchsorted(pd.Timestamp(bt.start)))
    assert_development_only(cal[dev_end_idx], holdout)
    return Setup(cfg, rules, data, cal, frames, index_close, start_idx, dev_end_idx, holdout, hashes, ids_by_name, universe,
                 {"instruments": len(ids), "repaired_bars": panel.repaired_bars, "band_source": "known exchange" if ex_band is not None else "inferred from trailing returns",
                  "open_missing_bars": int((panel.open.isna() & panel.close.notna()).to_numpy().sum())})


# ---- helpers ------------------------------------------------------------------------------------------------------------
def clean(o):
    """JSON-safe copy: numpy -> python, NaN/inf -> None, Timestamps -> ISO strings."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (pd.Timestamp, date)):
        return str(o)[:10]
    return o


def engine_config(cfg: AppConfig) -> EngineConfig:
    bt = cfg.backtest
    return EngineConfig(capital=bt.capital, rebalance_threshold=bt.rebalance_threshold, tie=bt.tie)


def config_hash(setup: Setup, spec: BaselineSpec) -> str:
    bt = setup.cfg.backtest
    return spec_hash({"v": BACKTEST_VERSION, "spec": asdict(spec), "rules": asdict(setup.rules), "engine": asdict(engine_config(setup.cfg)),
                      "window": [str(setup.calendar[setup.start_idx].date()), str(setup.calendar[setup.dev_end_idx].date())],
                      "holdout": str(setup.holdout.start.date()), "datasets": setup.dataset_hashes, "top_k": bt.top_k, "max_weight": bt.max_weight,
                      "multipliers": bt.cost_multipliers, "regime": [bt.regime_window, bt.regime_threshold], "rf": bt.risk_free_annual,
                      "folds": [bt.wf_train_sessions, bt.wf_test_sessions, bt.wf_embargo_sessions], "universe": setup.universe})


def _regimes(setup: Setup, index: pd.DatetimeIndex) -> pd.Series | None:
    ref = setup.index_close.get("VNINDEX")
    if ref is None:
        return None
    return M.classify_regimes(ref.reindex(index).ffill().bfill(), setup.cfg.backtest.regime_window, setup.cfg.backtest.regime_threshold)


def _core(equity: pd.Series) -> dict:
    return {k: v for k, v in M.compute_metrics(equity).items() if k in ("cagr", "sharpe", "max_drawdown", "total_return")}


def _era(setup: Setup, equity: pd.Series, sym: str = "VN30") -> dict | None:
    """Metrics over the window in which the VN30 exists (from its first value), for a like-for-like comparison with that benchmark."""
    s = setup.index_close.get(sym)
    if s is None or s.first_valid_index() is None:
        return None
    sub = equity[equity.index >= s.first_valid_index()]
    return {"from": str(sub.index[0].date()), **M.compute_metrics(sub, rf_annual=setup.cfg.backtest.risk_free_annual)} if len(sub) > 60 else None


def _folds(setup: Setup, equity: pd.Series) -> list[dict]:
    bt = setup.cfg.backtest
    folds = make_folds(setup.dev_sessions, scheme="expanding", train_sessions=bt.wf_train_sessions, test_sessions=bt.wf_test_sessions,
                       embargo_sessions=bt.wf_embargo_sessions, end=setup.holdout.start)
    out = []
    for f in folds:
        seg = equity[(equity.index >= f.test_start) & (equity.index <= f.test_end)]
        out.append({**f.as_dict(), **{k: v for k, v in M.compute_metrics(seg, rf_annual=bt.risk_free_annual).items() if k in ("cagr", "sharpe", "max_drawdown", "total_return")}})
    return out


def with_label_end(frame: pd.DataFrame) -> pd.DataFrame:
    """Adds ``label_end_max``: per row, the LATEST end date among all its labels (``fwd_end_*`` and ``tb_end``). Purging by it
    removes a training sample when ANY of its labels is still unresolved at the start of the test window."""
    cols = [c for c in frame.columns if c.startswith(("fwd_end_", "tb_end"))]
    out = frame[["trade_date"]].copy()
    out["label_end_max"] = frame[cols].apply(pd.to_datetime).max(axis=1) if cols else pd.NaT
    return out


def fold_purging(setup: Setup) -> list[dict]:
    """How many training samples each fold's purging removes, on the real datasets (label end columns of Phase 3)."""
    bt = setup.cfg.backtest
    ends = {name: with_label_end(f) for name, f in setup.frames.items()}
    out = []
    for f in make_folds(setup.dev_sessions, scheme="expanding", train_sessions=bt.wf_train_sessions, test_sessions=bt.wf_test_sessions,
                        embargo_sessions=bt.wf_embargo_sessions, end=setup.holdout.start):
        purged = {name: purged_count(e, f, "label_end_max") for name, e in ends.items()}
        trained = {name: int(((e["trade_date"] >= f.train_start) & (e["trade_date"] <= f.train_end)).sum()) for name, e in ends.items()}
        out.append({**f.as_dict(), "purged": purged, "train_rows": trained})
    return out


def _simulate(setup: Setup, signals, mult: float):
    return run_backtest(setup.data, signals, setup.rules.scaled_costs(mult), engine_config(setup.cfg), start=setup.start_idx, end=setup.dev_end_idx + 1)


def evaluate_portfolio(setup: Setup, spec: BaselineSpec) -> tuple[dict, dict[str, pd.Series]]:
    bt = setup.cfg.backtest
    frame = setup.frames[spec.dataset]
    spec_k = spec if spec.k is None else BaselineSpec(**{**asdict(spec), "k": bt.top_k})
    signals = make_signals(spec_k, frame, setup.calendar, setup.start_idx, setup.dev_end_idx, bt.max_weight)
    runs = {m: _simulate(setup, signals, m) for m in sorted(set(bt.cost_multipliers) | {0.0, 1.0})}
    net, gross = runs[1.0], runs[0.0]
    rf = bt.risk_free_annual
    full = lambda r: M.compute_metrics(r.equity, trips=r.round_trips, fills=r.fills, exposure=r.exposure, rf_annual=rf)
    reg = _regimes(setup, net.equity.index)
    orders = net.orders.groupby(["kind", "status"]).size().to_dict() if len(net.orders) else {}
    summary = {
        "key": spec.key, "title": spec.title, "description": spec.description, "kind": "portfolio", "config_hash": config_hash(setup, spec),
        "window": [str(net.equity.index[0].date()), str(net.equity.index[-1].date())], "n_rebalances": len(signals),
        "net": full(net), "gross": full(gross),
        "sensitivity": {str(m): {**{k: v for k, v in full(r).items() if k in ("cagr", "sharpe", "max_drawdown", "turnover_annual", "costs_paid", "slippage_paid")}} for m, r in runs.items()},
        "vn30_era_net": _era(setup, net.equity), "regimes": M.by_regime(net.equity, reg, rf) if reg is not None else {}, "years": M.by_year(net.equity, rf),
        "folds": _folds(setup, net.equity), "orders": {f"{k[0]}/{k[1]}": int(v) for k, v in orders.items()},
        "blocked": net.stats["blocked"], "open_positions_at_end": net.stats["open_positions"],
        "open_missing_bars": setup.panel_info.get("open_missing_bars"),
    }
    return clean(summary), {"net": net.equity, "gross": gross.equity, "exposure": net.exposure}


def evaluate_index(setup: Setup, spec: BaselineSpec) -> tuple[dict, dict[str, pd.Series]]:
    bt, rules = setup.cfg.backtest, setup.rules
    px = setup.index_close[spec.index_symbol].iloc[setup.start_idx: setup.dev_end_idx + 1]
    px = px[px.index >= px.first_valid_index()].dropna()
    gross = bt.capital * px / px.iloc[0]

    def with_costs(k: float) -> pd.Series:
        c = rules.scaled_costs(k)
        e = gross * (1 - c.fee_rate - c.slippage_rate)                        # buy once at the start
        e.iloc[-1] = e.iloc[-1] * (1 - c.fee_rate - c.sell_tax_rate - c.slippage_rate)   # sell at the end (for the final figure)
        return e
    net = with_costs(1.0)
    rf = bt.risk_free_annual
    reg = _regimes(setup, net.index)
    zero_fills = pd.DataFrame(columns=["value", "fee", "tax", "slippage_cost"])
    summary = {"key": spec.key, "title": spec.title, "description": spec.description, "kind": "index", "config_hash": config_hash(setup, spec),
               "window": [str(net.index[0].date()), str(net.index[-1].date())],
               "net": M.compute_metrics(net, fills=zero_fills, rf_annual=rf), "gross": M.compute_metrics(gross, fills=zero_fills, rf_annual=rf),
               "sensitivity": {str(m): {k: v for k, v in M.compute_metrics(with_costs(m), rf_annual=rf).items() if k in ("cagr", "sharpe", "max_drawdown")} for m in bt.cost_multipliers},
               "vn30_era_net": _era(setup, net), "regimes": M.by_regime(net, reg, rf) if reg is not None else {}, "years": M.by_year(net, rf),
               "folds": _folds(setup, net), "orders": {}, "blocked": {}, "note": "buy once, sell at the end; a price index (no dividends)"}
    return clean(summary), {"net": net, "gross": gross}


def run_all(setup: Setup, keys: list[str] | None = None) -> dict:
    """{key: (summary, equities)} for the requested baselines (default: all)."""
    out = {}
    for key in keys or list(BASELINES):
        spec = BASELINES[key]
        if spec.kind == "index" and spec.index_symbol not in setup.index_close:
            continue
        out[key] = evaluate_portfolio(setup, spec) if spec.kind == "portfolio" else evaluate_index(setup, spec)
    return out
