"""Backtests of the SWING strategies and of the baselines over the SAME window, and the pre-registered decision rule."""
from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from predict_stock.backtest import metrics as M
from predict_stock.backtest.baselines import BASELINES, PORTFOLIO_KEYS, BaselineSpec, make_signals
from predict_stock.backtest.engine import BacktestResult, Signal, run_backtest
from predict_stock.backtest.runner import Setup, clean, engine_config, evaluate_index
from predict_stock.config import SwingDecision

RANDOM_KEYS = ("random_weekly", "random_monthly")


def run_signals(setup: Setup, signals: dict[int, Signal], first_idx: int, mult: float, max_positions: int | None = None) -> BacktestResult:
    cfg = replace(engine_config(setup.cfg), max_positions=max_positions)
    return run_backtest(setup.data, signals, setup.rules.scaled_costs(mult), cfg, start=first_idx, end=setup.dev_end_idx + 1)


def _fold_metrics(equity: pd.Series, windows: list[dict], rf: float) -> list[dict]:
    out = []
    for w in windows:
        seg = equity[(equity.index >= pd.Timestamp(w["test"][0])) & (equity.index <= pd.Timestamp(w["test"][1]))]
        m = M.compute_metrics(seg, rf_annual=rf) if len(seg) > 20 else {}
        out.append({"fold": w["fold"], "test": w["test"], **{k: m.get(k) for k in ("cagr", "sharpe", "max_drawdown", "total_return")}})
    return out


def evaluate_signals(setup: Setup, signals: dict[int, Signal], first_idx: int, windows: list[dict], *, key: str, title: str,
                     multipliers: list[float], max_positions: int | None = None) -> tuple[dict, dict[str, pd.Series]]:
    """Net and gross metrics, cost sensitivity and per-walk-forward-window results of one signal set."""
    bt = setup.cfg.backtest
    runs = {m: run_signals(setup, signals, first_idx, m, max_positions) for m in sorted(set(multipliers) | {0.0, 1.0})}
    net, gross = runs[1.0], runs[0.0]
    rf = bt.risk_free_annual
    full = lambda r: M.compute_metrics(r.equity, trips=r.round_trips, fills=r.fills, exposure=r.exposure, rf_annual=rf)
    orders = net.orders.groupby(["kind", "status"]).size().to_dict() if len(net.orders) else {}
    summary = {"key": key, "title": title, "window": [str(net.equity.index[0].date()), str(net.equity.index[-1].date())], "n_signals": len(signals),
               "net": full(net), "gross": full(gross),
               "sensitivity": {str(m): {k: v for k, v in full(r).items() if k in ("cagr", "sharpe", "max_drawdown", "turnover_annual", "costs_paid", "slippage_paid")}
                               for m, r in runs.items()},
               "folds": _fold_metrics(net.equity, windows, rf), "orders": {f"{k[0]}/{k[1]}": int(v) for k, v in orders.items()},
               "blocked": net.stats["blocked"], "open_positions_at_end": net.stats["open_positions"],
               "exposure": {"mean": float(net.exposure.mean()), "idle_share": float((net.exposure < 0.01).mean())},
               "exit_reasons": net.round_trips["exit_reason"].value_counts().to_dict() if len(net.round_trips) else {}}
    return clean(summary), {"net": net.equity, "gross": gross.equity, "exposure": net.exposure}


def evaluate_baselines(setup: Setup, first_idx: int, windows: list[dict], multipliers: list[float]) -> dict[str, tuple[dict, dict]]:
    """Every Phase 4 baseline over the window that starts at ``first_idx`` (the first walk-forward test session), same engine, same costs."""
    bt = setup.cfg.backtest
    out = {}
    local = replace(setup, start_idx=first_idx)
    for key in PORTFOLIO_KEYS:
        spec = BASELINES[key]
        spec_k = spec if spec.k is None else BaselineSpec(**{**asdict(spec), "k": bt.top_k})
        sig = make_signals(spec_k, setup.frames[spec.dataset], setup.calendar, first_idx, setup.dev_end_idx, bt.max_weight)
        out[key] = evaluate_signals(setup, sig, first_idx, windows, key=key, title=spec.title, multipliers=multipliers)
    for key, spec in BASELINES.items():
        if spec.kind == "index" and spec.index_symbol in setup.index_close:
            summary, eq = evaluate_index(local, spec)
            summary["folds"] = _fold_metrics(eq["net"], windows, bt.risk_free_annual)
            out[key] = (summary, eq)
    return out


# ---- the pre-registered decision --------------------------------------------------------------------------------------------------
def decide(model: dict, baselines: dict[str, dict], noise: dict | None, ic: dict, rule: SwingDecision) -> dict:
    """PASS only if ALL of these hold (registered before any model was trained, config ``swing.decision``):
      1. net Sharpe of the primary strategy > the net Sharpe of EVERY portfolio baseline (not the random ones) over the same window;
      2. and > the 95th percentile of the random weekly portfolios' net Sharpe (luck with the same schedule, turnover and costs);
      3. mean daily rank IC > 0 with an overlap-adjusted t-statistic >= ``min_ic_tstat``;
      4. at least ``min_positive_fold_share`` of the walk-forward test windows have a positive net Sharpe;
      5. the net Sharpe is still > 0 at ``stress_cost_multiple`` times the configured costs.
    ``model`` / ``baselines`` are strategy summaries (``evaluate_signals``)."""
    sharpe = model["net"]["sharpe"]
    rivals = {k: v["net"]["sharpe"] for k, v in baselines.items() if k in PORTFOLIO_KEYS and k not in RANDOM_KEYS and v["net"].get("sharpe") is not None}
    best_key = max(rivals, key=rivals.get) if rivals else None
    p95 = (noise or {}).get("schedules", {}).get("weekly", {}).get("net", {}).get("sharpe", {}).get("p95")
    folds = [f["sharpe"] for f in model["folds"] if f.get("sharpe") is not None]
    pos_share = float(np.mean([s > 0 for s in folds])) if folds else float("nan")
    stress = model["sensitivity"].get(str(float(rule.stress_cost_multiple)), {}).get("sharpe")
    c = [
        {"name": "net Sharpe above every portfolio baseline", "value": sharpe, "threshold": rivals.get(best_key), "detail": f"best baseline: {best_key}",
         "ok": bool(best_key is not None and sharpe is not None and sharpe > rivals[best_key])},
        {"name": "net Sharpe above the random-weekly 95th percentile", "value": sharpe, "threshold": p95, "detail": "20 seeds, same window and costs",
         "ok": bool(p95 is not None and sharpe is not None and sharpe > p95)},
        {"name": f"rank IC t-statistic >= {rule.min_ic_tstat}", "value": ic.get("t_stat"), "threshold": rule.min_ic_tstat, "detail": f"mean IC {ic.get('mean')}",
         "ok": bool(ic.get("mean") is not None and ic["mean"] > 0 and ic.get("t_stat") is not None and ic["t_stat"] >= rule.min_ic_tstat)},
        {"name": f"share of test windows with positive net Sharpe >= {rule.min_positive_fold_share}", "value": pos_share, "threshold": rule.min_positive_fold_share,
         "detail": f"{sum(s > 0 for s in folds)} of {len(folds)} windows", "ok": bool(folds and pos_share >= rule.min_positive_fold_share)},
        {"name": f"net Sharpe > 0 at {rule.stress_cost_multiple:g}x costs", "value": stress, "threshold": 0.0, "detail": "", "ok": bool(stress is not None and stress > 0)},
    ]
    return clean({"passed": all(x["ok"] for x in c), "criteria": c})
