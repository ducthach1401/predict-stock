"""The `backtest` jobs: baselines on the development period (+ report, charts, DB) and the one-time held-out evaluation."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from predict_stock.backtest import metrics as M
from predict_stock.backtest.baselines import BASELINES, PORTFOLIO_KEYS, BaselineSpec, make_signals
from predict_stock.backtest.report import (
    plot_equity, plot_sensitivity, render_markdown, save_equity_artifacts, store_experiments, write_if_changed,
)
from predict_stock.backtest.runner import Setup, _simulate, clean, config_hash, fold_purging, load_setup, run_all
from predict_stock.backtest.walkforward import oos_status, reveal_holdout
from predict_stock.config import PROJECT_ROOT, AppConfig
from predict_stock.db.session import session_scope
from predict_stock.features.registry import spec_hash
from predict_stock.runs import save_config_snapshot, tracked_run

ORDER = ["equal_weight", "random_weekly", "random_monthly", "mom_short", "mean_reversion", "mom_long", "mom_long_invvol", "bh_vnindex", "bh_vn30"]
LIMITS = [
    "**Universe look-ahead / survivorship.** LARGE50 = the 50 most traded names *today*, applied back to 2019 (Phase 1). Every strategy here is more attractive than it could have been in real time.",
    "**Prices are the vendor's back-adjusted series** (Phase 0). Returns are right (dividends are effectively reinvested); lots, tick sizes and price bands are applied to adjusted levels, so they are right in logic but only approximate in level. The benchmarks are price indices without dividends.",
    "**Price band inferred**, not known: no exchange history exists for LARGE50, so each instrument's band on a date is the smallest of 7 / 10 / 15% that contains its largest move in the previous 252 sessions (a change of exchange is picked up with a lag). Bars whose open was unreliable (repair rule) cannot fill market orders that day.",
    "**T+2 for the whole period.** The settlement cycle before 2022 may have been longer (not verified); shares are treated as sellable for the whole session two sessions after purchase (slightly optimistic), and sale proceeds are reusable at once.",
    "**Costs are the brief's defaults** (fee 0.15% a side, tax 0.1% on sells, slippage 0.1% against you), not re-verified. There is no market-impact or volume-participation model: with about 100 M VND per position against billions of VND traded per session it is negligible here, but it would not be for larger capital.",
    "**Limit / stop / target machinery is tested but the baselines do not use it** (they trade at the next open). Daily bars cannot show the intraday order of a stop and a target; the stop is taken first.",
    "**Small sample.** ~6.5 years, 50 names, one path of history. The standard error of a Sharpe ratio over that span is roughly 0.4, so differences of a few tenths between strategies are not evidence.",
    "**Regime labels are ex-post** (they use the future of each window) and only describe results.",
]


def noise_floor(setup: Setup, n_seeds: int) -> dict:
    """Distribution of random top-K portfolios (weekly and monthly), net and gross, over ``n_seeds`` seeds."""
    bt = setup.cfg.backtest
    out = {"n_seeds": n_seeds, "schedules": {}}
    for sched, key in (("weekly", "random_weekly"), ("monthly", "random_monthly")):
        base = BASELINES[key]
        rows = {"net": [], "gross": []}
        turnovers = []
        for s in range(n_seeds):
            spec = BaselineSpec(**{**asdict(base), "seed": 1000 + s})
            sig = make_signals(spec, setup.frames[spec.dataset], setup.calendar, setup.start_idx, setup.dev_end_idx, bt.max_weight)
            for name, mult in (("net", 1.0), ("gross", 0.0)):
                r = _simulate(setup, sig, mult)
                m = M.compute_metrics(r.equity, fills=r.fills)
                rows[name].append((m["sharpe"], m["cagr"]))
                if name == "net":
                    turnovers.append(m["turnover_annual"])
        q = lambda arr: {"p05": float(np.nanpercentile(arr, 5)), "p50": float(np.nanpercentile(arr, 50)), "p95": float(np.nanpercentile(arr, 95))}
        out["schedules"][sched] = {k: {"sharpe": q([a for a, _ in v]), "cagr": q([b for _, b in v])} for k, v in rows.items()}
        out["schedules"][sched]["turnover"] = float(np.mean(turnovers))
    out["config_hash"] = spec_hash({"noise": n_seeds, "setup": config_hash(setup, BASELINES["random_weekly"])})
    return clean(out)


def build_payload(setup: Setup, results: dict, noise: dict | None) -> dict:
    cfg, bt = setup.cfg, setup.cfg.backtest
    summaries = {k: v[0] for k, v in results.items()}
    run_hash = spec_hash({k: s["config_hash"] for k, s in summaries.items()})[:12]
    meta = {"window": [str(setup.calendar[setup.start_idx].date()), str(setup.calendar[setup.dev_end_idx].date())], "universe": setup.universe,
            "instruments": setup.panel_info["instruments"], "capital": bt.capital, "top_k": bt.top_k, "max_weight": bt.max_weight,
            "rebalance_threshold": bt.rebalance_threshold, "run_hash": run_hash, "holdout": [str(setup.holdout.start.date()), str(setup.holdout.end.date())],
            "fee": setup.rules.fee_rate, "tax": setup.rules.sell_tax_rate, "slippage": setup.rules.slippage_rate, "lot": setup.rules.lot_size,
            "settle": setup.rules.settlement_days, "band_source": setup.panel_info["band_source"], "multipliers": bt.cost_multipliers,
            "regime_window": bt.regime_window, "regime_threshold": bt.regime_threshold,
            "wf": {"train": bt.wf_train_sessions, "test": bt.wf_test_sessions, "embargo": bt.wf_embargo_sessions}, "limits": LIMITS,
            "datasets": setup.dataset_hashes}
    return {"meta": meta, "results": summaries, "noise": noise, "order": ORDER, "purging": fold_purging(setup)}


def run_baselines_job(engine: Engine, cfg: AppConfig, *, keys: list[str] | None = None, noise_seeds: int = 20, write: bool = True) -> dict:
    with tracked_run(engine, "backtest_baselines", cfg, {"keys": keys, "noise_seeds": noise_seeds}) as (run_id, stats):
        setup = load_setup(engine, cfg)
        results = run_all(setup, keys)
        noise = noise_floor(setup, noise_seeds) if noise_seeds > 0 and (keys is None or any(k.startswith("random") for k in keys)) else None
        payload = build_payload(setup, results, noise)
        equities = {k: v[1] for k, v in results.items()}
        artifacts = save_equity_artifacts(cfg, payload["meta"]["run_hash"], equities)
        ids = store_experiments(engine, cfg, payload, artifacts, run_id)
        if write:
            img = PROJECT_ROOT / cfg.backtest.image_dir
            img.mkdir(parents=True, exist_ok=True)
            plot_equity({k: e["net"] for k, e in equities.items()}, img / "baselines_equity.png",
                        f"Baselines, development period {payload['meta']['window'][0]} → {payload['meta']['window'][1]}")
            plot_sensitivity(payload["results"], cfg.backtest.cost_multipliers, img / "baselines_cost_sensitivity.png")
            write_if_changed(PROJECT_ROOT / cfg.backtest.report_path, render_markdown(payload))
        stats.update(run_hash=payload["meta"]["run_hash"], experiments=ids, baselines=list(results))
    return {"payload": payload, "equities": equities, "experiments": ids, "run_id": run_id}


# ---- the held-out final period: ONE evaluation ------------------------------------------------------------------------------------
def run_holdout_once(engine: Engine, cfg: AppConfig, *, keys: list[str] | None = None, extra_candidates: list[str] | None = None) -> dict:
    """Evaluate the baselines on the held-out period. Allowed ONCE per (universe, holdout): the first call records itself, any
    later call raises ``OOSAlreadyUsed``. Run it for the final model AND every baseline in the same call."""
    setup = load_setup(engine, cfg)
    with session_scope(engine) as s:
        if oos_status(s, setup.holdout, setup.universe) is not None:
            reveal_holdout(s, setup.holdout, setup.universe, candidates=[], summary={})     # raises OOSAlreadyUsed with the details
    start = int(setup.calendar.get_loc(setup.holdout.start))
    end = len(setup.calendar)
    out = {}
    for key in keys or PORTFOLIO_KEYS:
        spec = BASELINES[key]
        spec_k = spec if spec.k is None else BaselineSpec(**{**asdict(spec), "k": cfg.backtest.top_k})
        sig = make_signals(spec_k, setup.frames[spec.dataset], setup.calendar, start, end - 1, cfg.backtest.max_weight)
        res = {}
        for name, mult in (("net", 1.0), ("gross", 0.0)):
            from predict_stock.backtest.engine import run_backtest
            from predict_stock.backtest.runner import engine_config
            r = run_backtest(setup.data, sig, setup.rules.scaled_costs(mult), engine_config(cfg), start=start, end=end)
            res[name] = M.compute_metrics(r.equity, trips=r.round_trips, fills=r.fills, exposure=r.exposure, rf_annual=cfg.backtest.risk_free_annual)
        out[key] = res
    with tracked_run(engine, "backtest_holdout", cfg, {"holdout": [str(setup.holdout.start.date()), str(setup.holdout.end.date())]}) as (run_id, stats):
        with session_scope(engine) as s:
            row = reveal_holdout(s, setup.holdout, setup.universe, candidates=list(out) + (extra_candidates or []), summary=clean(out),
                                 run_id=run_id, config_snapshot_id=save_config_snapshot(s, cfg.snapshot()))
            stats["experiment_id"] = row.id
    return clean(out)
