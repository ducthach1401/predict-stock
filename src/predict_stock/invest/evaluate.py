"""INVEST evaluation helpers: the noise floor at the preset's own schedule, the pre-registered decision, per-signal details and the fundamentals ablation."""
from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from predict_stock.backtest import metrics as M
from predict_stock.backtest.baselines import BASELINES, PORTFOLIO_KEYS, BaselineSpec, make_signals
from predict_stock.backtest.runner import Setup, clean
from predict_stock.config import InvestConfig, InvestDecision, InvestPreset
from predict_stock.invest import walkforward as W
from predict_stock.invest.stats import thesis_flags
from predict_stock.swing.evaluate import RANDOM_KEYS, run_signals
from predict_stock.swing.metrics import daily_rank_ic, ic_summary


def random_noise(setup: Setup, first_idx: int, freq: str, k: int, n_seeds: int) -> dict:
    """Net Sharpe / CAGR of random top-K portfolios on the same schedule, window and costs: what luck produces."""
    base = BASELINES["random_monthly"]
    sharpes, cagrs = [], []
    for s in range(n_seeds):
        spec = BaselineSpec(**{**asdict(base), "seed": 5000 + s, "rebalance": freq, "k": k})
        sig = make_signals(spec, setup.frames["invest"], setup.calendar, first_idx, setup.dev_end_idx, setup.cfg.backtest.max_weight)
        m = M.compute_metrics(run_signals(setup, sig, first_idx, 1.0).equity)
        sharpes.append(m["sharpe"])
        cagrs.append(m["cagr"])
    q = lambda a: {"p05": float(np.nanpercentile(a, 5)), "p50": float(np.nanpercentile(a, 50)), "p95": float(np.nanpercentile(a, 95))}
    return {"n_seeds": n_seeds, "schedule": freq, "sharpe": q(sharpes), "cagr": q(cagrs)}


def decide(best: str, models: dict[str, dict], baselines: dict[str, dict], noise: dict | None, ic: dict, rc: dict, rolling: dict, rule: InvestDecision) -> dict:
    """PASS only if ALL hold for ``best`` (the candidate with the highest net Sharpe; registered before any model was trained, config ``invest.decision``):
      1. net Sharpe above EVERY portfolio baseline (the random ones are not baselines) over the same window;
      2. White's reality check over the candidates against equal-weight: p <= ``reality_check_p``;
      3. mean daily rank IC > 0 with an overlap-adjusted t-statistic >= ``min_ic_tstat``;
      4. net return ahead of equal-weight in >= ``min_rolling_share`` of the rolling 1-year windows;
      5. net Sharpe > 0 at ``stress_cost_multiple`` x costs."""
    s = models[best]
    sharpe = s["net"]["sharpe"]
    rivals = {k: v["net"]["sharpe"] for k, v in baselines.items() if k in PORTFOLIO_KEYS and k not in RANDOM_KEYS and v["net"].get("sharpe") is not None}
    top = max(rivals, key=rivals.get) if rivals else None
    stress = s["sensitivity"].get(str(float(rule.stress_cost_multiple)), {}).get("sharpe")
    roll = rolling.get("252", {})
    share = roll.get("share_ahead")
    icb = ic.get(best, {})
    c = [
        {"name": "net Sharpe above every portfolio baseline", "value": sharpe, "threshold": rivals.get(top), "detail": f"best baseline: {top}",
         "ok": bool(top is not None and sharpe is not None and sharpe > rivals[top])},
        {"name": f"reality check p <= {rule.reality_check_p} (Sharpe vs equal-weight, {len(rc['candidates'])} candidates)", "value": rc["p_value"], "threshold": rule.reality_check_p,
         "detail": f"best by the check: {rc['best']}", "ok": bool(rc["p_value"] <= rule.reality_check_p and rc["observed"][rc["best"]] > 0)},
        {"name": f"rank IC t-statistic >= {rule.min_ic_tstat}", "value": icb.get("t_stat"), "threshold": rule.min_ic_tstat, "detail": "mean IC " + ("n/a" if icb.get("mean") is None else f"{icb['mean']:.4f}"),
         "ok": bool(icb.get("mean") is not None and icb["mean"] > 0 and icb.get("t_stat") is not None and icb["t_stat"] >= rule.min_ic_tstat)},
        {"name": f"share of rolling 1y windows ahead of equal-weight >= {rule.min_rolling_share}", "value": share, "threshold": rule.min_rolling_share,
         "detail": f"{roll.get('n_windows', 0)} overlapping windows", "ok": bool(share is not None and share >= rule.min_rolling_share)},
        {"name": f"net Sharpe > 0 at {rule.stress_cost_multiple:g}x costs", "value": stress, "threshold": 0.0, "detail": "", "ok": bool(stress is not None and stress > 0)},
    ]
    return clean({"best": best, "passed": all(x["ok"] for x in c), "criteria": c})


def build_details(rows: pd.DataFrame, pred: pd.DataFrame, model, preset: InvestPreset, cfg: InvestConfig, top: int = 5) -> list[dict]:
    """Per signal: scenario quantiles (bear / base / bull at the horizon), horizon, thesis-break flags with their thresholds and the top contributions
    (name, input value, contribution)."""
    contrib, names = model.contributions(rows)
    inputs = model.inputs()
    flags = thesis_flags(rows, preset.drawdown_break, cfg.thesis_rs_floor)
    out = []
    for i in range(len(rows)):
        order = np.argsort(-np.abs(contrib[i]), kind="stable")[:top]
        x = rows.iloc[i]
        val = lambda j: None if not np.isfinite(x[inputs[j]]) else round(float(x[inputs[j]]), 6)
        active = [c for c in flags.columns if bool(flags.iloc[i][c])]
        out.append({"scenario": {"bear_q10": round(float(pred.iloc[i]["q10"]), 6), "base_q50": round(float(pred.iloc[i]["q50"]), 6), "bull_q90": round(float(pred.iloc[i]["q90"]), 6)},
                    "horizon_sessions": preset.horizon, "thesis_break": {"triggered": active, "sma_ratio_200": val_of(x, "sma_ratio_200"), "rs_rank": val_of(x, "mom_6m_csrank"),
                                                                        "drawdown_252": val_of(x, "dd_252"), "limits": {"rs_rank_below": cfg.thesis_rs_floor, "drawdown_beyond": -preset.drawdown_break}},
                    "contributions": [[names[j], val(j), round(float(contrib[i][j]), 6)] for j in order]})
    return out


def val_of(x: pd.Series, col: str):
    v = x.get(col)
    return None if v is None or not np.isfinite(v) else round(float(v), 6)


# ---- fundamentals ----------------------------------------------------------------------------------------------------------------------------
def fundamental_columns(frame: pd.DataFrame, prefixes: list[str]) -> list[str]:
    return [c for c in frame.columns if c.startswith(tuple(prefixes))]


def fundamentals_ablation(frame: pd.DataFrame, cfg: InvestConfig, key: str, preset: InvestPreset, hypers: dict, end: pd.Timestamp) -> dict:
    """Walk-forward IC of every learned candidate with and without the fundamental columns (the factor score never uses them). The DIFFERENCE is the
    contribution of the fundamentals. Only meaningful if such columns exist; otherwise ``{"available": False}``."""
    cols = fundamental_columns(frame, cfg.fundamental_prefixes)
    if not cols:
        return {"available": False, "note": "no fundamental columns in the dataset: the INVEST models ran on prices only"}
    out = {"available": True, "columns": cols, "candidates": {}}
    learned = replace_candidates(cfg, [c for c in cfg.candidates if c != "factor"])
    _, ret, _ = W.targets(preset)
    for label, f in (("with", frame), ("without", frame.drop(columns=cols))):
        runs = W.run_walk_forward(W.prepare(f, preset), learned, key, preset, hypers, end)
        for cand in learned.candidates:
            ev = W.evaluable(W.concat(runs, cand), preset, end)
            out["candidates"].setdefault(cand, {})[label] = ic_summary(daily_rank_ic(ev, "score", ret), preset.horizon)
    for cand, d in out["candidates"].items():
        d["ic_gain"] = d["with"]["mean"] - d["without"]["mean"]
    return clean(out)


def replace_candidates(cfg: InvestConfig, names: list[str]) -> InvestConfig:
    return cfg.model_copy(update={"candidates": names})
