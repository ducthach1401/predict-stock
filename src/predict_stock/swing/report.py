"""docs/SWING.md and its charts, built from the payload of a `swing run` (the numbers are the ones stored in `experiments`).

Charts follow the same rules as the baseline charts: one axis per panel, the model in the first categorical slot (blue), the closest
like-for-like baseline in the second (orange), the secondary strategy in the third (aqua), the main benchmark in neutral ink, random portfolios
in gray dashes; a legend AND direct labels (three of the four hues are under 3:1 contrast on the surface); text in ink tokens."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predict_stock.backtest.report import GRID, INK, INK2, MUTED, SURFACE, _num, _pct, _style, _table, write_if_changed
from predict_stock.config import PROJECT_ROOT, AppConfig

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COLORS = {"swing_lgbm_topk": BLUE, "mom_short": ORANGE, "swing_lgbm_barrier": AQUA, "equal_weight": INK2, "random_weekly": "#a9a8a1", "bh_vnindex": "#c4c3bc"}
NAMES = {"swing_lgbm_topk": "SWING top-K (weekly)", "swing_lgbm_barrier": "SWING barrier trades", "equal_weight": "Equal-weight universe", "mom_short": "Momentum 10d",
         "mean_reversion": "Mean reversion", "mom_long": "Momentum 6-12m", "mom_long_invvol": "Momentum 6-12m inv-vol", "random_weekly": "Random weekly",
         "random_monthly": "Random monthly", "bh_vnindex": "VNINDEX buy&hold", "bh_vn30": "VN30 buy&hold"}
ORDER = ["swing_lgbm_topk", "swing_lgbm_barrier", "equal_weight", "mom_short", "mean_reversion", "mom_long", "mom_long_invvol", "random_weekly", "random_monthly",
         "bh_vnindex", "bh_vn30"]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=SURFACE, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def _end_labels(ax, ends: list[tuple[float, str, object]], fs: float = 8.5) -> None:
    lo, hi = ax.get_ylim()
    gap = ((max(v for v, _, _ in ends) - min(v for v, _, _ in ends)) or (hi - lo)) * 0.06
    prev = None
    for v, k, x in sorted(ends, key=lambda t: t[0]):
        y = v if prev is None else max(v, prev + gap)
        ax.annotate(NAMES[k], (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=fs, color=INK2, annotation_clip=False)
        prev = y


def plot_equity(equities: dict[str, pd.Series], path: Path, title: str) -> None:
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.6], "hspace": 0.08}, facecolor=SURFACE)
    ends = []
    for k in ("random_weekly", "equal_weight", "mom_short", "swing_lgbm_barrier", "swing_lgbm_topk"):
        if k not in equities:
            continue
        e = equities[k]
        idx, dd = 100 * e / e.iloc[0], e / e.cummax() - 1
        hero = k.startswith("swing")
        kw = dict(color=COLORS[k], lw=2 if k != "random_weekly" else 1.3, ls="--" if k == "random_weekly" else "-", solid_capstyle="round", zorder=3 if hero else 2)
        a.plot(idx.index, idx.values, label=NAMES[k], **kw)
        b.plot(dd.index, dd.values * 100, **{**kw, "lw": kw["lw"] * 0.8})
        ends.append((float(idx.iloc[-1]), k, idx.index[-1]))
    _style(a); _style(b)
    a.set_ylabel("Equity, net of costs (start = 100)", color=INK2, fontsize=9)
    b.set_ylabel("Drawdown (%)", color=INK2, fontsize=9)
    a.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    a.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2, ncol=2)
    a.margins(x=0.01)
    _end_labels(a, ends)
    a.set_xlim(right=a.get_xlim()[1] + (a.get_xlim()[1] - a.get_xlim()[0]) * 0.16)
    _save(fig, path)


def plot_calibration(calib: dict, path: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.15, 1]})
    _style(a); _style(b)
    lim = max(max(r["mean_pred"] for r in c) for c in calib["curves"].values()) * 1.05
    top = max(max(r["observed"] for r in c) for c in calib["curves"].values()) * 1.05
    top = max(top, lim)
    a.plot([0, top], [0, top], color=MUTED, lw=1.2, ls=":", label="perfect calibration")
    style = {"raw": ("#a9a8a1", "--", "raw classifier"), "isotonic": (BLUE, "-", "isotonic (used)" if calib["used"] == "isotonic" else "isotonic"),
             "platt": (ORANGE, "-", "Platt" if calib["used"] != "platt" else "Platt (used)")}
    ends = []
    for name, (color, ls, label) in style.items():
        c = calib["curves"][name]
        xs, ys = [r["mean_pred"] for r in c], [r["observed"] for r in c]
        a.plot(xs, ys, color=color, lw=2 if name != "raw" else 1.4, ls=ls, marker="o", ms=4, markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
    a.set_xlim(0, top); a.set_ylim(0, top)
    a.set_xlabel("Predicted probability (10 equal-count bins)", color=INK2, fontsize=9)
    a.set_ylabel("Observed frequency of target-before-stop", color=INK2, fontsize=9)
    a.set_title("Reliability on out-of-sample predictions", loc="left", fontsize=10, color=INK)
    a.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
    f = calib["by_fold"]
    b.bar([r["fold"] for r in f], [r["ece"] for r in f], color=BLUE, width=0.62)
    b.set_xticks([r["fold"] for r in f])
    b.axhline(calib["variants"][calib["used"]]["ece"], color=INK2, lw=1.2, ls="--")
    b.set_xlabel("Walk-forward fold", color=INK2, fontsize=9)
    b.set_ylabel("ECE of the calibrated probability", color=INK2, fontsize=9)
    b.set_title("Calibration error by fold (dashed: all folds pooled)", loc="left", fontsize=10, color=INK)
    _save(fig, path)


def plot_ic(ic: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.8), facecolor=SURFACE)
    _style(ax)
    rows = ic["by_fold"]
    h = ic["horizon"]
    means = [r["mean"] for r in rows]
    se = [r["std"] / np.sqrt(max(r["n_days"] / h, 1)) for r in rows]
    ax.set_xticks([r["fold"] for r in rows])
    ax.bar([r["fold"] for r in rows], means, yerr=[1.96 * s for s in se], color=BLUE, width=0.62, error_kw={"ecolor": INK2, "lw": 1.2, "capsize": 3})
    ax.axhline(0, color=MUTED, lw=1.2)
    total = ic["scores"]["swing_lgbm"]["mean"]
    ax.axhline(total, color=ORANGE, lw=1.6, ls="--")
    ax.annotate(f"all folds: {total:.3f}", (len(rows) - 0.5, total), xytext=(4, 6), textcoords="offset points", ha="right", fontsize=8.5, color=INK2)
    ax.set_xlabel("Walk-forward fold (test windows in time order)", color=INK2, fontsize=9)
    ax.set_ylabel(f"Mean daily rank IC vs {ic['target']}", color=INK2, fontsize=9)
    ax.set_title("Rank information coefficient by fold (bars: mean, whiskers: 95% interval, overlap-adjusted)", loc="left", fontsize=10, color=INK)
    _save(fig, path)


def plot_sensitivity(payload: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=SURFACE)
    _style(ax)
    multipliers = payload["primary"]["sensitivity"].keys()
    xs = sorted(float(m) for m in multipliers)
    ends = []
    series = {"swing_lgbm_topk": payload["primary"], "swing_lgbm_barrier": payload["secondary"], **{k: payload["baselines"][k] for k in ("equal_weight", "mom_short")}}
    for k, s in series.items():
        ys = [s["sensitivity"][str(x)]["sharpe"] for x in xs]
        ax.plot(xs, ys, color=COLORS[k], lw=2, marker="o", ms=4.5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=NAMES[k], zorder=3)
        ends.append((ys[-1], k, xs[-1]))
    ns = (payload.get("noise") or {}).get("schedules", {}).get("weekly")
    if ns:
        ax.axhline(ns["net"]["sharpe"]["p95"], color="#a9a8a1", lw=1.3, ls="--")
        ax.annotate("random weekly, 95th pct (net)", (xs[0], ns["net"]["sharpe"]["p95"]), xytext=(4, 5), textcoords="offset points", fontsize=8, color=INK2)
    ax.axhline(0, color=GRID, lw=1.2)
    ax.set_xticks(xs)
    ax.set_xlim(right=max(xs) * 1.5)
    _end_labels(ax, ends, 8.5)
    ax.set_xlabel("Costs as a multiple of the configured fee / tax / slippage", color=INK2, fontsize=9)
    ax.set_ylabel("Sharpe ratio", color=INK2, fontsize=9)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK2)
    ax.set_title("Sensitivity to trading costs (walk-forward window)", loc="left", fontsize=11, color=INK)
    _save(fig, path)


# ---- markdown ---------------------------------------------------------------------------------------------------------------------------
def _m(s: dict, key: str, kind="net"):
    return s[kind].get(key)


def render_markdown(p: dict, examples: list[dict] | None = None) -> str:
    A, B, base, dec = p["primary"], p["secondary"], p["baselines"], p["decision"]
    ic, cal, q, hold, pre = p["ic"], p["calibration"], p["quantiles"], p["holding"], p["preregistration"]
    L = [f"# SWING model — walk-forward {p['window'][0]} → {p['window'][1]}\n"]
    L.append(f"Universe **{p['universe']}** · run `{p['run_key']}` · dataset `{p['dataset_hash'][:12]}` · seed {p['seed']} · "
             f"held-out period {p['holdout'][0]} → {p['holdout'][1]} **untouched**.\n")
    c0 = dec["criteria"][0]
    eqf = {x["fold"]: x.get("sharpe") for x in base["equal_weight"]["folds"]}
    wins = [f for f in A["folds"] if f.get("sharpe") is not None and eqf.get(f["fold"]) is not None and f["sharpe"] > eqf[f["fold"]]]
    L.append("## Verdict\n")
    if dec["passed"]:
        L.append(f"**The pre-registered criteria are all met on the development walk-forward** (net Sharpe {_num(_m(A, 'sharpe'))}, best baseline {c0['detail'].split(': ')[-1]} "
                 f"{_num(c0['threshold'])}). That only *permits* opening the held-out period, once, together with all baselines (`swing oos --final`).\n")
    else:
        failed = [c for c in dec["criteria"] if not c["ok"]]
        L.append(f"**The SWING model does not beat the baselines after costs.** Primary strategy (top-{pre['primary_strategy']['top_k']} by rank score, weekly): net Sharpe {_num(_m(A, 'sharpe'))}, "
                 f"CAGR {_pct(_m(A, 'cagr'))}, max drawdown {_pct(_m(A, 'max_drawdown'))}; {c0['detail'].split(': ')[-1]} over the same window: {_num(c0['threshold'])}. "
                 f"{len(failed)} of {len(dec['criteria'])} pre-registered criteria fail; it beats equal-weight in {len(wins)} of {len(A['folds'])} test windows. "
                 "The held-out period stays closed and the configuration is not tuned further.\n")
    L.append(_table(["pre-registered criterion", "value", "threshold", "detail", "met"],
                    [[c["name"], _num(c["value"], 3), _num(c["threshold"], 3), c["detail"], "yes" if c["ok"] else "**no**"] for c in dec["criteria"]]))
    L.append("## Results against the baselines (same window, same engine, same costs)\n")
    rows = []
    for k in ORDER:
        s = A if k == "swing_lgbm_topk" else B if k == "swing_lgbm_barrier" else base.get(k)
        if s is None:
            continue
        n, g = s["net"], s["gross"]
        rows.append([f"**{NAMES[k]}**" if k.startswith("swing") else NAMES[k], _pct(n["cagr"]), _num(n["sharpe"]), _pct(n["max_drawdown"], 0), _num(n.get("turnover_annual"), 1),
                     _pct(g["cagr"]), _num(g["sharpe"]), _num(s["sensitivity"].get("2.0", {}).get("sharpe"))])
    L.append(_table(["strategy", "CAGR net", "Sharpe net", "max DD", "turnover (×/yr)", "CAGR gross", "Sharpe gross", "Sharpe at 2× costs"], rows))
    L.append(f"Average invested fraction: top-K {_pct(A['exposure']['mean'], 0)}, barrier trades {_pct(B['exposure']['mean'], 0)} "
             f"(no position at all on {_pct(B['exposure']['idle_share'], 0)} of the barrier strategy's sessions).\n")
    noise = p.get("noise")
    if noise:
        w = noise["schedules"]["weekly"]
        L.append(f"Random top-10 portfolios, weekly, {noise['n_seeds']} seeds over the same window: net Sharpe median {_num(w['net']['sharpe']['p50'])}, 5th–95th percentile "
                 f"{_num(w['net']['sharpe']['p05'])} … {_num(w['net']['sharpe']['p95'])}; gross median {_num(w['gross']['sharpe']['p50'])}.\n")
    L.append("![equity](img/swing_equity.png)\n")
    L.append("![costs](img/swing_cost_sensitivity.png)\n")
    L.append("Sharpe ratios over this window have a standard error of roughly 0.5: differences of a few tenths are not evidence. "
             "All levels carry the universe look-ahead and adjusted-price caveats of the baselines report.\n")

    L.append("## Ranking quality (rank IC)\n")
    L.append(f"Daily cross-sectional Spearman correlation between the score at the close and the realised `{ic['target']}`; t-statistic uses days / {ic['horizon']} as the effective sample (overlapping windows).\n")
    rows = [[k, v["n_days"], _num(v["mean"], 4), _num(v["std"], 3), _num(v["icir"], 2), _num(v["t_stat"], 2), _pct(v["hit_rate"], 0)] for k, v in ic["scores"].items()]
    L.append(_table(["score", "days", "mean IC", "std", "ICIR", "t-stat", "days IC>0"], rows))
    L.append("![ic](img/swing_ic_folds.png)\n")

    L.append("## Walk-forward folds\n")
    rows = []
    for f, i in zip(p["folds"], ic["by_fold"]):
        eq = next((x for x in base["equal_weight"]["folds"] if x["fold"] == f["fold"]), {})
        sa = next((x for x in A["folds"] if x["fold"] == f["fold"]), {})
        rows.append([f["fold"], f"{f['train'][0]} → {f['train'][1]}", f"{f['test'][0]} → {f['test'][1]}", f["n_fit"], f["n_val"], f["purged_rows"], f["unlabelled_rows"], f["best_iteration"]["rank"],
                     _num(i["mean"], 3), _num(sa.get("sharpe")), _num(eq.get("sharpe"))])
    L.append(_table(["fold", "train", "test", "fit rows", "val rows", "purged (label open at test start)", "unlabelled rows dropped", "rank trees", "rank IC", "SWING Sharpe", "equal-weight Sharpe"], rows))
    L.append(f"Embargo {pre['folds']['embargo_sessions']} sessions, validation = last {pre['folds']['val_sessions']} sessions of each training window, samples purged by the end date of their labels. "
             "With an embargo equal to the barrier horizon no label is still open when the test window starts (0 purged in every fold; the tests check the purge on data where it does bite). "
             "The few unlabelled rows are instrument-days with no label at all (three days of one suspended instrument in October 2020).\n")

    L.append("## Calibration of the probability \"target before stop within 10 sessions\"\n")
    v, k = cal["variants"], cal["constants"]
    rows = [[n, _num(x["brier"], 4), _num(x["log_loss"], 4), _num(x["ece"], 4), _num(x["auc"], 3), _num(x["mean_pred"], 3)] for n, x in v.items()]
    rows.append(["constant: the training base rate of each model", _num(k["train_base_rate"]["brier"], 4), _num(k["train_base_rate"]["log_loss"], 4), "", "0.500", ""])
    rows.append(["constant: the calibration-window base rate of each model", _num(k["calibration_base_rate"]["brier"], 4), _num(k["calibration_base_rate"]["log_loss"], 4), "", "0.500", ""])
    rows.append(["constant: the pooled out-of-sample rate (hindsight)", _num(v["raw"]["brier_constant"], 4), _num(v["raw"]["log_loss_constant"], 4), "", "0.500", _num(cal["base_rate"], 3)])
    L.append(f"Out-of-sample: {cal['n']:,} predictions, observed base rate {_pct(cal['base_rate'])}. The calibrator is fitted on each fold's validation rows and never sees its test window. "
             f"Lower Brier / log loss / ECE is better; AUC is of the ranking (0.5 = none). The mean of the per-fold AUCs of the raw classifier is "
             f"{_num(np.mean([f['auc'] for f in cal['by_fold']]), 3)} (range {_num(min(f['auc'] for f in cal['by_fold']), 3)} … {_num(max(f['auc'] for f in cal['by_fold']), 3)}); "
             "the pooled AUC mixes folds whose probabilities are centred differently and is shown only for completeness.\n")
    L.append(_table(["probability", "Brier", "log loss", "ECE (10 equal-count bins)", "AUC (pooled)", "mean predicted"], rows))
    L.append(_table(["fold", "event rate in the test window", "calibration-window rate", "training rate", "AUC (raw)", "highest calibrated probability", "ECE"],
                    [[f["fold"], _pct(f["observed"]), _pct(f["calibration_base_rate"]), _pct(pf["base_rate"]), _num(f["auc"], 3), _num(f["max_proba"], 2), _num(f["ece"], 3)]
                     for f, pf in zip(cal["by_fold"], p["folds"])]))
    L.append("![calibration](img/swing_calibration.png)\n")
    L.append("## Return quantiles and holding time\n")
    L.append(f"Quantiles of `{ic['target']}` (sorted per row so they never cross): observed share below q10 / q50 / q90 = "
             f"{_pct(q['coverage']['q10'])} / {_pct(q['coverage']['q50'])} / {_pct(q['coverage']['q90'])} (nominal 10 / 50 / 90%); q10–q90 interval covers {_pct(q['interval_coverage'])} (nominal {_pct(q['nominal_interval'], 0)}). "
             f"Pinball loss model / constant: " + ", ".join(f"{k} {_num(q['pinball'][k], 5)} / {_num(q['pinball_constant'][k], 5)}" for k in q["pinball"]) + ".\n")
    rows = [[b["bucket"], b["n"], _num(b["pred_median"], 1), _num(b["pred_p75"], 1), _num(b["realised_median"], 1), _num(b["realised_p75"], 1)] for b in hold["buckets"]]
    L.append(f"Expected holding time (sessions until a barrier is touched, time-outs counted at 10) from similar past signals; mean absolute error of the median {_num(hold['mae_median'], 2)} sessions "
             f"against {_num(hold['mae_constant'], 2)} for always guessing the overall median.\n")
    L.append(_table(["bucket (by predicted median)", "n", "predicted median", "predicted p75", "realised median", "realised p75"], rows))

    L.append("## What drives the score\n")
    L.append(_table(["feature", "share of gain (final model, rank booster)"], [[f["feature"], _pct(f["gain_share"])] for f in p["feature_importance"][:10]]))
    if examples:
        L.append("A stored signal with its explanation (`predictions.details`; TreeSHAP contributions of the rank score, feature value → contribution):\n")
        for e in examples:
            d = e["details"]
            L.append(f"* {e['as_of']} · {e['symbol']} · rank {e['rank']} · P(target before stop) {d['proba_raw']:.2f} raw → {e['proba']:.2f} calibrated · "
                     f"q10/q50/q90 {d['q10'] * 100:.1f}% / {d['q50'] * 100:.1f}% / {d['q90'] * 100:.1f}% · holding median {d['hold_median']:.0f}, p75 {d['hold_p75']:.0f} sessions (n = {d['hold_n']}) · "
                     + "; ".join(f"{n}={v if v is None else round(v, 3)} → {c:+.3f}" for n, v, c in d["shap_rank"][:4]))
        L.append("")

    L.append("## Sensitivities (report only, nothing here chose the configuration)\n")
    s = p["sensitivity"]
    hdr = ["setting", "CAGR net", "Sharpe net", "max DD", "turnover", "Sharpe gross", "Sharpe at 2× costs"]
    fmt = lambda name, r: [name, _pct(r["net"]["cagr"]), _num(r["net"]["sharpe"]), _pct(r["net"]["max_drawdown"], 0), _num(r["net"].get("turnover_annual"), 1), _num(r["gross"]["sharpe"]), _num(r["stress2"])]
    L.append("**Top-K weekly, number of names K**\n")
    L.append(_table(hdr, [fmt(f"K = {k}", r) for k, r in s["top_k"].items()]))
    L.append("**Top-K weekly, minimum calibrated probability (multiple of the calibration window's base rate; fewer names pass → the rest stays in cash)**\n")
    L.append(_table(hdr, [fmt(("no filter" if float(k) == 0 else f"≥ {k}× base rate"), r) for k, r in s["top_k_min_prob"].items()]))
    L.append("**Barrier trades, target / stop in ATR** (the probability was calibrated for the 2 / 1 barrier only; other pairs use it as a ranking filter)\n")
    L.append(_table(hdr, [fmt(f"{k} ATR", r) for k, r in s["barrier_atr"].items()]))
    L.append("**Barrier trades, minimum calibrated probability**\n")
    L.append(_table(hdr, [fmt(("no filter" if float(k) == 0 else f"≥ {k}× base rate"), r) for k, r in s["barrier_min_prob"].items()]))
    L.append("**Primary strategy, one cost component at a time** (the other components stay at the configured values)\n")
    L.append(_table(["setting", "CAGR net", "Sharpe net", "max DD", "turnover"],
                    [[k, _pct(r["cagr"]), _num(r["sharpe"]), _pct(r["max_drawdown"], 0), _num(r.get("turnover_annual"), 1)] for k, r in s["cost_components"].items()]))
    L.append("Fee, tax and slippage all scaled together (0×, 0.5×, 1×, 2×, 3×) is in the results table and the chart above.\n")

    L.append("## Changes made after the first run had been looked at\n")
    L.append("Both are mechanical corrections of things the first run exposed; neither touches the rank score, the primary strategy or the pre-registered decision rule. "
             "The first run's figures are kept next to each. The primary strategy's numbers are identical in both runs.\n")
    for a in p.get("amendments", []):
        L.append(f"{a['id']}. **Found:** {a['found']}. **Change:** {a['change']}. **Affects:** {a['affects']}. **First run:** {a['first_run_result']}.")
    L.append("")
    L.append("## Tuning and reproducibility\n")
    t = p["tuning"]
    L.append(f"Optuna: {t.get('trials')} trials on the first fold's training/validation rows ({t.get('fit_rows')} / {t.get('validation_rows')}); {t.get('n_complete')} completed, {t.get('n_failed')} failed; "
             f"best validation rank IC {_num(t.get('best_validation_ic'), 4)}. Every trial is an `experiments` row (`swing:optuna:{t['key']}:trialNNN`). Frozen parameters: `{json.dumps(t.get('best_params'))}`. "
             f"The tuning used validation rows only; the ICs above come from test windows the tuning never saw, but the choice of *which* fold to tune on was made once and in advance.\n")
    m = p["models"]
    L.append(f"Models are registered in `models` as `swing_lgbm_fNN` (one per fold) and `swing_lgbm_final` (all development data, v{m['final']['version']}, sha256 `{m['final']['sha256'][:16]}…`), status `candidate`; "
             f"{p['predictions']['rows']:,} out-of-sample predictions are stored in `predictions` with q10/q50/q90, holding time and SHAP contributions in `details`. "
             "Re-running with unchanged data and configuration reproduces identical model files (same sha256) and reuses the registered rows.\n")
    L.append("## Assumptions and limits\n")
    for x in LIMITS:
        L.append(f"* {x}")
    L.append("")
    return "\n".join(L).rstrip() + "\n"


LIMITS = [
    "Everything in [BASELINES.md](BASELINES.md) applies: universe look-ahead/survivorship (LARGE50 = today's 50 most traded names), dividend-adjusted vendor prices, inferred price bands, T+2 for the whole period, brief-default costs (not re-verified), no market-impact model, ~6 years of one history.",
    "The comparison window starts at the first walk-forward test session (the model needs ≥ 500 sessions of training data), so the baselines were re-run over that same window; numbers differ from BASELINES.md, which starts in 2019.",
    "Labels use ATR-based barriers measured from the signal-day close; the barrier strategy trades from the next open with the stop and target expressed as a percentage of the signal-day close, so its realised barriers differ slightly from the label's.",
    "The quantile boosters use early stopping on the validation window; a booster that stops after a handful of trees is a near-constant quantile (a sign of weak signal, not of a bug).",
    "The test windows (~6 months each, 11 of them, one path of history) are a small sample of market regimes; they include the March 2020 crash and rebound, the 2021 boom and the 2022 bear market.",
    "The held-out period was not used for any model choice, tuning or figure in this report.",
]


def write_report(cfg: AppConfig, payload: dict, equities: dict[str, dict[str, pd.Series]], engine=None) -> str:
    img = PROJECT_ROOT / cfg.backtest.image_dir
    plot_equity({k: e["net"] for k, e in equities.items()}, img / "swing_equity.png",
                f"SWING model vs baselines, walk-forward window {payload['window'][0]} → {payload['window'][1]}")
    plot_calibration(payload["calibration"], img / "swing_calibration.png")
    plot_ic(payload["ic"], img / "swing_ic_folds.png")
    plot_sensitivity(payload, img / "swing_cost_sensitivity.png")
    examples = _examples(engine, payload) if engine is not None else None
    write_if_changed(PROJECT_ROOT / cfg.swing.report_path, render_markdown(payload, examples))
    return cfg.swing.report_path


def _examples(engine, payload: dict) -> list[dict]:
    """The best-ranked stored signals of the last day predicted by the last fold model (read back from the database)."""
    from sqlalchemy import func, select
    from predict_stock.db.models import InstrumentSymbolHistory as SymbolRow
    from predict_stock.db.models import Prediction
    from predict_stock.db.session import session_scope
    last = payload["models"]["folds"][-1]["model_id"]
    with session_scope(engine) as s:
        d = s.scalar(select(func.max(Prediction.as_of_date)).where(Prediction.model_id == last))
        rows = s.execute(select(Prediction.instrument_id, Prediction.as_of_date, Prediction.rank_in_universe, Prediction.proba, Prediction.details)
                         .where(Prediction.model_id == last, Prediction.as_of_date == d).order_by(Prediction.rank_in_universe).limit(2)).all()
        out = []
        for r in rows:
            sym = s.scalar(select(SymbolRow.symbol).where(SymbolRow.instrument_id == r.instrument_id, SymbolRow.valid_from <= d)
                           .order_by(SymbolRow.valid_from.desc()).limit(1))
            out.append({"instrument_id": r.instrument_id, "symbol": sym or f"instrument {r.instrument_id}", "as_of": str(r.as_of_date), "rank": r.rank_in_universe,
                        "proba": r.proba, "details": r.details})
    return out


def latest_payload_path(cfg: AppConfig) -> Path:
    files = sorted((PROJECT_ROOT / cfg.swing.artifacts_dir / "runs").glob("*/payload.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError("no stored swing run: run `python -m predict_stock swing run` first")
    return files[-1]


def rewrite_latest(cfg: AppConfig) -> str:
    from predict_stock.db.session import make_engine
    payload = json.loads(latest_payload_path(cfg).read_text(encoding="utf-8"))
    equities = {}
    for k, art in payload["artifacts"].items():
        df = pd.read_csv(PROJECT_ROOT / art["path"], index_col="date", parse_dates=True)
        equities[k] = {c: df[c] for c in df.columns}
    return write_report(cfg, payload, equities, make_engine("app"))
