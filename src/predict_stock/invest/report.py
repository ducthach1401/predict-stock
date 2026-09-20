"""docs/INVEST_B1.md and docs/INVEST_B2.md (one report per preset) and their charts, built from the stored payloads.

Chart rules as in the SWING report: one axis per panel, the four candidates in the four categorical slots in a fixed order (factor blue, ridge orange,
elastic net aqua, LightGBM yellow), benchmarks in neutral inks, legend AND direct labels (three of the four hues are under 3:1 contrast on the surface)."""
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
from predict_stock.invest import stats as S

SLOT = {"factor": "#2a78d6", "ridge": "#eb6834", "elasticnet": "#1baf7a", "lgbm": "#eda100"}
CAND_NAMES = {"factor": "Factor score (rule-based)", "ridge": "Ridge", "elasticnet": "ElasticNet", "lgbm": "LightGBM (shallow)"}
BASE_NAMES = {"equal_weight": "Equal-weight universe", "mom_long": "Momentum 6-12m", "mom_long_invvol": "Momentum 6-12m inv-vol", "mom_short": "Momentum 10d",
              "mean_reversion": "Mean reversion", "bh_vnindex": "VNINDEX buy&hold", "bh_vn30": "VN30 buy&hold"}
NEUTRAL = {"equal_weight": INK2, "mom_long": "#8a8983", "bh_vn30": "#b9b8b1"}
BASE_ORDER = ["equal_weight", "mom_long", "mom_long_invvol", "mom_short", "mean_reversion", "bh_vnindex", "bh_vn30"]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=SURFACE, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def _spread(ax, ends, fs=8.5):
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.045                                   # one text line of the axis, so labels of lines that end together never overlap
    prev = None
    for v, label, x in sorted(ends, key=lambda t: t[0]):
        y = v if prev is None else max(v, prev + gap)
        ax.annotate(label, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=fs, color=INK2, annotation_clip=False)
        prev = y


def plot_equity(key: str, equities: dict[str, dict[str, pd.Series]], path: Path, title: str) -> None:
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.5], "hspace": 0.08}, facecolor=SURFACE)
    ends = []
    series = [(k, NEUTRAL[k], BASE_NAMES[k], "--" if k != "equal_weight" else "-", 1.6) for k in ("bh_vn30", "mom_long", "equal_weight") if k in equities]
    series += [(f"invest_{key}_{c}", SLOT[c], CAND_NAMES[c], "-", 2.0) for c in SLOT if f"invest_{key}_{c}" in equities]
    for k, color, label, ls, lw in series:
        e = equities[k]["net"]
        e = e[e.index >= e.first_valid_index()]
        idx, dd = 100 * e / e.iloc[0], e / e.cummax() - 1
        a.plot(idx.index, idx.values, color=color, lw=lw, ls=ls, label=label, solid_capstyle="round")
        b.plot(dd.index, dd.values * 100, color=color, lw=lw * 0.8, ls=ls)
        ends.append((float(idx.iloc[-1]), label, idx.index[-1]))
    _style(a); _style(b)
    a.set_ylabel("Equity, net of costs (start = 100)", color=INK2, fontsize=9)
    b.set_ylabel("Drawdown (%)", color=INK2, fontsize=9)
    a.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    a.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2, ncol=2)
    a.margins(x=0.01)
    _spread(a, ends)
    a.set_xlim(right=a.get_xlim()[1] + (a.get_xlim()[1] - a.get_xlim()[0]) * 0.2)
    _save(fig, path)


def plot_forest(p: dict, path: Path) -> None:
    """Sharpe difference against equal-weight with its bootstrap interval, one row per candidate (and the best one against the other benchmarks)."""
    rows = [(CAND_NAMES[c], SLOT[c], p["bootstrap"]["vs_equal_weight"][c]["sharpe_diff"]) for c in p["candidates"]]
    for b, label in (("mom_long", "best vs Momentum 6-12m"), ("bh_vn30", "best vs VN30 buy&hold"), ("bh_vnindex", "best vs VNINDEX buy&hold")):
        if b in p["bootstrap"]["best_vs_others"]:
            rows.append((label, INK2, p["bootstrap"]["best_vs_others"][b]["sharpe_diff"]))
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 1.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(False, axis="y")
    for i, (label, color, d) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([d["lo"], d["hi"]], [y, y], color=color, lw=2.2, solid_capstyle="round")
        ax.plot(d["point"], y, marker="o", ms=7, color=color, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.annotate(f"{d['point']:+.2f}  [{d['lo']:+.2f}, {d['hi']:+.2f}]", (d["hi"], y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    ax.axvline(0, color=MUTED, lw=1.3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9, color=INK2)
    lo = min(r[2]["lo"] for r in rows)
    hi = max(r[2]["hi"] for r in rows)
    ax.set_xlim(lo - 0.15 * (hi - lo), hi + 0.45 * (hi - lo))
    ax.set_xlabel(f"Difference in net Sharpe ratio, {int(p['bootstrap']['config']['level'] * 100)}% block-bootstrap interval (right of the line = better than the benchmark)", color=INK2, fontsize=9)
    ax.set_title("Is the difference distinguishable from zero?", loc="left", fontsize=11, color=INK)
    _save(fig, path)


def plot_rolling(p: dict, equities: dict[str, dict[str, pd.Series]], key: str, path: Path) -> None:
    best = p["best"]
    s, b = equities[f"invest_{key}_{best}"]["net"], equities["equal_weight"]["net"]
    fig, ax = plt.subplots(figsize=(10, 3.9), facecolor=SURFACE)
    _style(ax)
    ends = []
    for w, ls in ((252, "-"), (756, "--")):
        both = pd.concat([s.rename("s"), b.rename("b")], axis=1, join="inner").dropna()
        if len(both) <= w:
            continue
        r = (both / both.shift(w) - 1).dropna()
        ex = (r["s"] - r["b"]) * 100
        ax.plot(ex.index, ex.values, color=SLOT[best], lw=2 if w == 252 else 1.6, ls=ls, label=f"rolling {w // 252}-year excess return")
        ends.append((float(ex.iloc[-1]), f"{w // 252}y", ex.index[-1]))
    ax.axhline(0, color=MUTED, lw=1.3)
    ax.set_ylabel(f"{CAND_NAMES[best]} minus equal-weight, net return (points)", color=INK2, fontsize=9)
    ax.set_title("Rolling excess return over the equal-weight universe", loc="left", fontsize=11, color=INK)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
    ax.margins(x=0.01)
    if ends:
        _spread(ax, ends)
        ax.set_xlim(right=ax.get_xlim()[1] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.06)
    _save(fig, path)


# ---- markdown -------------------------------------------------------------------------------------------------------------------------------
def _ci(d: dict, pct=False, dec=2) -> str:
    f = (lambda v: f"{v * 100:.1f}%") if pct else (lambda v: f"{v:.{dec}f}")
    return f"{f(d['point'])} [{f(d['lo'])}, {f(d['hi'])}]"


def render(key: str, p: dict, examples: list[dict] | None) -> str:
    C, base, dec, bs, pre = p["candidates"], p["baselines"], p["decision"], p["bootstrap"], p["preregistration"]
    best = p["best"]
    L = [f"# {p['label']} — walk-forward {p['window'][0]} → {p['window'][1]}\n"]
    L.append(f"Universe **{p['universe']}** · run `{p['run_key']}` · dataset `{p['dataset_hash'][:12]}` · horizon {p['config']['horizon']} sessions · rebalance {p['config']['rebalance']} in "
             f"{p['config']['tranches']} tranches · top {pre['portfolio']['top_k']}, {pre['portfolio']['weighting']} weight, cap {_pct(pre['portfolio']['max_weight'], 0)} · "
             f"held-out period {p['holdout'][0]} → {p['holdout'][1]} **untouched**. Prices only: no fundamental data is used.\n")
    L.append("## Verdict\n")
    top = dec["criteria"][0]
    if dec["passed"]:
        L.append(f"**All pre-registered criteria are met for {CAND_NAMES[best]}** (net Sharpe {_num(C[best]['net']['sharpe'])}). That only *permits* opening the held-out period, once, with all baselines together.\n")
    else:
        failed = [c for c in dec["criteria"] if not c["ok"]]
        L.append(f"**{p['label']} does not beat the baselines after costs.** The best candidate by net Sharpe is {CAND_NAMES[best]}: net Sharpe {_num(C[best]['net']['sharpe'])} "
                 f"(90% interval {_num(bs['vs_equal_weight'][best]['sharpe']['lo'])} … {_num(bs['vs_equal_weight'][best]['sharpe']['hi'])}), CAGR {_pct(C[best]['net']['cagr'])}, max drawdown "
                 f"{_pct(C[best]['net']['max_drawdown'])}; {top['detail'].split(': ')[-1]} over the same window: {_num(top['threshold'])}. {len(failed)} of {len(dec['criteria'])} "
                 "pre-registered criteria fail. The held-out period stays closed and nothing is tuned further.\n")
    L.append(_table(["pre-registered criterion (for the best candidate)", "value", "threshold", "detail", "met"],
                    [[c["name"], _num(c["value"], 3), _num(c["threshold"], 3), c["detail"], "yes" if c["ok"] else "**no**"] for c in dec["criteria"]]))

    L.append("## Results (net of costs unless stated; baselines re-run over the same window, engine and costs)\n")
    rows = []
    for k, s in [(f"c:{c}", C[c]) for c in C] + [(f"b:{b}", base[b]) for b in BASE_ORDER if b in base]:
        n, g = s["net"], s["gross"]
        name = CAND_NAMES[k[2:]] if k.startswith("c:") else BASE_NAMES[k[2:]]
        if k == f"c:{best}":
            name = f"**{name}** (best)"
        drag = (g["cagr"] - n["cagr"]) * 100 if g.get("cagr") is not None and n.get("cagr") is not None else None
        rows.append([name, _pct(n["cagr"]), _num(n["sharpe"]), _num(n.get("sortino")), _pct(n["max_drawdown"], 0), _num(n.get("calmar")), _num(n.get("turnover_annual"), 1),
                     _pct(g["cagr"]), "n/a" if drag is None else f"{drag:.1f}", _num(s["sensitivity"].get("2.0", {}).get("sharpe"))])
    L.append(_table(["strategy", "CAGR", "Sharpe", "Sortino", "max DD", "Calmar", "turnover ×/yr", "CAGR gross", "cost drag (points/yr)", "Sharpe at 2× costs"], rows))
    nz = p.get("noise")
    if nz:
        L.append(f"Random top-{pre['portfolio']['top_k']} portfolios on the same {nz['schedule']} schedule ({nz['n_seeds']} seeds): net Sharpe median {_num(nz['sharpe']['p50'])}, "
                 f"5th–95th percentile {_num(nz['sharpe']['p05'])} … {_num(nz['sharpe']['p95'])}.\n")
    L.append(f"![equity](img/invest_{key}_equity.png)\n")

    L.append("## Confidence intervals (stationary block bootstrap)\n")
    L.append(f"{bs['config']['resamples']} resamples, mean block {bs['config']['block']} sessions, {int(bs['config']['level'] * 100)}% two-sided intervals, on {bs['vs_equal_weight'][best]['n_days']} daily returns. "
             "Differences are computed on the same resampled days as the benchmark (paired).\n")
    rows = []
    for c in C:
        v = bs["vs_equal_weight"][c]
        rows.append([CAND_NAMES[c], _ci(v["sharpe"]), _ci(v["cagr"], True), _ci(v["max_drawdown"], True), _ci(v["sharpe_diff"]), _pct(v["sharpe_diff"]["share_positive"], 0)])
    L.append(_table(["candidate", "Sharpe", "CAGR", "max drawdown", "Sharpe − equal-weight", "resamples with a positive difference"], rows))
    rc = bs["reality_check"]
    L.append(f"**White's reality check** over the {len(rc['candidates'])} candidates against equal-weight (Sharpe difference): largest observed difference {_num(rc['statistic'])} "
             f"({CAND_NAMES[rc['best']]}), **p = {_num(rc['p_value'], 3)}** — the probability of seeing a best-of-{len(rc['candidates'])} difference this large if none of them had any edge. "
             f"Pre-registered threshold {pre['decision_rule']['reality_check_p']}.\n")
    L.append(f"![forest](img/invest_{key}_forest.png)\n")

    L.append("## Rolling windows\n")
    rows = []
    for c in C:
        for w in ("252", "756"):
            r = p["rolling"]["all"][c][w]
            rows.append([CAND_NAMES[c], f"{int(w) // 252}y", r.get("n_windows"), "n/a" if not r.get("n_windows") else _pct(r["share_ahead"], 0), "n/a" if not r.get("n_windows") else _pct(r["median_excess"]),
                         "n/a" if not r.get("n_windows") else _pct(r["worst_excess"]), "n/a" if not r.get("n_windows") else _pct(r["median_return"]),
                         "n/a" if not r.get("n_windows") else _pct(r["share_negative"], 0)])
    L.append(_table(["candidate", "window", "windows", "share ahead of equal-weight", "median excess", "worst excess", "median return", "share with a loss"], rows))
    for b, r in p["rolling"]["best_vs"].items():
        for w, x in r.items():
            if x.get("n_windows"):
                L.append(f"* {CAND_NAMES[best]} against {BASE_NAMES[b]}, {int(w) // 252}-year windows: ahead in {_pct(x['share_ahead'], 0)} of {x['n_windows']} (overlapping) windows, median excess {_pct(x['median_excess'])}.")
    L.append("\nRolling windows overlap almost completely, so their number overstates the evidence; a 3-year window needs 3 years of out-of-sample data and there are only a few.\n")
    L.append(f"![rolling](img/invest_{key}_rolling.png)\n")

    L.append("## Ranking quality\n")
    L.append(f"Daily cross-sectional Spearman IC between the score and the realised forward return over {p['config']['horizon']} sessions (only labels that end before the held-out period). "
             f"t-statistic with an effective sample of days / {p['config']['horizon']}.\n")
    L.append(_table(["candidate", "days", "mean IC", "std", "ICIR", "t-stat", "days IC>0"],
                    [[CAND_NAMES[c], v["n_days"], _num(v["mean"], 4), _num(v["std"], 3), _num(v["icir"], 2), _num(v["t_stat"], 2), _pct(v["hit_rate"], 0)] for c, v in p["ic"].items()]))
    L.append("Mean IC by fold:\n")
    folds = p["folds"]
    L.append(_table(["fold", "test window", "fit rows", "val rows", "LightGBM trees kept"] + [CAND_NAMES[c] for c in C],
                    [[f["fold"], f"{f['test'][0]} → {f['test'][1]}", f["n_fit"], f["n_val"], f.get("lgbm_trees")] +
                     [_num(next((x["mean"] for x in p["ic_by_fold"][c] if x["fold"] == f["fold"]), None), 3) for c in C] for f in folds]))
    few = [f["fold"] for f in folds if f.get("lgbm_trees") is not None and f["lgbm_trees"] <= 5]
    if few:
        L.append(f"In fold(s) {', '.join(map(str, few))} early stopping kept 5 trees or fewer: the shallow LightGBM found almost nothing on the validation window there, so its scores are close to constant "
                 "(ties are broken by instrument id) and its portfolio in those windows is close to arbitrary.\n")
    L.append(f"Embargo = {p['config']['horizon']} sessions (the label horizon); samples purged by the end date of the {p['config']['horizon']}-session label. "
             "The four candidates are compared on the same folds; the ridge / elastic-net penalty was chosen once on the first fold: "
             + "; ".join(f"{k} {json.dumps({a: (round(b, 5) if isinstance(b, float) else b) for a, b in v.items()})}" for k, v in p["hypers"].items()) + ".\n")
    lr = pre["learned"]
    edge = [k for k, grid in (("ridge", lr["ridge_alphas"]), ("elasticnet", lr["enet_alphas"])) if p["hypers"][k]["alpha"] in (min(grid), max(grid))]
    neg = [k for k, v in p["hypers"].items() if v["validation_ic"] is not None and v["validation_ic"] <= 0]
    if edge or neg:
        L.append("Grid caveat: " + ("the chosen penalty of " + " and ".join(edge) + " sits at the edge of the pre-registered grid; " if edge else "") +
                 ("the best validation IC of " + " and ".join(neg) + " was not even positive, so the selection among grid points is mostly noise. " if neg else "") +
                 "The grid was fixed in advance and is not widened after the fact.\n")

    L.append("## Scenario quantiles (bear / base / bull) at the horizon\n")
    q = p["quantiles"]
    L.append(f"q10 / q50 / q90 of the {p['config']['horizon']}-session forward return from three shallow quantile boosters (sorted so they never cross): observed share below each = "
             f"{_pct(q['coverage']['q10'])} / {_pct(q['coverage']['q50'])} / {_pct(q['coverage']['q90'])} (nominal 10 / 50 / 90%); the q10–q90 band covers {_pct(q['interval_coverage'])} "
             f"(nominal {_pct(q['nominal_interval'], 0)}), mean width {_pct(q['mean_interval_width'])}. Pinball loss model / constant: "
             + ", ".join(f"{k} {_num(q['pinball'][k], 4)} / {_num(q['pinball_constant'][k], 4)}" for k in q["pinball"]) + ". Overlapping labels make these intervals less reliable than the sample size suggests.\n")
    worse = [k for k in q["pinball"] if q["pinball"][k] > q["pinball_constant"][k]]
    if worse:
        L.append(f"**The scenario quantiles do not beat a constant** at {', '.join(worse)} (pinball loss above that of the pooled out-of-sample quantile, itself a yardstick that sees the whole sample): "
                 "read the bear / base / bull figures as a rough spread of outcomes, not as a forecast.\n")

    L.append("## Variants of the best candidate (information only; nothing here chose the configuration)\n")
    v = p["variants"]
    hdr = ["variant", "CAGR", "Sharpe", "Sortino", "max DD", "Calmar", "turnover", "Sharpe gross", "Sharpe at 2× costs"]
    fmt = lambda name, r: [name, _pct(r["net"]["cagr"]), _num(r["net"]["sharpe"]), _num(r["net"].get("sortino")), _pct(r["net"]["max_drawdown"], 0), _num(r["net"].get("calmar")),
                           _num(r["net"].get("turnover_annual"), 1), _num(r["gross"]["sharpe"]), _num(r["stress2"])]
    rows = [fmt(f"primary: K = {pre['portfolio']['top_k']}, {pre['portfolio']['weighting']}, {p['config']['tranches']} tranches", {"net": C[best]["net"], "gross": C[best]["gross"],
                                                                                                                                    "stress2": C[best]["sensitivity"].get("2.0", {}).get("sharpe")})]
    rows += [fmt(f"K = {k}", r) for k, r in v["top_k"].items()]
    rows += [fmt(f"weights: {w}", r) for w, r in v["weighting"].items()]
    rows += [fmt(f"{t} tranche(s)", r) for t, r in v["tranches"].items()]
    rows += [fmt(f"regime filter (index < SMA200 ⇒ {int(p['regime_config']['equity_share'] * 100)}% of the stock weights, rest cash)", v["regime"]["filter_on"])]
    L.append(_table(hdr, rows))
    L.append("These variants differ by amounts of the same size as the bootstrap intervals above; a difference between two rows is not evidence that one setting is better.\n")
    L.append(f"The index was below its 200-day average on {_pct(v['regime']['share_of_sessions_risk_off'], 0)} of the sessions in the window (VN30, or VNINDEX where VN30 has no 200-day average yet).\n")
    d = p["dca"]
    L.append(f"**Periodic contribution (DCA)** of {d['contribution'] / 1e6:.0f} M VND on the first session of each month, invested at each strategy's own daily return "
             "(an approximation: no lot rounding on the marginal contribution; costs are inside the returns):\n")
    rows = [[n, f"{x['contributed'] / 1e6:,.0f} M", f"{x['final_value'] / 1e6:,.0f} M", f"{x['gain'] / 1e6:,.0f} M", _pct(x["irr_annual"])] for n, x in
            (("best candidate", d["strategy"]), ("equal-weight", d["equal_weight"]), *((("VN30 buy&hold", d["bh_vn30"]),) if "bh_vn30" in d else ()))]
    L.append(_table(["portfolio", "paid in", "final value", "gain", "money-weighted return / yr"], rows))

    L.append("## Signal outputs and thesis-break conditions\n")
    th = p.get("thesis") or {}
    if th:
        lim = th["limits"]
        L.append(f"Each stored signal carries its score, the bear / base / bull quantiles, the horizon, the factor contributions and three thesis-break flags: close below the 200-day average; "
                 f"relative strength (rank of 6-month momentum in the universe) below {lim['rs_rank_below']}; drawdown from the 252-session high deeper than {_pct(lim['drawdown_beyond'], 0)}. "
                 f"Among the {th['n_holdings']} holdings the best candidate would have taken at rebalance dates, the forward {p['config']['horizon']}-session return was:\n")
        names = {"below_sma200": "below SMA200", "weak_relative_strength": "weak relative strength", "deep_drawdown": "deep drawdown", "any": "any flag"}
        rows = [[names[k], x["n_flagged"], _pct(x["mean_flagged"]), _pct(x["share_negative_flagged"], 0), x["n_clear"], _pct(x["mean_clear"]), _pct(x["share_negative_clear"], 0)]
                for k, x in th["information"].items()]
        L.append(_table(["flag", "flagged: n", "flagged: mean return", "flagged: share negative", "clear: n", "clear: mean return", "clear: share negative"], rows))
        L.append("Holdings overlap from one rebalance to the next, so these are descriptive, not independent tests. The flags are an output for the reader; they are not a trading rule in the backtest.\n")
    if examples:
        L.append("Stored signals (`predictions.details`, latest rebalance date of the last fold model, best candidate):\n")
        for e in examples:
            d = e["details"]
            sc, tb = d["scenario"], d["thesis_break"]
            L.append(f"* {e['as_of']} · {e['symbol']} · rank {e['rank']} · score {e['score']:.3f} · bear/base/bull {sc['bear_q10'] * 100:.0f}% / {sc['base_q50'] * 100:.0f}% / {sc['bull_q90'] * 100:.0f}% over {d['horizon_sessions']} sessions · "
                     f"thesis-break triggers: {', '.join(tb['triggered']) or 'none'} · contributions: " + "; ".join(f"{n}={'n/a' if v is None else round(v, 2)} → {c:+.3f}" for n, v, c in d["contributions"][:4]))
        L.append("")

    L.append("## Fundamentals\n")
    f = p["fundamentals"]
    if not f.get("available"):
        L.append(f"{f.get('note', 'not available')}. The pipeline is ready for them: if the dataset gains columns starting with `fund_`, `invest run` re-fits the learned candidates with and without them and this section reports the difference in out-of-sample rank IC.\n")
    else:
        L.append(f"Columns: {', '.join(f['columns'])}.\n")
        L.append(_table(["candidate", "IC with", "IC without", "gain from fundamentals"], [[CAND_NAMES[c], _num(x["with"]["mean"], 4), _num(x["without"]["mean"], 4), _num(x["ic_gain"], 4)] for c, x in f["candidates"].items()]))

    L.append("## Reproducibility and records\n")
    m = p["models"]
    L.append(f"Models (`invest_{key}_<candidate>_fNN` per fold, `_final` on all development data) are registered in `models` with status `candidate`; {sum(x['predictions'] for x in m['folds']):,} predictions "
             f"(rebalance dates only) are in `predictions` with scenario quantiles, horizon, thesis flags and contributions in `details`. The hyper-parameter grid (one row per point) is in `experiments` "
             f"(`invest:grid:{p['tuning_study']}:NNN`), the pre-registration in `invest:preregistration`. Re-running with unchanged data and configuration reproduces identical model files and reuses the rows.\n")
    L.append("## Assumptions and limits\n")
    for x in LIMITS:
        L.append(f"* {x}")
    L.append("")
    return "\n".join(L).rstrip() + "\n"


LIMITS = [
    "Everything in [BASELINES.md](BASELINES.md) applies: universe look-ahead/survivorship (LARGE50 = today's 50 most traded names), dividend-adjusted vendor prices, inferred price bands, T+2 for the whole period, brief-default costs (not re-verified), no market impact.",
    "**Small sample.** About 4 years of out-of-sample returns, up to 50 names, one path of history, and labels of 63 / 126 sessions that overlap almost completely: the number of *independent* observations is a handful. The intervals above are the honest way to read the point estimates.",
    "The comparison window starts at the first test session (≥ 500 training sessions + an embargo equal to the horizon), later than the Phase 4 baselines' window; the baselines were re-run over it.",
    "The regime filter uses VN30 only from when it has a 200-day average (VN30 exists from 2020-05); before that VNINDEX.",
    "DCA is a return-series approximation (marginal contributions are assumed to earn the strategy's return, without lot rounding).",
    "Forward-return labels that would end inside the held-out period are excluded from every IC / quantile figure; the portfolio backtests stop at the last development session. The held-out period was not used for any choice.",
]


def _examples(engine, p: dict) -> list[dict]:
    from sqlalchemy import func, select
    from predict_stock.db.models import InstrumentSymbolHistory as SymbolRow
    from predict_stock.db.models import Prediction
    from predict_stock.db.session import session_scope
    last = next(x for x in reversed(p["models"]["folds"]) if x["candidate"] == p["best"])["model_id"]
    with session_scope(engine) as s:
        d = s.scalar(select(func.max(Prediction.as_of_date)).where(Prediction.model_id == last))
        rows = s.execute(select(Prediction.instrument_id, Prediction.as_of_date, Prediction.rank_in_universe, Prediction.score, Prediction.details)
                         .where(Prediction.model_id == last, Prediction.as_of_date == d).order_by(Prediction.rank_in_universe).limit(2)).all()
        out = []
        for r in rows:
            sym = s.scalar(select(SymbolRow.symbol).where(SymbolRow.instrument_id == r.instrument_id, SymbolRow.valid_from <= d).order_by(SymbolRow.valid_from.desc()).limit(1))
            out.append({"symbol": sym or f"instrument {r.instrument_id}", "as_of": str(r.as_of_date), "rank": r.rank_in_universe, "score": r.score, "details": r.details})
    return out


def write_reports(cfg: AppConfig, payloads: dict[str, dict], equities: dict[str, dict], engine=None) -> list[str]:
    img = PROJECT_ROOT / cfg.backtest.image_dir
    out = []
    for key, p in payloads.items():
        eq = equities[key]
        plot_equity(key, eq, img / f"invest_{key}_equity.png", f"{p['label']}: candidates and baselines, walk-forward {p['window'][0]} → {p['window'][1]}")
        plot_forest(p, img / f"invest_{key}_forest.png")
        plot_rolling(p, eq, key, img / f"invest_{key}_rolling.png")
        path = f"{cfg.invest.report_dir}/INVEST_{key.upper()}.md"
        write_if_changed(PROJECT_ROOT / path, render(key, p, _examples(engine, p) if engine is not None else None))
        out.append(path)
    return out


def latest_payloads(cfg: AppConfig) -> dict[str, dict]:
    """The most recent stored payload of each preset."""
    found: dict[str, tuple[float, Path]] = {}
    for f in (PROJECT_ROOT / cfg.invest.artifacts_dir / "runs").glob("invest_*/payload.json"):
        key = f.parent.name.split("_")[1]
        if key not in found or f.stat().st_mtime > found[key][0]:
            found[key] = (f.stat().st_mtime, f)
    if not found:
        raise FileNotFoundError("no stored invest run: run `python -m predict_stock invest run` first")
    return {k: json.loads(v[1].read_text(encoding="utf-8")) for k, v in sorted(found.items())}


def rewrite_latest(cfg: AppConfig) -> list[str]:
    from predict_stock.db.session import make_engine
    payloads = latest_payloads(cfg)
    equities = {}
    for key, p in payloads.items():
        equities[key] = {}
        for k, art in p["artifacts"].items():
            df = pd.read_csv(PROJECT_ROOT / art["path"], index_col="date", parse_dates=True)
            equities[key][k] = {c: df[c] for c in df.columns}
    return write_reports(cfg, payloads, equities, make_engine("app"))
