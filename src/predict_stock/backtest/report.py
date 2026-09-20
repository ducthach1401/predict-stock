"""Baseline report: Markdown (docs/BASELINES.md), charts (docs/img), and the results stored in the database.

Charts follow the dataviz rules: one axis per panel, categorical hues in fixed slot order (blue, orange, aqua, yellow), the
benchmark in neutral ink, the noise floor in de-emphasis gray, a legend AND direct end labels (three of the four hues are
below 3:1 contrast on the surface, so visible labels and the tables carry the values), 2px lines, hairline recessive grid,
text in ink tokens (never the series colour). Light surface only (these are static images for the docs).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import Engine, select

from predict_stock.config import PROJECT_ROOT, AppConfig
from predict_stock.db.models import Experiment
from predict_stock.db.session import session_scope
from predict_stock.runs import save_config_snapshot

SURFACE, GRID, INK, INK2, MUTED = "#fcfcfb", "#e7e6e1", "#0b0b0b", "#52514e", "#8a8983"
SLOTS = {"equal_weight": "#2a78d6", "mom_short": "#eb6834", "mean_reversion": "#1baf7a", "mom_long": "#eda100"}   # fixed order
LABELS = {"equal_weight": "Equal-weight", "mom_short": "Momentum 10d", "mean_reversion": "Mean reversion", "mom_long": "Momentum 6-12m",
          "bh_vnindex": "VNINDEX buy&hold", "random_weekly": "Random weekly", "random_monthly": "Random monthly"}
NEUTRAL = {"bh_vnindex": INK2, "random_weekly": "#a9a8a1", "random_monthly": "#c4c3bc"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9, length=0)


def _pct(x, d=1):
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def _num(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def _spread_labels(ax, ends, fontsize):
    """Direct labels at the line ends, nudged apart in data units so that two lines that finish together stay readable."""
    if not ends:
        return
    ends = sorted(ends)
    lo, hi = ax.get_ylim()
    gap = (max(v for v, _, _ in ends) - min(v for v, _, _ in ends) or (hi - lo)) * 0.06
    prev = None
    for v, k, x in ends:
        y = v if prev is None else max(v, prev + gap)
        ax.annotate(LABELS[k], (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=fontsize, color=INK2, annotation_clip=False)
        prev = y


def plot_equity(equities: dict[str, pd.Series], path: Path, title: str) -> None:
    """Net equity indexed to 100 and drawdown below it, sharing the time axis. Each panel has ONE y axis."""
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.6], "hspace": 0.08}, facecolor=SURFACE)
    order = ["random_weekly", "random_monthly", "bh_vnindex", "equal_weight", "mom_short", "mean_reversion", "mom_long"]
    ends = []
    for k in order:
        if k not in equities:
            continue
        e = equities[k]
        idx = 100 * e / e.iloc[0]
        dd = e / e.cummax() - 1
        color = SLOTS.get(k) or NEUTRAL[k]
        style = "--" if k.startswith("random") else "-"
        a.plot(idx.index, idx.values, color=color, lw=2 if not k.startswith("random") else 1.4, ls=style, solid_capstyle="round", label=LABELS[k], zorder=3 if k in SLOTS else 2)
        b.plot(dd.index, dd.values * 100, color=color, lw=1.6 if not k.startswith("random") else 1.1, ls=style, zorder=3 if k in SLOTS else 2)
        ends.append((float(idx.iloc[-1]), k, idx.index[-1]))
    _style(a); _style(b)
    a.set_ylabel("Equity, net of costs (start = 100)", color=INK2, fontsize=9)
    b.set_ylabel("Drawdown (%)", color=INK2, fontsize=9)
    _spread_labels(a, ends, 8.5)
    a.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    a.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2, ncol=2)
    a.margins(x=0.01)
    a.set_xlim(right=a.get_xlim()[1] + (a.get_xlim()[1] - a.get_xlim()[0]) * 0.12)
    fig.savefig(path, dpi=110, facecolor=SURFACE, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def plot_sensitivity(results: dict, multipliers: list[float], path: Path) -> None:
    """Sharpe and CAGR against the multiple of the configured costs; one axis per panel."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), facecolor=SURFACE)
    for ax, metric, label in ((axes[0], "sharpe", "Sharpe ratio"), (axes[1], "cagr", "CAGR (%)")):
        _style(ax)
        ends = []
        for k in ("random_weekly", "random_monthly", "equal_weight", "mom_short", "mean_reversion", "mom_long"):
            if k not in results:
                continue
            sens = results[k]["sensitivity"]
            xs = [m for m in multipliers if str(m) in sens]
            ys = [sens[str(m)][metric] * (100 if metric == "cagr" else 1) for m in xs]
            color = SLOTS.get(k) or NEUTRAL[k]
            ax.plot(xs, ys, color=color, lw=2 if k in SLOTS else 1.4, ls="-" if k in SLOTS else "--", marker="o", ms=4.5 if k in SLOTS else 3.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[k], zorder=3 if k in SLOTS else 2)
            ends.append((ys[-1], k, xs[-1]))
        _spread_labels(ax, ends, 8)
        ax.set_xlabel("Costs as a multiple of the configured fee / tax / slippage", color=INK2, fontsize=9)
        ax.set_ylabel(label, color=INK2, fontsize=9)
        ax.axhline(0, color=GRID, lw=1.2)
        ax.set_xticks(multipliers)
        ax.set_xlim(right=max(multipliers) * 1.55)
    axes[0].legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Sensitivity to trading costs (development period)", x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


# ---- Markdown ------------------------------------------------------------------------------------------------------------
def _table(header, rows) -> str:
    line = lambda cells: "| " + " | ".join(str(c) for c in cells) + " |"
    return "\n".join([line(header), line(["---"] * len(header))] + [line(r) for r in rows]) + "\n"


def render_markdown(payload: dict) -> str:
    meta, res, noise = payload["meta"], payload["results"], payload.get("noise", {})
    keys = [k for k in payload["order"] if k in res]
    L = [f"# Baselines — development period {meta['window'][0]} → {meta['window'][1]}\n"]
    L.append(f"Universe **{meta['universe']}** ({meta['instruments']} instruments) · capital {meta['capital']:,.0f} VND · top-K {meta['top_k']} · weight cap {_pct(meta['max_weight'], 0)} · "
             f"rebalance when a position is off target by {_pct(meta['rebalance_threshold'], 0)} · config hash `{meta['run_hash']}`\n")
    L.append(f"**Held-out final period {meta['holdout'][0]} → {meta['holdout'][1]}: not touched by any figure below.** It can be revealed once (`backtest oos`).\n")
    L.append("Costs: fee {fee} per side, tax {tax} on sells, slippage {slip} against you on market orders, lot {lot}, T+{t}, price band inferred per date ({bands}). "
             "*Costs, tax, tick table and settlement are the brief's defaults, not re-verified against current rules.*\n".format(
                 fee=_pct(meta["fee"], 2), tax=_pct(meta["tax"], 2), slip=_pct(meta["slippage"], 2), lot=meta["lot"], t=meta["settle"], bands=meta["band_source"]))

    L.append("## How to read these numbers\n")
    L.append("* **They are yardsticks, not results.** A model earns attention only by beating the relevant baseline *after costs* and beating the random noise floor below.\n"
             "* **Absolute levels are inflated.** LARGE50 was chosen by *today's* traded value and applied backwards (look-ahead / survivorship), and prices are the vendor's back-adjusted series "
             "(dividends effectively reinvested), while the VNINDEX / VN30 benchmarks are price indices. Compare strategies with each other and with the noise floor, not with the index.\n"
             "* Lots, tick sizes and price bands are applied to *adjusted* prices, so they are exact in the logic but approximate in level (see limits).\n")

    L.append("## After costs (development period)\n")
    def name(k):
        w = res[k]["window"][0]
        return res[k]["title"] + (f" (from {w})" if w != meta["window"][0] else "")

    rows = []
    for k in keys:
        n = res[k]["net"]
        few = (n.get("trades") or 0) < 30                                        # win rate / profit factor of a handful of trades mean nothing
        rows.append([name(k), _pct(n["cagr"]), _num(n["sharpe"]), _num(n["sortino"]), _pct(n["max_drawdown"]), _num(n["calmar"]), _num(n.get("turnover_annual"), 1) if n.get("turnover_annual") is not None else "—",
                     "n<30" if few and n.get("trades") else (_pct(n.get("win_rate"), 0) if n.get("win_rate") is not None else "—"),
                     "n<30" if few and n.get("trades") else (_num(n.get("profit_factor")) if n.get("profit_factor") is not None else "—"),
                     "n<30" if few and n.get("trades") else (_pct(n.get("expectancy"), 2) if n.get("expectancy") is not None else "—"), n.get("trades", "—") if n.get("trades") is not None else "—",
                     _pct(n.get("avg_exposure"), 0) if n.get("avg_exposure") is not None else "—"])
    L.append(_table(["strategy", "CAGR", "Sharpe", "Sortino", "MDD", "Calmar", "turnover/yr", "win rate", "profit factor", "expectancy/trade", "trades", "exposure"], rows))
    L.append("Turnover = (buys + sells) / 2 / average equity per year. A *trade* is a closed position (first buy → full exit); win rate, profit factor and expectancy (mean net return per trade) use closed trades only "
             "and are shown only from 30 trades on (a buy-and-hold-like strategy closes almost nothing). "
             "Sharpe/Sortino use a 0% risk-free rate, 252 sessions a year.\n")

    L.append("## Before vs after costs\n")
    rows = []
    for k in keys:
        g, n = res[k]["gross"], res[k]["net"]
        paid = n.get("costs_paid")
        rows.append([res[k]["title"], _pct(g["cagr"]), _pct(n["cagr"]), _pct(g["cagr"] - n["cagr"]) if g["cagr"] is not None and n["cagr"] is not None else "n/a",
                     _num(g["sharpe"]), _num(n["sharpe"]), f"{paid / 1e6:,.0f}" if paid is not None else "—",
                     f"{n['slippage_paid'] / 1e6:,.0f}" if n.get("slippage_paid") is not None else "—"])
    L.append(_table(["strategy", "CAGR before", "CAGR after", "cost drag / yr", "Sharpe before", "Sharpe after", "fees+tax paid (M VND)", "slippage paid (M VND)"], rows))

    L.append("## Sensitivity to costs\n")
    L.append("![cost sensitivity](img/baselines_cost_sensitivity.png)\n")
    mults = meta["multipliers"]
    for metric, name, fmt in (("sharpe", "Sharpe", lambda v: _num(v)), ("cagr", "CAGR", lambda v: _pct(v))):
        L.append(f"**{name} by cost multiple** (×1 = configured costs, ×0 = before costs)\n")
        L.append(_table(["strategy"] + [f"×{m:g}" for m in mults],
                        [[res[k]["title"]] + [fmt(res[k]["sensitivity"].get(str(m), {}).get(metric)) for m in mults] for k in keys]))
    be = []
    for k in keys:
        s = res[k]["sensitivity"]
        if res[k]["kind"] != "portfolio":
            continue
        pos = [m for m in mults if s.get(str(m), {}).get("sharpe") is not None and s[str(m)]["sharpe"] > 0]
        be.append([res[k]["title"], f"×{max(pos):g}" if pos else "never positive"])
    L.append("Highest tested cost multiple at which the Sharpe ratio is still positive: " + "; ".join(f"{a} {b}" for a, b in be) + ".\n")

    L.append("## Equity and drawdown\n")
    L.append("![equity and drawdown](img/baselines_equity.png)\n")
    L.append("Lines: the four headline strategies (fixed colours), VNINDEX buy & hold (dark grey), the two random baselines (light grey dashed; one seed each, see the noise floor for the spread).\n")

    L.append("## By market condition (ex-post, VNINDEX)\n")
    L.append(f"Sessions are split into consecutive {meta['regime_window']}-session windows and labelled by VNINDEX's move in the window: up > +{meta['regime_threshold'] * 100:.0f}%, down < -{meta['regime_threshold'] * 100:.0f}%, otherwise sideways. "
             "The label uses the future of each window: it is for *reporting* only. Figures are annualised over the days spent in the regime.\n")
    regs = ["up", "sideways", "down"]
    L.append(_table(["strategy"] + [f"{r}: return / Sharpe / MDD" for r in regs],
                    [[res[k]["title"]] + [(f"{_pct(res[k]['regimes'][r].get('annualised_return'), 0)} / {_num(res[k]['regimes'][r].get('sharpe'), 1)} / {_pct(res[k]['regimes'][r].get('max_drawdown'), 0)}"
                                          if r in res[k]["regimes"] else "—") for r in regs] for k in keys]))
    days = {r: next((res[k]["regimes"][r]["sessions"] for k in keys if r in res[k]["regimes"]), 0) for r in regs}
    L.append(f"Sessions in each regime: {', '.join(f'{r} {n}' for r, n in days.items())}.\n")
    years = sorted({int(y) for k in keys for y in res[k]["years"]})
    L.append("**Calendar-year return (net)**\n")
    L.append(_table(["strategy"] + [str(y) for y in years], [[res[k]["title"]] + [_pct(res[k]["years"].get(str(y), {}).get("return"), 0) if str(y) in res[k]["years"] else "—" for y in years] for k in keys]))

    era = [k for k in keys if res[k].get("vn30_era_net")]
    if era:
        L.append(f"## Like-for-like with VN30 (from {res[era[0]]['vn30_era_net']['from']})\n")
        L.append(_table(["strategy", "CAGR", "Sharpe", "MDD"], [[res[k]["title"], _pct(res[k]["vn30_era_net"]["cagr"]), _num(res[k]["vn30_era_net"]["sharpe"]), _pct(res[k]["vn30_era_net"]["max_drawdown"])] for k in era]))

    L.append("## Noise floor: random top-K over many seeds\n")
    if noise:
        L.append(f"{noise['n_seeds']} seeded random portfolios per schedule (K = {meta['top_k']}), same engine, same costs. A model's out-of-sample Sharpe should sit above the 95th percentile of the matching schedule.\n")
        rows = []
        for sched, d in noise["schedules"].items():
            for kind in ("net", "gross"):
                q = d[kind]
                rows.append([sched, "after costs" if kind == "net" else "before costs"] + [_num(q["sharpe"][p]) for p in ("p05", "p50", "p95")] + [_pct(q["cagr"][p]) for p in ("p05", "p50", "p95")])
        L.append(_table(["schedule", "costs", "Sharpe p5", "p50", "p95", "CAGR p5", "p50", "p95"], rows))

    if noise:
        L.append("**Against the noise floor** (net Sharpe of each strategy vs the 95th percentile of random portfolios with the same rebalance schedule; turnover shown because the floor's turnover is what it is)\n")
        rows = []
        for k in keys:
            if res[k]["kind"] != "portfolio" or k.startswith("random"):
                continue
            sched = "weekly" if k in ("mom_short", "mean_reversion") else "monthly"
            p95 = noise["schedules"][sched]["net"]["sharpe"]["p95"]
            sh = res[k]["net"]["sharpe"]
            rows.append([res[k]["title"], sched, _num(sh), _num(p95), "above" if sh > p95 else "not above", _num(res[k]["net"].get("turnover_annual"), 1),
                         _num(noise["schedules"][sched].get("turnover"), 1) if noise["schedules"][sched].get("turnover") is not None else "—"])
        L.append(_table(["strategy", "schedule", "net Sharpe", "random p95", "vs p95", "turnover/yr", "random turnover/yr"], rows))
        L.append("Equal-weight has no schedule of its own (monthly rebalance, almost no turnover), so the monthly floor is the closest reference. "
                 "A baseline being *above* the floor only says it is unlikely to be luck within this universe; it says nothing about the future.\n")

    L.append("## Walk-forward folds (expanding, purged, embargoed)\n")
    L.append(f"Development sessions only. Train ≥ {meta['wf']['train']} sessions, test {meta['wf']['test']} sessions, embargo {meta['wf']['embargo']} sessions between them; a training sample is purged when its label ends on or after the test start. "
             "The baselines have nothing to fit, so folds here only show how stable they are through time; they are the folds a model will be trained and tested on. "
             "Purging removes almost nothing from the swing dataset (labels of at most 10 sessions, and the embargo of 10 sessions already covers them) and thousands of rows from the invest dataset (126-session labels).\n")
    pcols = list(payload["purging"][0]["purged"]) if payload["purging"] else []
    L.append(_table(["fold", "train", "test"] + [f"purged rows ({c} dataset)" for c in pcols],
                    [[f["fold"], f"{f['train'][0]} → {f['train'][1]}", f"{f['test'][0]} → {f['test'][1]}"] + [f["purged"][c] for c in pcols] for f in payload["purging"]]))
    L.append(_table(["strategy (net Sharpe per test window)"] + [f"fold {f['fold']}" for f in payload["purging"]],
                    [[res[k]["title"]] + [_num(fo.get("sharpe"), 2) for fo in res[k]["folds"]] for k in keys]))

    L.append("## Execution\n")
    rows = []
    for k in keys:
        if res[k]["kind"] != "portfolio":
            continue
        o, b = res[k]["orders"], res[k]["blocked"]
        filled = sum(v for kk, v in o.items() if kk.endswith("/filled"))
        total = sum(v for kk, v in o.items() if not kk.endswith(("/superseded", "/open_at_end", "/cancelled")))
        rows.append([res[k]["title"], total, f"{filled / total * 100:.1f}%" if total else "n/a", ", ".join(f"{a} {n}" for a, n in sorted(b.items())) or "none"])
    L.append(_table(["strategy", "orders", "fill rate", "blocked attempts (reason count)"], rows))
    L.append("All baselines use market orders at the next open (the fill rate of *limit* orders is recorded by the engine and matters for later strategies). "
             "Buys that hit the ceiling, sells at the floor, suspended sessions and T+2 are the blocking reasons above.\n")

    L.append("## Assumptions and limits\n")
    for x in meta["limits"]:
        L.append(f"* {x}")
    L.append("")
    return "\n".join(L).rstrip() + "\n"


# ---- files and database --------------------------------------------------------------------------------------------------------
def write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")


def save_equity_artifacts(cfg: AppConfig, run_hash: str, equities: dict[str, dict[str, pd.Series]]) -> dict[str, dict]:
    """Daily equity curves (net and gross) as CSV under artifacts/backtests/<hash>/; returns path + sha256 per baseline."""
    base = PROJECT_ROOT / cfg.backtest.artifacts_dir / run_hash[:12]
    base.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, eq in equities.items():
        df = pd.DataFrame({name: s for name, s in eq.items() if name in ("net", "gross")})
        text = df.to_csv(float_format="%.6f", index_label="date")
        p = base / f"equity_{k}.csv"
        write_if_changed(p, text)
        try:
            shown = str(p.relative_to(PROJECT_ROOT))
        except ValueError:                                                     # a directory outside the project (tests, custom config)
            shown = str(p)
        out[k] = {"path": shown, "sha256": hashlib.sha256(text.encode()).hexdigest()}
    return out


def store_experiments(engine: Engine, cfg: AppConfig, payload: dict, artifacts: dict[str, dict], run_id: int | None) -> dict[str, int]:
    """One ``experiments`` row per baseline (+ the noise floor). Idempotent: a row with the same name and config hash is reused."""
    out = {}
    with session_scope(engine) as s:
        snap = save_config_snapshot(s, cfg.snapshot())
        rows = [(f"baseline:{k}", v["config_hash"], v, artifacts.get(k)) for k, v in payload["results"].items()]
        if payload.get("noise"):
            rows.append(("baseline:noise_floor", payload["noise"]["config_hash"], payload["noise"], None))
        for name, chash, summary, art in rows:
            prior = s.scalars(select(Experiment).where(Experiment.name == name).order_by(Experiment.id.desc())).first()
            if prior is not None and (prior.params or {}).get("config_hash") == chash:
                prior.summary = {**summary, "artifact": art}                      # same inputs: refresh the stored figures in place, no new row
                out[name] = prior.id
                continue
            row = Experiment(name=name, description=summary.get("title") or name, run_id=run_id, config_snapshot_id=snap,
                             params={"config_hash": chash, "window": summary.get("window"), "universe": payload["meta"]["universe"]},
                             summary={**summary, "artifact": art}, status="success")
            s.add(row)
            s.flush()
            out[name] = row.id
    return out
