"""docs/RECOMMENDATIONS.md and its charts, from the stored payload of `reco backtest` and the latest live cards."""
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

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def _pcts(x, d: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:+.{d}f}%"


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=SURFACE, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def _spread(ax, ends, fs=8.5):
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.045
    prev = None
    for v, label, x in sorted(ends, key=lambda t: t[0]):
        y = v if prev is None else max(v, prev + gap)
        ax.annotate(label, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=fs, color=INK2, annotation_clip=False)
        prev = y


def plot_equity(eq: dict, path: Path, title: str) -> None:
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.5], "hspace": 0.08}, facecolor=SURFACE)
    series = [("bh_vnindex", "#c4c3bc", "VNINDEX buy&hold", "--", 1.4), ("bh_vn30", "#a9a8a1", "VN30 buy&hold", "--", 1.4), ("equal_weight", INK2, "Equal-weight universe", "-", 1.8),
              ("portfolio_no_kill", "#8bb6ea", "Combined, no kill-switch", "--", 1.5), ("portfolio_net", BLUE, "Combined portfolio (cards, kill-switch)", "-", 2.2)]
    ends = []
    for k, color, label, ls, lw in series:
        if k not in eq:
            continue
        e = eq[k]["net"]
        idx, dd = 100 * e / e.iloc[0], e / e.cummax() - 1
        a.plot(idx.index, idx.values, color=color, lw=lw, ls=ls, label=label, solid_capstyle="round")
        b.plot(dd.index, dd.values * 100, color=color, lw=lw * 0.8, ls=ls)
        ends.append((float(idx.iloc[-1]), label.split(" (")[0], idx.index[-1]))
    _style(a); _style(b)
    a.set_ylabel("Equity, net of costs (start = 100)", color=INK2, fontsize=9)
    b.set_ylabel("Drawdown (%)", color=INK2, fontsize=9)
    a.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    a.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK2)          # the lines are all above 95 in the first months: the corner is free
    a.margins(x=0.01)
    _spread(a, ends)
    a.set_xlim(right=a.get_xlim()[1] + (a.get_xlim()[1] - a.get_xlim()[0]) * 0.2)
    _save(fig, path)


def plot_sleeves(eq: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.9), facecolor=SURFACE)
    _style(ax)
    ends = []
    for k, color, label in (("sleeve_swing", ORANGE, "SWING sleeve"), ("sleeve_invest_b1", AQUA, "INVEST B1 sleeve"), ("sleeve_invest_b2", YELLOW, "INVEST B2 sleeve")):
        if k in eq:
            e = eq[k]["net"]
            y = (e / e.iloc[0] - 1) * 100
            ax.plot(y.index, y.values, color=color, lw=2, label=label, solid_capstyle="round")
            ends.append((float(y.iloc[-1]), label, y.index[-1]))
    ax.axhline(0, color=MUTED, lw=1.2)
    ax.set_ylabel("Profit or loss, % of total capital", color=INK2, fontsize=9)
    ax.set_title("Each sleeve run alone with its own cards (no kill-switch)", loc="left", fontsize=11, color=INK)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
    ax.margins(x=0.01)
    _spread(ax, ends)
    ax.set_xlim(right=ax.get_xlim()[1] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.16)
    _save(fig, path)


def plot_calibration(cal: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor=SURFACE)
    _style(ax)
    hi = 0.2
    for name, color, label in (("p_model", ORANGE, "calibrated model probability"), ("p_display", BLUE, "probability shown on the card")):
        if name in cal:
            c = cal[name]["curve"]
            xs, ys = [r["mean_pred"] for r in c], [r["observed"] for r in c]
            hi = max(hi, max(xs + ys))
            ax.plot(xs, ys, color=color, lw=2, marker="o", ms=4.5, markeredgecolor=SURFACE, markeredgewidth=1.3, label=label)
    hi *= 1.05
    ax.plot([0, hi], [0, hi], color=MUTED, lw=1.2, ls=":", label="perfect calibration")
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("Stated probability (equal-count bins)", color=INK2, fontsize=9)
    ax.set_ylabel("Realised frequency of target 2 before the stop", color=INK2, fontsize=9)
    ax.set_title("Stated probability vs what happened (issued SWING cards)", loc="left", fontsize=10, color=INK)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
    _save(fig, path)


def plot_r(payload: dict, path: Path) -> None:
    r = payload["swing"].get("R")
    if not r:
        return
    share = payload["swing"]["outcome_share"]
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    _style(ax)
    order = ["target2", "stop_after_target1", "stop", "time_stop", "kill_switch"]
    names = {"target2": "target 2", "stop_after_target1": "stop after target 1", "stop": "stop", "time_stop": "time-stop", "kill_switch": "kill-switch"}
    vals = [share.get(k, 0) * 100 for k in order]
    ax.bar([names[k] for k in order], vals, color=BLUE, width=0.6)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.0f}%", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color=INK2)
    ax.set_ylabel("Share of filled SWING trades (%)", color=INK2, fontsize=9)
    ax.set_title("How the filled SWING cards ended", loc="left", fontsize=11, color=INK)
    ax.set_ylim(0, max(vals) * 1.2 + 1)
    _save(fig, path)


# ---- markdown ---------------------------------------------------------------------------------------------------------------------------------------------
def _live_samples(cfg: AppConfig) -> tuple[str, list[str]]:
    root = PROJECT_ROOT / cfg.reco.artifacts_dir / "live"
    dirs = sorted(p for p in root.glob("*") if p.is_dir()) if root.exists() else []
    if not dirs:
        return "", []
    d = dirs[-1]
    texts = {}
    for f in sorted(d.glob("*.txt")):
        j = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
        texts.setdefault((j["strategy"], j["action"]), []).append((f.read_text(encoding="utf-8"), j))
    out = []
    for key in (("SWING", "BUY"), ("INVEST_B1", "BUY"), ("INVEST_B2", "BUY")):
        if key in texts:
            out.append(texts[key][0][0])
    rejected = [j for (s, a), lst in texts.items() if a == "NO_TRADE" for _, j in lst]
    return d.name, out + [f"[không phát hành] {j['symbol']} ({j['strategy']}): {j['rejected']}" for j in rejected[:4]]


def render(cfg: AppConfig, p: dict) -> str:
    rc = p["rules"]
    pf, sw, bench = p["portfolio"], p["swing"], p["benchmarks"]
    L = [f"# Recommendation cards and the combined portfolio — backtest {p['window'][0]} → {p['window'][1]}\n"]
    L.append(f"Universe **{p['universe']}** · run `{p['run_key']}` · held-out period {p['holdout'][0]} → {p['holdout'][1]} **untouched**. "
             "Every card is a statistical estimate, not a promise; paper trading only. The models behind the cards have NOT passed their pre-registered criteria: " +
             "; ".join(f"{k.replace('_', ' ')}: {'passed' if (v or {}).get('passed') else 'FAILED (' + str(len((v or {}).get('failed', []))) + ' criteria)'}" for k, v in p["verdicts"].items() if v) +
             ". Every card says so.\n")
    L.append("## What a card contains\n")
    L.append("`recommendations` row + JSON (`artifacts/reco/`) + Vietnamese text. Sections, all mandatory for a BUY / WATCH card (otherwise the card is NO_TRADE with the reason):\n")
    L.append(_table(["part", "SWING", "INVEST B1 / B2"], [
        ["identity", "symbol, strategy, creation date, valid-until session, action (MUA / THEO DÕI / KHÔNG GIAO DỊCH), model id, universe id", "same"],
        ["entry", "zone [low, high] on the HOSE tick grid inside the next session's ±7% band; order type: pullback limit / breakout buy-stop / ATO; cancel conditions (opens above the zone: not chased; expires after N sessions)",
         "limit at the zone's high, no chasing; 2 tranches, the second a few sessions later if the stock is still in the top 10"],
        ["exits", "stop, target 1 and 2 with the probability of touching, reward:risk (no card below 1.5), capped under the nearest resistance", "bear / base / bull price zones (q10 / q50 / q90) and thesis-break conditions instead of a hard stop"],
        ["holding", "expected (median) and 75th percentile, maximum (time-stop), earliest sale under T+2", "horizon, maximum, next review / rebalance date, earliest sale under T+2"],
        ["confidence", "calibrated probability, the probability actually SHOWN (and why), score percentile, similar past signals with n (flagged when small), grade of the evidence", "score percentile, past picks with n, evidence grade"],
        ["rationale", "template text from real numbers: top SHAP contributions with their values, trend / RSI / volume / relative strength / regime; ALWAYS risks and the conditions that void the signal", "same, with the factor contributions"],
        ["sizing", "loss-if-stop = 0.75% of capital, per-name cap 5%, whole lots of 100", "target weight of the sleeve, whole lots, loss if the bear scenario materialises"]]))
    L.append("Rules (`config/default.yaml`, section `reco`; fixed before this backtest): " + f"stop {rc['swing']['stop_atr']} ATR, target 1 / 2 = {rc['swing']['target1_atr']} / {rc['swing']['target2_atr']} ATR, min R:R {rc['swing']['min_rr']}, "
             f"time-stop {rc['swing']['max_hold']} sessions, at most {rc['swing']['max_positions']} SWING positions, risk per trade {_pct(rc['swing']['risk_per_trade'], 2)}, sleeves "
             f"{_pct(rc['portfolio']['allocation']['swing'], 0)} SWING / {_pct(rc['portfolio']['allocation']['invest'], 0)} INVEST, ≤ {_pct(rc['portfolio']['max_weight_name'], 0)} of capital in one name over all sleeves, "
             f"kill-switch at {_pct(rc['portfolio']['kill_drawdown'], 0)} drawdown ({rc['portfolio']['kill_cooldown']} sessions flat).\n")

    date, samples = _live_samples(cfg)
    if samples:
        L.append(f"## Cards issued on {date}\n")
        L.append("Produced by `python -m predict_stock reco generate` from the final models; BUY and WATCH cards are in the `recommendations` table, the rest are listed with their reason. "
                 "The as-of date lies inside the held-out period only in the sense that these are forward-looking cards: no return after it is used or evaluated anywhere.\n")
        for t in samples:
            L.append("```\n" + t + "\n```\n")

    L.append("## Backtest of the cards, exactly as issued\n")
    L.append(f"{p['cards']['buy']:,} BUY cards were issued ({sw['orders']['cards_issued']:,} SWING, {p['plan']['invest_b1']['buy_cards']} INVEST B1, {p['plan']['invest_b2']['buy_cards']} INVEST B2 incl. second tranches), "
             f"{p['cards']['watch']:,} WATCH and {p['cards']['no_trade']:,} NO_TRADE (with reasons). Each card became the engine order it states — zone, order type, cancel condition, validity, stop, two targets, time-stop, "
             "size — in one shared book of cash where each sleeve is its own set of positions. Costs: fee 0.15% a side, tax 0.1% on sells, slippage 0.1% on market fills; T+2, lot 100, tick and band from the config.\n")
    rows = []
    for name, m, g in [("**Combined portfolio (cards)**", pf["net"], pf["gross"])] + [(v["title"], v["net"], v["gross"]) for v in bench.values()]:
        rows.append([name, _pct(m["cagr"]), _num(m["sharpe"]), _num(m.get("sortino")), _pct(m["max_drawdown"], 0), _num(m.get("calmar")), _num(m.get("turnover_annual"), 1), _pct(g["cagr"]), _num(g["sharpe"])])
    nk = pf["no_kill_switch"]
    rows.insert(1, ["Combined, no kill-switch", _pct(nk["cagr"]), _num(nk["sharpe"]), _num(nk.get("sortino")), _pct(nk["max_drawdown"], 0), _num(nk.get("calmar")), _num(nk.get("turnover_annual"), 1), "", ""])
    L.append(_table(["portfolio", "CAGR", "Sharpe", "Sortino", "max DD", "Calmar", "turnover ×/yr", "CAGR gross", "Sharpe gross"], rows))
    ex = pf["exposure"]
    L.append(f"Average invested fraction {_pct(ex['mean_invested'], 0)} (the rest is cash: the sleeves are budgets, not promises to be fully invested). Kill-switch fired {len(ex['kill_events'])} time(s)"
             + (f" ({', '.join(ex['kill_events'])})" if ex["kill_events"] else "") + f"; {ex['blocked'].get('kill_switch', 0)} buy decisions were ignored while it was active. "
             f"Sharpe at 2× costs: {_num(pf['sensitivity'].get('2.0', {}).get('sharpe'))}.\n")
    ci = p["bootstrap"]
    a, d = ci["alone"], ci["vs_equal_weight"]
    L.append(f"**Confidence intervals** ({int(a['level'] * 100)}%, stationary block bootstrap, {a['n_days']} days): combined Sharpe {_num(a['sharpe']['point'])} [{_num(a['sharpe']['lo'])}, {_num(a['sharpe']['hi'])}], "
             f"CAGR {_pct(a['cagr']['point'])} [{_pct(a['cagr']['lo'])}, {_pct(a['cagr']['hi'])}]; **Sharpe minus equal-weight {_num(d['sharpe_diff']['point'])} [{_num(d['sharpe_diff']['lo'])}, {_num(d['sharpe_diff']['hi'])}]**"
             + "".join(f"; minus {'VN30' if k == 'vs_bh_vn30' else 'VNINDEX'} {_num(ci[k]['sharpe_diff']['point'])} [{_num(ci[k]['sharpe_diff']['lo'])}, {_num(ci[k]['sharpe_diff']['hi'])}]" for k in ("vs_bh_vn30", "vs_bh_vnindex") if k in ci) + ".\n")
    L.append("![equity](img/reco_equity.png)\n")
    L.append("### Rolling windows against the benchmarks\n")
    rows = []
    for b, r in p["rolling"].items():
        for w, x in r.items():
            if x.get("n_windows"):
                rows.append([bench[b]["title"], f"{int(w) // 252}y", x["n_windows"], _pct(x["share_ahead"], 0), _pct(x["median_excess"]), _pct(x["worst_excess"])])
    L.append(_table(["benchmark", "window", "windows", "share of windows the combined portfolio is ahead", "median excess return", "worst excess"], rows))
    L.append("### The sleeves separately\n")
    ib = p["sleeves_in_book"]
    rows = [[k.replace("_", " "), _pct(v["pnl_pct_capital"]), _pct(ib["without_kill_switch"].get(k)), _pct(ib["with_kill_switch"].get(k)), _pct(v["cagr"]), _num(v["sharpe"]), _pct(v["max_drawdown"], 0),
             _num(v.get("turnover_annual"), 1), _pct(v["avg_exposure"], 0)] for k, v in p["sleeves"].items()]
    L.append(_table(["sleeve", "run alone: profit, % of capital", "inside the shared book, no kill-switch", "inside the book, with kill-switch", "alone: CAGR", "alone: Sharpe", "alone: max DD", "alone: turnover", "alone: avg. exposure"], rows))
    L.append("The sleeves are **not additive**: weights are shares of current equity, so a sleeve bought at a peak of the combined equity (end of 2021) is larger than the same sleeve run alone, and a cash shortage can shrink a buy. "
             "Read the 'inside the shared book' columns as the real contribution of each sleeve to the combined result, and the 'alone' columns as what its cards would do on their own.\n")
    L.append("![sleeves](img/reco_sleeves.png)\n")
    ov = p["overlap"]
    L.append(f"A stock was a target of SWING and INVEST on the same day in {ov['cards_overlapping_invest_targets']} cases ({ov['days_with_same_stock_in_swing_and_invest']} days); the largest combined target weight of a name was "
             f"{_pct(ov['max_combined_target_weight'], 1)} against the {_pct(ov['cap'], 0)} cap. Positions are tracked per sleeve, so each has its own entry, stop and exit.\n")

    L.append("## What happened to the SWING cards\n")
    o = sw["orders"]
    L.append(_table(["cards issued", "orders placed", "not ordered (already held / sleeve full / kill-switch)", "filled", "expired unfilled", "cancelled (no chase, kill-switch)", "fill rate"],
                    [[o["cards_issued"], o["orders_placed"], o["cards_not_ordered"], o["filled"], o["unfilled_expired"], o["cancelled"], _pct(o["fill_rate"], 0)]]))
    if sw.get("trades"):
        sh, n = sw["outcome_share"], sw["outcome_n"]
        names = {"target2": "reached target 2", "stop_after_target1": "target 1, then the (break-even) stop", "stop": "stop before target 1", "time_stop": "time-stop", "kill_switch": "kill-switch"}
        ef = sw["exit_fills"]
        L.append(f"Exits: {ef['stop_fills']} original-stop fills (before target 1), {_pct(ef['stop_gap_share'], 0)} of them gap-throughs at the open; the average fill was {_pcts(ef['stop_fill_vs_stop_mean'], 2)} from the stated stop "
                 f"(10th percentile {_pcts(ef['stop_fill_vs_stop_p10'], 2)}); target fills {_pcts(ef['target_fill_vs_level_mean'], 2)} from the stated level.\n")
        L.append(_table(["outcome", "trades", "mean R", "mean net return", "mean sessions held"], [[names.get(k, k), v["n"], _num(v["mean_R"]), _pct(v["mean_net_return"], 2), _num(v["mean_sessions"], 1)] for k, v in sw["by_outcome"].items()]))
        L.append(_table(["how the filled trades ended", "n", "share"], [[names.get(k, k), n[k], _pct(v, 0)] for k, v in sh.items()] + [["touched target 1 at all", "", _pct(sw["t1_touch_rate"], 0)]]))
        h, R = sw["hold"], sw["R"]
        L.append(f"**Holding time:** actual mean {_num(h['actual_mean'], 1)} sessions (median {_num(h['actual_median'], 1)}) against a stated expectation of {_num(h['expected_mean'], 1)} (median {_num(h['expected_median'], 1)}); "
                 f"{_pct(h['share_beyond_expected'], 0)} of trades lasted longer than stated; the time-stop is {h['max']} sessions.\n")
        L.append(f"**Reward:risk:** stated R:R to target 2 averaged {_num(R['stated_rr_mean'])}; realised (net P&L / the risk stated on the card, per trade) averaged {_num(R['realised_R_mean'])} R "
                 f"(median {_num(R['realised_R_median'])}, 10th–90th percentile {_num(R['realised_R_p10'])} … {_num(R['realised_R_p90'])}); win rate {_pct(R['win_rate'], 0)}, average win {_num(R['avg_win_R'])} R, "
                 f"average loss {_num(R['avg_loss_R'])} R, profit factor {_num(R['profit_factor'])}. Mean fill against the stated reference entry: {_pcts(sw['entry_vs_reference']['mean_fill_vs_reference'], 2)}.\n")
        L.append(_table(["entry style", "trades", "reached target 2", "ended at a stop", "mean R", "mean net return"],
                        [[k, v["n"], _pct(v["t2"], 0), _pct(v["stop_share"], 0), _num(v["mean_R"]), _pct(v["mean_return"], 2)] for k, v in sw["by_style"].items()]))
        L.append("![outcomes](img/reco_outcomes.png)\n")
    cal = sw.get("calibration") or {}
    if cal:
        L.append("## Are the stated probabilities right?\n")
        L.append(f"On the {cal['n_events']:,} issued SWING cards the event the model's probability is about (target 2 ATR before stop 1 ATR within 10 sessions from the signal-day close) happened {_pct(cal['realised_rate'], 1)} of the time. "
                 f"Grades of the evidence at issue time: " + ", ".join(f"{k} {v:,}" for k, v in cal["by_grade"].items()) + "; probability shown as: " + ", ".join(f"{k} {v:,}" for k, v in cal["by_display_kind"].items()) + ".\n")
        rows = []
        for name, label in (("p_model", "calibrated model probability"), ("p_display", "probability shown on the card"), ("p_trailing", "alternative: the trailing realised rate of past out-of-sample signals")):
            if name in cal:
                c = cal[name]
                rows.append([label, _pct(c["mean_stated"], 1), _pct(c["base_rate"], 1), _num(c["brier"], 4), _num(c["brier_constant"], 4), _num(c["ece"], 4), _num(c["auc"], 3)])
        L.append(_table(["probability", "mean stated", "realised", "Brier", "Brier of a constant", "ECE", "AUC"], rows))
        L.append(_table(["evidence grade at issue", "cards", "mean model probability", "mean shown", "realised"],
                        [[g, v["n"], _pct(v["mean_p_model"], 1), _pct(v["mean_p_display"], 1), _pct(v["realised"], 1)] for g, v in cal["by_grade_detail"].items()]))
        L.append("![calibration](img/reco_calibration.png)\n")
        gap = (cal.get("p_model", {}).get("mean_stated", 0) - cal["realised_rate"])
        L.append(f"The calibrated model probability differed from the realised rate by {_pcts(gap, 1)} on average; its AUC on the issued cards is {_num(cal.get('p_model', {}).get('auc'), 3)}. "
                 "That is why the card does not show it as the headline number unless the model's PAST out-of-sample predictions had AUC ≥ 0.55 (they never did): it shows the historical win rate of similar signals with n, and grades itself.\n")
    L.append("## INVEST cards\n")
    for k, v in p["invest"].items():
        o = v["orders"]
        line = f"**{k.upper()}**: {o['cards_issued']} BUY cards (tranches included), {o['orders_placed']} orders, fill rate {_pct(o['fill_rate'], 0)}"
        if v.get("closed_positions"):
            c = v["closed_positions"]
            line += f"; {c['n']} positions closed in the window, mean net return {_pct(c['mean_net_return'])}, {_pct(c['share_positive'], 0)} positive, median {_num(c['median_sessions'], 0)} sessions held"
        if v.get("scenario_coverage"):
            s = v["scenario_coverage"]
            line += f"; realised {s['horizon']}-session return below the card's bear / base / bull = {_pct(s['below_bear_q10'], 0)} / {_pct(s['below_base_q50'], 0)} / {_pct(s['below_bull_q90'], 0)} (nominal 10 / 50 / 90%, n = {s['n']})"
        L.append(line + ".\n")
    L.append("## Findings and limits\n")
    for x in findings(p) + LIMITS:
        L.append(f"* {x}")
    L.append("")
    return "\n".join(L).rstrip() + "\n"


def findings(p: dict) -> list[str]:
    """What the numbers above say, in words; every clause is conditional on the figures it quotes."""
    out = []
    pf, sw, bs = p["portfolio"], p["swing"], p["bootstrap"]
    d = bs["vs_equal_weight"]["sharpe_diff"]
    ew = p["benchmarks"]["equal_weight"]["net"]
    verdict = "is not distinguishable from" if d["lo"] <= 0 <= d["hi"] else ("is above" if d["lo"] > 0 else "is below")
    out.append(f"**Combined portfolio.** Net Sharpe {_num(pf['net']['sharpe'])} against {_num(ew['sharpe'])} for holding the whole universe equal-weight; the difference {_num(d['point'])} [{_num(d['lo'])}, {_num(d['hi'])}] "
               f"{verdict} zero. CAGR {_pct(pf['net']['cagr'])} against {_pct(ew['cagr'])}; at 2× costs the Sharpe is {_num(pf['sensitivity'].get('2.0', {}).get('sharpe'))}.")
    nk = pf["no_kill_switch"]
    out.append(f"**Kill-switch.** With it max drawdown {_pct(pf['net']['max_drawdown'], 0)} and Sharpe {_num(pf['net']['sharpe'])}; without it {_pct(nk['max_drawdown'], 0)} and {_num(nk['sharpe'])}. "
               f"It fired {len(pf['exposure']['kill_events'])} time(s); one window is one path, so this is an illustration, not a measurement of its value.")
    if sw.get("trades"):
        R, h = sw["R"], sw["hold"]
        out.append(f"**Stated vs realised reward:risk.** Cards stated {_num(R['stated_rr_mean'])} R to target 2; the realised expectancy was {_num(R['realised_R_mean'])} R per trade. "
                   f"Only {_pct(sw['t2_rate'], 0)} of filled trades reached target 2 (target 1 was touched by {_pct(sw['t1_touch_rate'], 0)}), so the average win was {_num(R['avg_win_R'])} R, "
                   f"while the average loss was {_num(R['avg_loss_R'])} R rather than -1 R: trades ended at the original stop averaged {_num(sw['by_outcome'].get('stop', {}).get('mean_R'))} R "
                   f"(fills {_pcts(sw['exit_fills']['stop_fill_vs_stop_mean'], 2)} from the stated stop, {_pct(sw['exit_fills']['stop_gap_share'], 0)} of them gaps at the open), time-stops "
                   f"{_num(sw['by_outcome'].get('time_stop', {}).get('mean_R'))} R, and costs are inside every figure. The stated R:R is the reward if target 2 is reached, not what the sleeve earned.")
        out.append(f"**Holding time** was as stated on average ({_num(h['actual_mean'], 1)} vs {_num(h['expected_mean'], 1)} sessions), but that is largely the exit rules at work (stop, targets and time-stop cap it), "
                   "and the model's holding-time estimate had no skill over a constant in the SWING report.")
    cal = sw.get("calibration") or {}
    if cal.get("p_model"):
        pm = cal["p_model"]
        out.append(f"**Probabilities.** Stated {_pct(pm['mean_stated'], 1)} on average against {_pct(cal['realised_rate'], 1)} realised; ECE {_num(pm['ece'], 3)}, AUC {_num(pm['auc'], 3)} — no discrimination. "
                   "The policy therefore downgraded every card to grade THẤP and shows the historical rate of similar signals with n instead of the model probability as the headline. "
                   + (f"The alternative of quoting the trailing realised rate has ECE {_num(cal['p_trailing']['ece'], 3)}: a better-calibrated number, but by construction the same for every card." if cal.get("p_trailing") else ""))
    z = sw.get("sizing")
    if z:
        out.append(f"**Position size.** The loss-if-stop budget is {_pct(z['target_risk'], 2)} of capital per trade, but a per-name cap of {_pct(p['rules']['swing']['max_weight'], 0)} (the sleeve is {_pct(p['rules']['portfolio']['allocation']['swing'], 0)} "
                   f"over {p['rules']['swing']['max_positions']} names) applied to {_pct(z['share_capped_by_name_limit'], 0)} of the cards: the mean weight was {_pct(z['mean_weight'], 1)} and the mean loss if the stop is hit "
                   f"{_pct(z['mean_risk_if_stop_pct_capital'], 2)} of capital. With stops of about 1 ATR (2-4% of the price) the cap, not the risk budget, sets the size; loosening the cap would raise the risk per trade towards the budget.")
    o = sw["orders"]
    out.append(f"**From cards to orders.** {o['cards_issued']:,} SWING cards became {o['orders_placed']:,} orders: the rest were for stocks the sleeve already held, arrived when its {p['rules']['swing']['max_positions']} positions were full, "
               f"or came while the kill-switch was active. {_pct(o['fill_rate'], 0)} of the orders filled; the others expired or were cancelled without chasing the price.")
    return out


LIMITS = [
    "Everything in [BASELINES.md](BASELINES.md), [SWING.md](SWING.md), [INVEST_B1.md](INVEST_B1.md) and [INVEST_B2.md](INVEST_B2.md) applies: LARGE50 hindsight, adjusted prices (ticks, lots and bands approximate in level), inferred price bands, T+2 for the whole period, brief-default costs, no market impact.",
    "The window is where all three sleeves have out-of-sample predictions (the INVEST models need ≥ 500 sessions of training plus an embargo of 63 / 126 sessions), about 3.75 years: short, one path, overlapping labels. The intervals above are the honest way to read the point estimates.",
    "SWING cards are issued every day for the top-ranked names; the engine ignores a card for a stock the sleeve already holds and stops at the sleeve's position limit, so 'cards issued' overstates the number of independent decisions.",
    "Stops and targets are levels checked on daily highs / lows; with both inside one bar the stop is taken first (pessimistic). A stop does not fire during the T+2 period, exactly as the card says.",
    "The rules of the cards were stated before the backtest and are not tuned to it. A different entry style, stop multiple or R:R gate would change the numbers; none was tried to improve them.",
    "The probability grade uses only past out-of-sample predictions whose labels had ended (a monthly refresh); before enough evidence exists the card shows the historical rate of similar signals and says so.",
    "The kill-switch is an equity rule inside the simulation; it sells at the next open (subject to T+2 like any sale) and can be too late in a fast fall.",
]


def write_report(cfg: AppConfig, payload: dict, equities: dict, engine=None) -> str:
    img = PROJECT_ROOT / cfg.backtest.image_dir
    plot_equity(equities, img / "reco_equity.png", f"Combined portfolio of recommendation cards vs benchmarks, {payload['window'][0]} → {payload['window'][1]}")
    plot_sleeves(equities, img / "reco_sleeves.png")
    if payload["swing"].get("calibration"):
        plot_calibration(payload["swing"]["calibration"], img / "reco_calibration.png")
    plot_r(payload, img / "reco_outcomes.png")
    write_if_changed(PROJECT_ROOT / cfg.reco.report_path, render(cfg, payload))
    return cfg.reco.report_path


def latest_payload(cfg: AppConfig) -> dict:
    files = sorted((PROJECT_ROOT / cfg.reco.artifacts_dir / "runs").glob("*/payload.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError("no stored reco backtest: run `python -m predict_stock reco backtest` first")
    return json.loads(files[-1].read_text(encoding="utf-8"))


def rewrite_latest(cfg: AppConfig) -> str:
    payload = latest_payload(cfg)
    eq = {}
    for k, art in payload["artifacts"].items():
        df = pd.read_csv(PROJECT_ROOT / art["path"], index_col="date", parse_dates=True)
        eq[k] = {"net": df["net"]}
    return write_report(cfg, payload, eq)
