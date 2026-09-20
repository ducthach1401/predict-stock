"""What happened to the cards in the backtest: fills, targets, stops, expiries, realised holding time and reward:risk against the stated ones, and the
calibration of the probabilities that were shown."""
from __future__ import annotations

import numpy as np
import pandas as pd

from predict_stock.reco.backtest import CardPlan
from predict_stock.swing import calibration as cal


def _share(n, d):
    return float(n / d) if d else None


def order_stats(res, cards: dict, prefix: str) -> dict:
    ids = {k for k, c in cards.items() if c.strategy.startswith(prefix)}
    o = res.orders[res.orders["tag"].isin(ids) & res.orders["kind"].isin(["open", "limit", "stop", "ato"])] if len(res.orders) else pd.DataFrame(columns=["tag", "status"])
    st = o["status"].value_counts().to_dict()
    filled = int(st.get("filled", 0))
    done = filled + int(st.get("unfilled", 0)) + int(st.get("cancelled", 0))
    return {"cards_issued": len(ids), "orders_placed": int(len(o)), "cards_not_ordered": len(ids) - int(len(o)), "filled": filled, "unfilled_expired": int(st.get("unfilled", 0)),
            "cancelled": int(st.get("cancelled", 0)), "open_at_end": int(st.get("open_at_end", 0)), "superseded": int(st.get("superseded", 0)), "fill_rate": _share(filled, done),
            "cancel_reasons": o[o["status"] == "cancelled"]["reason"].value_counts().to_dict() if len(o) else {}}


def swing_outcomes(res, cards: dict, labels: pd.DataFrame) -> dict:
    """Outcomes of the filled SWING cards. ``labels``: index (date, instrument_id) with the model's event label (tb_label) for the calibration of the probabilities."""
    sw = {k: c for k, c in cards.items() if c.strategy == "SWING"}
    out = {"orders": order_stats(res, cards, "SWING")}
    sizes = [c.sizing for c in sw.values()]
    if sizes:
        out["sizing"] = {"mean_weight": float(np.mean([z["weight"] for z in sizes])), "mean_risk_if_stop_pct_capital": float(np.mean([z["risk_pct_capital"] for z in sizes])),
                         "target_risk": float(sizes[0]["risk_target"]), "share_capped_by_name_limit": float(np.mean([bool(z["caps_applied"]) for z in sizes]))}
    trips = res.round_trips[res.round_trips["tag"].isin(sw)] if len(res.round_trips) else pd.DataFrame()
    if trips.empty:
        return {**out, "trades": 0}
    fills = res.fills[res.fills["tag"].isin(sw)]
    t1 = set(fills[fills["reason"].astype(str).str.startswith("target1")]["tag"])
    style = {k: c.entry["style"] for k, c in sw.items()}
    rows = []
    for t in trips.itertuples():
        c = sw[t.tag]
        risk = t.qty * (t.avg_entry - c.exits["stop"])
        cls = ("target2" if t.exit_reason in ("target", "target_gap") else
               "stop_after_target1" if (t.exit_reason in ("stop", "stop_gap") and t.tag in t1) else "stop" if t.exit_reason in ("stop", "stop_gap") else
               "time_stop" if t.exit_reason == "time" else "kill_switch" if t.exit_reason == "kill_switch" else t.exit_reason)
        rows.append({"tag": t.tag, "cls": cls, "t1": t.tag in t1, "sessions": t.sessions, "net_return": t.net_return, "pnl": t.pnl, "R": t.pnl / risk if risk > 0 else np.nan,
                     "stated_rr": c.exits["rr_target2"], "expected_hold": c.holding["expected_sessions"], "max_hold": c.holding["max_sessions"], "style": style[t.tag],
                     "stated_stop_pct": (c.exits["stop"] / c.exits["reference"] - 1), "avg_entry": t.avg_entry, "ref_entry": c.exits["reference"], "grade": c.confidence["grade"]})
    d = pd.DataFrame(rows)
    n = len(d)
    ex = fills[fills["side"] == "sell"].copy()
    ex["level"] = [sw[t].exits["stop"] if r in ("stop", "stop_gap") else sw[t].exits["target2"] if r in ("target", "target_gap") else sw[t].exits["target1"] if str(r).startswith("target1") else np.nan
                   for t, r in zip(ex["tag"], ex["reason"])]
    stops = ex[ex["reason"].isin(["stop", "stop_gap"]) & ~ex["tag"].isin(t1)]           # the ORIGINAL stop: after target 1 the stop is the break-even level, not the card's
    exits = {"stop_fills": int(len(stops)), "stop_gap_share": _share((stops["reason"] == "stop_gap").sum(), len(stops)),
             "stop_fill_vs_stop_mean": float((stops["price"] / stops["level"] - 1).mean()) if len(stops) else None,
             "stop_fill_vs_stop_p10": float((stops["price"] / stops["level"] - 1).quantile(0.1)) if len(stops) else None,
             "target_fill_vs_level_mean": float((ex[ex["reason"].isin(["target", "target_gap"])]["price"] / ex[ex["reason"].isin(["target", "target_gap"])]["level"] - 1).mean()) if ex["reason"].isin(["target", "target_gap"]).any() else None}
    out.update({"trades": n, "exit_fills": exits, "open_at_end": int(res.stats.get("open_positions", 0)),
                "by_outcome": {k: {"n": int(len(g)), "mean_R": float(g["R"].mean()), "mean_net_return": float(g["net_return"].mean()), "mean_sessions": float(g["sessions"].mean())} for k, g in d.groupby("cls")},
                "outcome_share": {k: float(v / n) for k, v in d["cls"].value_counts().items()}, "outcome_n": {k: int(v) for k, v in d["cls"].value_counts().items()},
                "t1_touch_rate": float(d["t1"].mean()), "t2_rate": float((d["cls"] == "target2").mean()),
                "hold": {"actual_mean": float(d["sessions"].mean()), "actual_median": float(d["sessions"].median()), "expected_mean": float(d["expected_hold"].mean()),
                         "expected_median": float(d["expected_hold"].median()), "share_beyond_expected": float((d["sessions"] > d["expected_hold"]).mean()), "max": int(d["max_hold"].iloc[0])},
                "R": {"stated_rr_mean": float(d["stated_rr"].mean()), "realised_R_mean": float(d["R"].mean()), "realised_R_median": float(d["R"].median()),
                      "realised_R_p10": float(d["R"].quantile(0.1)), "realised_R_p90": float(d["R"].quantile(0.9)), "win_rate": float((d["pnl"] > 0).mean()),
                      "profit_factor": float(d.loc[d["pnl"] > 0, "pnl"].sum() / -d.loc[d["pnl"] < 0, "pnl"].sum()) if (d["pnl"] < 0).any() else None,
                      "avg_win_R": float(d.loc[d["R"] > 0, "R"].mean()), "avg_loss_R": float(d.loc[d["R"] <= 0, "R"].mean()),
                      "expectancy_R": float(d["R"].mean()), "stated_expectancy_R_if_p": None},
                "net_return": {"mean": float(d["net_return"].mean()), "median": float(d["net_return"].median())},
                "by_style": {s: {"n": int(len(g)), "t2": float((g["cls"] == "target2").mean()), "stop_share": float(g["cls"].isin(["stop", "stop_after_target1"]).mean()),
                                 "mean_R": float(g["R"].mean()), "mean_return": float(g["net_return"].mean())} for s, g in d.groupby("style")},
                "entry_vs_reference": {"mean_fill_vs_reference": float((d["avg_entry"] / d["ref_entry"] - 1).mean())}})
    out["calibration"] = probability_calibration(sw, labels, d)
    return out


def probability_calibration(sw: dict, labels: pd.DataFrame, trades: pd.DataFrame) -> dict:
    """Stated probability against what happened. Event level: every issued BUY card, the model's own event (target 2 ATR before stop 1 ATR within 10 sessions
    from the signal-day close). Trade level: the filled trades that reached target 2."""
    rows = []
    for c in sw.values():
        y = labels.get((pd.Timestamp(c.as_of), c.instrument_id)) if isinstance(labels, dict) else None
        if y is None or not np.isfinite(y):
            continue
        k = c.confidence
        rows.append({"y": float(y == 1), "p_model": k["p_model"], "p_display": k["p_display"], "p_trailing": (k.get("evidence") or {}).get("realised_rate"), "kind": k["display_kind"], "grade": k["grade"], "n_similar": (k.get("similar") or {}).get("n")})
    d = pd.DataFrame(rows)
    if d.empty:
        return {}
    out = {"n_events": int(len(d)), "realised_rate": float(d["y"].mean()), "by_grade": {g: int(n) for g, n in d["grade"].value_counts().items()}, "by_display_kind": {g: int(n) for g, n in d["kind"].value_counts().items()}}
    for name in ("p_model", "p_display", "p_trailing"):
        x = d[name].dropna()
        y = d.loc[x.index, "y"]
        if len(x) > 50:
            out[name] = {**cal.summary(y, x, 10), "auc": cal.auc(y, x), "curve": cal.reliability(y, x, 10).round(6).to_dict("records"), "mean_stated": float(x.mean())}
    g = {}
    for grade, s in d.groupby("grade"):
        g[grade] = {"n": int(len(s)), "mean_p_model": float(s["p_model"].mean()), "mean_p_display": float(s["p_display"].mean()) if s["p_display"].notna().any() else None, "realised": float(s["y"].mean())}
    out["by_grade_detail"] = g
    if not trades.empty:
        out["trade_level"] = {"n": int(len(trades)), "reached_target2": float((trades["cls"] == "target2").mean()), "reached_target1": float(trades["t1"].mean())}
    return out


def invest_outcomes(res, cards: dict, key: str, quant_rows: pd.DataFrame | None, horizon: int) -> dict:
    prefix = f"INVEST_{key.upper()}"
    inv = {k: c for k, c in cards.items() if c.strategy == prefix}
    out = {"orders": order_stats(res, cards, prefix)}
    trips = res.round_trips[res.round_trips["tag"].isin(inv)] if len(res.round_trips) else pd.DataFrame()
    if not trips.empty:
        out["closed_positions"] = {"n": int(len(trips)), "mean_net_return": float(trips["net_return"].mean()), "median_net_return": float(trips["net_return"].median()),
                                   "share_positive": float((trips["net_return"] > 0).mean()), "mean_sessions": float(trips["sessions"].mean()), "median_sessions": float(trips["sessions"].median()),
                                   "exit_reasons": {k: int(v) for k, v in trips["exit_reason"].value_counts().items()}}
    if quant_rows is not None and len(quant_rows):
        y = quant_rows["y"]
        out["scenario_coverage"] = {"n": int(len(y)), "below_bear_q10": float((y <= quant_rows["q10"]).mean()), "below_base_q50": float((y <= quant_rows["q50"]).mean()), "below_bull_q90": float((y <= quant_rows["q90"]).mean()),
                                    "nominal": [0.1, 0.5, 0.9], "horizon": horizon}
    return out


def exposure_stats(res, plan: CardPlan) -> dict:
    return {"mean_invested": float(res.exposure.mean()), "max_invested": float(res.exposure.max()), "min_invested": float(res.exposure.min()), "kill_events": res.stats["kill_events"],
            "blocked": res.stats["blocked"], "fills": int(res.stats["n_fills"]), "round_trips": int(res.stats["n_round_trips"])}
