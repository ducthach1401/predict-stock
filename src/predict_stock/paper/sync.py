"""Writes the replayed state into the tables: paper_orders, paper_positions, portfolio_snapshots, recommendation_outcomes and the status of the recommendations.
Everything is an upsert keyed on something reproducible, and rows of the paper record that a replay no longer produces are removed, so the tables always equal the replay."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import Engine, delete, select

from predict_stock.backtest.runner import Setup, clean
from predict_stock.config import AppConfig
from predict_stock.db.models import (
    PaperOrder, PaperPosition, PortfolioSnapshot, Recommendation, RecommendationOutcome,
)
from predict_stock.db.session import session_scope
from predict_stock.paper.replay import Replay
from predict_stock.reco import backtest as BT

ORDER_STATUS = {"filled": "filled", "unfilled": "expired", "cancelled": "cancelled", "superseded": "cancelled", "open_at_end": "pending", "expired": "expired"}


def _d(ts) -> date | None:
    return None if ts is None or pd.isna(ts) else pd.Timestamp(ts).date()


def _rec_id(rp: Replay, tag) -> int | None:
    sc = rp.cards.get(tag) if isinstance(tag, str) else None
    return sc.rec_id if sc else None


def sync_orders(engine: Engine, cfg: AppConfig, setup: Setup, rp: Replay, run_id: int | None) -> dict:
    pc = cfg.paper.portfolio_code
    res, cal = rp.res, setup.calendar
    fills = res.fills.copy() if len(res.fills) else pd.DataFrame(columns=["order_id", "idx", "date"])
    by_order = {oid: g for oid, g in fills.groupby("order_id")} if "order_id" in fills and fills["order_id"].notna().any() else {}
    rows: dict[str, dict] = {}
    for o in res.orders.itertuples() if len(res.orders) else []:
        sleeve, iid = BT.unvid(o.instrument_id)
        g = by_order.get(o.order_id)
        card = rp.cards.get(o.tag).card if isinstance(o.tag, str) and o.tag in rp.cards else None
        if g is not None and len(g):
            qty, value = int(g["qty"].sum()), float(g["value"].sum())
            fill_price, fee, tax, slip = value / qty, float(g["fee"].sum()), float(g["tax"].sum()), float(g["slippage_cost"].sum())
        else:
            qty = int(o.qty) if o.side == "sell" else (card.sizing["shares"] if card else 0)
            fill_price = fee = tax = slip = None
        rows[f"{pc}:{o.instrument_id}:{pd.Timestamp(o.created):%Y%m%d}:{o.side}:{o.kind}:{o.order_id}"] = dict(
            portfolio_code=f"{pc}:{sleeve}", recommendation_id=_rec_id(rp, o.tag), instrument_id=iid, side=o.side.upper(), order_type=o.kind, quantity=int(qty),
            limit_price=None if o.limit is None or pd.isna(o.limit) else float(o.limit), status=ORDER_STATUS.get(o.status, o.status), placed_date=_d(o.created), executed_date=_d(o.executed) if o.status == "filled" else None,
            fill_price=fill_price, fee=fee, tax=tax, slippage_cost=slip, reject_reason=None if o.status in ("filled", "open_at_end") else str(o.reason)[:255])
    for f in fills[fills["kind"] == "exit_rule"].itertuples() if len(fills) else []:
        sleeve, iid = BT.unvid(f.instrument_id)
        rows[f"{pc}:x:{f.instrument_id}:{pd.Timestamp(f.date):%Y%m%d}:{f.reason}:{f.qty}"] = dict(
            portfolio_code=f"{pc}:{sleeve}", recommendation_id=_rec_id(rp, f.tag), instrument_id=iid, side="SELL", order_type=str(f.reason), quantity=int(f.qty), limit_price=float(f.price),
            status="filled", placed_date=_d(f.date), executed_date=_d(f.date), fill_price=float(f.price), fee=float(f.fee), tax=float(f.tax), slippage_cost=float(f.slippage_cost), reject_reason=None)
    for sc in rp.cards.values():                                       # decided at the as-of close: the order for the next session is pending
        if sc.as_of == rp.as_of:
            c = sc.card
            e = c.entry
            rows[f"{pc}:card:{c.card_id}"] = dict(portfolio_code=f"{pc}:{c.sleeve}", recommendation_id=sc.rec_id, instrument_id=c.instrument_id, side="BUY", order_type=e["order_type"],
                                                 quantity=int(c.sizing["shares"]), limit_price=e.get("trigger") or e.get("zone_high"), status="pending", placed_date=_d(sc.as_of),
                                                 executed_date=None, fill_price=None, fee=None, tax=None, slippage_cost=None, reject_reason=None)
    start = _d(cal[rp.first_idx])
    with session_scope(engine) as s:
        existing = {o.order_key: o for o in s.scalars(select(PaperOrder).where(PaperOrder.order_key.like(f"{pc}:%")))}
        for k, v in rows.items():
            o = existing.pop(k, None)
            if o is None:
                s.add(PaperOrder(order_key=k, run_id=run_id, **v))
            else:
                for a, b in v.items():
                    setattr(o, a, b)
        for o in existing.values():                                    # no longer produced by the replay
            if o.placed_date >= start:
                s.delete(o)
    return {"orders": len(rows)}


def positions_from_fills(rp: Replay) -> list[dict]:
    """Lifecycle of every position (per sleeve and stock) rebuilt from the fills: opened at the first buy, closed when the quantity returns to zero."""
    out, live = [], {}
    fills = rp.res.fills
    for f in fills.itertuples() if len(fills) else []:
        v = f.instrument_id
        st = live.get(v)
        if f.side == "buy":
            if st is None:
                st = live[v] = dict(vid=v, opened=pd.Timestamp(f.date), qty=0, bought=0, buy_gross=0.0, sold=0, sell_gross=0.0, tag=f.tag)
            st["qty"] += int(f.qty); st["bought"] += int(f.qty); st["buy_gross"] += float(f.value)
        elif st is not None:
            st["qty"] -= int(f.qty); st["sold"] += int(f.qty); st["sell_gross"] += float(f.value)
            if st["qty"] <= 0:
                out.append({**st, "closed": pd.Timestamp(f.date)})
                del live[v]
    out += [{**st, "closed": None} for st in live.values()]
    return out


def sync_positions(engine: Engine, cfg: AppConfig, setup: Setup, rp: Replay) -> dict:
    pc = cfg.paper.portfolio_code
    pos = positions_from_fills(rp)
    start = _d(setup.calendar[rp.first_idx])
    keys = set()
    with session_scope(engine) as s:
        existing = {(p.portfolio_code, p.instrument_id, p.opened_date): p for p in s.scalars(select(PaperPosition).where(PaperPosition.portfolio_code.like(f"{pc}:%")))}
        for p in pos:
            sleeve, iid = BT.unvid(p["vid"])
            key = (f"{pc}:{sleeve}", iid, _d(p["opened"]))
            keys.add(key)
            vals = dict(recommendation_id=_rec_id(rp, p["tag"]), quantity=int(p["qty"] if p["closed"] is None else p["bought"]), avg_cost=p["buy_gross"] / p["bought"],
                        closed_date=_d(p["closed"]), close_price=(p["sell_gross"] / p["sold"]) if p["closed"] is not None and p["sold"] else None, status="open" if p["closed"] is None else "closed")
            row = existing.get(key)
            if row is None:
                s.add(PaperPosition(portfolio_code=key[0], instrument_id=iid, opened_date=key[2], **vals))
            else:
                for a, b in vals.items():
                    setattr(row, a, b)
        for k, row in existing.items():
            if k not in keys and row.opened_date >= start:
                s.delete(row)
    return {"positions": len(pos), "open": sum(p["closed"] is None for p in pos)}


def sync_snapshots(engine: Engine, cfg: AppConfig, setup: Setup, rp: Replay, symbols) -> dict:
    pc = cfg.paper.portfolio_code
    res, cal = rp.res, setup.calendar
    eq, cash = res.equity, res.cash
    peak = eq.cummax()
    last_close = setup.data.close
    d_last = cal[rp.last_idx]
    open_pos = [p for p in positions_from_fills(rp) if p["closed"] is None]
    plist = []
    for p in open_pos:
        sleeve, iid = BT.unvid(p["vid"])
        px = float(last_close.iloc[rp.last_idx][iid]) if iid in last_close.columns else float("nan")
        plist.append({"sleeve": sleeve, "instrument_id": iid, "symbol": symbols.at(iid, d_last), "quantity": int(p["qty"]), "avg_cost": p["buy_gross"] / p["bought"], "last_close": px,
                      "value": p["qty"] * px, "weight": p["qty"] * px / float(eq.iloc[-1]), "opened": str(p["opened"].date())})
    with session_scope(engine) as s:
        existing = {r.snapshot_date: r for r in s.scalars(select(PortfolioSnapshot).where(PortfolioSnapshot.portfolio_code == pc))}
        for d, e in eq.items():
            dd = d.date()
            c = float(cash.loc[d])
            m = {"peak": float(peak.loc[d]), "drawdown": float(e / peak.loc[d] - 1), "exposure": float(res.exposure.loc[d])}
            if d == d_last:
                m.update({"kill_events": res.stats["kill_events"], "blocked": res.stats["blocked"], "sleeve_value": {sl: sum(x["value"] for x in plist if x["sleeve"] == sl) for sl in BT.SLEEVE_CODE}})
            vals = dict(cash=c, market_value=float(e) - c, equity=float(e), positions=clean(plist) if d == d_last else None, metrics=clean(m))
            row = existing.get(dd)
            if row is None:
                s.add(PortfolioSnapshot(portfolio_code=pc, snapshot_date=dd, **vals))
            else:
                for a, b in vals.items():
                    if a == "positions" and b is None:
                        continue
                    setattr(row, a, b)
    return {"snapshots": len(eq)}


def _bench(setup: Setup, d0, d1) -> float | None:
    for sym in ("VN30", "VNINDEX"):
        s = setup.index_close.get(sym)
        if s is not None:
            a, b = s.reindex([pd.Timestamp(d0), pd.Timestamp(d1)]).to_numpy()
            if np.isfinite(a) and np.isfinite(b) and a > 0:
                return float(b / a - 1)
    return None


def sync_recommendations(engine: Engine, cfg: AppConfig, setup: Setup, rp: Replay) -> dict:
    """Status lifecycle: pending (waiting to fill) -> holding -> target / stopped / time_exit / kill_switch / closed; unfilled: expired / cancelled / skipped."""
    res, cal = rp.res, setup.calendar
    trips = {t.tag: t for t in res.round_trips.itertuples()} if len(res.round_trips) else {}
    fills = res.fills
    fills_by_tag = {tag: g for tag, g in fills.groupby("tag")} if len(fills) else {}
    orders = res.orders[res.orders["kind"].isin(["open", "limit", "stop", "ato"]) & (res.orders["side"] == "buy")] if len(res.orders) else res.orders
    ord_by_tag = {tag: g for tag, g in orders.groupby("tag")} if len(orders) else {}
    counts: dict[str, int] = {}
    with session_scope(engine) as s:
        for tag, sc in rp.cards.items():
            rec = s.get(Recommendation, sc.rec_id)
            c = sc.card
            g = fills_by_tag.get(tag)
            buys = g[g["side"] == "buy"] if g is not None else None
            details: dict = {"filled": False}
            exit_date = exit_price = holding = gross = net = mfe = mae = bench = None
            if sc.as_of == rp.as_of:
                status = "pending"
            elif buys is not None and len(buys):
                entry_date = pd.Timestamp(buys["date"].iloc[0])
                avg = float((buys["value"].sum()) / buys["qty"].sum())
                sells = g[g["side"] == "sell"]
                t = trips.get(tag)
                last_px = float(setup.data.close.iloc[rp.last_idx][c.instrument_id])
                if t is not None:
                    status = {"target": "target", "target_gap": "target", "stop": "stopped", "stop_gap": "stopped", "time": "time_exit", "kill_switch": "kill_switch"}.get(t.exit_reason, "closed")
                    exit_date = pd.Timestamp(sells["date"].iloc[-1])
                    exit_price = float(sells["value"].sum() / sells["qty"].sum())
                    holding, gross, net = int(t.sessions), float(t.gross_return), float(t.net_return)
                else:
                    status = "holding"
                    exit_date = None
                    gross = last_px / avg - 1                          # marked to the last close, before selling costs
                end = exit_date or cal[rp.last_idx]
                hi = setup.data.high[c.instrument_id].loc[entry_date:end]
                lo = setup.data.low[c.instrument_id].loc[entry_date:end]
                mfe, mae = float(hi.max() / avg - 1), float(lo.min() / avg - 1)
                bench = _bench(setup, entry_date, end)
                t1 = bool(len(sells) and sells["reason"].astype(str).str.startswith("target1").any())
                details = {"filled": True, "entry_date": str(entry_date.date()), "avg_entry": avg, "quantity": int(buys["qty"].sum()), "target1_hit": t1, "exit_reason": getattr(t, "exit_reason", None),
                           "unrealised": t is None}
                if c.strategy == "SWING" and c.exits.get("stop"):
                    risk = avg - c.exits["stop"]
                    details["stated"] = {"stop": c.exits["stop"], "target1": c.exits["target1"], "target2": c.exits["target2"], "rr": c.exits["rr_target2"]}
                    if t is not None and risk > 0:
                        details["realised_R"] = float(t.pnl / (t.qty * risk))
            else:
                o = ord_by_tag.get(tag)
                sts = set(o["status"]) if o is not None else set()
                status = "pending" if "open_at_end" in sts else "cancelled" if "cancelled" in sts else "expired" if "unfilled" in sts else "skipped"
                details = {"filled": False, "order_status": sorted(sts), "note": {"skipped": "no order was placed (stock already held in the sleeve, sleeve full or kill-switch)"}.get(status)}
            rec.status = status
            counts[status] = counts.get(status, 0) + 1
            out = s.get(RecommendationOutcome, sc.rec_id)
            vals = dict(exit_date=_d(exit_date), exit_price=exit_price, exit_reason=(details.get("exit_reason") or ("open" if status == "holding" else status)) if status not in ("pending",) else None,
                        holding_days=holding, gross_return=gross, net_return=net, max_favorable=mfe, max_adverse=mae, benchmark_return=bench, details=clean(details))
            if out is None:
                s.add(RecommendationOutcome(recommendation_id=sc.rec_id, **vals))
            else:
                for a, b in vals.items():
                    setattr(out, a, b)
    return {"status": counts}


def sync_all(engine: Engine, cfg: AppConfig, setup: Setup, rp: Replay, symbols, run_id: int | None = None) -> dict:
    out = {}
    out.update(sync_orders(engine, cfg, setup, rp, run_id))
    out.update(sync_positions(engine, cfg, setup, rp))
    out.update(sync_snapshots(engine, cfg, setup, rp, symbols))
    out.update(sync_recommendations(engine, cfg, setup, rp))
    return out
