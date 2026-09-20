"""Event-driven daily simulator for a LONG-ONLY portfolio under Vietnamese market rules.

Timeline of one session ``i``
  0. time-stops that fall due become sell orders;
  1. orders decided at the close of ``i-1`` execute: SELLS first (they free cash), then BUYS.
     market orders fill at the OPEN of ``i`` (plus adverse slippage); a limit order fills only if the session's
     low/high reaches the limit, at the better of the limit and the open (fill rate is recorded);
  2. exit rules (stop / target) are checked on the session's open/high/low for positions whose shares are SELLABLE
     (bought at least ``settlement_days`` sessions ago: T+2). Gap through a level -> exit at the open; both levels inside
     one bar -> the STOP first (configurable);
  3. the portfolio is marked to market at the close (a suspended instrument keeps its last close);
  4. the strategy's signal for the close of ``i`` (computed with data up to ``i`` only) becomes orders for ``i+1``.

A signal never fills in the session it is computed in. Nothing is short. Costs: fee both ways, tax on sells,
slippage against you on market orders. A run with all costs set to zero gives the "before costs" figures.

What blocks a fill (each is counted): no bar (suspended) or no valid open; buy at the ceiling / sell at the floor
(locked, cannot be filled); T+2 (sell only what is sellable); lot too small; not enough cash; limit not reached.
Buys that fail are dropped after their validity; sells that fail persist until done (up to ``max_pending_sell_sessions``).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from predict_stock.backtest.market import MarketRules


@dataclass
class SignalItem:
    instrument_id: int
    weight: float                       # target fraction of equity; 0 = exit
    stop_pct: float | None = None       # exit levels relative to the entry fill price
    target_pct: float | None = None
    max_hold: int | None = None         # sessions; the exit executes at the open of session entry + max_hold
    order: str = "open"                 # "open" (market at next open) | "limit"
    limit_offset: float = 0.0           # buy limit = signal-day close x (1 - offset)
    valid_sessions: int = 1
    only_if_flat: bool = False          # entry only: ignored while the instrument is held or has a pending buy (no top-up, no reset of its exit levels)


@dataclass
class Signal:
    items: list[SignalItem]
    full_rebalance: bool = True         # True: any held instrument not listed is sold


@dataclass
class EngineConfig:
    capital: float = 1_000_000_000.0
    rebalance_threshold: float = 0.0    # skip a change smaller than this FRACTION OF THE TARGET position value (never blocks an exit to 0)
    tie: str = "stop_first"             # both barriers inside one bar: "stop_first" | "target_first"
    max_pending_sell_sessions: int = 20
    max_positions: int | None = None    # at most this many instruments held or awaiting a buy; a new entry beyond it is skipped (counted as `max_positions`)


@dataclass
class MarketData:
    """Wide (date x instrument) prices as the simulator sees them. ``bands`` = daily price band known at the start of each date."""
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    bands: pd.DataFrame


@dataclass
class BacktestResult:
    equity: pd.Series
    cash: pd.Series
    exposure: pd.Series                 # invested fraction at each close
    fills: pd.DataFrame
    orders: pd.DataFrame
    round_trips: pd.DataFrame
    open_positions: pd.DataFrame
    stats: dict = field(default_factory=dict)


@dataclass
class _Order:
    oid: int
    j: int
    side: str
    kind: str                           # open | limit | exit_rule | time | rebalance
    created: int
    first: int
    last: int
    value: float = 0.0                  # buys: cash to deploy
    qty: int = 0                        # sells: shares
    limit: float | None = None
    reason: str = ""
    item: SignalItem | None = None
    tries: int = 0


class _Pos:
    __slots__ = ("j", "lots", "cost", "proceeds", "buy_gross", "sell_gross", "entry", "stop", "target", "time_idx", "last_reason", "fills")

    def __init__(self, j: int, entry: int):
        self.j, self.lots, self.entry = j, [], entry
        self.cost = self.proceeds = self.buy_gross = self.sell_gross = 0.0
        self.stop = self.target = None
        self.time_idx = None
        self.last_reason = ""
        self.fills = 0

    @property
    def qty(self) -> int:
        return sum(q for q, _ in self.lots)

    def sellable(self, i: int, settle: int) -> int:
        return sum(q for q, idx in self.lots if i - idx >= settle)

    def remove(self, qty: int) -> None:
        """FIFO."""
        left = qty
        for lot in self.lots:
            take = min(lot[0], left)
            lot[0] -= take
            left -= take
            if left == 0:
                break
        self.lots = [l for l in self.lots if l[0] > 0]


def run_backtest(data: MarketData, signals: dict[int, Signal], rules: MarketRules, cfg: EngineConfig | None = None,
                 start: int = 0, end: int | None = None) -> BacktestResult:
    """Simulate sessions ``start .. end-1``. ``signals`` maps a session index (its CLOSE) to the signal computed then."""
    cfg = cfg or EngineConfig()
    cal, ids = data.close.index, list(data.close.columns)
    O, H, L, C = (getattr(data, k).to_numpy(float) for k in ("open", "high", "low", "close"))
    B = data.bands.reindex(index=cal, columns=ids).to_numpy(float)
    T, N = C.shape
    end = T if end is None else min(end, T)
    col = {iid: j for j, iid in enumerate(ids)}
    fee, settle, lot = rules.fee_rate, rules.settlement_days, rules.lot_size

    for sig in signals.values():
        for it in sig.items:
            if it.weight < 0:
                raise ValueError(f"instrument {it.instrument_id}: weight {it.weight} is negative; the engine is long-only")
            if it.order not in ("open", "limit"):
                raise ValueError(f"unknown order type {it.order!r}")
    cash = float(cfg.capital)
    pos: dict[int, _Pos] = {}
    pending: list[_Order] = []
    last_close = np.full(N, np.nan)
    if start > 0:                                                         # reference prices carried in from before the window
        last_close = np.array(pd.DataFrame(C[:start]).ffill().iloc[-1], dtype=float)
    fills, orders_log, trips = [], [], []
    blocked = Counter()
    oid = 0
    eq_rows, cash_rows, expo_rows = [], [], []

    def log_order(o: _Order, status: str, i: int | None, reason: str = ""):
        orders_log.append({"order_id": o.oid, "instrument_id": ids[o.j], "side": o.side, "kind": o.kind, "created": cal[o.created],
                           "executed": cal[i] if i is not None else pd.NaT, "status": status, "reason": reason or o.reason})

    def band_prices(i: int, j: int) -> tuple[float, float]:
        ref = last_close[j]                                              # the last traded close before this session
        if not np.isfinite(ref) or not np.isfinite(B[i, j]):
            return np.inf, 0.0
        return float(rules.ceiling(ref, B[i, j])), float(rules.floor(ref, B[i, j]))

    def record_fill(i: int, j: int, side: str, qty: int, price: float, ref: float, reason: str, kind: str):
        nonlocal cash
        value = qty * price
        if side == "buy":
            f, tax = rules.buy_fee(value), 0.0
            cash -= value + f
        else:
            f, tax = rules.sell_costs(value)
            cash += value - f - tax
        adverse = (price - ref) if side == "buy" else (ref - price)       # against you > 0; a limit fill better than the open < 0
        fills.append({"date": cal[i], "idx": i, "instrument_id": ids[j], "side": side, "qty": qty, "price": price, "ref_price": ref,
                      "value": value, "fee": f, "tax": tax, "slippage_cost": adverse * qty, "reason": reason, "kind": kind})
        return value, f, tax

    def close_position(i: int, j: int, p: _Pos, reason: str):
        net_cost, net_proceeds = p.cost, p.proceeds
        trips.append({"instrument_id": ids[j], "entry_date": cal[p.entry], "exit_date": cal[i], "sessions": i - p.entry,
                      "cost": net_cost, "proceeds": net_proceeds, "pnl": net_proceeds - net_cost,
                      "net_return": net_proceeds / net_cost - 1 if net_cost > 0 else np.nan,
                      "gross_return": p.sell_gross / p.buy_gross - 1 if p.buy_gross > 0 else np.nan, "exit_reason": reason})
        del pos[j]

    def exec_sell(i: int, j: int, qty: int, price: float, ref: float, reason: str, kind: str):
        p = pos[j]
        value, f, tax = record_fill(i, j, "sell", qty, price, ref, reason, kind)
        p.proceeds += value - f - tax
        p.sell_gross += value
        p.remove(qty)
        p.last_reason = reason
        if p.qty == 0:
            close_position(i, j, p, reason)

    def exec_buy(i: int, j: int, qty: int, price: float, ref: float, o: _Order):
        p = pos.get(j)
        if p is None:
            p = pos[j] = _Pos(j, i)
        value, f, tax = record_fill(i, j, "buy", qty, price, ref, o.reason or o.kind, o.kind)
        p.cost += value + f
        p.buy_gross += value
        p.lots.append([qty, i])
        p.fills += 1
        it = o.item
        if it is not None and (p.stop is None and p.target is None and p.time_idx is None):
            if it.stop_pct is not None:
                p.stop = rules.round_down(price * (1 - it.stop_pct))
            if it.target_pct is not None:
                p.target = rules.round_up(price * (1 + it.target_pct))
            if it.max_hold is not None:
                p.time_idx = i + it.max_hold

    def open_ok(i: int, j: int) -> bool:
        return bool(np.isfinite(O[i, j]) and np.isfinite(C[i, j]))

    # ------------------------------------------------------------------------------------------------------------------
    for i in range(start, end):
        # 0. time-stops that fall due
        for j, p in list(pos.items()):
            if p.time_idx is not None and i >= p.time_idx and not any(o.j == j and o.side == "sell" for o in pending):
                oid += 1
                pending.append(_Order(oid, j, "sell", "time", i - 1, i, i + cfg.max_pending_sell_sessions, qty=p.qty, reason="time"))

        # 1. execute orders decided at the previous close: sells first, then buys
        still: list[_Order] = []
        for o in [x for x in pending if x.side == "sell"] + [x for x in pending if x.side == "buy"]:
            if o.first > i:
                still.append(o)
                continue
            j, done = o.j, False
            if o.side == "sell":
                p = pos.get(j)
                if p is None:
                    log_order(o, "cancelled", i, "position already closed")
                    continue
                ok, why = open_ok(i, j), ""
                ceil_, floor_ = band_prices(i, j)
                if not ok:
                    why = "suspended_or_no_open"
                elif O[i, j] <= floor_ * (1 + rules.lock_tolerance):
                    why = "limit_down_locked"
                else:
                    sellable = p.sellable(i, settle)
                    qty = min(o.qty, sellable, p.qty)
                    if qty <= 0:
                        why = "t_plus_settlement"
                    else:
                        price = max(rules.sell_price(O[i, j]), floor_)
                        exec_sell(i, j, qty, price, float(O[i, j]), o.reason, o.kind)
                        o.qty -= qty
                        done = o.qty <= 0 or j not in pos
                if why:
                    blocked[why] += 1
                    o.tries += 1
                if done:
                    log_order(o, "filled", i)
                elif i >= o.last:
                    log_order(o, "expired", i, "sell persisted too long")
                else:
                    still.append(o)
            else:  # buy
                why = ""
                ceil_, floor_ = band_prices(i, j)
                has_bar = np.isfinite(C[i, j])
                if not has_bar:
                    why = "suspended_or_no_open"
                elif o.kind == "limit":
                    lim = o.limit
                    if not np.isfinite(L[i, j]) or L[i, j] > lim:
                        why = "limit_not_reached"
                    else:
                        price = min(lim, O[i, j]) if np.isfinite(O[i, j]) else lim
                        price = min(price, ceil_)
                elif not open_ok(i, j):
                    why = "suspended_or_no_open"
                elif O[i, j] >= ceil_ * (1 - rules.lock_tolerance):
                    why = "limit_up_locked"
                else:
                    price = min(rules.buy_price(O[i, j]), ceil_)
                if not why:
                    qty = rules.buy_quantity(min(o.value, cash), price)
                    if qty < lot:
                        why = "lot_too_small" if o.value < price * lot * (1 + fee) else "not_enough_cash"
                    else:
                        exec_buy(i, j, qty, price, float(O[i, j]) if np.isfinite(O[i, j]) else price, o)
                        done = True
                if why:
                    blocked[why] += 1
                if done:
                    log_order(o, "filled", i)
                elif i >= o.last:
                    log_order(o, "unfilled", i, why)
                else:
                    still.append(o)
        pending = still

        # 2. exit levels on positions whose shares are sellable
        for j, p in list(pos.items()):
            if (p.stop is None and p.target is None) or not open_ok(i, j) or not (np.isfinite(H[i, j]) and np.isfinite(L[i, j])):
                continue
            sellable = p.sellable(i, settle)
            if sellable <= 0:
                continue
            ceil_, floor_ = band_prices(i, j)
            if H[i, j] <= floor_ * (1 + rules.lock_tolerance):          # locked at the floor all session: cannot sell
                blocked["limit_down_locked"] += 1
                continue
            o_, h_, l_ = O[i, j], H[i, j], L[i, j]
            stop, tgt = p.stop, p.target
            price = reason = None
            if stop is not None and o_ <= stop:
                price, reason = max(rules.sell_price(o_), floor_), "stop_gap"
            elif tgt is not None and o_ >= tgt:
                price, reason = o_, "target_gap"
            else:
                hit_s = stop is not None and l_ <= stop
                hit_t = tgt is not None and h_ >= tgt
                if hit_s and hit_t:
                    hit_s, hit_t = (True, False) if cfg.tie == "stop_first" else (False, True)
                if hit_s:
                    price, reason = max(rules.sell_price(stop), l_, floor_), "stop"
                elif hit_t:
                    price, reason = float(tgt), "target"
            if price is not None:
                exec_sell(i, j, sellable, float(price), float(o_), reason, "exit_rule")

        # 3. mark to market at the close
        for j in range(N):
            if np.isfinite(C[i, j]):
                last_close[j] = C[i, j]
        held = sum(p.qty * last_close[j] for j, p in pos.items())
        equity = cash + held
        eq_rows.append((cal[i], equity)); cash_rows.append(cash); expo_rows.append(held / equity if equity > 0 else 0.0)

        # 4. the signal computed at this close becomes orders for the next session
        sig = signals.get(i)
        if sig is not None and i + 1 < end:                              # a decision at the last session cannot execute
            listed = {col[it.instrument_id] for it in sig.items if it.instrument_id in col}
            for it in sig.items:
                j = col.get(it.instrument_id)
                if j is None or not np.isfinite(last_close[j]):
                    continue
                if it.only_if_flat and (j in pos or any(x.j == j for x in pending)):
                    continue
                if cfg.max_positions is not None and it.weight > 0 and j not in pos:
                    occupied = set(pos) | {x.j for x in pending if x.side == "buy"}
                    if j not in occupied and len(occupied) >= cfg.max_positions:
                        blocked["max_positions"] += 1
                        continue
                for o in [x for x in pending if x.j == j]:                      # a new decision supersedes old pending orders
                    log_order(o, "superseded", i)
                pending = [x for x in pending if x.j != j]
                cur_qty = pos[j].qty if j in pos else 0
                cur_val = cur_qty * last_close[j]
                target_val = it.weight * equity
                delta = target_val - cur_val
                if it.weight > 0 and cur_qty > 0 and abs(delta) < cfg.rebalance_threshold * target_val:      # relative to the TARGET, so a 2% weight is not frozen by a 2%-of-equity band
                    continue
                oid += 1
                if delta > 0 and it.weight > 0:
                    lim = None
                    if it.order == "limit":
                        lim = rules.round_down(last_close[j] * (1 - it.limit_offset))
                    pending.append(_Order(oid, j, "buy", it.order, i, i + 1, i + max(1, it.valid_sessions), value=delta, limit=lim,
                                          reason="rebalance" if cur_qty else "entry", item=it))
                elif delta < 0 or it.weight == 0:
                    qty = cur_qty if it.weight == 0 else rules.lots(-delta / last_close[j])
                    if qty >= (1 if it.weight == 0 else lot) and cur_qty > 0:
                        pending.append(_Order(oid, j, "sell", "rebalance", i, i + 1, i + 1 + cfg.max_pending_sell_sessions, qty=min(qty, cur_qty),
                                              reason="rebalance"))
            if sig.full_rebalance:
                for j in list(pos):
                    if j in listed:
                        continue
                    for o in [x for x in pending if x.j == j]:
                        log_order(o, "superseded", i)
                    pending = [x for x in pending if x.j != j]
                    oid += 1
                    pending.append(_Order(oid, j, "sell", "rebalance", i, i + 1, i + 1 + cfg.max_pending_sell_sessions, qty=pos[j].qty, reason="rebalance"))

    for o in pending:
        log_order(o, "open_at_end", None)
    equity_s = pd.Series([e for _, e in eq_rows], index=pd.DatetimeIndex([d for d, _ in eq_rows], name="date"), name="equity")
    open_pos = pd.DataFrame([{"instrument_id": ids[j], "qty": p.qty, "entry_date": cal[p.entry], "value": p.qty * last_close[j]} for j, p in pos.items()])
    fills_df = pd.DataFrame(fills)
    orders_df = pd.DataFrame(orders_log)
    stats = {"blocked": dict(blocked), "n_fills": len(fills), "n_round_trips": len(trips), "open_positions": len(pos)}
    return BacktestResult(equity_s, pd.Series(cash_rows, index=equity_s.index, name="cash"), pd.Series(expo_rows, index=equity_s.index, name="exposure"),
                          fills_df, orders_df, pd.DataFrame(trips), open_pos, stats)
