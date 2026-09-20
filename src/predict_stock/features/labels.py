"""Labels. They look FORWARD from the decision date t (the close of session t) by design; the decision is taken
after the close, and execution costs/slippage are the backtester's job (Phase 4), not part of the label.

* ``fwd_rank_return``  forward return over h sessions and its cross-sectional rank among the universe members ON t.
* ``triple_barrier``   target = close + m_t x ATR(t), stop = close - m_s x ATR(t), time barrier after H sessions.
  Barriers are checked on the daily high/low of sessions t+1 .. t+H. If both are touched on the same day the STOP
  wins (daily bars do not show the intraday order: this is the conservative choice for a long-only position). If a
  session opens beyond a barrier the exit is at the open (gap-through), not at the barrier. Bars without data
  (suspension) are skipped. If the window does not fit inside the data the label is NaN (never a partial window).

Every label also gives its END date (when its outcome was known) for purging / embargo in walk-forward CV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from predict_stock.features.cs import masked_rank
from predict_stock.features.registry import LabelBuilder, RegistryError, register_label
from predict_stock.features.ts import atr


def _shift_up(a: np.ndarray, k: int) -> np.ndarray:
    """Row t gets row t+k (NaN where t+k is beyond the data)."""
    out = np.full_like(a, np.nan, dtype=float)
    if k < a.shape[0]:
        out[: a.shape[0] - k] = a[k:]
    return out


def _end_dates(calendar: pd.DatetimeIndex, steps: np.ndarray) -> pd.DataFrame:
    """steps[t, j] = number of sessions until the outcome (0 or negative = unknown) -> the calendar date."""
    T, N = steps.shape
    idx = np.arange(T)[:, None] + steps
    ok = (steps > 0) & (idx < T)
    vals = np.where(ok, calendar.to_numpy()[np.clip(idx, 0, T - 1)], np.datetime64("NaT", "ns"))
    return pd.DataFrame(vals, index=calendar)


@register_label
class ForwardRankReturn(LabelBuilder):
    name, version = "fwd_rank_return", 1
    DEFAULT_PARAMS = {"horizons": [3, 5], "min_count": 5}

    def validate(self):
        h = self.params["horizons"]
        if not h or any((not isinstance(x, int)) or x < 1 for x in h):
            raise RegistryError(f"horizons must be positive integers, got {h!r}")

    def columns(self):
        return [c for h in self.params["horizons"] for c in (f"fwd_ret_{h}", f"fwd_rank_{h}", f"fwd_end_{h}")]

    def horizon(self): return max(self.params["horizons"])

    def compute(self, panel, mask):
        out: dict[str, pd.DataFrame] = {}
        close = panel.close
        for h in self.params["horizons"]:
            fwd = close.shift(-h) / close - 1                      # NaN when either close is missing
            steps = np.where(fwd.notna().to_numpy(), h, 0)
            out[f"fwd_ret_{h}"] = fwd
            out[f"fwd_rank_{h}"] = masked_rank(fwd, mask, self.params["min_count"])
            out[f"fwd_end_{h}"] = _end_dates(panel.calendar, steps).set_axis(close.columns, axis=1)
        return out


@register_label
class TripleBarrier(LabelBuilder):
    name, version = "triple_barrier", 1
    DEFAULT_PARAMS = {"horizon": 10, "atr_window": 14, "target_mult": 2.0, "stop_mult": 1.0, "suffix": ""}

    def validate(self):
        p = self.params
        if not isinstance(p["horizon"], int) or p["horizon"] < 1 or not isinstance(p["atr_window"], int) or p["atr_window"] < 1:
            raise RegistryError("horizon and atr_window must be positive integers")
        if p["target_mult"] <= 0 or p["stop_mult"] <= 0:
            raise RegistryError("target_mult and stop_mult must be positive")

    def columns(self):
        s = self.params["suffix"]
        return [f"tb_label{s}", f"tb_time{s}", f"tb_ret{s}", f"tb_end{s}"]

    def horizon(self): return self.params["horizon"]

    def compute(self, panel, mask):
        p, hz = self.params, self.params["horizon"]
        cal = panel.calendar
        H, L, O, C = (panel.high.to_numpy(float), panel.low.to_numpy(float), panel.open.to_numpy(float), panel.close.to_numpy(float))
        T, N = C.shape
        A = np.full((T, N), np.nan)
        for j, iid in enumerate(panel.close.columns):
            a = atr(panel.bars_of(iid), p["atr_window"])
            A[:, j] = a.reindex(cal).to_numpy()
        target, stop = C + p["target_mult"] * A, C - p["stop_mult"] * A
        active = np.isfinite(C) & np.isfinite(A) & ((np.arange(T) + hz) < T)[:, None]
        label, time, ret = (np.full((T, N), np.nan) for _ in range(3))
        steps = np.zeros((T, N), dtype=int)
        resolved = np.zeros((T, N), dtype=bool)
        for k in range(1, hz + 1):
            hk, lk, ok = _shift_up(H, k), _shift_up(L, k), _shift_up(O, k)
            free = active & ~resolved & np.isfinite(hk) & np.isfinite(lk)
            hit_stop = free & (lk <= stop)
            hit_tgt = free & (hk >= target) & ~hit_stop            # same session: the stop wins
            exit_stop = np.where(np.isfinite(ok) & (ok <= stop), ok, stop)
            exit_tgt = np.where(np.isfinite(ok) & (ok >= target), ok, target)
            for hit, lab, ex in ((hit_stop, -1.0, exit_stop), (hit_tgt, 1.0, exit_tgt)):
                label[hit], time[hit], steps[hit] = lab, k, k
                ret[hit] = (ex / C - 1)[hit]
            resolved |= hit_stop | hit_tgt
        c_end = _shift_up(C, hz)
        timed_out = active & ~resolved & np.isfinite(c_end)
        label[timed_out], time[timed_out], steps[timed_out] = 0.0, hz, hz
        ret[timed_out] = (c_end / C - 1)[timed_out]
        s = p["suffix"]
        ends = _end_dates(cal, steps).set_axis(panel.close.columns, axis=1)
        mk = lambda a: pd.DataFrame(a, index=cal, columns=panel.close.columns)
        return {f"tb_label{s}": mk(label), f"tb_time{s}": mk(time), f"tb_ret{s}": mk(ret), f"tb_end{s}": ends}
