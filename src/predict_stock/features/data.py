"""Price panel for features and labels: wide (calendar date x instrument) frames plus point-in-time membership.

Policies applied here (and reported in the dataset manifest):
* Only bars on trading-calendar days are used (bars outside the calendar are dropped and counted).
* Bars whose OHLC is inconsistent (Phase 2 known issues) are repaired by a fixed rule, not left to chance:
  ``high := max(high, low, close)``, ``low := min(low, close, high)`` so the range contains the close, and
  ``open := NaN`` when it still lies outside the range. A bar with a non-positive price becomes an empty bar.
* Prices are the vendor's back-adjusted series: ratios and returns are meaningful, absolute levels are not
  point-in-time (see docs/DNSE_API_NOTES.md). Features here are ratios; the one level-based feature (traded
  value, for liquidity) says so.
* The benchmark is aligned to the calendar with a *forward fill* of at most ``limit`` sessions: the value used at
  date t is the last one known at t, never a later one.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, select, text
from sqlalchemy.orm import Session

from predict_stock.db.models import TradingCalendar, Universe, UniverseMembership
from predict_stock.db.repo import find_symbol_row

FIELDS = ("open", "high", "low", "close", "volume")


@dataclass
class Panel:
    calendar: pd.DatetimeIndex
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    bench: pd.Series | None = None  # benchmark close on the calendar (point-in-time)
    repaired_bars: int = 0
    dropped_non_calendar: int = 0
    bench_info: dict | None = None

    @property
    def instrument_ids(self) -> list[int]:
        return list(self.close.columns)

    def truncate(self, last: pd.Timestamp) -> "Panel":
        """The same panel as it would have looked with no data after ``last`` (used by the look-ahead audit)."""
        cut = self.calendar <= last
        return Panel(self.calendar[cut], *(getattr(self, f).loc[cut] for f in FIELDS),
                     bench=None if self.bench is None else self.bench.loc[cut],
                     repaired_bars=self.repaired_bars, dropped_non_calendar=self.dropped_non_calendar, bench_info=self.bench_info)

    def bars_of(self, iid: int) -> pd.DataFrame:
        """One instrument's rows (only sessions it has a bar), columns open/high/low/close/volume."""
        f = pd.DataFrame({k: getattr(self, k)[iid] for k in FIELDS})
        return f[f["close"].notna()]


def repair_ohlc(o: pd.DataFrame, h: pd.DataFrame, l: pd.DataFrame, c: pd.DataFrame, v: pd.DataFrame):
    """Apply the repair rule to wide frames. Returns (o, h, l, c, v, number_of_repaired_bars)."""
    nonpos = (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)
    if nonpos.to_numpy().any():
        o, h, l, c, v = (x.mask(nonpos) for x in (o, h, l, c, v))
    has = c.notna()
    h2 = np.fmax(np.fmax(h, c), l)
    l2 = np.fmin(np.fmin(l, c), h)
    o2 = o.where((o >= l2) & (o <= h2))
    repaired = ((h2 != h) | (l2 != l) | (o.notna() & o2.isna())) & has
    return o2, h2, l2, c, v, int(repaired.to_numpy().sum()) + int(nonpos.to_numpy().sum())


def splice_benchmark(primary: pd.Series, fallback: pd.Series | None) -> pd.Series:
    """Where ``primary`` has no value yet, use ``fallback`` rescaled so the two agree on the first date the primary
    exists (levels stay continuous; returns before that date are the fallback's). Never uses a later value."""
    if fallback is None:
        return primary
    first = primary.first_valid_index()
    if first is None:
        return fallback
    f0 = fallback.get(first)
    if f0 is None or pd.isna(f0) or f0 == 0:
        return primary
    scaled = fallback * (primary[first] / f0)
    return primary.where(primary.index >= first, scaled.reindex(primary.index))


def load_calendar(session: Session, code: str, start: date, end: date) -> pd.DatetimeIndex:
    rows = session.scalars(select(TradingCalendar.trade_date).where(
        TradingCalendar.calendar_code == code, TradingCalendar.trade_date >= start, TradingCalendar.trade_date <= end).order_by(TradingCalendar.trade_date))
    return pd.DatetimeIndex(pd.to_datetime(list(rows)), name="trade_date")


def _bars(session: Session, ids: list[int], start: date, end: date) -> pd.DataFrame:
    df = pd.read_sql(
        text("SELECT instrument_id, trade_date, open, high, low, close, volume FROM price_bar "
             "WHERE instrument_id IN :ids AND trade_date BETWEEN :s AND :e").bindparams(bindparam("ids", expanding=True)),
        session.connection(), params={"ids": ids, "s": start, "e": end})
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def load_panel(session: Session, *, calendar_code: str, instrument_ids: list[int], load_start: date, cutoff: date,
               bench_symbols: tuple[str | None, str | None] = (None, None), bench_ffill_limit: int = 5) -> Panel:
    """Wide price panel for ``instrument_ids`` on the trading calendar in [load_start, cutoff]."""
    cal = load_calendar(session, calendar_code, load_start, cutoff)
    ids = sorted(instrument_ids)
    df = _bars(session, ids, load_start, cutoff) if ids else pd.DataFrame(columns=["instrument_id", "trade_date", *FIELDS])
    on_cal = df["trade_date"].isin(cal)
    dropped = int((~on_cal).sum())
    df = df[on_cal]
    wide = {f: df.pivot(index="trade_date", columns="instrument_id", values=f).reindex(index=cal, columns=ids) for f in FIELDS}
    o, h, l, c, v, n_rep = repair_ohlc(wide["open"], wide["high"], wide["low"], wide["close"], wide["volume"])
    bench, info = None, None
    primary_sym, fallback_sym = bench_symbols
    if primary_sym:
        series = {}
        for sym in (primary_sym, fallback_sym):
            if not sym:
                continue
            row = find_symbol_row(session, sym)
            if row is None:
                continue
            b = _bars(session, [row.instrument_id], load_start, cutoff)
            s = b.set_index("trade_date")["close"].reindex(cal).ffill(limit=bench_ffill_limit)
            series[sym] = s
        if primary_sym in series:
            bench = splice_benchmark(series[primary_sym], series.get(fallback_sym) if fallback_sym else None)
        elif fallback_sym in series:
            bench = series[fallback_sym]
        if bench is not None:
            info = {"primary": primary_sym, "fallback": fallback_sym, "first_valid": str(bench.first_valid_index().date()) if bench.first_valid_index() is not None else None,
                    "primary_first_valid": str(series[primary_sym].first_valid_index().date()) if primary_sym in series and series[primary_sym].first_valid_index() is not None else None,
                    "coverage": round(float(bench.notna().mean()), 4)}
    return Panel(cal, o, h, l, c, v, bench=bench, repaired_bars=n_rep, dropped_non_calendar=dropped, bench_info=info)


def membership_mask(session: Session, universe_code: str, calendar: pd.DatetimeIndex, instrument_ids: list[int]) -> pd.DataFrame:
    """Boolean (date x instrument): True when the instrument is a member of the universe on that date
    (valid_from <= d and (valid_to is NULL or d < valid_to)); computed from the dated membership rows only."""
    mask = pd.DataFrame(False, index=calendar, columns=sorted(instrument_ids))
    rows = session.execute(
        select(UniverseMembership.instrument_id, UniverseMembership.valid_from, UniverseMembership.valid_to)
        .join(Universe, Universe.id == UniverseMembership.universe_id)
        .where(Universe.code == universe_code, UniverseMembership.instrument_id.in_(instrument_ids or [0]))).all()
    return mask_from_intervals(calendar, mask.columns, rows)


def mask_from_intervals(calendar: pd.DatetimeIndex, instrument_ids, rows) -> pd.DataFrame:
    """``rows``: (instrument_id, valid_from, valid_to|None) with half-open [valid_from, valid_to)."""
    mask = pd.DataFrame(False, index=calendar, columns=list(instrument_ids))
    for iid, vf, vt in rows:
        if iid not in mask.columns:
            continue
        sel = calendar >= pd.Timestamp(vf)
        if vt is not None:
            sel &= calendar < pd.Timestamp(vt)
        mask.loc[sel, iid] = True
    return mask
