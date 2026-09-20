"""Read-only data-quality checks on stored daily bars.

Findings are *reported, never silently fixed*: a flagged bar might be a real event (suspension,
first listing day, a stock that traded on another exchange with another price band) and the
decision belongs to whoever reads the report. Persistence, known-issue matching and the Markdown
report live in quality_store.py / quality_report.py.

Check                 severity  what it looks for
--------------------  --------  -------------------------------------------------------------------------
no_data               error     an instrument with no bars at all
duplicate_bar         error     more than one row per (instrument, date)  (the primary key forbids it)
ohlc_inconsistent     error     high < max(open, close, low) or low > min(open, close, high)
nonpositive_price     error     any price <= 0
price_unit            error     stock price outside the plausible VND range, or a day-over-day jump of
                                about x1000 (thousand-VND vs VND mix-up)
zero_volume           warn      volume == 0 on a stock
repeated_bar          warn      a bar identical (OHLCV) to the previous one: copied/stale data
big_move              warn      |close return| beyond the price band (+ tolerance)
return_outlier        warn      robust z-score (median/MAD) of the daily log return beyond the threshold
volume_spike          warn      volume far above the median of the previous sessions
missing_trading_day   warn      trading-calendar day without a bar between an instrument's first/last bar
extra_day             warn      bar on a date outside the trading calendar
stale_series          warn      last bar older than the latest trading day
vendor_duplicate      info      same-date rows the vendor sent twice and the client merged (from ingest runs)
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from predict_stock.config import AppConfig
from predict_stock.universe import Member

SEVERITY = {
    "no_data": "error", "duplicate_bar": "error", "ohlc_inconsistent": "error", "nonpositive_price": "error",
    "price_unit": "error",
    "zero_volume": "warn", "repeated_bar": "warn", "big_move": "warn", "return_outlier": "warn", "volume_spike": "warn",
    "missing_trading_day": "warn", "extra_day": "warn", "stale_series": "warn",
    "vendor_duplicate": "info",
}
COLUMNS = ["instrument_id", "symbol", "trade_date", "check", "severity", "detail"]
_MERGED = re.compile(r"merged (\d+) same-date rows for (\d{4}-\d{2}-\d{2})")


def _row(iid: int, symbol: str, trade_date, check: str, detail: str) -> dict:
    return {"instrument_id": iid, "symbol": symbol, "trade_date": trade_date, "check": check,
            "severity": SEVERITY[check], "detail": detail}


def _load_bars(conn, ids: list[int]) -> pd.DataFrame:
    bars = pd.read_sql(
        text(
            "SELECT b.instrument_id, b.trade_date, b.open, b.high, b.low, b.close, b.volume, i.kind "
            "FROM price_bar b JOIN instruments i ON i.id = b.instrument_id "
            "WHERE b.instrument_id IN :ids ORDER BY b.instrument_id, b.trade_date"
        ).bindparams(bindparam("ids", expanding=True)),
        conn,
        params={"ids": ids},
    )
    for c in ("open", "high", "low", "close"):
        bars[c] = bars[c].astype(float)
    bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.date
    return bars


def _series_checks(iid: int, symbol: str, g: pd.DataFrame, cfg: AppConfig) -> list[dict]:
    q = cfg.quality
    out: list[dict] = []
    is_stock = g["kind"].iloc[0] == "stock"
    dates = g["trade_date"]

    # ---- bar consistency (all instrument kinds)
    hi_low = g[["open", "close", "low"]].max(axis=1)
    lo_high = g[["open", "close", "high"]].min(axis=1)
    for i in g.index[(g["high"] < hi_low) | (g["low"] > lo_high)]:
        r = g.loc[i]
        parts = []
        if not r.low <= r.open <= r.high:
            parts.append("open outside [low, high]")
        if not r.low <= r.close <= r.high:
            parts.append("close outside [low, high]")
        if r.high < r.low:
            parts.append("high < low")
        size = max(r.open - r.high, r.low - r.open, r.close - r.high, r.low - r.close, 0) / r.close
        out.append(_row(iid, symbol, dates[i], "ohlc_inconsistent",
                        f"{', '.join(parts) or 'inconsistent'}; o={r.open:g} h={r.high:g} l={r.low:g} c={r.close:g}; violation {size:.2%} of close"))
    for i in g.index[(g[["open", "high", "low", "close"]] <= 0).any(axis=1)]:
        out.append(_row(iid, symbol, dates[i], "nonpositive_price", "a price is <= 0"))
    for i in g.index[(g[["open", "high", "low", "close"]].isna()).any(axis=1)]:
        out.append(_row(iid, symbol, dates[i], "nonpositive_price", "a price is missing"))

    if not is_stock:
        return out

    # ---- price unit
    lo, hi = q.price_range_vnd
    for i in g.index[(g["close"] < lo) | (g["close"] > hi)]:
        out.append(_row(iid, symbol, dates[i], "price_unit", f"close {g.close[i]:g} VND outside the plausible range [{lo:g}, {hi:g}]"))
    ratio = g["close"] / g["close"].shift()
    for i in g.index[(ratio >= q.unit_jump_ratio) | (ratio <= 1 / q.unit_jump_ratio)]:
        out.append(_row(iid, symbol, dates[i], "price_unit", f"close changed x{ratio[i]:.4g} in one session (thousand-VND vs VND mix-up?)"))

    # ---- volume and repeated bars
    for i in g.index[g["volume"] == 0]:
        out.append(_row(iid, symbol, dates[i], "zero_volume", "volume == 0"))
    cols = ["open", "high", "low", "close", "volume"]
    for i in g.index[(g[cols] == g[cols].shift()).all(axis=1)]:
        out.append(_row(iid, symbol, dates[i], "repeated_bar", "identical to the previous bar (copied or stale data?)"))
    med = g["volume"].rolling(q.volume_spike_window, min_periods=q.volume_spike_window // 2).median().shift()
    spike = (med > 0) & (g["volume"] / med > q.volume_spike_ratio)
    for i in g.index[spike]:
        out.append(_row(iid, symbol, dates[i], "volume_spike", f"volume {g.volume[i]:,} = x{g.volume[i] / med[i]:.1f} the prior {q.volume_spike_window}-session median"))

    # ---- returns
    limit = cfg.market.price_limit()
    ret = g["close"].pct_change()
    for i in ret.index[ret.abs() > limit + q.big_move_tolerance]:
        gap = (dates[i] - dates[i - 1]).days
        out.append(_row(iid, symbol, dates[i], "big_move",
                        f"ret={ret[i]:+.2%} vs limit ±{limit:.0%} (+{q.big_move_tolerance:.0%} tol); {gap} calendar days since previous bar"))
    lr = np.log(g["close"] / g["close"].shift()).replace([np.inf, -np.inf], np.nan)
    if lr.notna().sum() >= 60:
        mad = (lr - lr.median()).abs().median() * 1.4826
        if mad > 0:
            z = (lr - lr.median()) / mad
            for i in z.index[z.abs() > q.return_outlier_z]:
                out.append(_row(iid, symbol, dates[i], "return_outlier", f"log-return {lr[i]:+.3f}, robust z={z[i]:+.1f} (threshold {q.return_outlier_z:g})"))
    return out


def run_quality_checks(session: Session, instruments: list[Member] | list[tuple[int, str]], cfg: AppConfig) -> pd.DataFrame:
    """``instruments``: Members or (instrument_id, symbol) pairs to check."""
    pairs = [(m.instrument_id, m.symbol) if isinstance(m, Member) else m for m in instruments]
    if not pairs:
        return pd.DataFrame(columns=COLUMNS)
    conn = session.connection()
    ids = [i for i, _ in pairs]
    bars = _load_bars(conn, ids)
    days = pd.read_sql(
        text("SELECT trade_date FROM trading_calendar WHERE calendar_code = :c ORDER BY trade_date"),
        conn, params={"c": cfg.ingest.calendar_code},
    )
    trading_days = sorted(pd.to_datetime(days["trade_date"]).dt.date)
    day_set = set(trading_days)
    out: list[dict] = []

    dup = conn.execute(text(
        "SELECT instrument_id, trade_date, COUNT(*) FROM price_bar WHERE instrument_id IN :ids GROUP BY 1, 2 HAVING COUNT(*) > 1"
    ).bindparams(bindparam("ids", expanding=True)), {"ids": ids}).all()
    label = dict(pairs)
    out += [_row(i, label[i], d, "duplicate_bar", f"{n} rows for one (instrument, date)") for i, d, n in dup]

    for iid, symbol in pairs:
        g = bars[bars["instrument_id"] == iid].reset_index(drop=True)
        if g.empty:
            out.append(_row(iid, symbol, None, "no_data", "no bars stored"))
            continue
        out += _series_checks(iid, symbol, g, cfg)
        if day_set:
            first, last = g["trade_date"].iloc[0], g["trade_date"].iloc[-1]
            have = set(g["trade_date"])
            out += [_row(iid, symbol, d, "missing_trading_day", "trading day without a bar (suspension or data gap)")
                    for d in trading_days if first <= d <= last and d not in have]
            out += [_row(iid, symbol, d, "extra_day", "bar on a date outside the trading calendar") for d in sorted(have - day_set)]
            if last < trading_days[-1]:
                out.append(_row(iid, symbol, last, "stale_series", f"last bar {last} < latest trading day {trading_days[-1]}"))

    # same-date rows the vendor sent twice (merged by the client): recorded by the ingest runs
    warn_rows = conn.execute(text(
        "SELECT DISTINCT instrument_id, warnings FROM data_ingest_runs WHERE warnings IS NOT NULL AND instrument_id IN :ids"
    ).bindparams(bindparam("ids", expanding=True)), {"ids": ids}).all()
    seen = set()
    for iid, warnings in warn_rows:
        for w in (json.loads(warnings) if isinstance(warnings, str) else warnings) or []:
            m = _MERGED.search(w)
            if m and (iid, m.group(2)) not in seen:
                seen.add((iid, m.group(2)))
                out.append(_row(iid, label[iid], pd.Timestamp(m.group(2)).date(), "vendor_duplicate",
                                f"{m.group(1)} same-date rows from the vendor were merged into one bar"))
    return pd.DataFrame(out, columns=COLUMNS)


def summarize(issues: pd.DataFrame) -> pd.DataFrame:
    if issues.empty:
        return pd.DataFrame(columns=["check", "severity", "count", "symbols"])
    order = {"error": 0, "warn": 1, "info": 2}
    return (
        issues.groupby(["check", "severity"])
        .agg(count=("symbol", "size"), symbols=("symbol", "nunique"))
        .reset_index()
        .assign(_o=lambda d: d.severity.map(order))
        .sort_values(["_o", "count"], ascending=[True, False])
        .drop(columns="_o")
    )
