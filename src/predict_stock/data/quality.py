"""Read-only data-quality checks on stored daily bars.

Findings are *reported, never silently fixed*: a flagged bar might be a real
event (suspension, first listing day, unadjusted corporate action) and the
decision belongs to whoever reads the report.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from predict_stock.config import AppConfig

COLUMNS = ["symbol", "trade_date", "check", "severity", "detail"]


def _issue(symbol: str, trade_date, check: str, severity: str, detail: str) -> dict:
    return {"symbol": symbol, "trade_date": trade_date, "check": check, "severity": severity, "detail": detail}


def run_quality_checks(session: Session, symbols: list[str], cfg: AppConfig) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=COLUMNS)
    conn = session.connection()
    bars = pd.read_sql(
        text(
            "SELECT o.symbol, o.trade_date, o.open, o.high, o.low, o.close, o.volume, i.kind "
            "FROM ohlcv_daily o JOIN instruments i ON i.symbol = o.symbol "
            "WHERE o.symbol IN :syms ORDER BY o.symbol, o.trade_date"
        ).bindparams(bindparam("syms", expanding=True)),
        conn,
        params={"syms": list(symbols)},
    )
    for c in ("open", "high", "low", "close"):
        bars[c] = bars[c].astype(float)
    bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.date
    days = pd.read_sql(text("SELECT trade_date FROM trading_days ORDER BY trade_date"), conn)
    trading_days = sorted(pd.to_datetime(days["trade_date"]).dt.date)
    day_set = set(trading_days)

    tol = cfg.quality.big_move_tolerance
    limit = cfg.market.price_limit()
    out: list[dict] = []

    for symbol in symbols:
        g = bars[bars["symbol"] == symbol].reset_index(drop=True)
        if g.empty:
            out.append(_issue(symbol, None, "no_data", "error", "no bars stored"))
            continue
        is_stock = g["kind"].iloc[0] == "stock"

        bad_ohlc = g[(g["high"] < g[["open", "close", "low"]].max(axis=1)) | (g["low"] > g[["open", "close", "high"]].min(axis=1))]
        out += [_issue(symbol, r.trade_date, "ohlc_inconsistent", "error", f"o={r.open} h={r.high} l={r.low} c={r.close}") for r in bad_ohlc.itertuples()]
        bad_px = g[(g[["open", "high", "low", "close"]] <= 0).any(axis=1)]
        out += [_issue(symbol, r.trade_date, "nonpositive_price", "error", "price <= 0") for r in bad_px.itertuples()]

        if is_stock:
            zero_v = g[g["volume"] == 0]
            out += [_issue(symbol, r.trade_date, "zero_volume", "warn", "volume == 0") for r in zero_v.itertuples()]
            ret = g["close"].pct_change()
            for i in ret.index[ret.abs() > limit + tol]:
                gap = (g["trade_date"][i] - g["trade_date"][i - 1]).days
                out.append(_issue(symbol, g["trade_date"][i], "big_move", "warn", f"ret={ret[i]:+.2%} vs limit ±{limit:.0%} (+{tol:.0%} tol); {gap} calendar days since previous bar"))

        if day_set:
            first, last = g["trade_date"].iloc[0], g["trade_date"].iloc[-1]
            have = set(g["trade_date"])
            missing = [d for d in trading_days if first <= d <= last and d not in have]
            out += [_issue(symbol, d, "missing_trading_day", "warn", "trading day without a bar (suspension or data gap)") for d in missing]
            extra = sorted(have - day_set)
            out += [_issue(symbol, d, "extra_day", "warn", "bar on a date outside the trading calendar") for d in extra]
            if last < trading_days[-1]:
                out.append(_issue(symbol, last, "stale_series", "warn", f"last bar {last} < latest trading day {trading_days[-1]}"))

    return pd.DataFrame(out, columns=COLUMNS)


def summarize(issues: pd.DataFrame) -> pd.DataFrame:
    if issues.empty:
        return pd.DataFrame(columns=["check", "severity", "count", "symbols"])
    return (
        issues.groupby(["check", "severity"])
        .agg(count=("symbol", "size"), symbols=("symbol", "nunique"))
        .reset_index()
        .sort_values(["severity", "count"], ascending=[True, False])
    )
