"""Trading calendar.

DNSE's official `get_working_dates` needs an API key (see docs/DNSE_API_NOTES.md) and the public
endpoint has no calendar, so `trading_calendar` is *observed*: a date is a trading day when at least
`ingest.calendar_min_stock_fraction` of the active stocks have a bar (see `ingest.sync_trading_days`).
Index series are not used: the VNINDEX series misses ~33 real trading days.

Known limits: future holidays are unknown, and a day on which the vendor lost most stocks' bars would
look like a holiday. `calendar_gaps` reports the weekday gaps so a human can tell the two apart.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from predict_stock.config import AppConfig
from predict_stock.data.ingest import sync_trading_days
from predict_stock.db.models import Instrument, PriceBar, TradingCalendar


def sync_trading_calendar(session: Session, cfg: AppConfig) -> dict[str, int]:
    return sync_trading_days(session, cfg.ingest.calendar_code, cfg.ingest.calendar_min_stock_fraction, cfg.ingest.calendar_grace_days)


def calendar_gaps(session: Session, cfg: AppConfig) -> dict:
    """Weekdays between the first and last calendar day that are not trading days.
    ``holiday_like`` = no stock has a bar; ``partial`` = some stocks do (below the consensus)."""
    days = sorted(session.scalars(select(TradingCalendar.trade_date).where(TradingCalendar.calendar_code == cfg.ingest.calendar_code)))
    if not days:
        return {"trading_days": 0, "first": None, "last": None, "by_year": {}, "holiday_like": [], "partial": []}
    have = set(days)
    bars_on = dict(session.execute(
        select(PriceBar.trade_date, func.count()).join(Instrument, Instrument.id == PriceBar.instrument_id)
        .where(Instrument.kind == "stock").group_by(PriceBar.trade_date)).all())
    holiday_like, partial = [], []
    d = days[0]
    while d <= days[-1]:
        if d.weekday() < 5 and d not in have:
            n = bars_on.get(d, 0)
            (partial if n else holiday_like).append((d, n) if n else d)
        d += timedelta(days=1)
    per_year = Counter(x.year for x in days)
    gaps_year = Counter(x.year for x in holiday_like)
    by_year = {y: {"trading_days": per_year[y], "weekday_gaps": gaps_year.get(y, 0)} for y in sorted(per_year)}
    return {"trading_days": len(days), "first": days[0], "last": days[-1], "by_year": by_year,
            "holiday_like": holiday_like, "partial": partial}
