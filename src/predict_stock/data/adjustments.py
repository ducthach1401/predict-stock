"""Corporate actions and price adjustment.

Phase 0 concluded (by inference) that DNSE serves back-adjusted prices, so by default nothing is adjusted
here. What this module does instead:

* `detect_gap_candidates` — an ordinary session cannot OPEN beyond the price band around the previous
  close (compounded over the sessions between the two bars when bars are missing, so a suspension or a data
  gap is not mistaken for an event). An open gap beyond it in the stored series is the signature of an unadjusted corporate action
  (stock dividend, bonus, split) or of a bad bar. Such days are stored in `corporate_actions` as
  status='candidate' and raise a warning alert. **Nothing is ever adjusted from a candidate.**
  Limits: only events larger than the band are visible (small cash dividends are not), and where the
  exchange history is unknown the widest configured band is used, so events below ~16% are invisible there.
* `apply_confirmed` — an operator supplies a file of confirmed events; each becomes a confirmed
  corporate action plus a new *version* of the instrument's adjustment factors.
* `set_price_basis` — marks an instrument's bars 'raw' so the view `v_price_adjusted` applies the factors
  (bars marked 'vendor_adjusted' are never re-adjusted). Ingest keeps writing new bars with the same basis.
"""
from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from predict_stock.config import AppConfig
from predict_stock.db.models import (
    AdjustmentFactor,
    Alert,
    CorporateAction,
    InstrumentSymbolHistory as SymbolRow,
    PriceBar,
    PriceBarIntraday,
    TradingCalendar,
)
from predict_stock.db.repo import find_symbol_row

ACTION_TYPES = ("stock_dividend", "bonus", "split", "cash_dividend", "rights", "other")
ALERT_CATEGORY = "corporate_action_candidate"


class AdjustmentError(ValueError):
    pass


@dataclass(frozen=True)
class Candidate:
    instrument_id: int
    symbol: str
    trade_date: date
    gap: float  # open / previous close - 1
    prev_close: float
    open: float
    band: float
    implied_factor: float  # open / previous close: the factor that would bring the two back in line
    sessions: int = 1  # trading sessions between the previous bar and this one (1 = consecutive)


def _band(cfg: AppConfig, exchange: str | None) -> float:
    m = cfg.market
    if exchange:
        return m.price_limit_pct[exchange]
    return max(m.price_limit_pct.values()) if cfg.adjustments.detect_max_band else m.price_limit()


def _allowed_move(band: float, sessions: int, tol: float) -> tuple[float, float]:
    """(lowest, highest) plausible open/prev_close - 1 after ``sessions`` sessions, the band compounding each session."""
    return (1 - band) ** sessions - 1 - tol, (1 + band) ** sessions - 1 + tol


def detect_gap_candidates(session: Session, cfg: AppConfig, pairs: list[tuple[int, str]]) -> list[Candidate]:
    out: list[Candidate] = []
    calendar = sorted(session.scalars(select(TradingCalendar.trade_date).where(TradingCalendar.calendar_code == cfg.ingest.calendar_code)))
    for iid, symbol in pairs:
        bars = pd.DataFrame(
            session.execute(select(PriceBar.trade_date, PriceBar.open, PriceBar.close).where(PriceBar.instrument_id == iid).order_by(PriceBar.trade_date)).all(),
            columns=["trade_date", "open", "close"])
        if len(bars) < 2:
            continue
        hist = list(session.scalars(select(SymbolRow).where(SymbolRow.instrument_id == iid)))
        bars["open"], bars["close"] = bars["open"].astype(float), bars["close"].astype(float)
        bars["prev"] = bars["close"].shift()
        bars["prev_date"] = bars["trade_date"].shift()
        bars["gap"] = bars["open"] / bars["prev"] - 1
        for r in bars.dropna().itertuples():
            ex = next((h.exchange for h in hist if h.valid_from <= r.trade_date and (h.valid_to is None or r.trade_date < h.valid_to)), None)
            band = _band(cfg, ex)
            k = 1
            if calendar:  # trading sessions in (prev_date, trade_date]
                k = max(1, bisect.bisect_right(calendar, r.trade_date) - bisect.bisect_right(calendar, r.prev_date))
            lo, hi = _allowed_move(band, k, cfg.adjustments.gap_tolerance)
            if not lo <= r.gap <= hi:
                out.append(Candidate(iid, symbol, r.trade_date, float(r.gap), float(r.prev), float(r.open), band, float(r.open / r.prev), k))
    return out


def record_candidates(session: Session, candidates: list[Candidate], pairs: list[tuple[int, str]], run_id: int | None) -> dict[str, int]:
    """Idempotently store candidates; warn once per new one. Candidates no longer detected become 'stale'."""
    new = 0
    seen = set()
    for c in candidates:
        seen.add((c.instrument_id, c.trade_date))
        row = session.scalars(select(CorporateAction).where(
            CorporateAction.instrument_id == c.instrument_id, CorporateAction.ex_date == c.trade_date,
            CorporateAction.action_type == "unknown_gap")).first()
        details = {"gap": round(c.gap, 6), "prev_close": c.prev_close, "open": c.open, "band": c.band, "sessions": c.sessions,
                   "implied_factor": round(c.implied_factor, 6), "detected": "open beyond the price band around the previous close"}
        if row is None:
            session.add(CorporateAction(instrument_id=c.instrument_id, action_type="unknown_gap", ex_date=c.trade_date,
                                        details=details, status="candidate", source="detect_gap_candidates"))
            session.add(Alert(severity="warn", category=ALERT_CATEGORY, instrument_id=c.instrument_id, run_id=run_id,
                              message=f"{c.symbol} {c.trade_date}: open {c.gap:+.1%} vs previous close over {c.sessions} session(s) (band ±{c.band:.0%} per session); "
                                      "possible unadjusted corporate action or bad bar — confirm with `adjustments apply` or explain"))
            new += 1
        elif row.status in ("candidate", "stale"):
            row.details, row.status = details, "candidate"
    stale = 0
    ids = [i for i, _ in pairs]
    for row in session.scalars(select(CorporateAction).where(
            CorporateAction.instrument_id.in_(ids), CorporateAction.action_type == "unknown_gap", CorporateAction.status == "candidate")):
        if (row.instrument_id, row.ex_date) not in seen:
            row.status = "stale"
            stale += 1
            for alert in session.scalars(select(Alert).where(
                    Alert.category == ALERT_CATEGORY, Alert.instrument_id == row.instrument_id,
                    Alert.acknowledged_at.is_(None), Alert.message.like(f"% {row.ex_date}:%"))):
                alert.acknowledged_at = datetime.now(timezone.utc).replace(tzinfo=None)  # the warning no longer applies
    session.flush()
    return {"candidates": len(candidates), "new": new, "stale": stale}


def compute_factor(action_type: str, ratio: Decimal | None, cash: Decimal | None, factor: Decimal | None,
                   prev_close: Decimal | None) -> Decimal:
    """Back-adjustment factor for bars BEFORE the ex-date. Explicit ``factor`` wins; stock-type actions use
    1/(1+ratio) (ratio = new shares per old share); a cash dividend uses (P - D) / P with P the last close
    before the ex-date; rights and others must give an explicit factor."""
    if factor is not None:
        f = factor
    elif action_type in ("stock_dividend", "bonus", "split") and ratio is not None:
        if ratio <= 0:
            raise AdjustmentError("ratio must be positive")
        f = Decimal(1) / (Decimal(1) + ratio)
    elif action_type == "cash_dividend" and cash is not None:
        if prev_close is None or prev_close <= cash:
            raise AdjustmentError("cash_dividend needs a previous close greater than the dividend")
        f = (prev_close - cash) / prev_close
    else:
        raise AdjustmentError(f"cannot compute a factor for {action_type!r} without an explicit factor (or ratio / cash_per_share)")
    if f <= 0 or f == 1:
        raise AdjustmentError(f"factor {f} is not a valid adjustment (must be > 0 and != 1)")
    return f.quantize(Decimal("0.000000000001"))


def _rebuild_factor_version(session: Session, instrument_id: int, source: str) -> int | None:
    """New version = every confirmed action that has a factor. Nothing is written if it equals the current version."""
    actions = list(session.scalars(select(CorporateAction).where(
        CorporateAction.instrument_id == instrument_id, CorporateAction.status == "confirmed").order_by(CorporateAction.ex_date)))
    want = {(a.ex_date, Decimal(a.details["factor"])): a.id for a in actions if a.details and "factor" in a.details}
    cur_version = session.scalar(select(AdjustmentFactor.version).where(AdjustmentFactor.instrument_id == instrument_id)
                                 .order_by(AdjustmentFactor.version.desc()).limit(1))
    have = set()
    if cur_version is not None:
        have = {(f.effective_date, Decimal(f.factor)) for f in session.scalars(select(AdjustmentFactor).where(
            AdjustmentFactor.instrument_id == instrument_id, AdjustmentFactor.version == cur_version))}
    if set(want) == have:
        return None
    version = (cur_version or 0) + 1
    for (d, f), aid in want.items():
        session.add(AdjustmentFactor(instrument_id=instrument_id, version=version, effective_date=d, factor=f,
                                     corporate_action_id=aid, source=source))
    session.flush()
    return version


def apply_confirmed(session: Session, path: str | Path) -> dict[str, list]:
    """Load confirmed corporate actions (CSV: symbol, ex_date, action_type, ratio, cash_per_share, factor, note)
    and create a new adjustment-factor version per affected instrument. Idempotent."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(line for line in fh if line.strip() and not line.lstrip().startswith("#")))
    if not rows:
        raise AdjustmentError(f"{path}: no data rows")
    touched: set[int] = set()
    for n, r in enumerate(rows, start=2):
        get = lambda k: (r.get(k) or "").strip() or None
        sym = (get("symbol") or "").upper()
        row = find_symbol_row(session, sym)
        if row is None:
            raise AdjustmentError(f"{path}:{n}: unknown symbol {sym!r}")
        try:
            ex = date.fromisoformat(get("ex_date"))
            ratio, cash, factor = (Decimal(get(k)) if get(k) else None for k in ("ratio", "cash_per_share", "factor"))
        except (TypeError, ValueError, InvalidOperation) as exc:
            raise AdjustmentError(f"{path}:{n}: bad ex_date/ratio/cash_per_share/factor ({exc})") from exc
        atype = get("action_type")
        if atype not in ACTION_TYPES:
            raise AdjustmentError(f"{path}:{n}: action_type must be one of {ACTION_TYPES}")
        iid = row.instrument_id
        prev = session.scalar(select(PriceBar.close).where(PriceBar.instrument_id == iid, PriceBar.trade_date < ex)
                              .order_by(PriceBar.trade_date.desc()).limit(1))
        try:
            f = compute_factor(atype, ratio, cash, factor, Decimal(prev) if prev is not None else None)
        except AdjustmentError as exc:
            raise AdjustmentError(f"{path}:{n}: {exc}") from exc
        details = {"factor": str(f), "note": get("note")}
        ca = session.scalars(select(CorporateAction).where(
            CorporateAction.instrument_id == iid, CorporateAction.ex_date == ex, CorporateAction.action_type == atype)).first()
        if ca is None:
            # a confirmed event replaces the gap candidate it explains
            cand = session.scalars(select(CorporateAction).where(
                CorporateAction.instrument_id == iid, CorporateAction.ex_date == ex, CorporateAction.action_type == "unknown_gap")).first()
            if cand is not None:
                cand.status = "rejected"
                details["explains_candidate"] = True
            ca = CorporateAction(instrument_id=iid, action_type=atype, ex_date=ex, ratio=ratio, cash_per_share=cash,
                                 details=details, status="confirmed", source=path.name)
            session.add(ca)
        else:
            ca.ratio, ca.cash_per_share, ca.details, ca.status, ca.source = ratio, cash, {**(ca.details or {}), **details}, "confirmed", path.name
        session.flush()
        touched.add(iid)
    versions = {}
    for iid in touched:
        v = _rebuild_factor_version(session, iid, path.name)
        if v is not None:
            versions[iid] = v
    return {"actions": len(rows), "new_versions": sorted(versions.items())}


def set_price_basis(session: Session, symbol: str, basis: str) -> int:
    """Mark every stored bar of an instrument 'raw' or 'vendor_adjusted' (daily and intraday). Returns rows changed."""
    if basis not in ("raw", "vendor_adjusted"):
        raise AdjustmentError("basis must be 'raw' or 'vendor_adjusted'")
    row = find_symbol_row(session, symbol.upper())
    if row is None:
        raise AdjustmentError(f"unknown symbol {symbol!r}")
    n = 0
    for model in (PriceBar, PriceBarIntraday):
        n += session.execute(update(model).where(model.instrument_id == row.instrument_id, model.price_basis != basis).values(price_basis=basis)).rowcount
    return n
