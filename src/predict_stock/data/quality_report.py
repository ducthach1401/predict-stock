"""Markdown data-quality report, built from the database (data_quality_findings et al.).

The report is deterministic: same data + same findings = same text (only the as-of date appears),
so re-running it changes nothing and diffs show real changes.
"""
from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from predict_stock.config import AppConfig
from predict_stock.data.calendar import calendar_gaps
from predict_stock.db.models import (
    CorporateAction,
    DataQualityFinding,
    DataQualityKnownIssue,
    PriceBar,
    PriceBarRevision,
)
from predict_stock.data.quality import SEVERITY

MAX_LISTED = 40


def _table(header: list[str], rows: list[list]) -> str:
    if not rows:
        return "_none_\n"
    line = lambda cells: "| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in cells) + " |"
    return "\n".join([line(header), line(["---"] * len(header))] + [line(r) for r in rows]) + "\n"


def build_report(session: Session, cfg: AppConfig, pairs: list[tuple[int, str]], as_of: date,
                 universes: list[str], blocked: dict[str, str] | None = None) -> str:
    ids = [i for i, _ in pairs]
    label = dict(pairs)
    blocked = blocked or {}
    findings = list(session.scalars(
        select(DataQualityFinding).where(DataQualityFinding.instrument_id.in_(ids), DataQualityFinding.status != "resolved")
        .order_by(DataQualityFinding.check_name, DataQualityFinding.instrument_id, DataQualityFinding.trade_date)))
    known = {k.id: k for k in session.scalars(select(DataQualityKnownIssue))}
    open_errors = [f for f in findings if f.status == "open" and f.severity == "error"]
    n_bars = session.scalar(select(func.count()).select_from(PriceBar).where(PriceBar.instrument_id.in_(ids)))
    lo, hi = session.execute(select(func.min(PriceBar.trade_date), func.max(PriceBar.trade_date)).where(PriceBar.instrument_id.in_(ids))).one()

    out = [f"# Data quality report — as of {as_of}\n"]
    out.append(f"Universes: {', '.join(universes)} · instruments checked: {len(pairs)} · daily bars: {n_bars:,} · "
               f"range: {lo} → {hi}\n")

    verdict = "PASS" if not open_errors else "FAIL"
    out.append(f"## Verdict: **{verdict}**\n")
    out.append(f"* Unexplained errors: **{len(open_errors)}**" + (" — every error-level finding is either absent or covered by a known issue." if not open_errors else "."))
    out.append(f"* Open warnings: {sum(1 for f in findings if f.status == 'open' and f.severity == 'warn')} "
               f"· explained findings: {sum(1 for f in findings if f.status == 'explained')} "
               f"· instruments blocked for insufficient history: {len(blocked)}\n")

    by = Counter((f.check_name, f.severity, f.status) for f in findings)
    checks = sorted({(c, s) for c, s, _ in by}, key=lambda t: ({"error": 0, "warn": 1, "info": 2}[t[1]], t[0]))
    out.append("## Findings by check\n")
    out.append(_table(["check", "severity", "open", "explained"],
                      [[c, s, by.get((c, s, "open"), 0), by.get((c, s, "explained"), 0)] for c, s in checks]))

    out.append("## Unexplained findings\n")
    if open_errors:
        out.append("**Errors (block use of the data until explained or fixed):**\n")
        out.append(_table(["symbol", "date", "check", "detail"],
                          [[label[f.instrument_id], f.trade_date or "—", f.check_name, f.detail] for f in open_errors[:MAX_LISTED]]))
        if len(open_errors) > MAX_LISTED:
            out.append(f"… and {len(open_errors) - MAX_LISTED} more.\n")
    else:
        out.append("No unexplained errors.\n")
    warns = [f for f in findings if f.status == "open" and f.severity == "warn"]
    if warns:
        out.append("**Warnings (not explained; they do not block):**\n")
        rows = []
        for check in sorted({f.check_name for f in warns}):
            fs = [f for f in warns if f.check_name == check]
            examples = "; ".join(f"{label[f.instrument_id]} {f.trade_date or ''}".strip() for f in fs[:6])
            rows.append([check, len(fs), len({f.instrument_id for f in fs}), examples + (" …" if len(fs) > 6 else "")])
        out.append(_table(["check", "findings", "symbols", "examples"], rows))

    out.append("## Explained findings (known issues)\n")
    cover = Counter(f.known_issue_id for f in findings if f.status == "explained")
    out.append(_table(["key", "check", "findings", "treatment", "explanation"],
                      [[known[k].issue_key, known[k].check_name, n, known[k].treatment,
                        known[k].explanation + (f" *Evidence:* {known[k].evidence}" if known[k].evidence else "")]
                       for k, n in sorted(cover.items(), key=lambda kv: known[kv[0]].issue_key)]))

    out.append("## History coverage\n")
    cov = session.execute(
        select(PriceBar.instrument_id, func.count(), func.min(PriceBar.trade_date), func.max(PriceBar.trade_date))
        .where(PriceBar.instrument_id.in_(ids)).group_by(PriceBar.instrument_id)).all()
    sessions = sorted(c[1] for c in cov)
    if sessions:
        out.append(f"Sessions per instrument: min {sessions[0]}, median {sessions[len(sessions) // 2]}, max {sessions[-1]} "
                   f"(`ingest.min_sessions` = {cfg.ingest.min_sessions}).\n")
        out.append("Shortest histories:\n")
        out.append(_table(["symbol", "sessions", "first bar", "last bar"],
                          [[label[i], n, a, b] for i, n, a, b in sorted(cov, key=lambda c: (c[1], label[c[0]]))[:8]]))
    out.append("Blocked (insufficient history):\n")
    out.append(_table(["symbol", "reason"], [[s, r] for s, r in sorted(blocked.items())]))

    out.append("## Trading calendar\n")
    cal = calendar_gaps(session, cfg)
    out.append(f"{cal['trading_days']:,} observed trading days, {cal['first']} → {cal['last']} (stock consensus; DNSE's official "
               "`get_working_dates` needs an API key, so it is not used). Future holidays are unknown.\n")
    out.append(_table(["year", "trading days", "weekday gaps"], [[y, v["trading_days"], v["weekday_gaps"]] for y, v in cal["by_year"].items()]))
    if cal["partial"]:
        out.append("Weekdays outside the calendar on which some stocks do have bars (below the consensus): "
                   + ", ".join(f"{d} ({n})" for d, n in cal["partial"][:20]) + ("…" if len(cal["partial"]) > 20 else "") + "\n")

    out.append("## Price adjustment\n")
    cands = list(session.scalars(select(CorporateAction).where(CorporateAction.instrument_id.in_(ids))
                                 .order_by(CorporateAction.status, CorporateAction.instrument_id, CorporateAction.ex_date)))
    n_rev = session.scalar(select(func.count()).select_from(PriceBarRevision).where(PriceBarRevision.instrument_id.in_(ids)))
    basis = dict(session.execute(select(PriceBar.price_basis, func.count()).where(PriceBar.instrument_id.in_(ids)).group_by(PriceBar.price_basis)).all())
    out.append(f"Price basis of stored bars: {', '.join(f'{k} = {v:,}' for k, v in sorted(basis.items()))}. DNSE serves back-adjusted prices "
               "(inferred in Phase 0: no gap beyond the price band that an unadjusted stock dividend or split would create), so no "
               "factors are applied. Adjustment candidates are only ever *reported*: none is applied without a confirmed file.\n")
    by_status = Counter(c.status for c in cands)
    out.append(f"Corporate-action rows: {', '.join(f'{k} = {v}' for k, v in sorted(by_status.items())) or 'none'} · "
               f"superseded bar revisions kept: {n_rev:,}.\n")
    if cands:
        out.append(_table(["symbol", "ex/gap date", "type", "status", "details"],
                          [[label[c.instrument_id], c.ex_date, c.action_type, c.status,
                           ", ".join(f"{k}={v}" for k, v in sorted((c.details or {}).items()) if k != "detected")] for c in cands[:MAX_LISTED]]))

    return "\n".join(out).rstrip() + "\n"


def write_report(path: str | Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != content:  # untouched file when nothing changed
        path.write_text(content, encoding="utf-8")
    return path
