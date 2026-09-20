"""The data job behind `make data`: backfill -> incremental ingest (+ calendar) -> corporate-action
candidates -> quality checks -> persisted findings -> Markdown report.

Every step is idempotent, so running it twice in a row writes nothing the second time (apart from the
job records themselves). A failing step is recorded and the later steps still run, so the report always
reflects the state of the data; `ok` is False if anything failed or an error-level finding is unexplained.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

from sqlalchemy import Engine

from predict_stock import universe as uni
from predict_stock.config import PROJECT_ROOT, AppConfig
from predict_stock.data.adjustments import detect_gap_candidates, record_candidates
from predict_stock.data.backfill import backfill, sync_readiness_alerts
from predict_stock.data.dnse_client import DnseClient, DnseError
from predict_stock.data.ingest import ingest_universe
from predict_stock.data.quality import run_quality_checks, summarize
from predict_stock.data.quality_report import build_report, write_report
from predict_stock.data.quality_store import load_known_issues, store_findings, unexplained_errors
from predict_stock.db.models import Instrument
from predict_stock.db.repo import find_symbol_row
from predict_stock.db.session import session_scope
from predict_stock.runs import tracked_run

log = logging.getLogger(__name__)


def quality_targets(session, cfg: AppConfig, codes: list[str], start: date, end: date) -> list[tuple[int, str]]:
    """(instrument_id, symbol) of every instrument in the universes during [start, end] plus the benchmarks."""
    found = {m.instrument_id: m.symbol for c in codes for m in uni.get_instruments_between(session, c, start, end)}
    for sym in cfg.ingest.benchmark_symbols:
        row = find_symbol_row(session, sym)
        if row is not None:
            found.setdefault(row.instrument_id, sym)
    return sorted(found.items(), key=lambda kv: kv[1])


def run_quality(engine: Engine, cfg: AppConfig, codes: list[str], start: date, end: date, *, as_of: date | None = None,
                known_issues_path: str | Path | None = None, report_path: str | Path | None = None, write: bool = True) -> dict:
    """Checks -> gap candidates -> persisted findings -> report. Returns a summary including ``unexplained_errors``."""
    as_of = as_of or end
    kpath = Path(known_issues_path or PROJECT_ROOT / cfg.quality.known_issues_path)
    params = {"universes": codes, "start": start.isoformat(), "end": end.isoformat(), "as_of": as_of.isoformat()}
    with tracked_run(engine, "data_quality", cfg, params) as (run_id, stats):
        with session_scope(engine) as s:
            known = load_known_issues(s, kpath) if kpath.exists() else 0
            pairs = quality_targets(s, cfg, codes, start, end)
        with session_scope(engine) as s:
            stock_pairs = [p for p in pairs if s.get(Instrument, p[0]).kind == "stock"]
            cands = detect_gap_candidates(s, cfg, stock_pairs)
            adj = record_candidates(s, cands, stock_pairs, run_id)
            issues = run_quality_checks(s, pairs, cfg)
            counts = store_findings(s, issues, [i for i, _ in pairs], run_id)
            members = uni.training_members(s, cfg, as_of)
            blocked = sync_readiness_alerts(s, cfg, members, as_of, run_id)
            unexplained = unexplained_errors(s, [i for i, _ in pairs])
            label = dict(pairs)
            unexplained_view = [f"{label[f.instrument_id]} {f.trade_date or ''} {f.check_name}: {f.detail}".strip() for f in unexplained]
            content = build_report(s, cfg, pairs, as_of, codes, blocked)
        path = write_report(report_path or PROJECT_ROOT / cfg.quality.report_path, content) if write else None
        summary = summarize(issues)
        stats.update(known_issues=known, findings=counts, adjustments=adj, blocked=blocked,
                     unexplained_errors=len(unexplained), report=str(path) if path else None)
    return {"findings": counts, "unexplained_errors": unexplained_view, "blocked": blocked, "adjustments": adj,
            "known_issues": known, "summary": summary, "report": path, "run_id": run_id}


def run_data_pipeline(engine: Engine, client: DnseClient, cfg: AppConfig, codes: list[str], start: date, end: date, *,
                      now: datetime | None = None, write_report_file: bool = True, intraday: bool | None = None,
                      known_issues_path: str | Path | None = None, report_path: str | Path | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    result: dict = {"errors": []}
    for name, step in (
        ("backfill", lambda: backfill(engine, client, cfg, codes, start, end, now=now)),
        ("ingest", lambda: ingest_universe(engine, client, cfg, codes, start, end, now=now, intraday=intraday)),
    ):
        try:
            result[name] = step()
        except DnseError as exc:
            log.error("%s failed: %s", name, exc)
            result["errors"].append(f"{name}: {exc}")
    result["quality"] = run_quality(engine, cfg, codes, start, end, known_issues_path=known_issues_path,
                                    report_path=report_path, write=write_report_file)
    result["ok"] = not result["errors"] and not result["quality"]["unexplained_errors"]
    return result
