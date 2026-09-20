"""Command line entry point:  python -m predict_stock <group> <command>

  universe apply|members|list|check|build     instrument rename|exchange|status
  ingest                                      quality
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from predict_stock import universe as uni
from predict_stock.config import PROJECT_ROOT, AppConfig, load_config
from predict_stock.data import adjustments as adj
from predict_stock.data import universe_builder
from predict_stock.data.backfill import backfill
from predict_stock.data.calendar import calendar_gaps, sync_trading_calendar
from predict_stock.data.dnse_client import DnseClient
from predict_stock.data.ingest import ingest_universe
from predict_stock.data.quality_store import load_known_issues
from predict_stock.db.models import Universe
from predict_stock.db.repo import find_symbol_row
from predict_stock.db.session import make_engine, session_scope
from predict_stock.pipeline import run_data_pipeline, run_quality
from predict_stock.instruments import EXCHANGES, STATUSES, change_exchange, rename_symbol, set_status
from predict_stock.runs import tracked_run
from predict_stock.universe_sync import SnapshotError, apply_snapshot, read_snapshot_csv


def _d(s: str) -> date:
    return date.fromisoformat(s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="predict_stock")
    p.add_argument("--config", help="YAML config (default: config/default.yaml)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="group", required=True)

    # ---- universe
    u = sub.add_parser("universe", help="universe membership").add_subparsers(dest="cmd", required=True)
    a = u.add_parser("apply", help="apply a membership snapshot CSV as of an effective date (sync_universe)")
    a.add_argument("--universe", required=True, help="universe code, e.g. LARGE50")
    a.add_argument("--file", required=True)
    a.add_argument("--effective-date", required=True, type=_d)
    a.add_argument("--dry-run", action="store_true", help="show the changes, write nothing")
    a.add_argument("--create", action="store_true", help="create the universe if it does not exist")
    a.add_argument("--name", help="display name when creating")
    a.add_argument("--no-backfill", action="store_true", help="do not fetch history for newly added members")
    m = u.add_parser("members", help="members of a universe on a date")
    m.add_argument("--universe", required=True)
    m.add_argument("--as-of", type=_d, default=date.today())
    m.add_argument("--tradable", action="store_true", help="exclude suspended/delisted instruments")
    u.add_parser("list", help="universes and their current size")
    u.add_parser("check", help="integrity of symbol/status/membership history and training/trading consistency")
    b = u.add_parser("build", help="rank a candidate list by traded value and write a snapshot CSV")
    b.add_argument("--candidates", required=True)
    b.add_argument("--code", required=True, help="universe code used in the note, e.g. LARGE50")
    b.add_argument("--top", type=int, default=50)
    b.add_argument("--window", type=int, default=60, help="sessions for the median traded value")
    b.add_argument("--start", type=_d)
    b.add_argument("--end", type=_d, default=date.today())
    b.add_argument("--out", required=True)

    # ---- instrument
    i = sub.add_parser("instrument", help="instrument life-cycle events").add_subparsers(dest="cmd", required=True)
    r = i.add_parser("rename", help="ticker change (đổi mã)")
    r.add_argument("--old", required=True); r.add_argument("--new", required=True)
    r.add_argument("--effective-date", required=True, type=_d)
    x = i.add_parser("exchange", help="exchange move (chuyển sàn)")
    x.add_argument("--symbol", required=True); x.add_argument("--exchange", required=True, choices=EXCHANGES)
    x.add_argument("--effective-date", required=True, type=_d)
    s = i.add_parser("status", help="suspension / delisting (tạm dừng / hủy niêm yết)")
    s.add_argument("--symbol", required=True); s.add_argument("--status", required=True, choices=STATUSES)
    s.add_argument("--effective-date", required=True, type=_d); s.add_argument("--note")

    # ---- data jobs
    def window(sp, refetch=False):
        sp.add_argument("--universe", action="append", help="universe code (repeatable; default: training + trading from config)")
        sp.add_argument("--start", type=_d, help="default: ingest.history_start")
        sp.add_argument("--end", type=_d, default=date.today())
        if refetch:
            sp.add_argument("--refetch", action="store_true", help="ignore stored data, re-fetch full history")
        sp.add_argument("--intraday", action=argparse.BooleanOptionalAction, default=None, help="override ingest.intraday.enabled")

    window(sub.add_parser("ingest", help="incremental daily OHLCV (+ 1H if enabled) for the universes and benchmarks"), refetch=True)
    window(sub.add_parser("backfill", help="full history for instruments that have none; blocks those with too few sessions"))
    dp = sub.add_parser("data", help="the whole data job: backfill, ingest, calendar, gap candidates, quality, report (make data)")
    window(dp)
    dp.add_argument("--no-report", action="store_true")

    c = sub.add_parser("calendar", help="trading calendar").add_subparsers(dest="cmd", required=True)
    c.add_parser("sync", help="recompute trading_calendar from stock bars")
    c.add_parser("gaps", help="weekday gaps and partial days")

    qq = sub.add_parser("quality", help="data quality").add_subparsers(dest="cmd", required=True)
    qr = qq.add_parser("run", help="run the checks, persist findings, write the Markdown report")
    window(qr)
    qr.add_argument("--details", type=int, default=0, help="print up to N unexplained findings")
    qr.add_argument("--no-report", action="store_true")
    ki = qq.add_parser("known-issues", help="load explained/characterised issues from a CSV")
    ki.add_argument("--file")

    ad = sub.add_parser("adjustments", help="corporate actions and price adjustment").add_subparsers(dest="cmd", required=True)
    ad.add_parser("detect", help="find open gaps beyond the price band (candidates are reported, never applied)")
    ap = ad.add_parser("apply", help="load confirmed corporate actions and create a new adjustment-factor version")
    ap.add_argument("--file", required=True)
    sb = ad.add_parser("set-basis", help="mark an instrument's bars 'raw' or 'vendor_adjusted'")
    sb.add_argument("--symbol", required=True); sb.add_argument("--basis", required=True, choices=("raw", "vendor_adjusted"))
    return p


def _codes(args, cfg: AppConfig) -> list[str]:
    return args.universe or sorted({cfg.universe.training_code, cfg.universe.trading_code})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    engine = make_engine("app")

    if args.group == "universe":
        return _universe(args, cfg, engine)
    if args.group == "instrument":
        return _instrument(args, cfg, engine)
    if args.group == "adjustments":
        return _adjustments(args, cfg, engine)
    if args.group == "calendar":
        with session_scope(engine) as s:
            if args.cmd == "sync":
                print(sync_trading_calendar(s, cfg))
            else:
                g = calendar_gaps(s, cfg)
                print(f"{g['trading_days']} trading days {g['first']} → {g['last']}; weekday gaps: {len(g['holiday_like'])}; partial days: {len(g['partial'])}")
                for y, v in g["by_year"].items():
                    print(f"  {y}: {v['trading_days']} trading days, {v['weekday_gaps']} weekday gaps")
        return 0
    if args.group == "quality" and args.cmd == "known-issues":
        with session_scope(engine) as s:
            print(f"loaded {load_known_issues(s, args.file or PROJECT_ROOT / cfg.quality.known_issues_path)} known issues")
        return 0

    codes = args.universe or sorted({cfg.universe.training_code, cfg.universe.trading_code})
    start = args.start or _d(cfg.ingest.history_start)
    if args.group == "ingest":
        stats = ingest_universe(engine, DnseClient(cfg.dnse), cfg, codes, start, args.end, refetch=args.refetch, intraday=args.intraday)
        print(json.dumps({k: stats[k] for k in ("totals", "totals_intraday", "drifted", "trading_days", "client_warnings")}, indent=2))
    elif args.group == "backfill":
        stats = backfill(engine, DnseClient(cfg.dnse), cfg, codes, start, args.end)
        print(json.dumps({k: stats[k] for k in ("new_instruments", "blocked", "duration_ms")}, indent=2))
    else:  # data / quality run
        if args.group == "data":
            res = run_data_pipeline(engine, DnseClient(cfg.dnse), cfg, codes, start, args.end, write_report_file=not args.no_report, intraday=args.intraday)
            q = res["quality"]
            for name in ("backfill", "ingest"):
                if name in res:
                    t = res[name].get("totals") or {"new_instruments": res[name]["new_instruments"], "blocked": res[name]["blocked"]}
                    print(f"{name}: {t}")
            for e in res["errors"]:
                print("ERROR", e)
        else:
            q = run_quality(engine, cfg, codes, start, args.end, write=not args.no_report)
            res = {"ok": not q["unexplained_errors"]}
        print(f"findings: {q['findings']} · known issues loaded: {q['known_issues']} · candidates: {q['adjustments']} · blocked: {len(q['blocked'])}")
        print(f"unexplained errors: {len(q['unexplained_errors'])}" + (f" · report: {q['report']}" if q["report"] else ""))
        for line in q["unexplained_errors"][: getattr(args, "details", 0) or 10]:
            print("  UNEXPLAINED", line)
        return 0 if res["ok"] else 1
    return 0


def _adjustments(args, cfg: AppConfig, engine) -> int:
    from predict_stock.pipeline import quality_targets
    codes = sorted({cfg.universe.training_code, cfg.universe.trading_code})
    try:
        with session_scope(engine) as s:
            if args.cmd == "detect":
                pairs = [p for p in quality_targets(s, cfg, codes, _d(cfg.ingest.history_start), date.today()) if p[1] not in cfg.ingest.benchmark_symbols]
                cands = adj.detect_gap_candidates(s, cfg, pairs)
                print(adj.record_candidates(s, cands, pairs, None))
                for c in cands:
                    print(f"  {c.symbol} {c.trade_date}: open {c.gap:+.1%} vs previous close (band ±{c.band:.0%}), implied factor {c.implied_factor:.4f}")
            elif args.cmd == "apply":
                print(adj.apply_confirmed(s, args.file))
            else:
                print(f"{adj.set_price_basis(s, args.symbol, args.basis)} bars set to {args.basis}")
    except adj.AdjustmentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _universe(args, cfg: AppConfig, engine) -> int:
    if args.cmd == "apply":
        path = Path(args.file)
        source = f"{path.name}@sha256:{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"
        rows = read_snapshot_csv(path)
        common = dict(source=source, floor=cfg.universe.floor(), create=args.create, name=args.name)
        try:
            if args.dry_run:
                with Session(engine) as s:
                    items = apply_snapshot(s, args.universe, rows, args.effective_date, **common)
                    s.rollback()
            else:
                params = {"universe": args.universe, "file": source, "effective_date": args.effective_date.isoformat()}
                with tracked_run(engine, "sync_universe", cfg, params) as (run_id, stats):
                    with session_scope(engine) as s:
                        items = apply_snapshot(s, args.universe, rows, args.effective_date, run_id=run_id, **common)
                    stats["changes"] = dict(Counter(i.action for i in items))
        except SnapshotError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        for it in items:
            print(it)
        counts = dict(Counter(i.action for i in items))
        verb = "would change" if args.dry_run else "applied"
        print(f"{verb}: {counts}" if items else f"{args.universe}: already up to date as of {args.effective_date}")
        if not args.dry_run and counts.get("add") and cfg.ingest.auto_backfill and not args.no_backfill:
            print(f"backfilling {counts['add']} new member(s) ...")
            rep = backfill(engine, DnseClient(cfg.dnse), cfg, [args.universe], _d(cfg.ingest.history_start), date.today())
            print(f"  backfilled {rep['new_instruments']} in ms {rep['duration_ms']}; blocked: {rep['blocked'] or 'none'}")
    elif args.cmd == "members":
        with session_scope(engine) as s:
            members = uni.get_member_records(s, args.universe, args.as_of, tradable_only=args.tradable)
        print(f"{args.universe} on {args.as_of}: {len(members)} members")
        print(" ".join(f"{m.symbol}" + (f"({m.exchange})" if m.exchange else "") for m in members))
    elif args.cmd == "list":
        with session_scope(engine) as s:
            for u in s.scalars(select(Universe).order_by(Universe.code)):
                n = len(uni.get_members(s, u.code, date.today()))
                roles = [r for r, c in (("training", cfg.universe.training_code), ("trading", cfg.universe.trading_code)) if c == u.code]
                print(f"{u.code:<12} members today: {n:<4} {'/'.join(roles)}")
    elif args.cmd == "check":
        with session_scope(engine) as s:
            problems = uni.check_integrity(s)
            extra = uni.trading_outside_training(s, cfg, date.today())
        for p in problems:
            print("PROBLEM:", p)
        if extra:
            print("PROBLEM: tradable but not in the training universe:", " ".join(extra))
        print("universe data healthy" if not (problems or extra) else f"{len(problems) + bool(extra)} problem(s)")
        return 1 if problems or extra else 0
    elif args.cmd == "build":
        start = args.start or _d(cfg.ingest.history_start)
        symbols = universe_builder.read_candidates(args.candidates)
        selected, ranked, excluded = universe_builder.build(
            DnseClient(cfg.dnse), symbols, start=start, end=args.end, window=args.window, top_n=args.top)
        picked = {c.symbol for c in selected}
        print(f"{len(symbols)} candidates -> {len(ranked)} rankable, {len(excluded)} excluded, {len(selected)} selected")
        print(f"rank symbol  median traded value/session ({args.window}d, billion VND)  first bar")
        for k, c in enumerate(ranked, 1):
            print(f"{k:>4} {c.symbol:<6} {c.median_value_vnd / 10**9:>14,.1f}   {c.first_date}   {'*' if c.symbol in picked else ''}")
        for sym, why in excluded.items():
            print(f"excluded {sym}: {why}")
        note = (f"{args.code}: fixed membership; top {args.top} of {len(symbols)} candidates by median daily traded value "
                f"(close*volume, last {args.window} sessions to {args.end}); liquidity proxy, NOT market cap; candidate list unverified")
        universe_builder.write_snapshot_csv(args.out, selected, start, note)
        print(f"wrote {args.out}")
    return 0


def _instrument(args, cfg: AppConfig, engine) -> int:
    from predict_stock.instruments import InstrumentError
    try:
        params = {k: (v.isoformat() if isinstance(v, date) else v) for k, v in vars(args).items() if k not in ("group", "config", "verbose")}
        with tracked_run(engine, f"instrument_{args.cmd}", cfg, params):
            with session_scope(engine) as s:
                if args.cmd == "rename":
                    iid = rename_symbol(s, args.old, args.new, args.effective_date)
                    print(f"instrument {iid}: {args.old} -> {args.new} effective {args.effective_date}")
                else:
                    row = find_symbol_row(s, args.symbol, args.effective_date) or find_symbol_row(s, args.symbol)
                    if row is None:
                        raise InstrumentError(f"unknown symbol {args.symbol!r}")
                    if args.cmd == "exchange":
                        ev = change_exchange(s, row.instrument_id, args.exchange, args.effective_date)
                        print(f"{args.symbol}: {ev or 'no change'}")
                    else:
                        changed = set_status(s, row.instrument_id, args.status, args.effective_date, args.note)
                        print(f"{args.symbol}: status {args.status} {'recorded' if changed else '(no change)'}")
    except InstrumentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
