"""Command line entry point:  python -m predict_stock <command>"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date

import pandas as pd

from predict_stock.config import load_config
from predict_stock.data.dnse_client import DnseClient
from predict_stock.data.ingest import ingest_universe
from predict_stock.data.quality import run_quality_checks, summarize
from predict_stock.db.session import make_engine, session_scope
from predict_stock.universe import get_members, get_symbols_between, load_membership_csv


def _d(s: str) -> date:
    return date.fromisoformat(s)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="predict_stock")
    p.add_argument("--config", help="YAML config (default: config/default.yaml)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("load-membership", help="load universe membership CSV (idempotent)")
    m.add_argument("file")
    m.add_argument("--source", help="provenance label stored with the rows (default: file name)")

    mem = sub.add_parser("members", help="print members of a universe on a date")
    mem.add_argument("--universe", required=True)
    mem.add_argument("--as-of", type=_d, default=date.today())

    ing = sub.add_parser("ingest", help="ingest daily OHLCV for a universe (+ benchmarks)")
    ing.add_argument("--universe", required=True)
    ing.add_argument("--start", type=_d, help="default: ingest.history_start")
    ing.add_argument("--end", type=_d, default=date.today())
    ing.add_argument("--refetch", action="store_true", help="ignore stored data, re-fetch full history")

    q = sub.add_parser("quality", help="data-quality report for a universe (+ benchmarks)")
    q.add_argument("--universe", required=True)
    q.add_argument("--start", type=_d)
    q.add_argument("--end", type=_d, default=date.today())
    q.add_argument("--details", type=int, default=0, help="print up to N individual issues")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    engine = make_engine("app")

    if args.cmd == "load-membership":
        with session_scope(engine) as s:
            n = load_membership_csv(s, args.file, args.source)
        print(f"processed {n} membership rows from {args.file}")
    elif args.cmd == "members":
        with session_scope(engine) as s:
            members = get_members(s, args.universe, args.as_of)
        print(f"{args.universe} on {args.as_of}: {len(members)} members\n{' '.join(members)}")
    elif args.cmd == "ingest":
        start = args.start or _d(cfg.ingest.history_start)
        stats = ingest_universe(engine, DnseClient(cfg.dnse), cfg, args.universe, start, args.end, refetch=args.refetch)
        print(json.dumps({k: stats[k] for k in ("totals", "drifted", "trading_days", "client_warnings")}, indent=2))
    elif args.cmd == "quality":
        start = args.start or _d(cfg.ingest.history_start)
        with session_scope(engine) as s:
            symbols = get_symbols_between(s, args.universe, start, args.end)
            symbols += [b for b in cfg.ingest.benchmark_symbols if b not in symbols]
            issues = run_quality_checks(s, symbols, cfg)
        print(f"checked {len(symbols)} symbols: {', '.join(symbols)}")
        with pd.option_context("display.width", 200, "display.max_colwidth", 120):
            summary = summarize(issues)
            print("no issues found" if summary.empty else summary.to_string(index=False))
            if args.details and not issues.empty:
                print(issues.head(args.details).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
