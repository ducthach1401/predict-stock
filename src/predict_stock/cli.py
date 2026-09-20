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
from predict_stock.backtest.walkforward import OOSAlreadyUsed
from predict_stock.features import registry as freg
from predict_stock.features.dataset import DatasetError, build_dataset, file_sha256, load_dataset
from predict_stock.features.sets import load_definitions, sync_definitions
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

    # ---- backtest
    bt = sub.add_parser("backtest", help="backtest engine and baselines").add_subparsers(dest="cmd", required=True)
    br = bt.add_parser("run", help="baselines on the development period: metrics before/after costs, sensitivity, sub-periods, report, DB")
    br.add_argument("--baseline", action="append", help="baseline key (repeatable; default: all)")
    br.add_argument("--noise-seeds", type=int, default=20, help="random portfolios for the noise floor (0 = skip)")
    br.add_argument("--no-report", action="store_true", help="do not write docs/BASELINES.md and the charts")
    bo = bt.add_parser("oos", help="evaluate the HELD-OUT final period. Allowed once; a second attempt is refused")
    bo.add_argument("--final", action="store_true", help="required: confirms this is the single final evaluation")
    bt.add_parser("list", help="the baselines")

    # ---- swing model
    sw = sub.add_parser("swing", help="SWING model: walk-forward, evaluation against the baselines").add_subparsers(dest="cmd", required=True)
    sr = sw.add_parser("run", help="pre-register, tune once, walk-forward, evaluate against the baselines, report, DB")
    sr.add_argument("--trials", type=int, help="Optuna trials (default: config swing.optuna.trials)")
    sr.add_argument("--noise-seeds", type=int, default=20, help="random portfolios for the noise floor")
    sr.add_argument("--no-report", action="store_true", help="do not write docs/SWING.md and the charts")
    so = sw.add_parser("oos", help="evaluate the HELD-OUT period for the final SWING model and all baselines. Only after a PASS of the pre-registered criteria; allowed once")
    so.add_argument("--final", action="store_true", help="required: confirms this is the single final evaluation")
    sw.add_parser("report", help="rewrite docs/SWING.md and the charts from the latest stored run (no training)")

    # ---- features / datasets
    ft = sub.add_parser("features", help="feature / label registry").add_subparsers(dest="cmd", required=True)
    ft.add_parser("list", help="registered plugins and the feature sets / label specs in config/feature_sets.yaml")
    ft.add_parser("sync", help="store feature sets and label specs in the database (idempotent; versions are immutable)")
    ds = sub.add_parser("dataset", help="feature+label datasets").add_subparsers(dest="cmd", required=True)
    bd = ds.add_parser("build", help="build (or reproduce) a Parquet dataset and register its manifest")
    bd.add_argument("--feature-set", required=True, help="name:version, e.g. swing:1")
    bd.add_argument("--label-spec", required=True, help="name:version, e.g. swing:1")
    bd.add_argument("--universe", help="default: universe.training_code")
    bd.add_argument("--start", type=_d, help="first decision date (default: ingest.history_start)")
    bd.add_argument("--end", type=_d, help="last decision date (default: today)")
    bd.add_argument("--data-cutoff", type=_d, help="latest data labels may use (default: latest trading day)")
    bd.add_argument("--name")
    ds.add_parser("list", help="registered datasets")
    vd = ds.add_parser("verify", help="check a dataset's file against its recorded sha256")
    vd.add_argument("--name", required=True); vd.add_argument("--version", type=int)

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
    if args.group in ("features", "dataset"):
        return _features_datasets(args, cfg, engine)
    if args.group == "backtest":
        return _backtest(args, cfg, engine)
    if args.group == "swing":
        return _swing(args, cfg, engine)
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


def _backtest(args, cfg: AppConfig, engine) -> int:
    from predict_stock.backtest.baselines import BASELINES
    from predict_stock.backtest.job import run_baselines_job, run_holdout_once
    if args.cmd == "list":
        for b in BASELINES.values():
            print(f"{b.key:18s} {b.title}: {b.description}")
        return 0
    if args.cmd == "oos":
        if not args.final:
            print("error: the held-out period can be evaluated ONCE; add --final to confirm this is the final evaluation", file=sys.stderr)
            return 2
        try:
            print(json.dumps(run_holdout_once(engine, cfg), indent=2))
        except OOSAlreadyUsed as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 3
        return 0
    out = run_baselines_job(engine, cfg, keys=args.baseline, noise_seeds=args.noise_seeds, write=not args.no_report)
    res = out["payload"]["results"]
    print(f"development period {out['payload']['meta']['window'][0]} → {out['payload']['meta']['window'][1]}; held-out {out['payload']['meta']['holdout'][0]} → {out['payload']['meta']['holdout'][1]} untouched")
    print(f"{'baseline':18s} {'CAGR net':>9s} {'Sharpe net':>10s} {'MDD':>7s} {'CAGR gross':>10s} {'Sharpe gross':>12s}")
    for k, r in res.items():
        n, g = r["net"], r["gross"]
        print(f"{k:18s} {n['cagr'] * 100:8.1f}% {n['sharpe']:10.2f} {n['max_drawdown'] * 100:6.1f}% {g['cagr'] * 100:9.1f}% {g['sharpe']:12.2f}")
    print(f"stored experiments: {out['experiments']}" + ("" if args.no_report else f"\nreport: {cfg.backtest.report_path}"))
    return 0


def _swing(args, cfg: AppConfig, engine) -> int:
    from predict_stock.swing.job import run_swing
    if args.cmd == "report":
        from predict_stock.swing.report import rewrite_latest
        print(rewrite_latest(cfg))
        return 0
    if args.cmd == "oos":
        from predict_stock.swing.job import HoldoutClosed, run_swing_oos
        if not args.final:
            print("error: the held-out period can be evaluated ONCE; add --final to confirm this is the final evaluation", file=sys.stderr)
            return 2
        try:
            print(json.dumps(run_swing_oos(engine, cfg), indent=2))
        except HoldoutClosed as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 4
        except OOSAlreadyUsed as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 3
        return 0
    out = run_swing(engine, cfg, trials=args.trials, noise_seeds=args.noise_seeds, write=not args.no_report)
    p = out["payload"]
    print(f"walk-forward {p['window'][0]} → {p['window'][1]}, {len(p['folds'])} folds; held-out {p['holdout'][0]} → {p['holdout'][1]} untouched")
    print(f"{'strategy':22s} {'CAGR net':>9s} {'Sharpe net':>10s} {'MDD':>7s} {'turnover':>8s} {'CAGR gross':>10s} {'Sharpe gross':>12s}")
    rows = {"SWING top-K": p["primary"], "SWING barrier": p["secondary"], **p["baselines"]}
    for k, r in rows.items():
        n, g = r["net"], r["gross"]
        print(f"{k:22s} {n['cagr'] * 100:8.1f}% {n['sharpe']:10.2f} {n['max_drawdown'] * 100:6.1f}% {n.get('turnover_annual') or 0:8.1f} {g['cagr'] * 100:9.1f}% {g['sharpe']:12.2f}")
    ic = p["ic"]["scores"]["swing_lgbm"]
    print(f"rank IC {ic['mean']:.4f} (t {ic['t_stat']:.2f}, hit {ic['hit_rate']:.2f}); calibration ECE raw {p['calibration']['variants']['raw']['ece']:.4f} → "
          f"{p['calibration']['variants'][p['calibration']['used']]['ece']:.4f}")
    d = p["decision"]
    print("decision:", "PASS - the held-out period may be opened" if d["passed"] else "FAIL - does not beat the baselines after costs; held-out period stays closed")
    for c in d["criteria"]:
        print(f"  [{'x' if c['ok'] else ' '}] {c['name']}: {c['value']} vs {c['threshold']} {c['detail']}")
    return 0


def _pair(text: str) -> tuple[str, int]:
    name, _, version = text.partition(":")
    return name, int(version or 1)


def _features_datasets(args, cfg: AppConfig, engine) -> int:
    from predict_stock.db.models import Dataset
    try:
        if args.group == "features":
            fsets, lspecs = load_definitions(PROJECT_ROOT / cfg.features.definitions_path)
            if args.cmd == "list":
                print("registered feature plugins (name, version, kind, group):")
                for row in freg.list_features():
                    print("  ", *row)
                print("registered label plugins:", ", ".join(f"{n} v{v}" for n, v in freg.list_labels()))
                for (n, v), spec in sorted(fsets.items()):
                    print(f"feature set {n}:{v}  {len(spec.columns())} columns  warm-up {max(f.warmup() for f in spec.instantiate())} sessions")
                for (n, v), spec in sorted(lspecs.items()):
                    print(f"label spec  {n}:{v}  horizon {spec.horizon()}  columns {len(spec.columns())}")
            else:
                with session_scope(engine) as s:
                    print(sync_definitions(s, fsets, lspecs))
            return 0
        if args.cmd == "build":
            res = build_dataset(engine, cfg, universe_code=args.universe, feature_set=_pair(args.feature_set), label_spec=_pair(args.label_spec),
                                start=args.start or _d(cfg.ingest.history_start), end=args.end or date.today(), data_cutoff=args.data_cutoff, name=args.name)
            m = res.manifest
            print(f"{'REUSED (identical)' if res.reused else 'BUILT'} {res.name} v{res.version}: {res.rows:,} rows, {m['instruments']} instruments, "
                  f"{m['first_date']} → {m['last_date']}, {len(m['feature_columns'])} features, {len(m['label_columns'])} label columns")
            print(f"  file {res.path}\n  sha256 {res.sha256}\n  content {res.content_hash}\n  look-ahead audit: {'passed' if (m.get('lookahead_audit') or {}).get('passed') else m.get('lookahead_audit')}")
        elif args.cmd == "list":
            with session_scope(engine) as s:
                for d in s.scalars(select(Dataset).order_by(Dataset.name, Dataset.version)):
                    print(f"{d.name} v{d.version}: {d.row_count:,} rows {d.start_date}→{d.end_date} {d.path} {d.sha256[:12]}")
        else:
            with session_scope(engine) as s:
                frame, manifest = load_dataset(s, args.name, args.version)   # raises if the sha256 does not match
            print(f"{args.name}: OK, {len(frame):,} rows, sha256 verified")
    except (DatasetError, freg.RegistryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
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
