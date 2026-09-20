"""The daily job (after the close): universe sync -> ingest -> quality -> features -> scoring -> cards for the next session -> replay of the paper portfolio ->
orders / positions / snapshots / outcomes -> report. Every step is its own `job_runs` row (status, duration), a failure raises an alert and never a half-written state
(the state is REPLAYED from the recommendations, so any run can be repeated safely). No step places a real order: there is no such code."""
from __future__ import annotations

import hashlib
import time
import traceback
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, func, select

from predict_stock.backtest.runner import Setup, load_setup
from predict_stock.config import PROJECT_ROOT, AppConfig
from predict_stock.data.backfill import backfill, sync_readiness_alerts
from predict_stock.data.dnse_client import DnseClient, DnseError
from predict_stock.db.models import Experiment, PriceBar, Recommendation
from predict_stock.db.session import session_scope
from predict_stock.paper import check_mode
from predict_stock.paper import live as LV
from predict_stock.paper import replay as RP
from predict_stock.paper import state as ST
from predict_stock.paper import sync as SY
from predict_stock.paper.alerts import raise_alert
from predict_stock.pipeline import run_data_pipeline
from predict_stock.reco.cards import Card
from predict_stock.reco.job import save_live_cards
from predict_stock.runs import tracked_run
from predict_stock.universe import get_member_records
from predict_stock.universe_sync import SnapshotError, apply_snapshot, read_snapshot_csv


class StepFailed(RuntimeError):
    pass


def paper_start(engine: Engine, cfg: AppConfig) -> pd.Timestamp | None:
    """First session of the paper record: the config value, else the one stored by `paper init`."""
    if cfg.paper.start_date:
        return pd.Timestamp(cfg.paper.start_date)
    with session_scope(engine) as s:
        row = s.scalars(select(Experiment).where(Experiment.name == "paper:start").order_by(Experiment.id.desc())).first()
        return pd.Timestamp(row.params["start_date"]) if row is not None else None


def init_paper(engine: Engine, cfg: AppConfig, start: date) -> None:
    """Fix the start of the paper record (idempotent; a different date is refused: the record's start never moves)."""
    cur = paper_start(engine, cfg)
    if cur is not None:
        if cur.date() != start:
            raise ValueError(f"the paper record already starts on {cur.date()}; it cannot be moved")
        return
    from predict_stock.swing.registry import log_experiment
    log_experiment(engine, cfg, "paper:start", key=f"start:{start}", params={"start_date": start.isoformat()}, summary={"note": "first session of the paper record"}, status="success",
                   description="Start of the paper trading record")


def run_step(engine: Engine, cfg: AppConfig, name: str, fn, results: dict, params: dict | None = None, *, critical: bool = False):
    """One job_runs row per step; an exception becomes an alert and a failed step (the next steps decide whether they can go on)."""
    t0 = time.time()
    out = {"ok": False, "seconds": 0.0}
    try:
        with tracked_run(engine, f"paper_{name}", cfg, params) as (run_id, stats):
            val = fn(run_id)
            if isinstance(val, dict):
                stats.update({k: v for k, v in val.items() if isinstance(v, (int, float, str, list, dict, bool)) or v is None})
            out.update(ok=True, result=val, run_id=run_id)
    except BaseException as exc:                                          # noqa: BLE001 - a failed step is reported, never silent
        if isinstance(exc, KeyboardInterrupt):
            raise
        out.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-1500:])
        raise_alert(engine, "critical" if critical else "error", "job_failed", f"paper step '{name}' failed: {out['error']}", details={"traceback": out["traceback"]})
    out["seconds"] = round(time.time() - t0, 2)
    results[name] = out
    return out


# ---- steps ---------------------------------------------------------------------------------------------------------------------------------------------
def step_universe(engine: Engine, cfg: AppConfig, client: DnseClient, d: date, run_id: int) -> dict:
    """Apply the user's universe snapshot file (if configured). Nothing changed -> nothing happens. New member: backfilled, and only scored once it has enough history.
    Removed member: NOT sold in a hurry; its open recommendations are flagged (see `flag_removed`)."""
    path = cfg.paper.universe_snapshot
    if not path or not (PROJECT_ROOT / path).exists():
        return {"skipped": "no universe snapshot file configured / found"}
    p = PROJECT_ROOT / path
    source = f"{p.name}@sha256:{hashlib.sha256(p.read_bytes()).hexdigest()[:12]}"
    code = cfg.universe.training_code
    try:
        rows = read_snapshot_csv(p)
        with session_scope(engine) as s:
            items = apply_snapshot(s, code, rows, d, source=source, floor=cfg.universe.floor(), run_id=run_id)
    except SnapshotError as exc:
        raise StepFailed(f"universe snapshot rejected: {exc}") from exc
    added = [i.symbol for i in items if i.action == "add"]
    removed = [i.symbol for i in items if i.action == "remove"]
    for sym in added:
        raise_alert(engine, "info", "universe_change", f"{sym} joined {code} on {d}: it is backfilled and scored once it has {cfg.ingest.min_sessions} sessions")
    for sym in removed:
        raise_alert(engine, "warn", "universe_change", f"{sym} left {code} on {d}: no urgent sale; open recommendations are flagged 'ra khỏi rổ' and handled by the configured policy")
    if added and cfg.ingest.auto_backfill:
        backfill(engine, client, cfg, [code], date.fromisoformat(cfg.ingest.history_start), d)
    return {"changes": [f"{i.action}:{i.symbol}" for i in items], "added": added, "removed": removed}


def step_ingest(engine: Engine, cfg: AppConfig, client: DnseClient, d: date) -> dict:
    codes = sorted({cfg.universe.training_code, cfg.universe.trading_code})
    try:
        res = run_data_pipeline(engine, client, cfg, codes, date.fromisoformat(cfg.ingest.history_start), d, write_report_file=False)
    except DnseError as exc:
        raise_alert(engine, "error", "api_error", f"DNSE request failed: {exc}")
        raise
    for e in res["errors"]:
        raise_alert(engine, "error", "api_error", f"data step error: {e}")
    q = res["quality"]
    if q["unexplained_errors"]:
        raise_alert(engine, "error", "data_quality", f"{len(q['unexplained_errors'])} unexplained data-quality error(s); first: {q['unexplained_errors'][0]}")
    return {"ok": res["ok"], "errors": res["errors"], "unexplained_errors": len(q["unexplained_errors"]), "findings": q["findings"]}


def data_checks(engine: Engine, cfg: AppConfig, d: pd.Timestamp) -> dict:
    """Do the members have a bar on the as-of date? Is the newest bar stale? Below the configured share no card is issued for that day."""
    members = ST.members_on(engine, cfg.universe.training_code, d.date())
    with session_scope(engine) as s:
        have = set(s.scalars(select(PriceBar.instrument_id).where(PriceBar.trade_date == d.date(), PriceBar.instrument_id.in_(list(members) or [0]))))
        newest = s.scalar(select(func.max(PriceBar.trade_date)).where(PriceBar.instrument_id.in_(list(members) or [0])))
    missing = sorted(members[i] for i in members if i not in have)
    share = len(have) / len(members) if members else 0.0
    if missing:
        raise_alert(engine, "warn" if share >= cfg.paper.min_share_with_bar else "error", "data_missing", f"{len(missing)} of {len(members)} members have no bar on {d.date()}: {', '.join(missing[:12])}")
    stale = newest is not None and (d.date() - newest).days > cfg.paper.stale_days
    if stale:
        raise_alert(engine, "error", "stale_data", f"the newest bar is from {newest}, more than {cfg.paper.stale_days} days before {d.date()}")
    return {"members": len(members), "with_bar": len(have), "missing": missing, "share": share, "newest_bar": str(newest), "ok": share >= cfg.paper.min_share_with_bar and not stale}


def flag_removed(engine: Engine, cfg: AppConfig, d: pd.Timestamp, run_id: int | None) -> dict:
    """Open recommendations of stocks that are no longer members: flagged 'ra khỏi rổ' (never an urgent sale unless the policy is close_now).
    close_now writes an 'exit' row in the sleeve's target book, which the replay turns into a sale at the next open."""
    members = ST.members_on(engine, cfg.universe.training_code, d.date())
    flagged, closing = [], []
    pol = cfg.paper.on_removal
    start = paper_start(engine, cfg)
    for rec in ST.open_recommendations(engine, start.date() if start is not None else None):
        if rec.instrument_id in members:
            ST.clear_flag(engine, rec.id, "left_universe")
            continue
        new = ST.set_flag(engine, rec.id, "left_universe", {"since": str(d.date()), "policy": pol.swing if rec.strategy == "SWING" else pol.invest, "note": "ra khỏi rổ"})
        flagged.append(f"{rec.strategy}:{rec.instrument_id}")
        if new:
            raise_alert(engine, "warn", "left_universe", f"recommendation {rec.strategy} #{rec.id} (instrument {rec.instrument_id}) is open but the stock left the universe; policy: "
                        f"{pol.swing if rec.strategy == 'SWING' else pol.invest}", instrument_id=rec.instrument_id, run_id=run_id)
    if pol.invest == "close_now":
        for sleeve in ("invest_b1", "invest_b2"):
            last = ST.latest_target(engine, sleeve, upto=d.date())
            if last is None:
                continue
            cur = ST._key(last.current)
            keep = {i: w for i, w in cur.items() if i in members}
            if len(keep) != len(cur):
                ST.save_target(engine, sleeve, d.date(), "exit", 1, 1, keep, None, None, run_id)
                closing.append(sleeve)
    return {"flagged": flagged, "close_now_sleeves": closing}


def step_cards(engine: Engine, cfg: AppConfig, setup: Setup, d: pd.Timestamp, checks: dict, run_id: int) -> dict:
    """Cards for the next session. SWING every day; INVEST only when a rebalance / tranche is due."""
    if not checks["ok"]:
        raise_alert(engine, "error", "cards_skipped", f"no cards for {d.date()}: data checks failed ({checks['with_bar']}/{checks['members']} members with a bar)")
        return {"skipped": "data checks failed", "cards": 0}
    dc = LV.build_day(engine, cfg, setup, d)
    if dc.d != d:
        return {"skipped": f"{d.date()} is not a trading session", "cards": 0}
    invest, steps = [], []
    for key in cfg.invest.presets:
        cards, info = LV.invest_cards(dc, key, run_id)
        invest += cards
        if info:
            steps.append(info)
    holdings = {}
    for key in cfg.invest.presets:
        t = ST.latest_target(engine, f"invest_{key}", upto=d.date())
        holdings[f"invest_{key}"] = ST._key(t.current) if t else {}
    swing = LV.swing_cards(dc, holdings, LV.get_evidence(dc))
    cards: list[Card] = swing + invest
    from predict_stock.reco.job import save_swing_predictions
    save_swing_predictions(engine, cfg, dc, run_id)
    keep = [c for c in cards if c.action in ("BUY", "WATCH")]
    n = save_live_cards(engine, cfg, keep, None, 0, dc.uid, run_id)
    return {"cards": len(cards), "buy": sum(c.action == "BUY" for c in cards), "watch": sum(c.action == "WATCH" for c in cards), "no_trade": sum(c.action == "NO_TRADE" for c in cards),
            "new_rows": n, "invest_steps": steps, "rejected": [f"{c.card_id}: {c.rejected}" for c in cards if c.action == "NO_TRADE"][:20]}


def step_state(engine: Engine, cfg: AppConfig, setup: Setup, d: pd.Timestamp, run_id: int) -> dict:
    start = paper_start(engine, cfg)
    if start is None:
        raise_alert(engine, "error", "paper_not_initialised", "the paper record has no start date: run `python -m predict_stock paper init`")
        return {"skipped": "not initialised"}
    if start > d:
        return {"skipped": f"the paper record starts on {start.date()}"}
    dc_symbols = S_symbols(engine, setup)
    rp = RP.replay(engine, cfg, setup, d, start)
    out = SY.sync_all(engine, cfg, setup, rp, dc_symbols, run_id)
    ke = rp.res.stats["kill_events"]
    if ke and pd.Timestamp(ke[-1]) >= d - pd.Timedelta(days=5):
        raise_alert(engine, "critical", "kill_switch", f"kill-switch fired on {ke[-1]}: all positions are sold at the next open and no new buys are made for {cfg.reco.portfolio.kill_cooldown} sessions")
    eq = rp.res.equity
    out.update({"equity": float(eq.iloc[-1]), "drawdown": float(eq.iloc[-1] / eq.max() - 1), "kill_events": ke})
    return out


def S_symbols(engine: Engine, setup: Setup):
    from predict_stock.reco import sources as S
    return S.Symbols(engine, list(setup.data.close.columns))


# ---- the whole day ---------------------------------------------------------------------------------------------------------------------------------------
def run_daily(engine: Engine, cfg: AppConfig, client: DnseClient, as_of: date | None = None, *, mode: str | None = None, ingest: bool = True, report: bool = True) -> dict:
    """The daily job for ``as_of`` (default: the latest session with data). Safe to repeat: every step is idempotent."""
    mode = check_mode(mode or cfg.run.mode)
    if mode != "paper":
        raise ValueError("`paper run` only runs in mode 'paper' (the 'backtest' mode is `backtest run`, `invest run`, `swing run`, `reco backtest`)")
    d = as_of or date.today()
    results: dict = {"as_of": str(d), "mode": mode}
    t0 = time.time()
    run_step(engine, cfg, "universe", lambda rid: step_universe(engine, cfg, client, d, rid), results, {"as_of": str(d)})
    if ingest:
        run_step(engine, cfg, "ingest", lambda rid: step_ingest(engine, cfg, client, d), results, {"as_of": str(d)})
    run_step(engine, cfg, "calendar", lambda rid: _calendar(engine, cfg), results)
    with session_scope(engine) as s:
        from predict_stock.db.models import TradingCalendar
        last = s.scalar(select(func.max(TradingCalendar.trade_date)).where(TradingCalendar.calendar_code == cfg.ingest.calendar_code, TradingCalendar.trade_date <= d))
    if last is None:
        raise_alert(engine, "critical", "no_calendar", "the trading calendar is empty")
        results["ok"] = False
        return results
    dts = pd.Timestamp(last)
    results["session"] = str(dts.date())
    checks = data_checks(engine, cfg, dts)
    results["checks"] = checks
    setup_holder: dict = {}

    def feat(rid):
        setup_holder["setup"] = load_setup(engine, cfg)
        with session_scope(engine) as s:
            sync_readiness_alerts(s, cfg, [m for m in get_member_records(s, cfg.universe.training_code, dts.date())], dts.date(), rid)
        return {"datasets": setup_holder["setup"].dataset_hashes, "instruments": setup_holder["setup"].panel_info["instruments"]}
    run_step(engine, cfg, "features", feat, results, {"as_of": str(dts.date())}, critical=True)
    setup = setup_holder.get("setup")
    if setup is not None:
        run_step(engine, cfg, "flags", lambda rid: flag_removed(engine, cfg, dts, rid), results)
        if cfg.paper.stop_on_quality_error and results.get("ingest", {}).get("result", {}).get("unexplained_errors"):
            raise_alert(engine, "error", "cards_skipped", f"no new cards for {dts.date()}: unexplained data-quality errors")
            results["cards"] = {"ok": True, "result": {"skipped": "quality errors", "cards": 0}}
        else:
            run_step(engine, cfg, "cards", lambda rid: step_cards(engine, cfg, setup, dts, checks, rid), results, {"session": str(dts.date())})
        run_step(engine, cfg, "state", lambda rid: step_state(engine, cfg, setup, dts, rid), results, {"session": str(dts.date())})
        if report:
            from predict_stock.paper.report import run_report
            run_step(engine, cfg, "report", lambda rid: run_report(engine, cfg, setup, dts, results), results, {"session": str(dts.date())})
    results["seconds"] = round(time.time() - t0, 1)
    results["ok"] = all(v.get("ok") for v in results.values() if isinstance(v, dict) and "ok" in v and "seconds" in v)
    return results


def _calendar(engine: Engine, cfg: AppConfig) -> dict:
    from predict_stock.data.calendar import sync_trading_calendar
    with session_scope(engine) as s:
        return sync_trading_calendar(s, cfg)
