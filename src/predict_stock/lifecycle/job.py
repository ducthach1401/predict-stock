"""The lifecycle steps of the daily paper job and the monthly evaluation.

daily   : `shadow` (predictions of the champions and of every shadow challenger; nothing else) and `monitor` (drift, rolling IC, calibration, fill rate, holding-time deviation; alerts).
monthly : on the first session of each month (`evaluate_day_of_month`): which retrains are due and why, the comparison of every shadow challenger with its champion (no promotion: that is a
          manual command), what the closed recommendations say about the champions. It only reports and alerts; with `lifecycle.auto_retrain: true` it may also START an allowed retrain
          (which still ends as a shadow model, never as a champion)."""
from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine, select

from predict_stock.backtest.runner import Setup, clean
from predict_stock.config import AppConfig
from predict_stock.db.models import Experiment
from predict_stock.db.session import session_scope
from predict_stock.lifecycle import compare as CMP
from predict_stock.lifecycle import monitor as MON
from predict_stock.lifecycle import outcomes as OUT
from predict_stock.lifecycle import registry as REG
from predict_stock.lifecycle import retrain as RT
from predict_stock.lifecycle import shadow as SH
from predict_stock.paper.alerts import raise_alert
from predict_stock.swing.registry import log_experiment


def step_shadow(engine: Engine, cfg: AppConfig, setup: Setup, d: pd.Timestamp, run_id: int) -> dict:
    return SH.shadow_step(engine, cfg, setup, d, run_id)


def step_monitor(engine: Engine, cfg: AppConfig, setup: Setup, d: pd.Timestamp, run_id: int) -> dict:
    from predict_stock.paper.daily import paper_start
    res = MON.run_monitor(engine, cfg, setup, d, perf_start=paper_start(engine, cfg))
    return {s: {"model": r.get("model"), "breaches": [b["message"] for b in r.get("breaches", [])], "triggers": r.get("triggers", []), "skipped": r.get("skipped")} for s, r in res.items()}


def evaluation_due(engine: Engine, cfg: AppConfig, d: pd.Timestamp) -> bool:
    """True on the first session on or after ``evaluate_day_of_month`` of a month with no evaluation recorded yet."""
    if d.day < cfg.lifecycle.evaluate_day_of_month:
        return False
    with session_scope(engine) as s:
        return s.scalar(select(Experiment.id).where(Experiment.name == f"lifecycle:evaluate:{d:%Y-%m}").limit(1)) is None


def evaluate(engine: Engine, cfg: AppConfig, setup: Setup, d: pd.Timestamp, run_id: int | None = None, *, monitor: dict | None = None, write: bool = True) -> dict:
    """The monthly evaluation of every strategy."""
    from predict_stock.paper.daily import paper_start
    monitor = monitor if monitor is not None else MON.run_monitor(engine, cfg, setup, d, perf_start=paper_start(engine, cfg), alert=False)
    out: dict = {"as_of": str(d.date()), "strategies": {}}
    for strategy in REG.STRATEGIES:
        champ = REG.champion(engine, strategy)
        entry: dict = {"champion": None if champ is None else f"{champ.name} v{champ.version}"}
        if champ is None:
            out["strategies"][strategy] = {**entry, "skipped": "no champion (mark the serving model with `lifecycle status` / the registry first)"}
            continue
        entry["due"] = RT.due(engine, cfg, strategy, d, monitor)
        entry["shadow"] = []
        for ch in REG.with_status(engine, strategy, "shadow"):
            try:
                rep = CMP.compare(engine, cfg, setup, ch.id, d)
                entry["shadow"].append({"model_id": ch.id, "passed": rep["passed"], "verdict": rep["verdict"], "shadow_weeks": rep.get("shadow_weeks"), "realised_days": rep.get("realised_days")})
            except REG.LifecycleError as exc:
                entry["shadow"].append({"model_id": ch.id, "error": str(exc)})
        start = paper_start(engine, cfg)
        entry["outcomes"] = OUT.assess(OUT.paper_table(engine, strategy, start), cfg)
        if entry["due"] and not entry["shadow"] and write:
            kinds = ", ".join(sorted({x["kind"] for x in entry["due"]}))
            raise_alert(engine, "warn", "retrain_due", f"{strategy}: a retrain is due ({kinds}): " + "; ".join(x["message"] for x in entry["due"][:3]) +
                        f". Start it with `retrain --strategy {strategy} --trigger {'drift' if 'drift' in kinds else 'schedule'}`"
                        + (" (training on data after the protected held-out period needs --consume-holdout: see docs/model_lifecycle.md)" if not CMP.holdout_consumed(engine) and d >= pd.Timestamp(cfg.lifecycle.protected_period[0]) else ""), details={"strategy": strategy, "reasons": entry["due"]})
            if cfg.lifecycle.auto_retrain:
                try:
                    entry["retrain"] = RT.retrain(engine, cfg, strategy, "drift" if any(x["kind"] == "drift" for x in entry["due"]) else "schedule", as_of=d, reasons=[x["message"] for x in entry["due"]],
                                                  actor="auto", setup=setup, run_id=run_id)
                except RT.RetrainRefused as exc:
                    entry["retrain"] = {"refused": str(exc)}
        for c in entry["shadow"]:
            if c.get("passed") and write:
                raise_alert(engine, "info", "challenger_ready", f"{strategy}: challenger #{c['model_id']} meets the promotion rule; confirm with `promote --model-id {c['model_id']}`", details=c)
        out["strategies"][strategy] = entry
    out = clean(out)
    if write:
        log_experiment(engine, cfg, f"lifecycle:evaluate:{d:%Y-%m}", key=f"{d:%Y-%m}", params={"as_of": str(d.date())}, summary=out, status="success", description="monthly model evaluation", run_id=run_id)
    return out
