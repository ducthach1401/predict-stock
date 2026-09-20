"""End-to-end demo of the paper bot on a simulated market: five consecutive daily runs without an error, and the universe changes two stocks on day 3 (a file is edited, no code):
the two new stocks are backfilled and scored, the two removed ones are flagged (not sold), and the next day's recommendations come out right. Running a day again changes nothing."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from conftest import FakeClient, apply
from predict_stock.backtest.runner import load_setup
from predict_stock.config import PROJECT_ROOT, load_config
from predict_stock.db.models import (
    Alert, Instrument, InstrumentSymbolHistory, JobRun, PaperOrder, PaperPosition, PortfolioSnapshot, Prediction, PriceBar, Recommendation, SleeveTarget,
)
from predict_stock.db.session import session_scope
from predict_stock.features.sets import load_definitions, sync_definitions
from predict_stock.invest import walkforward as IW
from predict_stock.paper import daily as D
from predict_stock.pipeline import run_data_pipeline
from predict_stock.reco.cards import Card
from predict_stock.swing.folds import with_ends
from predict_stock.swing.registry import register_artifact, register_model
from predict_stock.swing.walkforward import fit_final
from paper_env import market, write_snapshot

SYMBOLS = [f"S{k:02d}" for k in range(1, 15)]
MEMBERS0 = SYMBOLS[:12]


def rows(engine, stmt):
    with Session(engine) as s:
        return s.scalars(stmt).all()


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("paper_e2e")
    base = load_config()
    (tmp / "known.csv").write_text("key,check,symbol,date_from,date_to,treatment,explanation,evidence\n")
    write_snapshot(tmp / "snap.csv", MEMBERS0)
    inv = base.invest.model_copy(update={"artifacts_dir": str(tmp / "models"),
                                         "presets": {"b1": base.invest.presets["b1"].model_copy(update={"tranche_spacing": 2}), "b2": base.invest.presets["b2"].model_copy(update={"tranche_spacing": 3})}})
    cfg = base.model_copy(update={
        "ingest": base.ingest.model_copy(update={"benchmark_symbols": [], "min_sessions": 60, "history_start": "2022-01-03"}),
        "universe": base.universe.model_copy(update={"training_code": "U1", "trading_code": "U1"}),
        "features": base.features.model_copy(update={"benchmark_symbol": None}),
        "datasets": base.datasets.model_copy(update={"dir": str(tmp / "datasets"), "audit_lookahead": False}),
        "quality": base.quality.model_copy(update={"known_issues_path": str(tmp / "known.csv")}),
        "swing": base.swing.model_copy(update={"artifacts_dir": str(tmp / "models")}), "invest": inv,
        "paper": base.paper.model_copy(update={"universe_snapshot": str(tmp / "snap.csv"), "report_dir": str(tmp / "reports")}),
    })
    return {"tmp": tmp, "cfg": cfg}


def build(engine, sim):
    """History up to 2025-12-31 for the first 12 stocks, final models registered; returns the client (with data for all 14 stocks up to the end) and the five session dates."""
    cfg, tmp = sim["cfg"], sim["tmp"]
    data = market(SYMBOLS, date(2022, 1, 3), 1100)
    dates = [b.trade_date for b in data["S01"]]
    i0 = dates.index(date(2025, 12, 31))
    days = dates[i0 + 1: i0 + 6]
    apply(engine, "U1", MEMBERS0, date(2020, 1, 1))
    hist = {k: v for k, v in data.items() if k in MEMBERS0}
    res = run_data_pipeline(engine, FakeClient(hist), cfg, ["U1"], date(2022, 1, 3), dates[i0], write_report_file=False)
    assert res["ok"]
    fsets, lspecs = load_definitions(PROJECT_ROOT / cfg.features.definitions_path)
    with session_scope(engine) as s:
        sync_definitions(s, fsets, lspecs)
        from predict_stock.data.calendar import sync_trading_calendar
        sync_trading_calendar(s, cfg)
    setup = load_setup(engine, cfg)
    bundle, _ = fit_final(with_ends(setup.frames["swing"]), cfg.swing, {}, setup.holdout.start, setup.holdout.end)
    register_model(engine, cfg, bundle, "swing_lgbm_final", dataset_id=None, experiment_id=None, extra_params={})
    ivc = cfg.invest.model_copy(update={"candidates": ["factor"]})
    for key, preset in cfg.invest.presets.items():
        cands, _ = IW.fit_final(IW.prepare(setup.frames["invest"], preset), ivc, key, preset, {}, setup.holdout.start, setup.holdout.end)
        register_artifact(engine, cfg, cands.artifact("factor"), f"invest_{key}_factor_final", algo="invest_candidate", feature_set="invest:2", label_spec="invest:1", dataset_id=None,
                          experiment_id=None, params={}, seed=42)
    D.init_paper(engine, cfg, days[0])
    return FakeClient(data), days


def tables(engine):
    with engine.connect() as c:
        return {"orders": sorted(tuple(map(str, r)) for r in c.execute(select(PaperOrder.order_key, PaperOrder.status, PaperOrder.quantity, PaperOrder.fill_price)).all()),
                "positions": sorted(tuple(map(str, r)) for r in c.execute(select(PaperPosition.portfolio_code, PaperPosition.instrument_id, PaperPosition.quantity, PaperPosition.status)).all()),
                "snaps": sorted(tuple(map(str, r)) for r in c.execute(select(PortfolioSnapshot.snapshot_date, PortfolioSnapshot.equity)).all()),
                "recs": sorted(tuple(map(str, r)) for r in c.execute(select(Recommendation.strategy, Recommendation.instrument_id, Recommendation.as_of_date, Recommendation.action, Recommendation.status)).all()),
                "targets": sorted(tuple(map(str, r)) for r in c.execute(select(SleeveTarget.sleeve, SleeveTarget.as_of_date, SleeveTarget.kind, SleeveTarget.tranche_n)).all())}


def sym(engine):
    with engine.connect() as c:
        return dict(c.execute(select(InstrumentSymbolHistory.symbol, InstrumentSymbolHistory.instrument_id)).all())


def test_five_daily_runs_a_universe_change_and_a_repeat_without_touching_the_code(engine, sim):
    cfg, tmp = sim["cfg"], sim["tmp"]
    client, days = build(engine, sim)
    assert len(days) == 5 and days[0] == date(2026, 1, 1)                       # day 1 is a monthly AND quarterly rebalance date
    results = {}
    ids = sym(engine)
    inv = {v: k for k, v in ids.items()}

    # ---- days 1 and 2 ------------------------------------------------------------------------------------------------------------------------------------------
    for d in days[:2]:
        results[d] = D.run_daily(engine, cfg, client, d)
        assert results[d]["ok"], {k: v.get("error") for k, v in results[d].items() if isinstance(v, dict) and v.get("error")}
    with engine.connect() as c:
        bought = c.execute(select(Recommendation.instrument_id).where(Recommendation.strategy == "INVEST_B1", Recommendation.action == "BUY", Recommendation.as_of_date == days[0])).scalars().all()
    assert len(bought) == 10                                                     # the INVEST B1 rebalance of day 1 asked for its top 10
    removed = [inv[i] for i in bought[:2]]
    kept = [s for s in MEMBERS0 if s not in removed]
    assert {t[2] for t in tables(engine)["targets"]} == {"rebalance"} and len(tables(engine)["targets"]) == 2                 # B1 and B2, one rebalance each

    # ---- day 3: the universe file changes two stocks ---------------------------------------------------------------------------------------------------------
    write_snapshot(tmp / "snap.csv", kept + ["S13", "S14"])
    results[days[2]] = D.run_daily(engine, cfg, client, days[2])
    r3 = results[days[2]]
    assert r3["ok"], {k: v.get("error") for k, v in r3.items() if isinstance(v, dict) and v.get("error")}
    u = r3["universe"]["result"]
    assert sorted(u["added"]) == ["S13", "S14"] and sorted(u["removed"]) == sorted(removed)
    ids = sym(engine)
    with engine.connect() as c:
        for s13 in ("S13", "S14"):
            assert c.execute(select(func.count()).select_from(PriceBar).where(PriceBar.instrument_id == ids[s13])).scalar() >= 60           # backfilled with their history ...
            assert c.execute(select(func.count()).select_from(Prediction).where(Prediction.instrument_id == ids[s13], Prediction.as_of_date == days[2])).scalar() == 1   # ... and scored the same day
        for r in removed:
            assert c.execute(select(func.count()).select_from(Prediction).where(Prediction.instrument_id == ids[r], Prediction.as_of_date == days[2])).scalar() == 0     # no longer scored
        cats = c.execute(select(Alert.category, Alert.severity).where(Alert.category == "universe_change")).all()
    assert sorted(a[1] for a in cats) == ["info", "info", "warn", "warn"]
    flagged = rows(engine, select(Recommendation).where(Recommendation.instrument_id.in_([ids[r] for r in removed]), Recommendation.status != "watch"))
    assert flagged and all(r.flags and r.flags["left_universe"]["note"] == "ra khỏi rổ" and r.flags["left_universe"]["policy"] in ("hold_until_exit", "next_rebalance") for r in flagged
                           if r.status in ("holding", "pending"))
    step3 = tables(engine)

    # ---- days 4 and 5 ---------------------------------------------------------------------------------------------------------------------------------------------
    for d in days[3:]:
        results[d] = D.run_daily(engine, cfg, client, d)
        assert results[d]["ok"], {k: v.get("error") for k, v in results[d].items() if isinstance(v, dict) and v.get("error")}

    # ---- acceptance: nothing failed, every step is a job_runs row with a duration, the records are complete --------------------------------------------------------
    with engine.connect() as c:
        assert c.execute(select(func.count()).select_from(Alert).where(Alert.category == "job_failed")).scalar() == 0
        runs = c.execute(select(JobRun.job_name, JobRun.status, JobRun.finished_at, JobRun.started_at).where(JobRun.job_name.like("paper_%"))).all()
        snaps = c.execute(select(PortfolioSnapshot.snapshot_date, PortfolioSnapshot.equity, PortfolioSnapshot.cash).order_by(PortfolioSnapshot.snapshot_date)).all()
    assert all(r[1] == "success" and r[2] is not None and r[2] >= r[3] for r in runs)
    assert {r[0] for r in runs} == {"paper_universe", "paper_ingest", "paper_calendar", "paper_features", "paper_flags", "paper_cards", "paper_state", "paper_report"} and len(runs) == 5 * 8
    assert [s[0] for s in snaps] == days and all(s[1] > 0 for s in snaps) and snaps[0][1] == pytest.approx(1e9)
    t5 = tables(engine)
    assert {(t[0], t[1], t[2], t[3]) for t in t5["targets"]} == {("invest_b1", "2026-01-01", "rebalance", "1"), ("invest_b2", "2026-01-01", "rebalance", "1"),
                                                                 ("invest_b1", "2026-01-05", "tranche", "2"), ("invest_b2", "2026-01-06", "tranche", "2")}       # INVEST acts only on rebalance and tranche days
    assert t5["orders"] and t5["positions"]

    # the removed stocks: NOT sold in a hurry (their INVEST positions are still open on day 5; no rebalance sale), no new INVEST buys of them after they left
    with engine.connect() as c:
        for r in removed:
            pos = c.execute(select(PaperPosition.status, PaperPosition.portfolio_code).where(PaperPosition.instrument_id == ids[r], PaperPosition.portfolio_code == "paper:invest_b1")).all()
            assert all(p[0] == "open" for p in pos), (r, pos)
            sells = c.execute(select(PaperOrder.order_type).where(PaperOrder.instrument_id == ids[r], PaperOrder.side == "SELL", PaperOrder.portfolio_code == "paper:invest_b1")).scalars().all()
            assert "rebalance" not in sells
            late = c.execute(select(func.count()).select_from(Recommendation).where(Recommendation.instrument_id == ids[r], Recommendation.action == "BUY", Recommendation.as_of_date > days[1])).scalar()
            assert late == 0

    # tomorrow's recommendations are right: every stored BUY card of the last day is complete and consistent with its own rules, none for a removed stock
    from predict_stock.backtest.market import MarketRules
    rules = MarketRules.from_config(cfg.market)
    last = rows(engine, select(Recommendation).where(Recommendation.as_of_date == days[4], Recommendation.action == "BUY"))
    assert last and not {ids[r] for r in removed} & {r.instrument_id for r in last if r.strategy == "SWING"}
    for r in last:
        card = Card.from_dict(r.card)
        assert card.problems(rules) == [] and r.status == "pending" and pd.Timestamp(r.valid_until) > pd.Timestamp(days[4])

    # ---- the report of the last day exists in the three formats ------------------------------------------------------------------------------------------------
    out = Path(cfg.paper.report_dir) / str(days[4])
    assert (out / "report.md").exists() and (out / "report.html").exists() and (out / "cards.csv").exists() and (out / "equity_curve.csv").exists()
    assert "Paper trading report" in (out / "report.md").read_text(encoding="utf-8")

    # ---- a repeat changes nothing --------------------------------------------------------------------------------------------------------------------------------
    n_recs = len(t5["recs"])
    again = D.run_daily(engine, cfg, client, days[4])
    assert again["ok"]
    assert tables(engine) == t5 and len(tables(engine)["recs"]) == n_recs
    with engine.connect() as c:
        assert c.execute(select(func.count()).select_from(Alert).where(Alert.category.in_(("job_failed", "cards_skipped")))).scalar() == 0
