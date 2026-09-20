"""The held-out period opens only after a preset passed the pre-registered criteria under an unchanged pre-registration."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import select

from predict_stock.backtest.runner import clean
from predict_stock.backtest.walkforward import Holdout
from predict_stock.db.models import Experiment
from predict_stock.invest import job, report


@pytest.fixture
def stub(monkeypatch, cfg):
    setup = SimpleNamespace(holdout=Holdout(pd.Timestamp("2025-09-19"), pd.Timestamp("2026-09-18")), dataset_hashes={"invest": "abc"}, universe="LARGE50", cfg=cfg)
    monkeypatch.setattr(job, "load_setup", lambda engine, c: setup)
    return setup


def payload(setup, cfg, passed, prereg=None):
    crit = [{"name": "net Sharpe above every portfolio baseline", "ok": passed}, {"name": "reality check", "ok": passed}]
    return {"preregistration": prereg if prereg is not None else clean(job.preregistration(cfg, setup)), "decision": {"passed": passed, "criteria": crit}, "best": "factor"}


def none_recorded(engine):
    with engine.connect() as c:
        return c.execute(select(Experiment.id).where(Experiment.name.like("oos:%"))).all() == []


def test_no_preset_passing_keeps_it_closed_and_names_the_failures(engine, cfg, stub, monkeypatch):
    monkeypatch.setattr(report, "latest_payloads", lambda c: {"b1": payload(stub, cfg, False), "b2": payload(stub, cfg, False)})
    with pytest.raises(job.HoldoutClosed, match="reality check"):
        job.run_invest_oos(engine, cfg)
    assert none_recorded(engine)


def test_no_stored_run_means_closed(engine, cfg, stub, monkeypatch):
    def missing(c):
        raise FileNotFoundError("nothing stored")
    monkeypatch.setattr(report, "latest_payloads", missing)
    with pytest.raises(job.HoldoutClosed, match="no stored invest run"):
        job.run_invest_oos(engine, cfg)


def test_a_run_under_a_different_pre_registration_is_refused_even_if_it_passed(engine, cfg, stub, monkeypatch):
    other = clean(job.preregistration(cfg, stub))
    other["decision_rule"]["reality_check_p"] = 0.5                          # the rule was loosened after the fact
    monkeypatch.setattr(report, "latest_payloads", lambda c: {"b1": payload(stub, cfg, True, other)})
    with pytest.raises(job.HoldoutClosed, match="different pre-registration"):
        job.run_invest_oos(engine, cfg)
    assert none_recorded(engine)


def test_the_pre_registration_pins_presets_candidates_decision_and_data(cfg):
    setup = SimpleNamespace(holdout=Holdout(pd.Timestamp("2025-09-19"), pd.Timestamp("2026-09-18")), dataset_hashes={"invest": "abc"}, cfg=cfg)
    p = job.preregistration(cfg, setup)
    assert p["candidates"] == ["factor", "ridge", "elasticnet", "lgbm"] and p["presets"]["b1"]["horizon"] == 63 and p["presets"]["b2"]["rebalance"] == "quarterly"
    assert p["decision_rule"]["reality_check_p"] == 0.1 and p["portfolio"]["regime_filter"] == "variant only" and p["dataset_content_hash"] == "abc"
    assert p["factors"][2] == ["vol_126_csrank", -1]
    from predict_stock.features.registry import spec_hash
    changed = cfg.model_copy(update={"invest": cfg.invest.model_copy(update={"top_k": 12})})
    assert spec_hash(job.preregistration(changed, setup)) != spec_hash(p)
