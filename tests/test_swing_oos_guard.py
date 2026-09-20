"""The held-out period may be opened only after the pre-registered criteria were met on the development walk-forward."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import select

from predict_stock.backtest.runner import clean
from predict_stock.backtest.walkforward import Holdout
from predict_stock.db.models import Experiment
from predict_stock.swing import job, report


@pytest.fixture
def stub(monkeypatch, tmp_path, cfg):
    setup = SimpleNamespace(holdout=Holdout(pd.Timestamp("2025-09-19"), pd.Timestamp("2026-09-18")), dataset_hashes={"swing": "abc"}, universe="LARGE50")
    monkeypatch.setattr(job, "load_setup", lambda engine, c: setup)
    path = tmp_path / "payload.json"
    monkeypatch.setattr(report, "latest_payload_path", lambda c: path)
    return setup, path


def write(path, setup, cfg, passed=False, prereg=None):
    crit = [{"name": "net Sharpe above every portfolio baseline", "ok": passed}, {"name": "rank IC t-statistic >= 2.0", "ok": True}]
    path.write_text(json.dumps({"preregistration": prereg if prereg is not None else clean(job.preregistration(cfg, setup)), "decision": {"passed": passed, "criteria": crit}}))


def no_oos_recorded(engine):
    with engine.connect() as c:
        return c.execute(select(Experiment.id).where(Experiment.name.like("oos:%"))).all() == []


def test_a_failed_verdict_keeps_the_held_out_period_closed_and_names_the_failed_criterion(engine, cfg, stub):
    setup, path = stub
    write(path, setup, cfg, passed=False)
    with pytest.raises(job.HoldoutClosed, match="net Sharpe above every portfolio baseline"):
        job.run_swing_oos(engine, cfg)
    assert no_oos_recorded(engine)


def test_no_stored_run_means_closed(engine, cfg, stub):
    with pytest.raises(job.HoldoutClosed, match="no stored swing run"):
        job.run_swing_oos(engine, cfg)                                     # the payload file does not exist


def test_a_run_made_under_a_different_pre_registration_is_refused_even_if_it_passed(engine, cfg, stub):
    setup, path = stub
    other = clean(job.preregistration(cfg, setup))
    other["decision_rule"]["min_ic_tstat"] = 0.5                            # someone loosened the rule after the fact
    write(path, setup, cfg, passed=True, prereg=other)
    with pytest.raises(job.HoldoutClosed, match="different pre-registration"):
        job.run_swing_oos(engine, cfg)
    assert no_oos_recorded(engine)


def test_the_pre_registration_pins_the_decision_rule_the_primary_strategy_and_the_data(cfg):
    setup = SimpleNamespace(holdout=Holdout(pd.Timestamp("2025-09-19"), pd.Timestamp("2026-09-18")), dataset_hashes={"swing": "abc"})
    p = job.preregistration(cfg, setup)
    assert p["decision_rule"] == {"min_ic_tstat": 2.0, "min_positive_fold_share": 0.6, "stress_cost_multiple": 2.0}
    assert p["primary_strategy"]["top_k"] == 10 and p["primary_strategy"]["probability_filter"] is None and p["holdout"] == ["2025-09-19", "2026-09-18"]
    assert p["dataset_content_hash"] == "abc" and p["tuning"]["trials"] == 25
    changed = cfg.model_copy(update={"swing": cfg.swing.model_copy(update={"decision": cfg.swing.decision.model_copy(update={"min_ic_tstat": 1.0})})})
    from predict_stock.features.registry import spec_hash
    assert spec_hash(job.preregistration(changed, setup)) != spec_hash(p)         # a changed rule is a different registration
