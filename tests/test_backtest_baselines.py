"""Baselines: signal timing / point-in-time, schedules, the engine wiring, determinism, and the end-to-end job on synthetic data."""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select

import predict_stock.cli as cli
from conftest import FakeClient, apply, make_bars
from predict_stock.backtest.baselines import BASELINES, BaselineSpec, make_signals, rebalance_sessions, scores_for
from predict_stock.backtest.engine import EngineConfig, MarketData, run_backtest
from predict_stock.backtest.market import MarketRules
from predict_stock.config import load_config
from predict_stock.data.ingest import ingest_universe
from predict_stock.db.models import Experiment, JobRun
from predict_stock.db.session import session_scope

CAL = pd.bdate_range("2024-01-01", periods=80)


def frame(n_days=80, n_inst=6, seed=0):
    """Dataset-like frame with the columns the baselines read, one row per (date, instrument)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, d in enumerate(CAL[:n_days]):
        for k in range(1, n_inst + 1):
            rows.append({"trade_date": d, "instrument_id": k, "ret_10": rng.normal(), "zscore_20": rng.normal(), "mom_6m_csrank": rng.random(),
                         "mom_12m_csrank": rng.random(), "vol_126": rng.uniform(0.1, 0.5)})
    return pd.DataFrame(rows)


# ---- schedules ---------------------------------------------------------------------------------------------------------------------
def test_weekly_and_monthly_rebalance_dates_are_the_first_session_of_each_period():
    weekly = rebalance_sessions(CAL, "weekly", 0, len(CAL) - 1)
    assert [CAL[i].dayofweek for i in weekly] == [0] * len(weekly) and len(weekly) == 16
    monthly = rebalance_sessions(CAL, "monthly", 0, len(CAL) - 1)
    assert [CAL[i].strftime("%Y-%m-%d") for i in monthly] == ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
    window = rebalance_sessions(CAL, "monthly", 30, 60)
    assert window[0] == 30 and window[1:] == [i for i in monthly if 30 < i <= 60]        # the start of a window opens its first period
    assert rebalance_sessions(CAL, "weekly", 10, 9) == []


# ---- signals -----------------------------------------------------------------------------------------------------------------------
def weights(sig):
    return {it.instrument_id: it.weight for it in sig.items}


def test_top_k_by_a_column_and_reversed_for_mean_reversion():
    f = frame()
    sigs = make_signals(BASELINES["mom_short"], f, CAL, 5, 20, None)
    i = min(sigs)
    row = f[f.trade_date == CAL[i]].set_index("instrument_id")["ret_10"]
    assert set(weights(sigs[i])) == set(row.nlargest(6).index)                            # k = 10 > 6 members: every member
    small = replace(BASELINES["mom_short"], k=3)
    assert set(weights(make_signals(small, f, CAL, 5, 20, None)[i])) == set(row.nlargest(3).index)
    rev = replace(BASELINES["mean_reversion"], k=3)
    z = f[f.trade_date == CAL[i]].set_index("instrument_id")["zscore_20"]
    assert set(weights(make_signals(rev, f, CAL, 5, 20, None)[i])) == set(z.nsmallest(3).index)      # the MOST oversold, not the most extended


def test_the_signal_at_a_date_uses_only_that_dates_row():
    """Changing the dataset rows of every OTHER date must not change the signal of this date (no look-ahead in the wiring)."""
    f = frame()
    spec = replace(BASELINES["mom_short"], k=3)
    a = make_signals(spec, f, CAL, 5, 40, None)
    g = f.copy()
    i = min(a)
    g.loc[g.trade_date != CAL[i], "ret_10"] = -999.0
    b = make_signals(spec, g, CAL, i, i, None)
    assert weights(a[i]) == weights(b[i])


def test_weights_are_equal_capped_and_sum_to_at_most_one():
    f = frame()
    sig = make_signals(replace(BASELINES["mom_short"], k=4), f, CAL, 5, 5, None)[5]
    assert list(weights(sig).values()) == pytest.approx([0.25] * 4) and sig.full_rebalance
    capped = make_signals(replace(BASELINES["mom_short"], k=4), f, CAL, 5, 5, 0.2)[5]
    assert max(weights(capped).values()) <= 0.2 + 1e-9 and sum(weights(capped).values()) == pytest.approx(0.8)     # the excess stays in cash
    eq = make_signals(BASELINES["equal_weight"], f, CAL, 5, 5, None)[5]
    assert len(eq.items) == 6 and sum(weights(eq).values()) == pytest.approx(1.0)


def test_inverse_volatility_variant_weights_the_calm_names_more():
    f = frame()
    sig = make_signals(replace(BASELINES["mom_long_invvol"], k=4), f, CAL, 5, 5, None)[5]
    w = weights(sig)
    vol = f[f.trade_date == CAL[5]].set_index("instrument_id")["vol_126"]
    ordered = sorted(w, key=lambda k: vol[k])
    assert [w[k] for k in ordered] == sorted(w.values(), reverse=True) and sum(w.values()) == pytest.approx(1.0)


def test_momentum_long_uses_the_mean_of_the_two_ranks():
    f = frame()
    row = f[f.trade_date == CAL[5]]
    score, _ = scores_for(BASELINES["mom_long"], row, 5)
    assert score.to_numpy() == pytest.approx(((row.mom_6m_csrank + row.mom_12m_csrank) / 2).to_numpy())


def test_random_baselines_are_seeded_reproducible_and_change_with_time():
    f = frame()
    spec = replace(BASELINES["random_weekly"], k=3)
    a = make_signals(spec, f, CAL, 5, 60, None)
    b = make_signals(spec, f, CAL, 5, 60, None)
    assert {i: weights(s) for i, s in a.items()} == {i: weights(s) for i, s in b.items()}
    picks = {frozenset(weights(s)) for s in a.values()}
    assert len(picks) > 3                                                                # not the same names every week
    other = make_signals(replace(spec, seed=1), f, CAL, 5, 60, None)
    assert {i: weights(s) for i, s in a.items()} != {i: weights(s) for i, s in other.items()}


def test_missing_scores_are_never_selected_and_empty_dates_produce_no_signal():
    f = frame()
    f.loc[(f.trade_date == CAL[5]) & (f.instrument_id.isin([1, 2])), "ret_10"] = np.nan
    sig = make_signals(replace(BASELINES["mom_short"], k=6), f, CAL, 5, 5, None)[5]
    assert set(weights(sig)) == {3, 4, 5, 6}
    assert make_signals(BASELINES["mom_short"], f[f.trade_date != CAL[5]], CAL, 5, 5, None) == {}


def test_purging_uses_the_latest_end_date_of_all_of_a_rows_labels():
    from predict_stock.backtest.runner import with_label_end
    d = pd.bdate_range("2024-01-01", periods=5)
    f = pd.DataFrame({"trade_date": d, "tb_end": d + pd.Timedelta(days=3), "fwd_end_5": d + pd.Timedelta(days=7), "fwd_end_3": d + pd.Timedelta(days=4), "ret_5": 1.0})
    e = with_label_end(f)
    assert (e["label_end_max"] == d + pd.Timedelta(days=7)).all()
    f.loc[2, "tb_end"] = d[2] + pd.Timedelta(days=30)                                # one long-running barrier label decides for that row
    assert with_label_end(f).loc[2, "label_end_max"] == d[2] + pd.Timedelta(days=30)
    f.loc[1, "fwd_end_5"] = pd.NaT
    assert with_label_end(f).loc[1, "label_end_max"] == d[1] + pd.Timedelta(days=4)  # unresolved labels do not hide the others


def test_every_baseline_is_defined_and_the_registry_is_consistent():
    assert {"equal_weight", "mom_short", "mean_reversion", "mom_long", "bh_vnindex", "bh_vn30"} <= set(BASELINES)     # the five required + the noise floor
    for k, b in BASELINES.items():
        assert b.key == k and b.title and b.description
        assert b.kind in ("portfolio", "index") and (b.kind == "index") == (b.index_symbol is not None)
        if b.kind == "portfolio":
            assert b.rebalance in ("weekly", "monthly") and b.dataset in ("swing", "invest")
    assert BASELINES["random_weekly"].seed != BASELINES["random_monthly"].seed


# ---- signals through the engine: decided at the close, filled at the next open ----------------------------------------------------------
def market_from_frame(f, n_days=80):
    rng = np.random.default_rng(9)
    ids = sorted(f.instrument_id.unique())
    px = pd.DataFrame(20000 + 50 * rng.integers(-30, 30, (n_days, len(ids))).cumsum(0), index=CAL[:n_days], columns=ids).clip(lower=10100).astype(float)
    return MarketData(px, px * 1.005, px * 0.995, px, pd.DataFrame(0.07, index=CAL[:n_days], columns=ids)), px


def test_a_baseline_run_fills_at_the_next_open_and_is_deterministic(cfg):
    f = frame()
    data, px = market_from_frame(f)
    rules = MarketRules.from_config(cfg.market)
    sigs = make_signals(replace(BASELINES["mom_short"], k=3), f, CAL, 10, 60, 0.15)
    a = run_backtest(data, sigs, rules, EngineConfig(capital=1e9, rebalance_threshold=0.02), start=10, end=70)
    b = run_backtest(data, sigs, rules, EngineConfig(capital=1e9, rebalance_threshold=0.02), start=10, end=70)
    assert a.equity.equals(b.equity) and a.fills.equals(b.fills) and len(a.fills) > 5
    decision_idx = np.array(sorted(sigs))
    for _, fl in a.fills.iterrows():
        assert (fl["idx"] - 1) in decision_idx or fl["idx"] > 12                          # every first fill follows a decision session by at least one session
    first = a.fills.iloc[0]
    assert first["idx"] == min(sigs) + 1 and first["ref_price"] == px.iloc[first["idx"]][first["instrument_id"]]


def test_gross_equity_is_never_below_net_for_the_same_signals(cfg):
    f = frame()
    data, _ = market_from_frame(f)
    rules = MarketRules.from_config(cfg.market)
    sigs = make_signals(replace(BASELINES["random_weekly"], k=3), f, CAL, 10, 60, 0.15)
    net = run_backtest(data, sigs, rules, EngineConfig(), start=10, end=70)
    gross = run_backtest(data, sigs, rules.scaled_costs(0), EngineConfig(), start=10, end=70)
    assert gross.equity.iloc[-1] > net.equity.iloc[-1] and net.fills["fee"].sum() > 0


# ---- end to end on synthetic data in the test database --------------------------------------------------------------------------------------
@pytest.fixture
def bt_world(engine, cfg, tmp_path):
    """A tiny universe with 10 stocks + VNINDEX/VN30 over ~330 sessions, real datasets built from it, config pointed at tmp paths."""
    from datetime import timedelta
    start = date(2025, 1, 6)
    rng = np.random.default_rng(4)
    series = {}
    days = [b.trade_date for b in make_bars(start, 330)]
    from decimal import Decimal
    for i in range(10):
        px = 20000 * np.exp(np.cumsum(rng.normal(0.0004, 0.014, 330)))
        bars = []
        for j, b in enumerate(make_bars(start, 330, base=20000)):
            c = Decimal(str(round(px[j] / 50) * 50 + 50))
            o = Decimal(str(round(px[j - 1] / 50) * 50 + 50)) if j else c
            bars.append(replace(b, open=o, high=max(o, c) + 100, low=min(o, c) - 100, close=c, volume=1_000_000 + 1000 * (j % 5)))
        series[f"S{i:02d}"] = bars
    for sym, base in (("VNINDEX", 1200), ("VN30", 1300)):
        idx = 1200 * np.exp(np.cumsum(rng.normal(0.0002, 0.009, 330)))
        bars = [replace(b, open=Decimal(str(round(idx[j], 2))), high=Decimal(str(round(idx[j] * 1.004, 2))), low=Decimal(str(round(idx[j] * 0.996, 2))),
                        close=Decimal(str(round(idx[j], 2)))) for j, b in enumerate(make_bars(start, 330, base=base))]
        series[sym] = bars
    apply(engine, "TRAIN", [k for k in series if k.startswith("S")], days[0])
    c = cfg.model_copy(update={
        "ingest": cfg.ingest.model_copy(update={"benchmark_symbols": ["VNINDEX", "VN30"], "min_sessions": 30, "history_start": "2025-01-01"}),
        "features": cfg.features.model_copy(update={"benchmark_symbol": "VN30", "definitions_path": str(tmp_path / "defs.yaml")}),
        "universe": cfg.universe.model_copy(update={"training_code": "TRAIN", "trading_code": "TRAIN"}),
        "datasets": cfg.datasets.model_copy(update={"dir": str(tmp_path / "ds"), "audit_lookahead": False}),
        "backtest": cfg.backtest.model_copy(update={"start": "2025-04-01", "oos_months": 3, "wf_train_sessions": 60, "wf_test_sessions": 40, "wf_embargo_sessions": 5,
                                                    "top_k": 4, "cost_multipliers": [0.0, 1.0, 2.0], "report_path": str(tmp_path / "BASE.md"),
                                                    "image_dir": str(tmp_path / "img"), "artifacts_dir": str(tmp_path / "art"), "regime_window": 40}),
    })
    ingest_universe(engine, FakeClient(series), c, ["TRAIN"], start, date(2026, 12, 31), now=pd.Timestamp("2026-12-31 09:00", tz="UTC").to_pydatetime())
    (tmp_path / "defs.yaml").write_text("""
feature_sets:
  "swing:1":
    version: 1
    features:
      - {name: ret, version: 1, params: {windows: [5, 10]}}
      - {name: rsi, version: 1, params: {window: 14}}
      - {name: zscore, version: 1, params: {window: 20}}
      - {name: atr, version: 1, params: {window: 14}}
      - {name: cs_rank, version: 1, params: {inputs: [ret_5], min_count: 3}}
  "invest:2":
    version: 2
    features:
      - {name: momentum_skip, version: 1, params: {months: [3, 6, 12], skip: 5, days_per_month: 8}}
      - {name: volatility, version: 1, params: {windows: [20, 30]}}
      - {name: cs_rank, version: 1, params: {inputs: [mom_6m, mom_12m], min_count: 3}}
label_specs:
  "swing:1":
    version: 1
    labels:
      - {name: fwd_rank_return, version: 1, params: {horizons: [3, 5], min_count: 3}}
      - {name: triple_barrier, version: 1, params: {horizon: 10}}
  "invest:1":
    version: 1
    labels:
      - {name: fwd_rank_return, version: 1, params: {horizons: [5, 10], min_count: 3}}
""")
    return c, days


def _fix_baselines(monkeypatch):
    """The synthetic history is short: shrink the long-horizon columns the baselines expect (vol_126, invest names)."""
    import predict_stock.backtest.baselines as bl
    monkeypatch.setitem(bl.BASELINES, "mom_long_invvol", replace(bl.BASELINES["mom_long_invvol"], vol_col="vol_30"))


def test_the_job_runs_end_to_end_reports_stores_and_keeps_the_holdout_untouched(engine, bt_world, monkeypatch, tmp_path):
    from predict_stock.backtest import runner
    from predict_stock.backtest.job import run_baselines_job
    _fix_baselines(monkeypatch)
    monkeypatch.setattr(runner, "DATASETS", {"swing": (("swing", 1), ("swing", 1)), "invest": (("invest", 2), ("invest", 1))})
    cfg, days = bt_world
    out = run_baselines_job(engine, cfg, noise_seeds=3)
    p = out["payload"]
    res = p["results"]
    assert {"equal_weight", "mom_short", "mean_reversion", "mom_long", "bh_vnindex", "bh_vn30", "random_weekly", "random_monthly"} <= set(res)
    hold = pd.Timestamp(p["meta"]["holdout"][0])
    assert pd.Timestamp(p["meta"]["window"][1]) < hold                                         # the development window ends before the held-out period
    for k, eq in out["equities"].items():
        assert eq["net"].index.max() < hold, k
    for k in ("equal_weight", "mom_short"):
        n, g = res[k]["net"], res[k]["gross"]
        assert g["total_return"] >= n["total_return"] and n["sessions"] > 100 and set(res[k]["sensitivity"]) == {"0.0", "1.0", "2.0"}
        assert res[k]["sensitivity"]["1.0"]["cagr"] == pytest.approx(n["cagr"]) and res[k]["sensitivity"]["0.0"]["cagr"] == pytest.approx(g["cagr"])
        assert res[k]["sensitivity"]["2.0"]["sharpe"] <= res[k]["sensitivity"]["1.0"]["sharpe"] + 1e-9
    assert res["mom_short"]["net"]["turnover_annual"] > res["equal_weight"]["net"]["turnover_annual"]     # weekly top-K churns far more than monthly equal weight
    assert p["noise"]["n_seeds"] == 3 and set(p["noise"]["schedules"]) == {"weekly", "monthly"}
    assert p["purging"] and all(set(f["purged"]) == {"swing", "invest"} and all(v >= 0 for v in f["purged"].values()) for f in p["purging"])
    md = (tmp_path / "BASE.md").read_text()
    assert "held-out final period" in md.lower() and "not touched" in md and "Noise floor" in md and "Assumptions and limits" in md
    assert (tmp_path / "img" / "baselines_equity.png").stat().st_size > 5000 and (tmp_path / "img" / "baselines_cost_sensitivity.png").exists()
    with session_scope(engine) as s:
        names = set(s.scalars(select(Experiment.name)))
        assert {"baseline:equal_weight", "baseline:mom_short", "baseline:noise_floor", "baseline:bh_vnindex"} <= names
        exp = s.scalars(select(Experiment).where(Experiment.name == "baseline:equal_weight")).one()
        assert exp.summary["net"]["cagr"] == pytest.approx(res["equal_weight"]["net"]["cagr"]) and exp.summary["artifact"]["sha256"]
        assert s.scalars(select(JobRun).where(JobRun.job_name == "backtest_baselines")).one().status == "success"


def test_rerunning_the_job_reuses_the_stored_results(engine, bt_world, monkeypatch, tmp_path):
    from predict_stock.backtest import runner
    from predict_stock.backtest.job import run_baselines_job
    _fix_baselines(monkeypatch)
    cfg, _ = bt_world
    a = run_baselines_job(engine, cfg, noise_seeds=2)
    md1 = (tmp_path / "BASE.md").read_text()
    b = run_baselines_job(engine, cfg, noise_seeds=2)
    assert a["experiments"] == b["experiments"]                                                    # same rows, no duplicates
    assert (tmp_path / "BASE.md").read_text() == md1
    with session_scope(engine) as s:
        assert s.scalar(select(func.count()).select_from(Experiment)) == len(a["experiments"])
    for k in a["payload"]["results"]:
        assert a["payload"]["results"][k]["net"] == b["payload"]["results"][k]["net"]


def test_the_held_out_period_can_be_evaluated_only_once(engine, bt_world, monkeypatch):
    from predict_stock.backtest.job import run_baselines_job, run_holdout_once
    from predict_stock.backtest.walkforward import OOSAlreadyUsed
    _fix_baselines(monkeypatch)
    cfg, days = bt_world
    dev = run_baselines_job(engine, cfg, noise_seeds=0, write=False)
    with session_scope(engine) as s:
        assert not any(n.startswith("oos:") for n in s.scalars(select(Experiment.name)))         # the development run did not reveal it
    first = run_holdout_once(engine, cfg, keys=["equal_weight", "mom_short"])
    assert set(first) == {"equal_weight", "mom_short"} and first["equal_weight"]["net"]["start"] >= dev["payload"]["meta"]["holdout"][0]
    with pytest.raises(OOSAlreadyUsed, match="only once"):
        run_holdout_once(engine, cfg, keys=["equal_weight"])
    with session_scope(engine) as s:
        assert sum(n.startswith("oos:") for n in s.scalars(select(Experiment.name))) == 1


def test_the_cli_refuses_the_holdout_without_the_final_flag_and_after_use(engine, bt_world, monkeypatch, capsys):
    monkeypatch.setenv("MYSQL_DATABASE", "predict_stock_test")
    _fix_baselines(monkeypatch)
    cfg, _ = bt_world
    monkeypatch.setattr(cli, "load_config", lambda _p=None: cfg)
    assert cli.main(["backtest", "oos"]) == 2 and "--final" in capsys.readouterr().err
    assert cli.main(["backtest", "oos", "--final"]) == 0
    assert cli.main(["backtest", "oos", "--final"]) == 3 and "already evaluated" in capsys.readouterr().err


def test_the_cli_runs_the_baselines_and_lists_them(engine, bt_world, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("MYSQL_DATABASE", "predict_stock_test")
    _fix_baselines(monkeypatch)
    cfg, _ = bt_world
    monkeypatch.setattr(cli, "load_config", lambda _p=None: cfg)
    assert cli.main(["backtest", "list"]) == 0 and "mom_long" in capsys.readouterr().out
    assert cli.main(["backtest", "run", "--baseline", "equal_weight", "--baseline", "bh_vnindex", "--noise-seeds", "0"]) == 0
    out = capsys.readouterr().out
    assert "untouched" in out and "equal_weight" in out and "bh_vnindex" in out and "stored experiments" in out
    assert (tmp_path / "BASE.md").exists()
    with session_scope(engine) as s:
        assert not any(n.startswith("oos:") for n in s.scalars(select(Experiment.name)))          # running baselines never reveals the held-out period
