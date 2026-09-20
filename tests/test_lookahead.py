"""ACCEPTANCE: a feature at date t must not use data after t.

Method: cut all data after a date T and recompute; every feature value at dates <= T must be identical. The audit is
itself tested: deliberately leaky features (peeking one row ahead, using full-sample statistics, a centred window,
the last value of the series, the benchmark's future, a cross-sectional rank of tomorrow's values) must be caught."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import predict_stock.features.cs  # noqa: F401
import predict_stock.features.labels  # noqa: F401
import predict_stock.features.ts  # noqa: F401
from predict_stock.features.audit import audit_label_horizon, audit_no_lookahead, pick_cut_dates
from predict_stock.features.engine import assemble_frame, compute_feature_columns
from predict_stock.features.registry import (
    CrossSectionalFeature, FeatureSetSpec, LabelBuilder, LabelSpecSpec, TimeSeriesFeature, get_feature, register_feature,
    register_label, unregister,
)
from predict_stock.features.sets import load_definitions
from panel_helpers import all_members, random_panel


def small_sets():
    """The shipped SWING / INVEST sets with the cross-sectional minimum lowered for a 6-instrument panel."""
    fsets, lspecs = load_definitions("config/feature_sets.yaml")
    out = {}
    for key, spec in fsets.items():
        d = {"version": spec.version, "features": [{"name": n, "version": v, "params": {**dict(p), **({"min_count": 2} if n == "cs_rank" else {})}}
                                                   for n, v, p in spec.features]}
        out[key] = FeatureSetSpec.from_dict(spec.name, d)
    labels = {}
    for key, spec in lspecs.items():
        d = {"version": spec.version, "labels": [{"name": n, "version": v, "params": {**dict(p), **({"min_count": 2} if n == "fwd_rank_return" else {})}}
                                                 for n, v, p in spec.labels]}
        labels[key] = LabelSpecSpec.from_dict(spec.name, d)
    return out, labels


def membership_with_changes(panel):
    """Instrument 1 joins late, 2 leaves early, the rest are always members."""
    m = all_members(panel)
    m.loc[panel.calendar[:120], 1] = False
    m.loc[panel.calendar[200:], 2] = False
    return m


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("set_key", [("swing", 1), ("invest", 1), ("invest", 2)])
def test_shipped_feature_sets_do_not_use_future_data(set_key, seed):
    sets, _ = small_sets()
    panel = random_panel(n_days=340, n_inst=6, seed=seed, gaps=6)
    feats = sets[set_key].instantiate()
    mask = membership_with_changes(panel)
    cuts = list(pick_cut_dates(panel.calendar, panel.calendar[0], panel.calendar[-1], 5)) + [panel.calendar[-2], panel.calendar[150]]
    res = audit_no_lookahead(panel, mask, feats, cuts)
    assert res.passed, res.violations
    assert res.columns_checked == len(cuts) * len(sets[set_key].columns())


def test_removing_the_future_leaves_every_earlier_row_identical():
    """The same statement one level up: whole dataset rows (features), not just wide frames."""
    sets, labs = small_sets()
    panel = random_panel(n_days=340, n_inst=6, seed=7)
    mask = membership_with_changes(panel)
    feats, lbs = sets[("swing", 1)].instantiate(), labs[("swing", 1)].instantiate()
    fcols = sets[("swing", 1)].columns()
    full = assemble_frame(panel, feats, lbs, mask, panel.calendar[0], panel.calendar[-1])
    for T in (panel.calendar[120], panel.calendar[250], panel.calendar[300]):
        cut = assemble_frame(panel.truncate(T), feats, lbs, mask.loc[:T], panel.calendar[0], T)
        head = full[full["trade_date"] <= T].reset_index(drop=True)
        assert len(head) == len(cut) and (head[["trade_date", "instrument_id"]].to_numpy() == cut[["trade_date", "instrument_id"]].to_numpy()).all()
        a, b = head[fcols].to_numpy(float), cut[fcols].to_numpy(float)
        assert np.allclose(a, b, equal_nan=True, rtol=1e-9, atol=1e-12), f"features changed after cutting the data after {T.date()}"


def test_appending_future_bars_does_not_change_history():
    sets, labs = small_sets()
    long = random_panel(n_days=340, n_inst=5, seed=11, gaps=4)
    short = long.truncate(long.calendar[259])
    feats = sets[("invest", 1)].instantiate()
    a = compute_feature_columns(short, feats, all_members(short))
    b = compute_feature_columns(long, feats, all_members(long))
    for col, wide in a.items():
        assert np.allclose(wide.to_numpy(float), b[col].loc[:short.calendar[-1]].to_numpy(float), equal_nan=True), col


# ---- the audit must actually be able to fail ---------------------------------------------------------------------
@pytest.fixture
def leaky():
    made = []

    def reg(cls):
        register_feature(cls)
        made.append(cls)
        return cls
    yield reg
    for c in made:
        unregister("feature", c.name, c.version)


def ts_feature(name, fn):
    class F(TimeSeriesFeature):
        pass
    F.name, F.version, F.__qualname__ = name, 1, name
    F.columns = lambda self: [f"{name}_out"]
    F.compute = lambda self, bars, bench: pd.DataFrame({f"{name}_out": fn(bars, bench)}, index=bars.index)
    return F


LEAKS = {
    "leak_next_row": lambda b, bench: b["close"].shift(-1) / b["close"] - 1,
    "leak_two_ahead": lambda b, bench: b["close"].rolling(3).max().shift(-2) / b["close"],
    "leak_full_sample": lambda b, bench: (b["close"] - b["close"].mean()) / b["close"].std(),
    "leak_centered": lambda b, bench: b["close"].rolling(5, center=True).mean() / b["close"] - 1,
    "leak_last_value": lambda b, bench: b["close"] / b["close"].iloc[-1] - 1,
    "leak_bench_future": lambda b, bench: bench.shift(-1) / bench - 1,
    "leak_expanding_reverse": lambda b, bench: b["close"][::-1].expanding().max()[::-1] / b["close"],
}


@pytest.mark.parametrize("name", sorted(LEAKS))
def test_the_audit_catches_a_time_series_leak(leaky, name):
    leaky(ts_feature(name, LEAKS[name]))
    panel = random_panel(n_days=260, n_inst=4, seed=5)
    feats = FeatureSetSpec.from_dict("x", {"version": 1, "features": [{"name": name, "version": 1}]}).instantiate()
    res = audit_no_lookahead(panel, all_members(panel), feats, pick_cut_dates(panel.calendar, panel.calendar[0], panel.calendar[-1], 3))
    assert not res.passed and res.violations[0]["column"] == f"{name}_out" and res.violations[0]["kind"] == "feature"


def test_the_audit_catches_a_cross_sectional_leak(leaky):
    class TomorrowRank(CrossSectionalFeature):
        name, version = "leak_cs_tomorrow", 1
        def inputs(self): return ["ret_1"]
        def columns(self): return ["leak_cs_out"]
        def compute(self, inputs, mask): return {"leak_cs_out": inputs["ret_1"].shift(-1).rank(axis=1, pct=True)}
    class GlobalNorm(CrossSectionalFeature):
        name, version = "leak_cs_global", 1
        def inputs(self): return ["ret_1"]
        def columns(self): return ["leak_cs_global_out"]
        def compute(self, inputs, mask): return {"leak_cs_global_out": inputs["ret_1"] / inputs["ret_1"].abs().mean().mean()}
    leaky(TomorrowRank); leaky(GlobalNorm)
    panel = random_panel(n_days=260, n_inst=4, seed=6)
    for leak_feature, out_col in (("leak_cs_tomorrow", "leak_cs_out"), ("leak_cs_global", "leak_cs_global_out")):
        feats = FeatureSetSpec.from_dict("x", {"version": 1, "features": [{"name": "ret", "version": 1}, {"name": leak_feature, "version": 1}]}).instantiate()
        res = audit_no_lookahead(panel, all_members(panel), feats, pick_cut_dates(panel.calendar, panel.calendar[0], panel.calendar[-1], 3))
        assert not res.passed and {v["column"] for v in res.violations} == {out_col}      # ret itself is clean, only the leak is reported


def test_a_clean_control_feature_passes(leaky):
    leaky(ts_feature("ctrl_ok", lambda b, bench: b["close"].rolling(10).mean() / b["close"] - 1))
    panel = random_panel(n_days=260, n_inst=4, seed=5)
    feats = FeatureSetSpec.from_dict("x", {"version": 1, "features": [{"name": "ctrl_ok", "version": 1}]}).instantiate()
    assert audit_no_lookahead(panel, all_members(panel), feats, pick_cut_dates(panel.calendar, panel.calendar[0], panel.calendar[-1], 3)).passed


# ---- labels: they look forward, but only as far as they declare -------------------------------------------------------
def test_labels_use_nothing_beyond_their_declared_horizon():
    _, lspecs = small_sets()
    panel = random_panel(n_days=300, n_inst=5, seed=8)
    for key in (("swing", 1), ("invest", 1)):
        lbs = lspecs[key].instantiate()
        cuts = [panel.calendar[i] for i in (200, 250, 290)]
        res = audit_label_horizon(panel, all_members(panel), lbs, cuts)
        assert res.passed, res.violations


def test_the_label_audit_catches_a_label_that_looks_past_its_horizon():
    class Sneaky(LabelBuilder):
        name, version = "sneaky", 1
        def columns(self): return ["sneaky_ret"]
        def horizon(self): return 3
        def compute(self, panel, mask):
            return {"sneaky_ret": panel.close.shift(-8) / panel.close - 1}     # declares 3, uses 8
    register_label(Sneaky)
    try:
        panel = random_panel(n_days=200, n_inst=3, seed=9)
        res = audit_label_horizon(panel, all_members(panel), [Sneaky()], [panel.calendar[120], panel.calendar[150]])
        assert not res.passed and res.violations[0]["column"] == "sneaky_ret"
    finally:
        unregister("label", "sneaky", 1)


def test_cut_dates_are_inside_the_range_and_leave_data_after_them():
    cal = pd.bdate_range("2020-01-01", periods=400)
    cuts = pick_cut_dates(cal, cal[50], cal[350], 3)
    assert len(cuts) == 3 and all(cal[50] < c < cal[350] for c in cuts) and cuts == sorted(cuts)
    assert pick_cut_dates(cal, cal[0], cal[1], 3) == []
