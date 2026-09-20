"""Registry: plugins, versions, specs, validation; adding a feature needs no schema change."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select, text

import predict_stock.features.ts  # noqa: F401  (registers the built-in plugins)
from predict_stock.db.models import FeatureSet, LabelSpec
from predict_stock.db.session import session_scope
from predict_stock.features import registry as R
from predict_stock.features.registry import (
    FeatureSetSpec, LabelSpecSpec, RegistryError, TimeSeriesFeature, get_feature, list_features, register_feature, spec_hash, unregister,
)
from predict_stock.features.sets import load_definitions, sync_definitions
from panel_helpers import bars_from_close


@pytest.fixture
def throwaway():
    """Register plugins for one test and remove them afterwards."""
    made = []

    def reg(cls):
        register_feature(cls)
        made.append((cls.name, cls.version))
        return cls
    yield reg
    for n, v in made:
        unregister("feature", n, v)


def test_builtin_features_cover_the_brief():
    names = {n for n, _, _, _ in list_features()}
    swing = {"ret", "rsi", "macd", "atr", "bollinger_pctb", "donchian", "zscore", "gap", "volume_spike", "rel_strength", "cs_rank"}
    invest = {"momentum_skip", "volatility", "downside_beta", "drawdown", "sma_ratio", "liquidity", "fundamentals"}
    assert swing | invest <= names
    assert all(k in ("ts", "cs", "panel") for _, _, k, _ in list_features())


def test_a_new_version_lives_next_to_the_old_one_without_touching_it(throwaway):
    class RetV2(TimeSeriesFeature):
        name, version, group = "ret", 2, "swing"
        DEFAULT_PARAMS = {"windows": [1]}
        def columns(self): return ["ret_log_1"]
        def compute(self, bars, bench): return pd.DataFrame({"ret_log_1": np.log(bars["close"]).diff()})
    throwaway(RetV2)
    assert get_feature("ret", 1).version == 1 and get_feature("ret", 2) is RetV2
    close = [100, 110, 121]
    bars = bars_from_close(close)
    v1 = get_feature("ret", 1)(windows=[1]).compute(bars, None)["ret_1"]
    v2 = RetV2().compute(bars, None)["ret_log_1"]
    assert v1.iloc[1] == pytest.approx(0.1) and v2.iloc[1] == pytest.approx(np.log(1.1))     # both usable, neither changed


def test_a_set_pins_versions_so_new_versions_never_change_it(throwaway):
    spec = FeatureSetSpec.from_dict("s", {"version": 1, "features": [{"name": "ret", "version": 1, "params": {"windows": [1]}}]})
    class RetV3(TimeSeriesFeature):
        name, version = "ret", 3
        def columns(self): return ["ret_x"]
        def compute(self, bars, bench): return pd.DataFrame({"ret_x": 0.0}, index=bars.index)
    throwaway(RetV3)
    assert spec.columns() == ["ret_1"]                                                      # still v1


def test_duplicate_registration_and_unknown_plugins_are_errors(throwaway):
    class Dup(TimeSeriesFeature):
        name, version = "dup_feat", 1
        def columns(self): return ["dup_a"]
        def compute(self, bars, bench): return pd.DataFrame({"dup_a": 0.0}, index=bars.index)
    throwaway(Dup)
    class Dup2(TimeSeriesFeature):
        name, version = "dup_feat", 1
        def columns(self): return ["dup_b"]
        def compute(self, bars, bench): return pd.DataFrame({"dup_b": 0.0}, index=bars.index)
    with pytest.raises(RegistryError, match="already registered"):
        register_feature(Dup2)
    with pytest.raises(RegistryError, match=r"unknown feature 'nope' v1"):
        get_feature("nope", 1)
    with pytest.raises(RegistryError, match=r"registered versions: \[1\]"):
        get_feature("ret", 9)
    with pytest.raises(RegistryError, match="unknown label"):
        R.get_label("nope", 1)


def test_parameters_are_validated():
    with pytest.raises(RegistryError, match="unknown params"):
        get_feature("rsi", 1)(windw=14)
    with pytest.raises(RegistryError, match="positive integer"):
        get_feature("rsi", 1)(window=0)
    with pytest.raises(RegistryError, match="fast must be shorter"):
        get_feature("macd", 1)(fast=30, slow=10)
    with pytest.raises(RegistryError, match="non-empty list"):
        get_feature("ret", 1)(windows=[])
    with pytest.raises(RegistryError, match="needs a non-empty"):
        get_feature("cs_rank", 1)()
    with pytest.raises(RegistryError, match="positive"):
        R.get_label("triple_barrier", 1)(target_mult=0)


@pytest.mark.parametrize("bad", ["symbol_x", "ticker", "instrument_id", "id", "exchange_hose", "trade_date", "isin_code", "sector_code_a"])
def test_a_feature_cannot_expose_an_identifier(throwaway, bad):
    class Ident(TimeSeriesFeature):
        name, version = "ident_feat", 1
        def columns(self): return [bad]
        def compute(self, bars, bench): return pd.DataFrame({bad: 1.0}, index=bars.index)
    throwaway(Ident)
    spec = FeatureSetSpec.from_dict("x", {"version": 1, "features": [{"name": "ident_feat", "version": 1}]})
    with pytest.raises(RegistryError, match="identify an instrument|clash"):
        spec.instantiate()


def test_set_validation_catches_clashes_and_missing_inputs():
    with pytest.raises(RegistryError, match="produced by both"):
        FeatureSetSpec.from_dict("x", {"version": 1, "features": [{"name": "rsi", "version": 1}, {"name": "rsi", "version": 1}]}).instantiate()
    with pytest.raises(RegistryError, match="which no feature of the set produces"):
        FeatureSetSpec.from_dict("x", {"version": 1, "features": [{"name": "cs_rank", "version": 1, "params": {"inputs": ["rsi_14"]}}]}).instantiate()
    with pytest.raises(RegistryError, match="no features"):
        FeatureSetSpec.from_dict("x", {"version": 1, "features": []})
    with pytest.raises(RegistryError, match="needs name and version"):
        FeatureSetSpec.from_dict("x", {"version": 1, "features": [{"name": "rsi"}]})
    with pytest.raises(RegistryError, match="duplicate column"):
        LabelSpecSpec.from_dict("l", {"version": 1, "labels": [{"name": "triple_barrier", "version": 1}, {"name": "triple_barrier", "version": 1}]}).instantiate()


def test_spec_hash_ignores_key_order_and_detects_changes():
    a = {"features": [{"name": "rsi", "version": 1, "params": {"window": 14, "x": 1}}]}
    b = {"features": [{"params": {"x": 1, "window": 14}, "version": 1, "name": "rsi"}]}
    assert spec_hash(a) == spec_hash(b)
    assert spec_hash(a) != spec_hash({"features": [{"name": "rsi", "version": 1, "params": {"window": 15, "x": 1}}]})


def test_shipped_definitions_are_valid_and_price_only():
    fsets, lspecs = load_definitions("config/feature_sets.yaml")
    assert {("swing", 1), ("invest", 1), ("invest", 2)} <= set(fsets) and {("swing", 1), ("invest", 1)} <= set(lspecs)
    for spec in fsets.values():
        assert all(f[0] != "fundamentals" for f in spec.features), "strategies must run on prices alone"
        assert spec.columns()
    assert lspecs[("swing", 1)].horizon() == 10 and lspecs[("invest", 1)].horizon() == 126


def test_swing_and_invest_sets_contain_what_the_brief_lists():
    fsets, _ = load_definitions("config/feature_sets.yaml")
    swing = set(fsets[("swing", 1)].columns())
    for c in ("ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "rsi_14", "macd_pct", "macd_hist_pct", "atr_pct_14", "bb_pctb_20",
              "don_hi_dist_20", "zscore_20", "gap_1", "vol_spike_20", "rs_5", "rs_10", "ret_5_csrank"):
        assert c in swing, c
    invest = set(fsets[("invest", 2)].columns())
    for c in ("mom_3m", "mom_6m", "mom_12m", "vol_63", "dbeta_126", "dd_252", "sma_ratio_200", "liq_trend_60_252", "liq_zero_share_60", "mom_6m_csrank"):
        assert c in invest, c
    assert "liq_logvalue_60" not in invest            # the price-level liquidity column is not point-in-time (see ts.Liquidity)
    assert "liq_logvalue_60" in set(fsets[("invest", 1)].columns())


# ---- data-side registry: no schema change is needed to add anything -------------------------------------------------
def test_sync_stores_sets_and_is_idempotent_and_versions_are_immutable(engine, tmp_path):
    fsets, lspecs = load_definitions("config/feature_sets.yaml")
    with session_scope(engine) as s:
        assert sync_definitions(s, fsets, lspecs) == {"feature_sets": 3, "label_specs": 2}
    with session_scope(engine) as s:
        assert sync_definitions(s, fsets, lspecs) == {"feature_sets": 0, "label_specs": 0}
        row = s.scalars(select(FeatureSet).where(FeatureSet.name == "swing")).one()
        assert row.spec["features"][0]["name"] == "ret"
        assert s.scalars(select(LabelSpec).where(LabelSpec.name == "swing")).one().horizon_days == 10
    changed = tmp_path / "fs.yaml"
    changed.write_text("feature_sets:\n  swing:\n    version: 1\n    features:\n      - {name: rsi, version: 1}\nlabel_specs: {}\n")
    f2, _ = load_definitions(changed)
    with session_scope(engine) as s:
        with pytest.raises(RegistryError, match="different content; bump the version"):
            sync_definitions(s, f2, {})
    bumped = tmp_path / "fs2.yaml"
    bumped.write_text(changed.read_text().replace("version: 1\n    features", "version: 2\n    features"))
    f3, _ = load_definitions(bumped)
    with session_scope(engine) as s:
        assert sync_definitions(s, f3, {}) == {"feature_sets": 1, "label_specs": 0}          # a new version is just a new row


def test_adding_a_feature_does_not_change_the_schema(engine, throwaway):
    def tables():
        with engine.connect() as c:
            return sorted(r[0] for r in c.execute(text("SHOW TABLES")))
    before = tables()
    class Extra(TimeSeriesFeature):
        name, version = "extra_feat", 1
        def columns(self): return ["extra_a"]
        def compute(self, bars, bench): return pd.DataFrame({"extra_a": bars["close"] * 2}, index=bars.index)
    throwaway(Extra)
    spec = FeatureSetSpec.from_dict("with_extra", {"version": 1, "features": [{"name": "ret", "version": 1}, {"name": "extra_feat", "version": 1}]})
    with session_scope(engine) as s:
        sync_definitions(s, {("with_extra", 1): spec}, {})
        assert s.scalar(select(func.count()).select_from(FeatureSet)) == 1
    assert tables() == before


def test_definition_keys_are_name_colon_version(tmp_path):
    ok = tmp_path / "ok.yaml"
    ok.write_text("feature_sets:\n  \"a:1\": {version: 1, features: [{name: rsi, version: 1}]}\n  \"a:2\": {version: 2, features: [{name: rsi, version: 1, params: {window: 10}}]}\n  b: {version: 3, features: [{name: gap, version: 1}]}\n")
    f, _ = load_definitions(ok)
    assert sorted(f) == [("a", 1), ("a", 2), ("b", 3)] and f[("a", 2)].columns() == ["rsi_10"]
    bad = tmp_path / "bad.yaml"
    bad.write_text("feature_sets:\n  \"a:1\": {version: 2, features: [{name: rsi, version: 1}]}\n")
    with pytest.raises(RegistryError, match="key says version 1"):
        load_definitions(bad)
    dup = tmp_path / "dup.yaml"
    dup.write_text("feature_sets:\n  \"a:1\": {version: 1, features: [{name: rsi, version: 1}]}\n  a: {version: 1, features: [{name: gap, version: 1}]}\n")
    with pytest.raises(RegistryError, match="defined twice"):
        load_definitions(dup)
