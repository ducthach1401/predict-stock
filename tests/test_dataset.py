"""build_dataset end to end: Parquet + manifest, reproducibility, versions, membership, audit, fundamentals."""
from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select, text

import predict_stock.cli as cli
from conftest import FakeClient, apply, make_bars
from predict_stock.config import load_config
from predict_stock.data.ingest import ingest_universe
from predict_stock.db.models import Dataset, FeatureSet, JobRun, LabelSpec
from predict_stock.db.repo import find_symbol_row
from predict_stock.db.session import session_scope
from predict_stock.features.dataset import DatasetError, build_dataset, file_sha256, hash_frame, load_dataset, read_dataset
from predict_stock.features.fundamentals import FundamentalsDisabled
from predict_stock.features.registry import TimeSeriesFeature, register_feature, unregister

START, END = date(2026, 1, 5), date(2026, 9, 18)
N = 170                                                # weekdays from 2026-01-05, all before the simulated "today" (2026-09-18)
DEFS = """
feature_sets:
  mini:
    version: 1
    features:
      - {name: ret, version: 1, params: {windows: [1, 5]}}
      - {name: rsi, version: 1, params: {window: 14}}
      - {name: rel_strength, version: 1, params: {windows: [5]}}
      - {name: cs_rank, version: 1, params: {inputs: [ret_5, rsi_14], min_count: 3}}
label_specs:
  mini:
    version: 1
    labels:
      - {name: fwd_rank_return, version: 1, params: {horizons: [3], min_count: 3}}
      - {name: triple_barrier, version: 1, params: {horizon: 5}}
"""
WARMUP = 15                                          # rsi_14 -> 15 own bars


def wobbly(n, base, seed):
    """A bumpy but deterministic price path (make_bars is monotonic, which would make every rank trivial)."""
    rng = np.random.default_rng(seed)
    bars = make_bars(START, n, base=base)
    px = base * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    from dataclasses import replace
    from decimal import Decimal
    out = []
    for i, b in enumerate(bars):
        c = Decimal(str(round(px[i], 2)))
        o = Decimal(str(round(px[i - 1], 2))) if i else c
        out.append(replace(b, open=o, high=max(o, c) * Decimal("1.01"), low=min(o, c) * Decimal("0.99"), close=c, volume=1_000_000 + 1000 * (i % 7)))
    return out


@pytest.fixture
def world(engine, cfg, after_close, tmp_path):
    """Six stocks + an index. A,B,C,D always members; E joins on day 100; F is removed on day 150."""
    days = [b.trade_date for b in make_bars(START, N)]
    series = {s: wobbly(N, 20000 + 7000 * i, i + 1) for i, s in enumerate("ABCDEF")}
    series["IDX"] = wobbly(N, 1200, 99)
    apply(engine, "TRAIN", ["A", "B", "C", "D", "F"], days[0])
    apply(engine, "TRAIN", ["A", "B", "C", "D", "E", "F"], days[100])
    apply(engine, "TRAIN", ["A", "B", "C", "D", "E"], days[150])
    c = cfg.model_copy(update={
        "ingest": cfg.ingest.model_copy(update={"benchmark_symbols": ["IDX"], "min_sessions": 30}),
        "features": cfg.features.model_copy(update={"benchmark_symbol": "IDX"}),
        "universe": cfg.universe.model_copy(update={"training_code": "TRAIN", "trading_code": "TRAIN"}),
        "datasets": cfg.datasets.model_copy(update={"dir": str(tmp_path / "ds")}),
    })
    ingest_universe(engine, FakeClient(series), c, ["TRAIN"], START, END, now=after_close)
    defs = tmp_path / "defs.yaml"
    defs.write_text(DEFS)
    with session_scope(engine) as s:
        ids = {k: find_symbol_row(s, k).instrument_id for k in "ABCDEFI" if k != "I"}
    return {"cfg": c, "days": days, "defs": defs, "ids": ids, "series": series, "engine": engine, "out": tmp_path / "ds"}


def build(w, **kw):
    args = dict(feature_set=("mini", 1), label_spec=("mini", 1), start=w["days"][0], end=w["days"][-1], definitions_path=w["defs"], out_dir=w["out"])
    return build_dataset(w["engine"], w["cfg"], **{**args, **kw})


def n_rows(engine, model=Dataset):
    with session_scope(engine) as s:
        return s.scalar(select(func.count()).select_from(model))


# ---- basic build ----------------------------------------------------------------------------------------------
def test_build_writes_parquet_and_registers_the_manifest(world):
    res = build(world)
    assert not res.reused and res.version == 1 and res.path.exists() and res.name == "TRAIN__mini1__mini1"
    assert res.sha256 == file_sha256(res.path)
    frame = read_dataset(res.path, verify_sha256=res.sha256)
    assert len(frame) == res.rows and hash_frame(frame) == res.content_hash
    m = res.manifest
    assert m["feature_columns"] == ["ret_1", "ret_5", "rsi_14", "rs_5", "ret_5_csrank", "rsi_14_csrank"]
    assert m["label_columns"] == ["fwd_ret_3", "fwd_rank_3", "fwd_end_3", "tb_label", "tb_time", "tb_ret", "tb_end"]
    assert set(frame.columns) == set(m["key_columns"]) | set(m["feature_columns"]) | set(m["label_columns"])
    assert set(m["key_columns"]).isdisjoint(m["feature_columns"]) and set(m["key_columns"]).isdisjoint(m["label_columns"])
    assert m["label_end_columns"] == ["fwd_end_3", "tb_end"] and m["rows"] == len(frame) and m["instruments"] == 6
    assert m["symbols"] == list("ABCDEF") and m["universe"] == "TRAIN" and m["warmup_sessions"] == WARMUP
    assert m["lookahead_audit"]["passed"] is True and m["lookahead_audit"]["features"]["columns_checked"] > 0
    assert m["policies"]["repair_rule"] and m["benchmark"]["primary"] == "IDX" and m["environment"]["pyarrow"]
    assert frame["trade_date"].is_monotonic_increasing and not frame.duplicated(["trade_date", "instrument_id"]).any()
    with session_scope(world["engine"]) as s:
        row = s.scalars(select(Dataset)).one()
        assert (row.name, row.version, row.row_count, row.sha256, row.path) == (res.name, 1, len(frame), res.sha256, str(res.path))
        assert row.start_date == world["days"][0] and row.end_date == world["days"][-1] and row.manifest["config_hash"] == m["config_hash"]
        assert s.get(FeatureSet, row.feature_set_id).name == "mini" and s.get(LabelSpec, row.label_spec_id).horizon_days == 5
        assert row.config_snapshot_id is not None and s.scalars(select(JobRun).where(JobRun.job_name == "build_dataset")).one().status == "success"


def test_the_feature_columns_never_identify_a_stock(world):
    frame = build(world).manifest
    for c in frame["feature_columns"] + frame["label_columns"]:
        assert not any(w in c.lower() for w in ("symbol", "ticker", "instrument", "exchange")) and c != "id"
    df = read_dataset(build(world).path)
    for c in frame["feature_columns"]:                          # and no feature is a relabelled id: it varies within an instrument
        assert df.groupby("instrument_id")[c].nunique().min() > 1


def test_rows_are_exactly_members_with_a_bar_and_enough_history(world):
    df = read_dataset(build(world).path)
    days, ids = world["days"], world["ids"]
    interval = {"A": (0, N), "B": (0, N), "C": (0, N), "D": (0, N), "E": (100, N), "F": (0, 150)}
    for sym, (lo, hi) in interval.items():
        got = sorted(df[df["instrument_id"] == ids[sym]]["trade_date"].dt.date)
        first = max(lo, WARMUP - 1)                              # a member from day `lo`, but needs 15 own bars
        assert got == days[first:hi], sym
    e = df[df["instrument_id"] == ids["E"]].iloc[0]
    assert e["trade_date"].date() == days[100] and not np.isnan(e["rsi_14"]) and not np.isnan(e["ret_5"])   # history from before it joined


def test_cross_sectional_ranks_use_that_days_members_only(world):
    df = read_dataset(build(world).path)
    check = df.groupby("trade_date")["ret_5"].rank(pct=True)
    n_members = df.groupby("trade_date")["ret_5"].transform("count")
    ok = n_members >= 3
    assert np.allclose(df.loc[ok, "ret_5_csrank"], check[ok])
    assert df.loc[df["trade_date"] < pd.Timestamp(world["days"][100]), "instrument_id"].nunique() == 5      # E is absent before it joins
    after = df[df["trade_date"] >= pd.Timestamp(world["days"][150])]
    assert world["ids"]["F"] not in set(after["instrument_id"])                                              # F is gone after removal
    day = df[df["trade_date"] == pd.Timestamp(world["days"][120])]
    assert len(day) == 6 and set(np.round(day["fwd_rank_3"].dropna(), 6)) <= {round(k / 6, 6) for k in range(1, 7)}


def test_labels_are_known_only_when_their_window_fits(world):
    df = read_dataset(build(world).path)
    last = df[df["trade_date"] == pd.Timestamp(world["days"][-1])]
    assert last["fwd_ret_3"].isna().all() and last["tb_label"].isna().all()
    ok = df[df["trade_date"] <= pd.Timestamp(world["days"][-8])]
    assert ok["fwd_ret_3"].notna().all() and ok["tb_label"].notna().all()
    assert set(ok["tb_label"].unique()) <= {-1.0, 0.0, 1.0} and ((ok["tb_time"] >= 1) & (ok["tb_time"] <= 5)).all()
    assert (ok["tb_end"] > ok["trade_date"]).all() and (ok["fwd_end_3"] > ok["trade_date"]).all()          # outcomes lie in the future: purge by end date


# ---- reproducibility -----------------------------------------------------------------------------------------------------------
def test_same_config_and_data_reproduce_the_same_dataset_and_reuse_it(world):
    r1 = build(world)
    mtime = r1.path.stat().st_mtime_ns
    r2 = build(world)
    assert r2.reused and r2.dataset_id == r1.dataset_id and r2.version == 1 and r2.sha256 == r1.sha256 and r2.content_hash == r1.content_hash
    assert r1.path.stat().st_mtime_ns == mtime and n_rows(world["engine"]) == 1                                # nothing rewritten, no new row


def test_a_rebuild_from_nothing_gives_byte_identical_output(world):
    r1 = build(world)
    with world["engine"].begin() as c:                                                                        # forget the dataset entirely
        c.execute(text("DELETE FROM datasets"))
    os.remove(r1.path)
    r2 = build(world)
    assert not r2.reused and r2.version == 1
    assert (r2.sha256, r2.content_hash) == (r1.sha256, r1.content_hash) and r2.manifest["inputs_hash"] == r1.manifest["inputs_hash"]


def test_a_lost_file_is_restored_identically_and_a_damaged_one_is_detected(world):
    r1 = build(world)
    os.remove(r1.path)
    r2 = build(world)
    assert r2.reused and r1.path.exists() and file_sha256(r1.path) == r1.sha256                              # restored, same bytes
    with open(r1.path, "ab") as f:
        f.write(b"tamper")
    with session_scope(world["engine"]) as s:
        with pytest.raises(DatasetError, match="sha256 does not match"):
            load_dataset(s, r1.name)
    build(world)                                                                                             # rebuilding repairs the damaged file
    with session_scope(world["engine"]) as s:
        frame, manifest = load_dataset(s, r1.name)
    assert len(frame) == r1.rows and manifest["config_hash"] == r1.manifest["config_hash"]


def test_changed_data_gives_a_new_version_and_keeps_the_old_one(world):
    r1 = build(world)
    with world["engine"].begin() as c:                                                                        # e.g. the vendor re-adjusted a bar
        c.execute(text("UPDATE price_bar SET close = close * 1.05 WHERE trade_date = :d AND instrument_id = :i"), {"d": world["days"][60], "i": world["ids"]["B"]})
    r2 = build(world)
    assert not r2.reused and r2.version == 2 and r2.sha256 != r1.sha256 and r2.manifest["inputs_hash"] != r1.manifest["inputs_hash"]
    assert r2.manifest["config_hash"] == r1.manifest["config_hash"] and r2.manifest["superseded_version"] == 1     # same recipe, new data
    assert r1.path.exists() and r2.path.exists() and r1.path != r2.path and n_rows(world["engine"]) == 2
    with session_scope(world["engine"]) as s:
        assert load_dataset(s, r1.name, 1)[0].equals(read_dataset(r1.path)) and len(load_dataset(s, r1.name)[0]) == r2.rows                # latest by default
    assert build(world).reused                                                                                # and the new state is stable


def test_asking_for_more_days_than_exist_describes_the_same_dataset(world):
    """`--end` defaults to today; on a day without new bars that must not create a new version."""
    r1 = build(world)
    r2 = build(world, start=date(1990, 1, 1), end=date(2035, 1, 1))
    assert r2.reused and r2.version == 1 and r2.dataset_id == r1.dataset_id and n_rows(world["engine"]) == 1
    with session_scope(world["engine"]) as s:
        row = s.scalars(select(Dataset)).one()
        assert (row.start_date, row.end_date) == (world["days"][0], world["days"][-1])           # the real range is what is recorded


def test_a_different_recipe_is_a_different_dataset(world):
    r1 = build(world)
    r2 = build(world, end=world["days"][-20], name="other")
    r3 = build(world, data_cutoff=world["days"][-30])
    assert len({r1.manifest["config_hash"], r2.manifest["config_hash"], r3.manifest["config_hash"]}) == 3
    assert r3.version == 2 and r3.manifest["data_cutoff"] == str(world["days"][-30])
    a, b = read_dataset(r1.path), read_dataset(r3.path)
    assert b["trade_date"].max().date() == world["days"][-30] and len(b) < len(a)                              # no data after the cutoff: no rows either
    assert b[b["trade_date"] > pd.Timestamp(world["days"][-36])]["fwd_ret_3"].isna().any()                     # and labels near it are unknown, not guessed


def test_feature_values_do_not_depend_on_the_start_date_chosen(world):
    early = read_dataset(build(world, name="early").path)
    late = read_dataset(build(world, name="late", start=world["days"][60]).path)
    key = ["trade_date", "instrument_id"]
    a, b = early.set_index(key).sort_index(), late.set_index(key).sort_index()
    common = b.index
    assert len(late) < len(early)
    feats = ["ret_1", "ret_5", "rsi_14", "rs_5", "ret_5_csrank", "rsi_14_csrank", "fwd_ret_3", "tb_label"]
    assert np.allclose(a.loc[common, feats].to_numpy(float), b[feats].to_numpy(float), equal_nan=True)         # recursive features (RSI/ATR) identical


# ---- guards -------------------------------------------------------------------------------------------------------------------------
def test_instruments_without_enough_history_are_left_out_with_the_reason(world, engine):
    apply(engine, "TRAIN", ["A", "B", "C", "D", "E", "G"], world["days"][160])
    short = make_bars(world["days"][150], 20, base=9000)
    world["series"]["G"] = short
    ingest_universe(engine, FakeClient(world["series"]), world["cfg"], ["TRAIN"], START, END, now=pd.Timestamp("2026-09-18 09:00", tz="UTC").to_pydatetime())
    r = build(world)
    assert "G" in r.manifest["excluded_not_ready"] and "only 20 sessions" in r.manifest["excluded_not_ready"]["G"]
    assert "G" not in r.manifest["symbols"]
    off = world["cfg"].model_copy(update={"datasets": world["cfg"].datasets.model_copy(update={"require_ready": False})})
    r2 = build_dataset(engine, off, feature_set=("mini", 1), label_spec=("mini", 1), start=world["days"][0], end=world["days"][-1],
                       definitions_path=world["defs"], out_dir=world["out"], name="with_g")
    assert "G" in r2.manifest["symbols"]


def test_errors_are_explicit_and_write_nothing(world):
    for kw, msg in [(dict(universe_code="NOPE"), "unknown universe"), (dict(feature_set=("mini", 9)), "unknown feature set"),
                    (dict(label_spec=("zzz", 1)), r"unknown feature set .* or label spec"), (dict(name="x" * 65), "at most 64"),
                    (dict(start=date(2010, 1, 1), end=date(2010, 2, 1)), "no members")]:
        with pytest.raises(DatasetError, match=msg):
            build(world, **kw)
    assert n_rows(world["engine"]) == 0 and not world["out"].exists()


def test_a_leaky_feature_stops_the_build_and_nothing_is_written(world, tmp_path):
    class Peek(TimeSeriesFeature):
        name, version = "peek_ahead", 1
        def columns(self): return ["peek_out"]
        def compute(self, bars, bench): return pd.DataFrame({"peek_out": bars["close"].shift(-1) / bars["close"] - 1}, index=bars.index)
    register_feature(Peek)
    try:
        defs = tmp_path / "leaky.yaml"
        defs.write_text(DEFS.replace("      - {name: rsi, version: 1, params: {window: 14}}", "      - {name: rsi, version: 1, params: {window: 14}}\n      - {name: peek_ahead, version: 1}"))
        with pytest.raises(DatasetError, match="look-ahead audit FAILED"):
            build(world, definitions_path=defs)
        assert n_rows(world["engine"]) == 0 and not world["out"].exists()
        with session_scope(world["engine"]) as s:
            assert s.scalars(select(JobRun).where(JobRun.job_name == "build_dataset")).one().status == "failed"
    finally:
        unregister("feature", "peek_ahead", 1)


def test_the_audit_can_be_switched_off_but_is_then_not_claimed(world):
    off = world["cfg"].model_copy(update={"datasets": world["cfg"].datasets.model_copy(update={"audit_lookahead": False})})
    r = build_dataset(world["engine"], off, feature_set=("mini", 1), label_spec=("mini", 1), start=world["days"][0], end=world["days"][-1],
                      definitions_path=world["defs"], out_dir=world["out"])
    assert r.manifest["lookahead_audit"] is None


# ---- fundamentals: off by default, price-only strategies are complete without them ----------------------------------------------------------------
class FakeProvider:
    """Point-in-time by contract: returns the latest value PUBLISHED on or before `when`."""

    enabled = True

    def __init__(self, table):                                    # {instrument_id: [(publish_date, value), ...]}
        self.table, self.calls = table, []

    def as_of(self, when, instrument_ids):
        self.calls.append(pd.Timestamp(when))
        rows = {}
        for iid in instrument_ids:
            known = [v for d, v in self.table.get(iid, []) if pd.Timestamp(d) <= when]
            rows[iid] = {"pe": known[-1] if known else np.nan}
        return pd.DataFrame.from_dict(rows, orient="index")


FUND_DEFS = DEFS.replace("      - {name: rsi, version: 1, params: {window: 14}}", "      - {name: rsi, version: 1, params: {window: 14}}\n      - {name: fundamentals, version: 1, params: {fields: [pe]}}")


def test_fundamentals_are_off_by_default_and_the_shipped_sets_need_none(world, cfg):
    assert cfg.features.fundamentals.enabled is False
    r = build(world)                                                                # a complete dataset from prices alone
    assert not any(c.startswith("fund_") for c in r.manifest["feature_columns"])


def test_a_set_that_needs_fundamentals_fails_loudly_when_they_are_disabled(world, tmp_path):
    defs = tmp_path / "f.yaml"
    defs.write_text(FUND_DEFS)
    with pytest.raises(FundamentalsDisabled, match="no fundamental provider is enabled"):
        build(world, definitions_path=defs)
    assert n_rows(world["engine"]) == 0
    on = world["cfg"].model_copy(update={"features": world["cfg"].features.model_copy(update={"fundamentals": world["cfg"].features.fundamentals.model_copy(update={"enabled": True})})})
    with pytest.raises(DatasetError, match="no FundamentalProvider was supplied"):
        build_dataset(world["engine"], on, feature_set=("mini", 1), label_spec=("mini", 1), start=world["days"][0], end=world["days"][-1], definitions_path=defs, out_dir=world["out"])


def test_an_enabled_provider_is_used_point_in_time(world, tmp_path):
    defs = tmp_path / "f.yaml"
    defs.write_text(FUND_DEFS)
    a = world["ids"]["A"]
    table = {a: [(world["days"][50], 10.0), (world["days"][120], 20.0)]}          # two publications; the second is later
    provider = FakeProvider(table)
    on = world["cfg"].model_copy(update={"features": world["cfg"].features.model_copy(update={"fundamentals": world["cfg"].features.fundamentals.model_copy(update={"enabled": True})})})
    r = build_dataset(world["engine"], on, feature_set=("mini", 1), label_spec=("mini", 1), start=world["days"][0], end=world["days"][-1],
                      definitions_path=defs, out_dir=world["out"], provider=provider)
    df = read_dataset(r.path)
    assert "fund_pe" in r.manifest["feature_columns"]
    ra = df[df["instrument_id"] == a].set_index("trade_date")["fund_pe"]
    assert ra[pd.Timestamp(world["days"][49])] != ra[pd.Timestamp(world["days"][49])]                          # nothing published yet: NaN
    assert ra[pd.Timestamp(world["days"][50])] == 10.0 and ra[pd.Timestamp(world["days"][119])] == 10.0        # the old value until the new one is published
    assert ra[pd.Timestamp(world["days"][120])] == 20.0
    assert max(provider.calls) <= pd.Timestamp(world["days"][-1])                                              # never asked about a date after the dataset's range


# ---- CLI ---------------------------------------------------------------------------------------------------------------------------------------------
def test_cli_builds_lists_and_verifies_a_dataset(world, monkeypatch, capsys):
    monkeypatch.setenv("MYSQL_DATABASE", "predict_stock_test")
    monkeypatch.setattr(cli, "load_config", lambda _p=None: world["cfg"].model_copy(update={"features": world["cfg"].features.model_copy(update={"definitions_path": str(world["defs"])})}))
    argv = ["dataset", "build", "--feature-set", "mini:1", "--label-spec", "mini:1", "--start", str(world["days"][0]), "--end", str(world["days"][-1])]
    assert cli.main(argv) == 0 and "BUILT TRAIN__mini1__mini1 v1" in capsys.readouterr().out
    assert cli.main(argv) == 0 and "REUSED (identical)" in capsys.readouterr().out
    assert cli.main(["dataset", "list"]) == 0 and "TRAIN__mini1__mini1 v1" in capsys.readouterr().out
    assert cli.main(["dataset", "verify", "--name", "TRAIN__mini1__mini1"]) == 0 and "sha256 verified" in capsys.readouterr().out
    assert cli.main(["dataset", "build", "--feature-set", "nope:1", "--label-spec", "mini:1"]) == 2
    assert cli.main(["features", "sync"]) == 0
