"""build_dataset: features + labels for a universe over a date range -> Parquet snapshot + manifest in ``datasets``.

Reproducibility
* The dataset is a pure function of (universe membership, price bars, feature set spec, label spec, options).
  Those are hashed into ``config_hash`` (plus ``inputs_hash`` of the actual bars used).
* Same config and same data -> byte-identical Parquet -> same ``sha256``; the existing row is reused and nothing is
  written. If the data changed (e.g. the vendor re-adjusted prices) the same config produces a NEW version and the
  old one is kept.
* ``data_cutoff`` (default: the latest calendar day) pins how much future data labels may see; it is part of the hash.

Rows: one per (trade_date, instrument_id) where the instrument is a MEMBER of the universe on that date, has a bar and
has the feature set's warm-up of history. ``instrument_id`` and ``trade_date`` are keys, never features: the manifest
lists ``feature_columns`` and ``label_columns`` explicitly.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from predict_stock.config import PROJECT_ROOT, AppConfig
from predict_stock.data.backfill import readiness
from predict_stock.db.models import Dataset, FeatureSet, LabelSpec, TradingCalendar, Universe
from predict_stock.db.session import session_scope
from predict_stock.features.audit import audit_label_horizon, audit_no_lookahead, pick_cut_dates
from predict_stock.features.data import Panel, load_panel, membership_mask
from predict_stock.features.engine import KEY_COLUMNS, assemble_frame, required_warmup
from predict_stock.features.registry import FeatureSetSpec, LabelSpecSpec, canonical_json, spec_hash
from predict_stock.features.sets import load_definitions, sync_definitions
from predict_stock.runs import git_state, save_config_snapshot, tracked_run
from predict_stock.universe import get_instruments_between

BUILDER_VERSION = 1  # bump when the assembly semantics change (part of the config hash)


class DatasetError(RuntimeError):
    pass


@dataclass
class DatasetResult:
    dataset_id: int
    name: str
    version: int
    path: Path
    sha256: str
    content_hash: str
    rows: int
    reused: bool
    manifest: dict


def _sha256_bytes(*chunks: bytes) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def _canonical(series: pd.Series) -> tuple[str, np.ndarray]:
    """(kind, values) that do not depend on how pandas/Parquet happen to store the column (datetime unit, int width)."""
    v = series.to_numpy()
    if np.issubdtype(v.dtype, np.datetime64):
        return "datetime", v.astype("datetime64[ns]").astype("int64")       # NaT -> one fixed integer
    if np.issubdtype(v.dtype, np.floating):
        return "float", np.where(np.isnan(v), np.nan, v).astype("float64")  # one NaN bit pattern
    if np.issubdtype(v.dtype, np.integer):
        return "int", v.astype("int64")
    raise DatasetError(f"column {series.name!r} has unsupported dtype {v.dtype}")


def hash_frame(frame: pd.DataFrame) -> str:
    """Content hash independent of the file format and of storage details: column names, kinds and every value."""
    h = hashlib.sha256()
    parts = [(c, *_canonical(frame[c])) for c in frame.columns]
    h.update(canonical_json([[c, kind] for c, kind, _ in parts]).encode())
    for _, _, v in parts:
        h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


def hash_inputs(panel: Panel, mask: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for name in ("open", "high", "low", "close", "volume"):
        h.update(np.ascontiguousarray(getattr(panel, name).to_numpy(float)).tobytes())
    h.update(np.ascontiguousarray(mask.to_numpy(bool)).tobytes())
    if panel.bench is not None:
        h.update(np.ascontiguousarray(panel.bench.to_numpy(float)).tobytes())
    h.update(np.asarray(panel.instrument_ids, dtype="int64").tobytes())
    return h.hexdigest()


def dataset_name(universe: str, fs: FeatureSetSpec, ls: LabelSpecSpec) -> str:
    return f"{universe}__{fs.name}{fs.version}__{ls.name}{ls.version}"[:64]


def _write_parquet(frame: pd.DataFrame, path: Path) -> str:
    """Write atomically; returns the file's sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(tmp, index=False, engine="pyarrow", compression="zstd")
        sha = _sha256_bytes(Path(tmp).read_bytes())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return sha


def file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def build_dataset(
    engine: Engine,
    cfg: AppConfig,
    *,
    universe_code: str | None = None,
    feature_set: tuple[str, int],
    label_spec: tuple[str, int],
    start: date,
    end: date,
    data_cutoff: date | None = None,
    name: str | None = None,
    out_dir: str | Path | None = None,
    provider=None,
    definitions_path: str | Path | None = None,
) -> DatasetResult:
    """Build (or reproduce) a dataset. See the module docstring for the semantics."""
    universe_code = universe_code or cfg.universe.training_code
    fsets, lspecs = load_definitions(definitions_path or PROJECT_ROOT / cfg.features.definitions_path)
    if feature_set not in fsets or label_spec not in lspecs:
        raise DatasetError(f"unknown feature set {feature_set} or label spec {label_spec}; known: {sorted(fsets)} / {sorted(lspecs)}")
    fs, ls = fsets[feature_set], lspecs[label_spec]
    features, labels = fs.instantiate(), ls.instantiate()
    name = name or dataset_name(universe_code, fs, ls)
    if len(name) > 64:
        raise DatasetError("dataset name must be at most 64 characters")
    fcols, lcols = fs.columns(), ls.columns()
    out_root = Path(out_dir) if out_dir else PROJECT_ROOT / cfg.datasets.dir
    fcfg = cfg.features
    # The config flag is the master switch for fundamentals: a provider only counts when it is enabled there, and
    # features that need one fail loudly (FundamentalsDisabled) otherwise, so a price-only run can never silently
    # contain half-empty fundamental columns.
    if fcfg.fundamentals.enabled and provider is None:
        raise DatasetError("features.fundamentals.enabled is true but no FundamentalProvider was supplied")
    provider = provider if fcfg.fundamentals.enabled else None

    params = {"universe": universe_code, "feature_set": list(feature_set), "label_spec": list(label_spec), "name": name,
              "start": start.isoformat(), "end": end.isoformat(), "data_cutoff": data_cutoff.isoformat() if data_cutoff else None}
    with tracked_run(engine, "build_dataset", cfg, params) as (run_id, stats):
        with session_scope(engine) as s:
            sync_definitions(s, fsets, lspecs)
            uni = s.scalars(select(Universe).where(Universe.code == universe_code)).first()
            if uni is None:
                raise DatasetError(f"unknown universe {universe_code!r}")
            members = get_instruments_between(s, universe_code, start, end)
            if not members:
                raise DatasetError(f"universe {universe_code!r} has no members in {start}..{end}")
            not_ready = readiness(s, cfg, members, end) if cfg.datasets.require_ready else {}
            ids = [m.instrument_id for m in members if m.symbol not in not_ready]
            symbol_of = {m.instrument_id: m.symbol for m in members}
            last_day = s.scalar(select(func.max(TradingCalendar.trade_date)).where(TradingCalendar.calendar_code == cfg.ingest.calendar_code))
            cutoff = min(data_cutoff, last_day) if data_cutoff else last_day
            warm = required_warmup(features)
            # ALL history is loaded, not just the warm-up: recursive features (RSI, ATR, MACD use EWM) depend on the whole
            # past, so a feature value must not change with the ``start`` the caller happened to choose.
            load_start = date.fromisoformat(cfg.ingest.history_start)
            panel = load_panel(s, calendar_code=cfg.ingest.calendar_code, instrument_ids=ids, load_start=load_start, cutoff=cutoff,
                               bench_symbols=(fcfg.benchmark_symbol, fcfg.benchmark_fallback_symbol), bench_ffill_limit=fcfg.benchmark_ffill_limit)
            mask = membership_mask(s, universe_code, panel.calendar, ids)
            fs_row = s.scalars(select(FeatureSet).where(FeatureSet.name == fs.name, FeatureSet.version == fs.version)).one()
            ls_row = s.scalars(select(LabelSpec).where(LabelSpec.name == ls.name, LabelSpec.version == ls.version)).one()
            fs_id, ls_id, uni_id = fs_row.id, ls_row.id, uni.id
        if not ids:
            raise DatasetError(f"every member of {universe_code} is blocked for insufficient history: {not_ready}")

        # The range is clamped to the days that actually have data, so asking for "until today" on Sunday, or from 1990,
        # describes the same dataset as the real range does: the config hash (and the version) only change with the data.
        start = max(start, panel.calendar[0].date())
        end = min(end, cutoff)
        audit = None
        if cfg.datasets.audit_lookahead:
            cuts = pick_cut_dates(panel.calendar, start, end, cfg.datasets.audit_cuts)
            a = audit_no_lookahead(panel, mask, features, cuts, provider)
            b = audit_label_horizon(panel, mask, labels, cuts)
            audit = {"features": a.summary(), "labels": b.summary(), "passed": a.passed and b.passed}
            if not audit["passed"]:
                raise DatasetError(f"look-ahead audit FAILED, nothing written: {(a.violations + b.violations)[:5]}")

        frame = assemble_frame(panel, features, labels, mask, start, end, provider)
        if frame.empty:
            raise DatasetError("the dataset would have no rows (check the date range, membership and warm-up)")
        content_hash = hash_frame(frame)
        inputs_hash = hash_inputs(panel, mask)
        config = {"builder": BUILDER_VERSION, **params, "start": start.isoformat(), "end": end.isoformat(), "cutoff": str(cutoff), "feature_spec": fs.to_spec(), "label_spec_body": ls.to_spec(),
                  "instruments": sorted(ids), "benchmark": [fcfg.benchmark_symbol, fcfg.benchmark_fallback_symbol, fcfg.benchmark_ffill_limit],
                  "require_ready": cfg.datasets.require_ready, "min_sessions": cfg.ingest.min_sessions, "calendar": cfg.ingest.calendar_code}
        config.pop("data_cutoff")
        config_hash = spec_hash(config)

        nan_share = {c: round(float(frame[c].isna().mean()), 4) for c in fcols + lcols}
        manifest = {
            "config_hash": config_hash, "content_hash": content_hash, "inputs_hash": inputs_hash, "builder_version": BUILDER_VERSION,
            "universe": universe_code, "start": start.isoformat(), "end": end.isoformat(), "data_cutoff": str(cutoff),
            "feature_set": {"name": fs.name, "version": fs.version, "spec_hash": spec_hash(fs.to_spec())},
            "label_spec": {"name": ls.name, "version": ls.version, "spec_hash": spec_hash(ls.to_spec()), "horizon": ls.horizon()},
            "key_columns": KEY_COLUMNS, "feature_columns": fcols, "label_columns": lcols,
            "label_end_columns": [c for c in lcols if c.startswith(("fwd_end_", "tb_end"))],   # for purging / embargo
            "rows": int(len(frame)), "dates": int(frame["trade_date"].nunique()), "instruments": int(frame["instrument_id"].nunique()),
            "first_date": str(frame["trade_date"].min().date()), "last_date": str(frame["trade_date"].max().date()),
            "warmup_sessions": warm, "nan_share": nan_share,
            "excluded_not_ready": not_ready, "symbols": sorted(symbol_of[i] for i in ids),
            "policies": {"repaired_bars": panel.repaired_bars, "dropped_non_calendar_bars": panel.dropped_non_calendar,
                         "repair_rule": "high=max(high,low,close); low=min(low,close,high); open=NaN if outside range; non-positive price -> empty bar",
                         "prices": "vendor back-adjusted (ratios only; liq_logvalue_* is level-based)",
                         "label_entry": "close of the decision date; costs and slippage are applied by the backtester",
                         "barrier_tie": "stop wins when both barriers are touched in one session; gap-through exits at the open"},
            "benchmark": panel.bench_info, "lookahead_audit": audit,
            "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "pyarrow": pyarrow.__version__},
            "run_id": run_id,
        }

        with session_scope(engine) as s:
            rows = list(s.scalars(select(Dataset).where(Dataset.name == name).order_by(Dataset.version)))
            same_cfg = [r for r in rows if (r.manifest or {}).get("config_hash") == config_hash]
            reused = None
            for r in same_cfg:
                path = PROJECT_ROOT / r.path if not Path(r.path).is_absolute() else Path(r.path)
                if (r.manifest or {}).get("content_hash") == content_hash:
                    if not path.exists() or file_sha256(path) != r.sha256:   # row is right, file lost or damaged: restore it
                        if _write_parquet(frame, path) != r.sha256:
                            raise DatasetError("rewritten file does not match the recorded sha256 (different library version?)")
                    reused = r
                    break
            if reused is not None:
                stats.update(reused=True, dataset_id=reused.id, version=reused.version)
                return DatasetResult(reused.id, name, reused.version, PROJECT_ROOT / reused.path, reused.sha256, content_hash,
                                     reused.row_count, True, reused.manifest)
            version = (rows[-1].version + 1) if rows else 1
            path = out_root / name / f"v{version}" / "dataset.parquet"
            sha = _write_parquet(frame, path)
            commit, dirty = git_state()
            manifest.update(file_sha256=sha, version=version, git_commit=commit, git_dirty=dirty,
                            built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            superseded_version=same_cfg[-1].version if same_cfg else None)
            try:
                stored_path = str(path.relative_to(PROJECT_ROOT))
            except ValueError:
                stored_path = str(path)
            row = Dataset(name=name, version=version, universe_id=uni_id, start_date=start, end_date=end, feature_set_id=fs_id,
                          label_spec_id=ls_id, path=stored_path, sha256=sha, row_count=len(frame), manifest=manifest,
                          config_snapshot_id=save_config_snapshot(s, cfg.snapshot()), git_commit=commit)
            s.add(row)
            s.flush()
            stats.update(reused=False, dataset_id=row.id, version=version, rows=len(frame), sha256=sha)
            return DatasetResult(row.id, name, version, path, sha, content_hash, len(frame), False, manifest)


def read_dataset(path: str | Path, *, verify_sha256: str | None = None) -> pd.DataFrame:
    p = Path(path)
    p = p if p.is_absolute() else PROJECT_ROOT / p
    if verify_sha256 is not None and file_sha256(p) != verify_sha256:
        raise DatasetError(f"{p}: sha256 does not match the recorded value (file changed or damaged)")
    return pd.read_parquet(p, engine="pyarrow")


def load_dataset(session: Session, name: str, version: int | None = None) -> tuple[pd.DataFrame, dict]:
    """(frame, manifest) of a registered dataset (latest version by default), verifying the file's sha256."""
    q = select(Dataset).where(Dataset.name == name)
    q = q.where(Dataset.version == version) if version else q.order_by(Dataset.version.desc()).limit(1)
    row = session.scalars(q).first()
    if row is None:
        raise DatasetError(f"no dataset {name!r}" + (f" v{version}" if version else ""))
    return read_dataset(row.path, verify_sha256=row.sha256), row.manifest
