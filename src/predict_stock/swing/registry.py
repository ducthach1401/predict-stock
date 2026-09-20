"""Database side of the SWING model: experiments (pre-registration, Optuna trials, walk-forward, sensitivities), models and predictions."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.orm import Session

from predict_stock.config import PROJECT_ROOT, AppConfig
from predict_stock.db.models import Experiment, FeatureSet, LabelSpec, Model, Prediction
from predict_stock.db.session import session_scope
from predict_stock.runs import git_state, save_config_snapshot
from predict_stock.swing.model import SwingBundle

ALGO = "lightgbm_swing"
CHUNK = 4000


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---- experiments ---------------------------------------------------------------------------------------------------------------
def log_experiment(engine: Engine, cfg: AppConfig, name: str, *, params: dict, summary: dict | None = None, status: str = "success",
                   description: str | None = None, run_id: int | None = None, key: str | None = None) -> int:
    """One ``experiments`` row. ``key`` (a hash of what makes the run what it is) makes it idempotent: a row with the same name and
    ``params.key`` is refreshed in place instead of duplicated. Without a key a new row is always added."""
    with session_scope(engine) as s:
        snap = save_config_snapshot(s, cfg.snapshot())
        if key is not None:
            for prior in s.scalars(select(Experiment).where(Experiment.name == name).order_by(Experiment.id.desc())):
                if (prior.params or {}).get("key") == key:
                    prior.summary, prior.status, prior.finished_at = summary, status, _now()
                    return prior.id
        row = Experiment(name=name, description=description, run_id=run_id, config_snapshot_id=snap, params={**params, **({"key": key} if key else {})},
                         summary=summary, status=status, finished_at=_now() if status != "running" else None)
        s.add(row)
        s.flush()
        return row.id


def find_experiment(engine: Engine, name: str, key: str | None = None) -> Experiment | None:
    with session_scope(engine) as s:
        for row in s.scalars(select(Experiment).where(Experiment.name == name).order_by(Experiment.id)):
            if key is None or (row.params or {}).get("key") == key:
                s.expunge(row)
                return row
    return None


def trial_sink(engine: Engine, cfg: AppConfig, study_key: str, run_id: int | None):
    """A sink for ``tuning.run_study``: one row per trial, whatever its state."""
    def sink(rec: dict) -> None:
        status = {"COMPLETE": "success", "FAIL": "failed", "PRUNED": "pruned"}.get(rec["state"], rec["state"].lower())
        log_experiment(engine, cfg, f"swing:optuna:{study_key}:trial{rec['number']:03d}", key=f"{study_key}:{rec['number']}", status=status, run_id=run_id,
                       description=f"Optuna trial {rec['number']} ({rec['state']})",
                       params={"study": study_key, "trial": rec["number"], "hyperparameters": rec["params"]},
                       summary={"validation_rank_ic": rec["value"], "state": rec["state"], "error": rec["error"], "duration_s": rec["duration_s"]})
    return sink


# ---- models --------------------------------------------------------------------------------------------------------------------
def lookup_ids(session: Session, feature_set: str, label_spec: str) -> tuple[int, int]:
    fname, fver = feature_set.split(":")
    lname, lver = label_spec.split(":")
    fs = session.scalar(select(FeatureSet.id).where(FeatureSet.name == fname, FeatureSet.version == int(fver)))
    ls = session.scalar(select(LabelSpec.id).where(LabelSpec.name == lname, LabelSpec.version == int(lver)))
    if fs is None or ls is None:
        raise LookupError(f"feature set {feature_set} / label spec {label_spec} are not in the database (run `features sync`)")
    return fs, ls


def register_artifact(engine: Engine, cfg: AppConfig, raw: bytes, name: str, *, algo: str, feature_set: str, label_spec: str, dataset_id: int | None,
                      experiment_id: int | None, params: dict, seed: int, artifact_dir: Path | None = None, suffix: str = ".json.gz") -> tuple[int, int, str, bool]:
    """Save ``raw`` under ``<artifacts>/<name>/v<version><suffix>`` and register it in ``models`` (status ``candidate``).
    Returns (model_id, version, sha256, reused). If the latest version of ``name`` has the same sha256 it is reused; otherwise version + 1 and the older
    candidate versions are marked ``superseded`` (their files and predictions stay)."""
    sha = hashlib.sha256(raw).hexdigest()
    commit, _ = git_state()
    base = artifact_dir or PROJECT_ROOT / cfg.swing.artifacts_dir

    def write(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_bytes() != raw:
            path.write_bytes(raw)
    with session_scope(engine) as s:
        fs_id, ls_id = lookup_ids(s, feature_set, label_spec)
        latest = s.scalars(select(Model).where(Model.name == name).order_by(Model.version.desc())).first()
        if latest is not None and latest.artifact_sha256 == sha:
            path = base / name / f"v{latest.version}{suffix}"
            if not path.exists():
                write(path)
            return latest.id, latest.version, sha, True
        version = (latest.version + 1) if latest is not None else 1
        for old in s.scalars(select(Model).where(Model.name == name, Model.status == "candidate")):
            old.status = "superseded"
        path = base / name / f"v{version}{suffix}"
        write(path)
        try:
            shown = str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            shown = str(path)
        row = Model(name=name, version=version, algo=algo, feature_set_id=fs_id, label_spec_id=ls_id, dataset_id=dataset_id, experiment_id=experiment_id,
                    artifact_path=shown, artifact_sha256=sha, seed=seed, git_commit=commit, status="candidate", params=params)
        s.add(row)
        s.flush()
        return row.id, version, sha, False


def register_model(engine: Engine, cfg: AppConfig, bundle: SwingBundle, name: str, *, dataset_id: int | None, experiment_id: int | None,
                   extra_params: dict, artifact_dir: Path | None = None) -> tuple[int, int, str, bool]:
    """A SWING bundle as a model row (see ``register_artifact``)."""
    sw = cfg.swing
    params = {"tree_params": bundle.tree_params, "n_estimators": sw.n_estimators, "early_stopping_rounds": sw.early_stopping_rounds, "best_iteration": bundle.best_iteration,
              "calibration": bundle.calibration_used, "horizon": sw.horizon, "quantiles": sw.quantiles, "features": bundle.features, "meta": bundle.meta, **extra_params}
    return register_artifact(engine, cfg, bundle.to_bytes(), name, algo=ALGO, feature_set=sw.feature_set, label_spec=sw.label_spec, dataset_id=dataset_id,
                             experiment_id=experiment_id, params=params, seed=sw.seed, artifact_dir=artifact_dir)


def _rows(model_id: int, universe_id: int | None, horizon: int, frame: pd.DataFrame, run_id: int | None) -> list[dict]:
    out = []
    for r in frame.itertuples(index=False):
        out.append({"model_id": model_id, "instrument_id": int(r.instrument_id), "universe_id": universe_id, "as_of_date": pd.Timestamp(r.trade_date).date(),
                    "horizon_days": horizon, "score": float(r.score), "proba": None if pd.isna(r.proba) else float(r.proba),
                    "rank_in_universe": int(r.rank_in_universe), "details": r.details, "run_id": run_id})
    return out


def save_predictions(engine: Engine, model_id: int, universe_id: int | None, horizon: int, frame: pd.DataFrame, run_id: int | None = None) -> int:
    """Upsert the predictions of one model (unique per model, instrument, date and horizon). ``frame`` needs trade_date, instrument_id, score,
    proba, rank_in_universe and details (a dict per row). Returns the number of rows written."""
    rows = _rows(model_id, universe_id, horizon, frame, run_id)
    with session_scope(engine) as s:
        for i in range(0, len(rows), CHUNK):
            stmt = mysql_insert(Prediction).values(rows[i:i + CHUNK])
            s.execute(stmt.on_duplicate_key_update(score=stmt.inserted.score, proba=stmt.inserted.proba, rank_in_universe=stmt.inserted.rank_in_universe,
                                                   details=stmt.inserted.details, universe_id=stmt.inserted.universe_id, run_id=stmt.inserted.run_id))
    return len(rows)


def build_details(pred: pd.DataFrame, shap_rank: np.ndarray, shap_event: np.ndarray, features: list[str], x: np.ndarray, top: int = 5) -> list[dict]:
    """Per signal: raw and calibrated probability, q10/q50/q90, expected holding time (median, p75, n) and the ``top`` features by |TreeSHAP| for
    the rank score and for the event log-odds (feature, value, contribution)."""
    def top_features(contrib: np.ndarray, x_row: np.ndarray) -> list[list]:
        idx = np.argsort(-np.abs(contrib), kind="stable")[:top]
        return [[features[j], None if not np.isfinite(x_row[j]) else round(float(x_row[j]), 6), round(float(contrib[j]), 6)] for j in idx]
    out = []
    for i in range(len(pred)):
        p = pred.iloc[i]
        out.append({"proba_raw": round(float(p["proba_raw"]), 6), "proba_platt": round(float(p["proba_platt"]), 6), "proba_isotonic": round(float(p["proba_isotonic"]), 6),
                    "q10": round(float(p["q10"]), 6), "q50": round(float(p["q50"]), 6), "q90": round(float(p["q90"]), 6),
                    "hold_median": float(p["hold_median"]), "hold_p75": float(p["hold_p75"]), "hold_n": int(p["hold_n"]),
                    "shap_rank": top_features(shap_rank[i], x[i]), "shap_event_logodds": top_features(shap_event[i], x[i])})
    return out
