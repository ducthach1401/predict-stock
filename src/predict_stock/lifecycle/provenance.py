"""Where a recommendation came from, and proof of it: recommendation -> prediction -> model (file + sha256 + seed + training cut-off) -> dataset (file + sha256) ->
feature set / label spec -> git commit -> the price data exactly as it was on the day.

`build` runs when a card is stored and freezes the chain into `recommendations.provenance`; `trace` reads it back and CHECKS it: the model file still has its sha256, the dataset file
exists and matches, the commit exists in the repository, and the price bars, reconstructed as they were known when the card was issued (a later vendor re-adjustment is undone by the
revision log), hash to the same fingerprint."""
from __future__ import annotations

import hashlib
import subprocess
from datetime import date, datetime, timezone

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from predict_stock.config import PROJECT_ROOT
from predict_stock.db.models import (
    Dataset, FeatureSet, JobRun, LabelSpec, Model, PriceBar, PriceBarRevision, Recommendation,
)
from predict_stock.db.session import session_scope

FINGERPRINT_BARS = 300


def bars_as_known(session: Session, instrument_id: int, as_of: date, known_at: datetime | None, n: int = FINGERPRINT_BARS) -> list[tuple]:
    """The last ``n`` bars up to ``as_of`` as they were at ``known_at``: a bar the vendor rewrote after that moment is taken from the revision log (its earliest superseded value
    after ``known_at``), so re-adjusted history does not change the answer."""
    rows = session.execute(select(PriceBar).where(PriceBar.instrument_id == instrument_id, PriceBar.trade_date <= as_of).order_by(PriceBar.trade_date.desc()).limit(n)).scalars().all()
    if not rows:
        return []
    first = rows[-1].trade_date
    revs: dict = {}
    if known_at is not None:
        for r in session.execute(select(PriceBarRevision).where(PriceBarRevision.instrument_id == instrument_id, PriceBarRevision.trade_date >= first, PriceBarRevision.trade_date <= as_of,
                                                                 PriceBarRevision.superseded_at > known_at).order_by(PriceBarRevision.superseded_at)).scalars():
            revs.setdefault(r.trade_date, r)                                 # the oldest revision written after the card = the value that stood at issue time
    out = []
    for b in reversed(rows):
        src = revs.get(b.trade_date, b)
        out.append((b.trade_date.isoformat(), f"{float(src.open):.4f}", f"{float(src.high):.4f}", f"{float(src.low):.4f}", f"{float(src.close):.4f}", int(src.volume)))
    return out


def fingerprint(session: Session, instrument_id: int, as_of: date, known_at: datetime | None, n: int = FINGERPRINT_BARS) -> dict:
    bars = bars_as_known(session, instrument_id, as_of, known_at, n)
    h = hashlib.sha256("\n".join(",".join(map(str, b)) for b in bars).encode()).hexdigest()
    return {"instrument_id": instrument_id, "as_of": as_of.isoformat(), "bars": len(bars), "first": bars[0][0] if bars else None, "last": bars[-1][0] if bars else None, "sha256": h, "n_requested": n}


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build(engine: Engine, *, model_id: int | None, instrument_id: int, as_of: date, run_id: int | None, dataset_ids: dict | None = None, dataset_hashes: dict | None = None,
          strategy: str | None = None, known_at: datetime | None = None) -> dict:
    """The provenance record of one card (stored as JSON in recommendations.provenance)."""
    known_at = known_at or datetime.now(timezone.utc).replace(tzinfo=None)
    with session_scope(engine) as s:
        m = s.get(Model, model_id) if model_id else None
        run = s.get(JobRun, run_id) if run_id else None
        fs = s.get(FeatureSet, m.feature_set_id) if m else None
        ls = s.get(LabelSpec, m.label_spec_id) if m else None
        ds = s.get(Dataset, m.dataset_id) if m and m.dataset_id else None
        key = "swing" if (strategy or "").upper().startswith("SWING") else "invest"
        live_ds = s.get(Dataset, (dataset_ids or {}).get(key)) if dataset_ids and (dataset_ids or {}).get(key) else None
        fp = fingerprint(s, instrument_id, as_of, known_at)
        return {
            "as_of": as_of.isoformat(), "issued_at": known_at.isoformat(timespec="seconds"),
            "model": None if m is None else {"id": m.id, "name": m.name, "version": m.version, "status_at_issue": m.status, "strategy": m.strategy, "artifact_path": m.artifact_path,
                                             "sha256": m.artifact_sha256, "seed": m.seed, "git_commit": m.git_commit, "trained_until": None if m.trained_until is None else m.trained_until.isoformat()},
            "training_dataset": None if ds is None else {"id": ds.id, "name": ds.name, "version": ds.version, "path": ds.path, "sha256": ds.sha256, "rows": ds.row_count},
            "scoring_dataset": None if live_ds is None else {"id": live_ds.id, "name": live_ds.name, "version": live_ds.version, "path": live_ds.path, "sha256": live_ds.sha256,
                                                             "content_hash": (dataset_hashes or {}).get(key)},
            "feature_set": None if fs is None else {"id": fs.id, "name": fs.name, "version": fs.version}, "label_spec": None if ls is None else {"id": ls.id, "name": ls.name, "version": ls.version},
            "code": {"git_commit": None if run is None else run.git_commit, "git_dirty": None if run is None else run.git_dirty, "job_run_id": run_id, "config_snapshot_id": None if run is None else run.config_snapshot_id},
            "data": fp,
        }


def _file_ok(path: str | None, sha: str | None) -> dict:
    if not path:
        return {"exists": False, "note": "no file recorded"}
    p = PROJECT_ROOT / path if not path.startswith("/") else __import__("pathlib").Path(path)
    if not p.exists():
        return {"exists": False, "path": str(path), "note": "file is gone (pruned or moved)"}
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"exists": True, "path": str(path), "sha256_matches": h.hexdigest() == sha}


def trace(engine: Engine, recommendation_id: int) -> dict:
    """The full chain of one recommendation with the verification of every link."""
    with session_scope(engine) as s:
        r = s.get(Recommendation, recommendation_id)
        if r is None:
            raise LookupError(f"recommendation #{recommendation_id} does not exist")
        prov = r.provenance
        chain = {"recommendation": {"id": r.id, "strategy": r.strategy, "instrument_id": r.instrument_id, "as_of": r.as_of_date.isoformat(), "action": r.action, "status": r.status,
                                    "created_at": r.created_at.isoformat(timespec="seconds"), "prediction_id": r.prediction_id, "model_id": r.model_id}, "provenance_recorded": prov is not None}
        checks: dict = {}
        if prov is None:                                                     # an older card: rebuild what the tables still say
            m = s.get(Model, r.model_id) if r.model_id else None
            prov = {"model": None if m is None else {"id": m.id, "name": m.name, "version": m.version, "sha256": m.artifact_sha256, "artifact_path": m.artifact_path, "git_commit": m.git_commit,
                                                     "seed": m.seed, "status_at_issue": None, "trained_until": None if m.trained_until is None else m.trained_until.isoformat()},
                    "training_dataset": None, "scoring_dataset": None, "feature_set": None, "label_spec": None, "code": {"git_commit": None}, "data": None}
            if m is not None and m.dataset_id:
                d = s.get(Dataset, m.dataset_id)
                prov["training_dataset"] = {"id": d.id, "name": d.name, "version": d.version, "path": d.path, "sha256": d.sha256}
            if m is not None:
                fs, ls = s.get(FeatureSet, m.feature_set_id), s.get(LabelSpec, m.label_spec_id)
                prov["feature_set"], prov["label_spec"] = {"name": fs.name, "version": fs.version}, {"name": ls.name, "version": ls.version}
            chain["note"] = "this card predates provenance recording: the chain is rebuilt from the model row (no data fingerprint)"
        chain["provenance"] = prov
        pred = None
        if r.prediction_id:
            from predict_stock.db.models import Prediction
            pred = s.get(Prediction, r.prediction_id)
        chain["prediction"] = None if pred is None else {"id": pred.id, "model_id": pred.model_id, "as_of": pred.as_of_date.isoformat(), "score": pred.score, "proba": pred.proba}
        # ---- verification -----------------------------------------------------------------------------------------------------------------------------
        mdl = (prov or {}).get("model")
        if mdl:
            checks["model_file"] = _file_ok(mdl.get("artifact_path"), mdl.get("sha256"))
            m = s.get(Model, mdl["id"])
            checks["model_row_sha_matches_record"] = bool(m is not None and m.artifact_sha256 == mdl["sha256"])
            checks["prediction_from_this_model"] = None if pred is None else pred.model_id == mdl["id"]
            commit = mdl.get("git_commit")
            checks["model_git_commit_exists"] = None if not commit else _git("cat-file", "-e", f"{commit}^{{commit}}") is not None
        for k in ("training_dataset", "scoring_dataset"):
            d = prov.get(k) if prov else None
            if d:
                checks[f"{k}_file"] = _file_ok(d.get("path"), d.get("sha256"))
        code = (prov or {}).get("code") or {}
        if code.get("git_commit"):
            checks["code_git_commit_exists"] = _git("cat-file", "-e", f"{code['git_commit']}^{{commit}}") is not None
        data = (prov or {}).get("data")
        if data:
            known_at = datetime.fromisoformat(prov["issued_at"])
            now = fingerprint(s, data["instrument_id"], date.fromisoformat(data["as_of"]), known_at, data["n_requested"])
            checks["data_fingerprint_matches"] = now["sha256"] == data["sha256"]
            checks["data_bars"] = now["bars"]
            with_rev = s.scalar(select(__import__("sqlalchemy").func.count()).select_from(PriceBarRevision).where(
                PriceBarRevision.instrument_id == data["instrument_id"], PriceBarRevision.trade_date <= date.fromisoformat(data["as_of"]), PriceBarRevision.superseded_at > known_at))
            checks["bars_re_adjusted_since_issue"] = int(with_rev)
        chain["checks"] = checks
        chain["verified"] = bool(checks) and all(v is not False and (not isinstance(v, dict) or (v.get("exists") and v.get("sha256_matches", True))) for k, v in checks.items()
                                                 if k not in ("bars_re_adjusted_since_issue", "data_bars"))
        return chain
