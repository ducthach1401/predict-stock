"""The lifecycle inside the daily paper job on the simulated market of the paper e2e test: the champion serves the recommendations, a challenger only shadows (predictions and nothing else),
every recommendation is traceable to its origin with the links verified, and a challenger that has not run long enough cannot be promoted."""
from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from predict_stock.backtest.runner import load_setup
from predict_stock.db.models import Model, Prediction, Recommendation
from predict_stock.lifecycle import compare as CMP
from predict_stock.lifecycle import provenance as PV
from predict_stock.lifecycle import registry as REG
from predict_stock.paper import daily as D
from predict_stock.swing.folds import with_ends
from predict_stock.swing.registry import register_model
from predict_stock.swing.walkforward import fit_final
from test_paper_e2e import build, sim  # noqa: F401  (the module-scoped simulation fixture)


def test_champion_serves_challenger_only_shadows_and_every_recommendation_is_traceable(engine, sim):
    cfg = sim["cfg"]
    client, days = build(engine, sim)
    with Session(engine) as s:
        champ = s.scalars(select(Model).where(Model.name == "swing_lgbm_final")).one()
    with Session(engine) as s, s.begin():
        s.get(Model, champ.id).trained_until = pd.Timestamp("2025-12-31").date()
    REG.set_status(engine, champ.id, "champion", actor="test", reason="first champion")
    # a challenger trained differently (another seed), registered with the current protocol and put in shadow
    setup = load_setup(engine, cfg)
    other = cfg.swing.model_copy(update={"seed": 7, "lgbm": {**cfg.swing.lgbm, "num_leaves": 7}})
    bundle, _ = fit_final(with_ends(setup.frames["swing"]), other, {}, setup.holdout.start, setup.holdout.end)
    cid, ver, _, reused = register_model(engine, cfg.model_copy(update={"swing": other}), bundle, "swing_lgbm_final", dataset_id=None, experiment_id=None,
                                         extra_params={"protocol": CMP.protocol_hash(cfg, "swing")})
    assert ver == 2 and not reused
    with Session(engine) as s, s.begin():
        s.get(Model, cid).trained_until = pd.Timestamp("2025-12-31").date()
    REG.set_status(engine, cid, "shadow", actor="test", reason="retrain (manual)")

    for d in days[:3]:
        res = D.run_daily(engine, cfg, client, d)
        assert res["ok"], {k: v.get("error") for k, v in res.items() if isinstance(v, dict) and v.get("error")}
        assert res["shadow"]["ok"] and res["monitor"]["ok"]
    with engine.connect() as c:
        for d in days[:3]:
            n_ch = c.execute(select(func.count()).select_from(Prediction).where(Prediction.model_id == cid, Prediction.as_of_date == d)).scalar()
            n_ck = c.execute(select(func.count()).select_from(Prediction).where(Prediction.model_id == champ.id, Prediction.as_of_date == d)).scalar()
            assert n_ch == n_ck >= 12                                          # the challenger scored the universe every day, exactly like the champion
        served = set(c.execute(select(Recommendation.model_id).where(Recommendation.strategy == "SWING")).scalars().all())
        assert served == {champ.id}                                            # shadow = predictions only: no card ever came from the challenger

    # ---- every recommendation is traceable, with the links verified ------------------------------------------------------------------------------------------
    with Session(engine) as s:
        recs = s.scalars(select(Recommendation).where(Recommendation.action == "BUY")).all()
    assert recs
    for r in recs:
        assert r.provenance is not None and r.provenance["model"]["id"] == r.model_id and r.provenance["data"]["bars"] > 0
        t = PV.trace(engine, r.id)
        assert t["provenance_recorded"] and t["verified"], (r.id, t["checks"])
        assert t["checks"]["data_fingerprint_matches"] and t["checks"]["model_file"]["sha256_matches"]
    sw = next(r for r in recs if r.strategy == "SWING")
    prov = PV.trace(engine, sw.id)["provenance"]
    assert prov["model"]["sha256"] and prov["model"]["git_commit"] and prov["feature_set"]["name"] == "swing" and prov["scoring_dataset"]["sha256"] and prov["code"]["job_run_id"]

    # ---- a tampered model file breaks the chain ----------------------------------------------------------------------------------------------------------------
    from predict_stock.config import PROJECT_ROOT
    path = PROJECT_ROOT / prov["model"]["artifact_path"] if not prov["model"]["artifact_path"].startswith("/") else __import__("pathlib").Path(prov["model"]["artifact_path"])
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"x")
        broken = PV.trace(engine, sw.id)
        assert broken["verified"] is False and broken["checks"]["model_file"]["sha256_matches"] is False
    finally:
        path.write_bytes(original)

    # ---- three days of shadow are nowhere near enough to promote ---------------------------------------------------------------------------------------------
    setup = load_setup(engine, cfg)
    with pytest.raises(REG.LifecycleError, match="protected held-out period"):                # the simulated days fall in the real protected period: the guard refuses to compare there
        CMP.compare(engine, cfg, setup, cid, pd.Timestamp(days[2]))
    c2 = cfg.model_copy(update={"lifecycle": cfg.lifecycle.model_copy(update={"protected_period": ("2099-01-01", "2099-12-31")})})
    rep = CMP.compare(engine, c2, setup, cid, pd.Timestamp(days[2]))
    assert rep["passed"] is False and rep["verdict"].startswith("NOT ENOUGH SHADOW DATA")
    with pytest.raises(REG.LifecycleError):
        CMP.promote(engine, c2, setup, cid, pd.Timestamp(days[2]))
    assert REG.champion(engine, "swing").id == champ.id and REG.get(engine, cid).status == "shadow"
