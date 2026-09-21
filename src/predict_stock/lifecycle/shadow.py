"""Shadow running: a challenger scores the universe every day next to the champion and ONLY its predictions are stored (`predictions`); it never produces a card, an order or a position.
The champion's predictions are stored too, so both series exist in the same table for the comparison. `replay` fills a date range at once (a missed period, or a demonstration on
history); it uses the model artifact only, so a prediction can never depend on anything after its own date."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import Engine, select

from predict_stock.backtest.runner import Setup
from predict_stock.config import PROJECT_ROOT, AppConfig
from predict_stock.db.models import Model, Universe
from predict_stock.db.session import session_scope
from predict_stock.lifecycle import registry as REG
from predict_stock.swing.registry import save_predictions


def _universe_id(engine: Engine, code: str) -> int:
    with session_scope(engine) as s:
        return s.scalar(select(Universe.id).where(Universe.code == code))


def predict_range(engine: Engine, cfg: AppConfig, setup: Setup, ref: REG.ModelRef, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """The model's predictions for every dataset row dated in [start, end]: trade_date, instrument_id, score, proba, rank_in_universe, details."""
    key = "swing" if ref.strategy == "swing" else "invest"
    frame = setup.frames[key]
    d = pd.to_datetime(frame["trade_date"])
    rows = frame[(d >= start) & (d <= end)]
    if rows.empty:
        return pd.DataFrame()
    raw = (PROJECT_ROOT / ref.artifact_path).read_bytes() if not ref.artifact_path.startswith("/") else open(ref.artifact_path, "rb").read()
    if ref.strategy == "swing":
        from predict_stock.swing.model import load_bundle
        from predict_stock.swing.walkforward import predict_rows
        bundle = load_bundle(PROJECT_ROOT / ref.artifact_path if not ref.artifact_path.startswith("/") else ref.artifact_path, ref.sha256)
        out = predict_rows(bundle, rows)
        details = [{"shadow_or_daily": True, "q10": float(a), "q50": float(b), "q90": float(c), "hold_median": float(h)} for a, b, c, h in zip(out["q10"], out["q50"], out["q90"], out["hold_median"])]
        out = out[["trade_date", "instrument_id", "score", "proba", "rank_in_universe"]].assign(details=details)
    else:
        from predict_stock.invest.models import load_candidate, sha256
        if sha256(raw) != ref.sha256:
            raise ValueError(f"{ref.artifact_path}: sha256 does not match the recorded value")
        model, scen, _ = load_candidate(raw)
        q = scen.predict(rows)
        out = rows[["trade_date", "instrument_id"]].assign(score=model.predict(rows), proba=np.nan)
        out["rank_in_universe"] = out.groupby("trade_date")["score"].rank(ascending=False, method="first").astype(int)
        out["details"] = [{"shadow_or_daily": True, "q10": float(a), "q50": float(b), "q90": float(c)} for a, b, c in zip(q["q10"], q["q50"], q["q90"])]
    return out


def store(engine: Engine, cfg: AppConfig, setup: Setup, ref: REG.ModelRef, start: pd.Timestamp, end: pd.Timestamp, run_id: int | None = None) -> int:
    out = predict_range(engine, cfg, setup, ref, start, end)
    if out.empty:
        return 0
    horizon = cfg.swing.horizon if ref.strategy == "swing" else cfg.invest.presets[ref.strategy.split("_")[1]].horizon
    return save_predictions(engine, ref.id, _universe_id(engine, setup.universe), horizon, out, run_id)


def shadow_step(engine: Engine, cfg: AppConfig, setup: Setup, as_of: pd.Timestamp, run_id: int | None = None) -> dict:
    """Today's predictions of every champion and every shadow model (idempotent)."""
    res = {}
    for strategy in REG.STRATEGIES:
        for status in ("champion", "shadow"):
            for ref in REG.with_status(engine, strategy, status):
                res[f"{ref.name} v{ref.version} ({status})"] = store(engine, cfg, setup, ref, as_of, as_of, run_id)
    return res


def replay(engine: Engine, cfg: AppConfig, setup: Setup, model_id: int, start: pd.Timestamp, end: pd.Timestamp, run_id: int | None = None) -> int:
    """Predictions of one model over a date range (shadow catch-up / demonstration on history)."""
    return store(engine, cfg, setup, REG.get(engine, model_id), pd.Timestamp(start), pd.Timestamp(end), run_id)
