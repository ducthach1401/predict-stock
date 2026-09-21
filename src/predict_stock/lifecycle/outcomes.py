"""What the accumulated `recommendation_outcomes` say about the champion, and whether they justify a correction.

* `assess`      – hit rate, calibration of the stated probability, realised vs stated holding time, net return. Numbers only; nothing changes.
* `recalibrate` – a PROPOSAL of a probability recalibration (isotonic on the earlier trades, judged on the later ones) and of a holding-time scale. Never applied by itself.
* `meta_label`  – a secondary model that decides which of the champion's trades to take (does the trade end with a net gain?). Trained on the earliest trades, judged on the later ones, with a
                  bootstrap of the gain of the filter; one attempt = one `experiments` row (`lifecycle:meta_label`), so the number of tries stays visible.
Every test set is later in time than the data the correction was fitted on and is used once per attempt; when the sample is too small the function says so and computes nothing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import Engine, select

from predict_stock.backtest.runner import clean
from predict_stock.config import AppConfig
from predict_stock.db.models import Recommendation, RecommendationOutcome
from predict_stock.db.session import session_scope
from predict_stock.invest import stats as IS
from predict_stock.lifecycle.monitor import CARD_STRATEGY
from predict_stock.swing import calibration as cal
from predict_stock.swing.registry import log_experiment

FEATURES = ["p_model", "atr_pct", "rr_target2", "expected_hold", "rsi14", "vol_spike", "q50_5d", "regime_off"]


def card_row(card: dict) -> dict:
    """The facts of a card that were known when it was issued."""
    conf, sig, ex, hold, ent = (card.get("confidence") or {}), (card.get("signal") or {}), (card.get("exits") or {}), (card.get("holding") or {}), (card.get("entry") or {})
    return {"p_model": conf.get("p_model"), "p_display": conf.get("p_display"), "atr_pct": sig.get("atr_pct"), "rr_target2": ex.get("rr_target2"), "expected_hold": hold.get("expected_sessions"),
            "rsi14": sig.get("rsi14"), "vol_spike": sig.get("vol_spike"), "q50_5d": sig.get("q50_5d"), "regime_off": float(bool(sig.get("regime_off"))), "style": ent.get("style")}


def paper_table(engine: Engine, strategy: str, start=None) -> pd.DataFrame:
    """One row per closed paper recommendation of ``strategy`` (a champion key: swing | invest_b1 | invest_b2), in issue order."""
    with session_scope(engine) as s:
        q = (select(Recommendation.id, Recommendation.as_of_date, Recommendation.card, RecommendationOutcome.exit_reason, RecommendationOutcome.holding_days, RecommendationOutcome.net_return,
                    RecommendationOutcome.gross_return, RecommendationOutcome.exit_date)
             .join(RecommendationOutcome, RecommendationOutcome.recommendation_id == Recommendation.id)
             .where(Recommendation.strategy == CARD_STRATEGY[strategy], RecommendationOutcome.exit_reason.is_not(None), RecommendationOutcome.exit_reason != "open")
             .order_by(Recommendation.as_of_date, Recommendation.id))
        if start is not None:
            q = q.where(Recommendation.as_of_date >= pd.Timestamp(start).date())
        rows = s.execute(q).all()
    out = [{"recommendation_id": r.id, "as_of": pd.Timestamp(r.as_of_date), "exit_reason": r.exit_reason, "sessions": r.holding_days, "net_return": r.net_return, "gross_return": r.gross_return,
            **card_row(r.card or {})} for r in rows]
    d = pd.DataFrame(out)
    if not d.empty:
        d["hit"] = (d["exit_reason"] == "target").astype(float)
        d["win"] = (d["net_return"] > 0).astype(float)
    return d


def backtest_table(round_trips: pd.DataFrame, cards: dict) -> pd.DataFrame:
    """The same table from the Phase-7 backtest of the cards (``res.round_trips`` and ``plan.cards``), SWING cards only."""
    rows = []
    for t in round_trips.itertuples():
        c = cards.get(t.tag)
        if c is None or c.strategy != "SWING":
            continue
        cd = c.to_dict()
        rows.append({"recommendation_id": t.tag, "as_of": pd.Timestamp(c.as_of), "exit_reason": t.exit_reason, "sessions": t.sessions, "net_return": t.net_return, "gross_return": t.net_return,
                     **card_row(cd)})
    d = pd.DataFrame(rows)
    if not d.empty:
        d = d.sort_values("as_of", kind="stable").reset_index(drop=True)
        d["hit"] = d["exit_reason"].isin(["target", "target_gap"]).astype(float)
        d["win"] = (d["net_return"] > 0).astype(float)
    return d


def assess(df: pd.DataFrame, cfg: AppConfig) -> dict:
    if df.empty:
        return {"trades": 0, "note": "no closed recommendations yet"}
    out: dict = {"trades": int(len(df)), "from": str(df["as_of"].min().date()), "to": str(df["as_of"].max().date()), "win_rate": float(df["win"].mean()), "target_rate": float(df["hit"].mean()),
                 "net_return_mean": float(df["net_return"].mean()), "net_return_median": float(df["net_return"].median())}
    p = df["p_display"].dropna() if df["p_display"].notna().any() else df["p_model"].dropna()
    if len(p) >= cfg.lifecycle.recalibration_min_events:
        y = df.loc[p.index, "hit"]
        out["calibration"] = {**cal.summary(y, p, 10), "mean_stated": float(p.mean()), "realised": float(y.mean()), "n": int(len(p)),
                              "note": "the event is 'target 2 reached after the fill' (a paper trade), not the model's own event measured from the signal close"}
    else:
        out["calibration"] = {"n": int(len(p)), "note": f"fewer than {cfg.lifecycle.recalibration_min_events} events with a stated probability: no calibration figure is computed"}
    h = df.dropna(subset=["sessions", "expected_hold"])
    if len(h):
        out["holding"] = {"trades": int(len(h)), "actual_mean": float(h["sessions"].mean()), "expected_mean": float(h["expected_hold"].mean()), "median_ratio": float((h["sessions"] / h["expected_hold"]).median()),
                          "share_beyond_expected": float((h["sessions"] > h["expected_hold"]).mean())}
    return clean(out)


def recalibrate(df: pd.DataFrame, cfg: AppConfig) -> dict:
    """Proposal only: isotonic map fitted on the earlier ``meta_train_fraction`` of the trades, compared with the stated probability on the later ones (Brier, ECE)."""
    lc = cfg.lifecycle
    d = df.dropna(subset=["p_model", "hit"]).sort_values("as_of", kind="stable")
    if len(d) < lc.recalibration_min_events:
        return {"applied": False, "n": int(len(d)), "verdict": f"not enough trades ({len(d)} < {lc.recalibration_min_events}): the stated probability is left as it is"}
    cut = int(len(d) * lc.meta_train_fraction)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    fit = cal.fit_isotonic(tr["p_model"].to_numpy(float), tr["hit"].to_numpy(float), cfg.swing.isotonic_min_bin)
    new = cal.apply_calibrator(fit, te["p_model"].to_numpy(float))
    y = te["hit"].to_numpy(float)
    old_b, new_b = cal.brier(y, te["p_model"].to_numpy(float)), cal.brier(y, new)
    old_e, new_e = cal.ece(y, te["p_model"].to_numpy(float)), cal.ece(y, new)
    better = bool(new_b < old_b and new_e <= old_e)
    h = d.dropna(subset=["sessions", "expected_hold"])
    return clean({"applied": False, "n_fit": int(len(tr)), "n_test": int(len(te)), "test_from": str(te["as_of"].min().date()), "brier": {"stated": old_b, "recalibrated": new_b}, "ece": {"stated": old_e, "recalibrated": new_e},
                  "hold_scale": float((h["sessions"] / h["expected_hold"]).median()) if len(h) else None, "recommend": better,
                  "verdict": ("the recalibrated probability is better on the later trades: adopt it in the NEXT model version" if better else "no improvement on the later trades: keep the stated probability")})


def _design(df: pd.DataFrame) -> pd.DataFrame:
    x = df[FEATURES].astype(float)
    for s in sorted(df["style"].dropna().unique()):
        x[f"style_{s}"] = (df["style"] == s).astype(float)
    return x


def meta_label(engine: Engine | None, cfg: AppConfig, df: pd.DataFrame, *, write: bool = True, source: str = "paper") -> dict:
    """Secondary filter: a regularised logistic model predicts whether a SWING trade ends with a net gain from what was known when the card was issued. The threshold keeps the top
    ``keep_share`` of the TRAIN scores (fixed in advance, not tuned); the verdict is the bootstrap of the mean net return of the kept trades minus that of all trades on the later ones."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    lc = cfg.lifecycle
    keep_share = 0.6
    d = df.dropna(subset=["net_return"] + FEATURES).sort_values("as_of", kind="stable").reset_index(drop=True)
    res: dict = {"source": source, "trades": int(len(d)), "keep_share": keep_share, "min_trades": lc.meta_min_trades}
    if len(d) < lc.meta_min_trades:
        res.update({"tried": False, "verdict": f"not tried: {len(d)} trades, {lc.meta_min_trades} needed. Too few trades and the filter would learn noise"})
        return clean(res)
    cut = int(len(d) * lc.meta_train_fraction)
    while cut < len(d) and cut > 0 and d.loc[cut, "as_of"] == d.loc[cut - 1, "as_of"]:
        cut += 1                                                            # never split one issue date between train and test
    tr, te = d.iloc[:cut], d.iloc[cut:]
    x_all = _design(d).fillna(0.0)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    model.fit(x_all.iloc[:cut], tr["win"].astype(int))
    thr = float(np.quantile(model.predict_proba(x_all.iloc[:cut])[:, 1], 1 - keep_share))
    p_te = model.predict_proba(x_all.iloc[cut:])[:, 1]
    kept = p_te >= thr
    r_all, r_kept = te["net_return"].to_numpy(float), te["net_return"].to_numpy(float)[kept]
    res.update({"tried": True, "train": int(len(tr)), "test": int(len(te)), "test_from": str(te["as_of"].min().date()), "kept_test": int(kept.sum()), "threshold": thr,
                "test_mean_all": float(r_all.mean()), "test_mean_kept": float(r_kept.mean()) if kept.any() else None, "test_win_all": float(te["win"].mean()),
                "test_win_kept": float(te["win"].to_numpy()[kept].mean()) if kept.any() else None})
    if kept.sum() >= 20:
        idx = IS.bootstrap_indices(len(r_all), cfg.lifecycle.promotion.bootstrap_resamples, 1, cfg.lifecycle.promotion.seed)
        k = kept.astype(float)
        gains = []
        for row in idx:                                                    # resample TRADES; the kept-set mean over the resample minus the all-trades mean
            rr, kk = r_all[row], k[row]
            gains.append(rr[kk > 0].mean() - rr.mean() if kk.sum() > 0 else np.nan)
        lo = float(np.nanquantile(gains, cfg.lifecycle.promotion.alpha))
        res["gain"] = {"point": float(r_kept.mean() - r_all.mean()), "lower_bound": lo, "level": cfg.lifecycle.promotion.alpha}
        res["useful"] = bool(lo > 0)
        res["verdict"] = ("the filter improves the mean net return on later trades with a bootstrap lower bound above zero: it may be built into the NEXT model version as a challenger (never switched on directly)"
                          if lo > 0 else "the filter does NOT beat 'take every trade' on the later trades once the noise is accounted for: discarded")
    else:
        res["useful"] = False
        res["verdict"] = "fewer than 20 kept test trades: no conclusion"
    if write and engine is not None:
        res["experiment_id"] = log_experiment(engine, cfg, "lifecycle:meta_label", params={"source": source, "features": FEATURES, "trades": int(len(d)), "last": str(d["as_of"].max().date())},
                                              summary=clean(res), status="success" if res.get("useful") else "rejected", description=res["verdict"][:200])
    return clean(res)
