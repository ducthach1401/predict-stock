"""Synthetic datasets shaped like the invest:2 / invest:1 dataset (features, cross-sectional ranks, forward labels and their end dates)."""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (63, 126)


def synthetic_invest(n_days: int = 900, n_inst: int = 16, seed: int = 0, signal: float = 1.0, start: str = "2019-01-01", fund: bool = False) -> pd.DataFrame:
    """The truth: forward returns load on momentum (+), volatility (-) and trend (+) with strength ``signal`` (0 = pure noise). The last h sessions have no
    h-session label. ``fund=True`` adds a ``fund_quality`` column that carries a signal of its own."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for t, d in enumerate(dates):
        mom6, mom12, vol, dd, sma, dbeta = (rng.normal(size=n_inst) for _ in range(6))
        q = rng.normal(size=n_inst)
        latent = signal * (0.6 * mom6 + 0.4 * mom12 - 0.5 * vol + 0.4 * sma) + (0.8 * signal * q if fund else 0.0)
        for j in range(n_inst):
            row = {"trade_date": d, "instrument_id": j + 1, "_t": t, "mom_3m": mom6[j] * 0.5, "mom_6m": mom6[j], "mom_12m": mom12[j], "vol_63": vol[j] + rng.normal() * 0.2,
                   "vol_126": vol[j], "dbeta_126": dbeta[j] if rng.random() > 0.15 else np.nan, "dd_252": -abs(dd[j]), "sma_ratio_200": sma[j], "liq_zero_share_60": 0.0}
            if fund:
                row["fund_quality"] = q[j]
            row["_latent"] = latent[j]
            rows.append(row)
    df = pd.DataFrame(rows)
    for c in ("mom_6m", "mom_12m", "vol_126", "dd_252", "sma_ratio_200"):
        df[f"{c}_csrank"] = df.groupby("trade_date")[c].rank(pct=True)
    cal = pd.DatetimeIndex(dates)
    for h in HORIZONS:
        ret = 0.03 * df["_latent"] * np.sqrt(h / 63) + 0.10 * np.sqrt(h / 63) * rng.normal(size=len(df))
        df[f"fwd_ret_{h}"] = ret
        df[f"fwd_rank_{h}"] = df.groupby("trade_date")[f"fwd_ret_{h}"].rank(pct=True)
        idx = df["_t"] + h
        df[f"fwd_end_{h}"] = pd.Series(cal[np.clip(idx, 0, len(cal) - 1)].astype("datetime64[ns]"), index=df.index)
        late = idx >= len(cal)
        df.loc[late, [f"fwd_ret_{h}", f"fwd_rank_{h}", f"fwd_end_{h}"]] = [np.nan, np.nan, pd.NaT]
    return df.drop(columns=["_t", "_latent"])
