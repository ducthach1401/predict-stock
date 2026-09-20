"""Synthetic datasets shaped like the Phase 3 swing dataset (keys, features, forward labels and their end dates)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_frame(n_days: int = 420, n_inst: int = 20, seed: int = 0, signal: float = 1.0, start: str = "2020-01-01", horizon: int = 10) -> pd.DataFrame:
    """Feature ``f_a`` carries a planted signal of strength ``signal`` (0 = pure noise); ``f_b`` is weakly informative, ``f_c`` is noise.
    The last ``horizon`` sessions have no labels, exactly like the real dataset."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for t, d in enumerate(dates):
        f = rng.normal(size=(n_inst, 3))
        latent = signal * f[:, 0] + 0.3 * signal * f[:, 1]
        ret = 0.01 * latent + 0.03 * rng.normal(size=n_inst)
        p_event = 1 / (1 + np.exp(-(-0.9 + latent)))
        event = rng.random(n_inst) < p_event
        stop = (~event) & (rng.random(n_inst) < 0.6)
        label = np.where(event, 1.0, np.where(stop, -1.0, 0.0))
        time = np.where(label == 0, horizon, rng.integers(1, horizon + 1, n_inst)).astype(float)
        time = np.where(label == 1, np.minimum(time, 1 + rng.integers(0, 6, n_inst)), time)
        for j in range(n_inst):
            rows.append({"trade_date": d, "instrument_id": j + 1, "f_a": f[j, 0], "f_b": f[j, 1], "f_c": f[j, 2], "_t": t, "fwd_ret_5": ret[j],
                         "tb_label": label[j], "tb_time": time[j]})
    df = pd.DataFrame(rows)
    df["fwd_rank_5"] = df.groupby("trade_date")["fwd_ret_5"].rank(pct=True)
    cal = pd.DatetimeIndex(dates)

    def end(steps: pd.Series) -> pd.Series:
        idx = df["_t"] + steps
        return pd.Series(np.where((steps > 0) & (idx < len(cal)), cal[np.clip(idx.fillna(0).astype(int), 0, len(cal) - 1)], np.datetime64("NaT", "ns")), index=df.index)

    df["fwd_end_5"] = end(pd.Series(5, index=df.index)).astype("datetime64[ns]")
    df["tb_end"] = end(df["tb_time"]).astype("datetime64[ns]")
    late = df["_t"] + horizon >= len(cal)                                   # window does not fit inside the data: no label
    df.loc[late, ["tb_label", "tb_time", "tb_end"]] = np.nan
    df.loc[df["_t"] + 5 >= len(cal), ["fwd_ret_5", "fwd_rank_5", "fwd_end_5"]] = np.nan
    return df.drop(columns="_t")
