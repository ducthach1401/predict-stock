"""Turns a price panel + a feature set + a label spec into one row per (date, instrument).

A row exists only when the instrument is a MEMBER of the universe on that date, has a bar that day, and has at
least the feature set's warm-up of its own history. Features of a row are computed from data up to that date;
cross-sectional features look only at that date's members; labels look forward (see labels.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from predict_stock.features.data import Panel
from predict_stock.features.registry import CrossSectionalFeature, Feature, LabelBuilder, PanelFeature, TimeSeriesFeature

KEY_COLUMNS = ["trade_date", "instrument_id"]


def compute_feature_columns(panel: Panel, features: list[Feature], mask: pd.DataFrame, provider=None) -> dict[str, pd.DataFrame]:
    """Wide frames (date x instrument) for every feature column, in the order the features declare them."""
    cal, ids = panel.calendar, panel.instrument_ids
    ts = [f for f in features if isinstance(f, TimeSeriesFeature)]
    series: dict[str, dict[int, pd.Series]] = {c: {} for f in ts for c in f.columns()}
    for iid in ids:
        bars = panel.bars_of(iid)
        if bars.empty:
            continue
        bench = None if panel.bench is None else panel.bench.reindex(bars.index)
        for f in ts:
            res = f.compute(bars, bench)
            for c in f.columns():
                series[c][iid] = res[c]
    wide: dict[str, pd.DataFrame] = {}
    for f in features:
        if isinstance(f, TimeSeriesFeature):
            for c in f.columns():
                wide[c] = (pd.DataFrame(series[c]).reindex(index=cal, columns=ids) if series[c]
                           else pd.DataFrame(np.nan, index=cal, columns=ids))
        elif isinstance(f, CrossSectionalFeature):
            wide.update(f.compute({c: wide[c] for c in f.inputs()}, mask.reindex(index=cal, columns=ids).fillna(False)))
        elif isinstance(f, PanelFeature):
            wide.update(f.compute(cal, ids, provider))
    return wide


def required_warmup(features: list[Feature]) -> int:
    return max((f.warmup() for f in features), default=0)


def compute_label_columns(panel: Panel, labels: list[LabelBuilder], mask: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    m = mask.reindex(index=panel.calendar, columns=panel.instrument_ids).fillna(False)
    for lb in labels:
        out.update(lb.compute(panel, m))
    return out


def assemble_frame(panel: Panel, features: list[Feature], labels: list[LabelBuilder], mask: pd.DataFrame, start, end,
                   provider=None) -> pd.DataFrame:
    """One row per eligible (date, instrument) with date in [start, end], sorted by date then instrument id."""
    cal, ids = panel.calendar, panel.instrument_ids
    mask = mask.reindex(index=cal, columns=ids).fillna(False).astype(bool)
    feats = compute_feature_columns(panel, features, mask, provider)
    labs = compute_label_columns(panel, labels, mask)
    history = panel.close.notna().cumsum()                                  # own bars up to and including t
    in_range = (cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))
    eligible = mask & panel.close.notna() & (history >= required_warmup(features)) & in_range[:, None]
    rows, cols = np.nonzero(eligible.to_numpy())
    data: dict[str, np.ndarray] = {
        "trade_date": cal.to_numpy()[rows],
        "instrument_id": np.asarray(ids, dtype="int64")[cols],
    }
    for name, wide in {**feats, **labs}.items():
        vals = wide.reindex(index=cal, columns=ids).to_numpy()
        data[name] = vals[rows, cols]
    return pd.DataFrame(data)
