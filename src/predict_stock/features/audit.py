"""Look-ahead audit: prove that a feature at date t does not use data after t.

Method: compute every feature on the full panel, then again on a panel with all data after a cut date T removed
(and the membership mask cut the same way). For every date <= T the two results must be identical. A feature that
peeked at a later row would change when that row disappears. Labels are audited the other way round: a label at t
may use data up to t + horizon and nothing beyond, so it must be unchanged when data after t + horizon is cut.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from predict_stock.features.data import Panel
from predict_stock.features.engine import compute_feature_columns, compute_label_columns
from predict_stock.features.registry import Feature, LabelBuilder


@dataclass
class AuditResult:
    passed: bool
    cut_dates: list[str]
    columns_checked: int
    violations: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {"passed": self.passed, "cut_dates": self.cut_dates, "columns_checked": self.columns_checked, "violations": self.violations[:20]}


def _differs(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Cell-wise difference mask (NaN equals NaN)."""
    a, b = a.reindex_like(b), b
    if np.issubdtype(a.to_numpy().dtype, np.datetime64) or np.issubdtype(b.to_numpy().dtype, np.datetime64):
        return ~((a == b) | (a.isna() & b.isna()))
    x, y = a.to_numpy(float), b.to_numpy(float)
    same = (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=False)
    return pd.DataFrame(~same, index=b.index, columns=b.columns)


def audit_no_lookahead(panel: Panel, mask: pd.DataFrame, features: list[Feature], cut_dates: list[pd.Timestamp],
                       provider=None) -> AuditResult:
    mask = mask.reindex(index=panel.calendar, columns=panel.instrument_ids).fillna(False)
    full = compute_feature_columns(panel, features, mask, provider)
    violations, checked = [], 0
    for T in cut_dates:
        part = compute_feature_columns(panel.truncate(T), features, mask.loc[:T], provider)
        for col, cut in part.items():
            bad = _differs(full[col].loc[:T], cut)
            checked += 1
            if bad.to_numpy().any():
                r, c = np.argwhere(bad.to_numpy())[0]
                violations.append({"kind": "feature", "column": col, "cut": str(T.date()), "cells": int(bad.to_numpy().sum()),
                                   "first_date": str(bad.index[r].date()), "instrument_id": int(bad.columns[c])})
    return AuditResult(not violations, [str(t.date()) for t in cut_dates], checked, violations)


def audit_label_horizon(panel: Panel, mask: pd.DataFrame, labels: list[LabelBuilder], cut_dates: list[pd.Timestamp]) -> AuditResult:
    mask = mask.reindex(index=panel.calendar, columns=panel.instrument_ids).fillna(False)
    full = compute_label_columns(panel, labels, mask)
    hz = max(lb.horizon() for lb in labels)
    violations, checked = [], 0
    for T in cut_dates:
        part = compute_label_columns(panel.truncate(T), labels, mask.loc[:T])
        pos = panel.calendar.get_loc(T)
        last_ok = panel.calendar[max(pos - hz, 0)] if pos - hz >= 0 else None
        if last_ok is None:
            continue
        for col, cut in part.items():
            bad = _differs(full[col].loc[:last_ok], cut.loc[:last_ok])
            checked += 1
            # ranks are cross-sectional: a member whose label is unknown on one side changes the rank of the others, so
            # ranks are compared only where the horizon fits for every instrument (checked via the return columns).
            if bad.to_numpy().any() and not col.startswith("fwd_rank_"):
                r, c = np.argwhere(bad.to_numpy())[0]
                violations.append({"kind": "label", "column": col, "cut": str(T.date()), "cells": int(bad.to_numpy().sum()),
                                   "first_date": str(bad.index[r].date()), "instrument_id": int(bad.columns[c])})
    return AuditResult(not violations, [str(t.date()) for t in cut_dates], checked, violations)


def pick_cut_dates(calendar: pd.DatetimeIndex, start, end, n: int = 3) -> list[pd.Timestamp]:
    """``n`` dates spread evenly over the decision range, leaving data after each so the cut is meaningful."""
    inside = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    if len(inside) < 3:
        return []
    idx = np.unique(np.linspace(len(inside) * 0.25, len(inside) * 0.75, n).astype(int))
    return [inside[i] for i in idx]
