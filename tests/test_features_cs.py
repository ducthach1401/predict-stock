"""Cross-sectional ranks: only universe members on the date take part."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict_stock.features.cs import masked_rank
from predict_stock.features.registry import get_feature

CAL = pd.bdate_range("2024-01-01", periods=3)


def frame(rows):
    return pd.DataFrame(rows, index=CAL, columns=[1, 2, 3, 4])


def test_rank_is_percentile_among_members_only():
    values = frame([[10, 20, 30, 40], [40, 30, 20, 10], [1, 1, 1, 1]])
    mask = frame([[True, True, True, True], [True, True, True, False], [True, True, True, True]])
    r = masked_rank(values, mask, min_count=2)
    assert r.iloc[0].tolist() == [0.25, 0.5, 0.75, 1.0]
    assert r.iloc[1, :3].tolist() == pytest.approx([1.0, 2 / 3, 1 / 3]) and np.isnan(r.iloc[1, 3])      # 4th is not a member that day
    assert r.iloc[2].tolist() == [0.625] * 4                                                              # ties: average rank


def test_a_non_member_never_influences_the_ranks_of_members():
    values = frame([[10, 20, 30, 999], [10, 20, 30, -999], [5, 6, 7, 8]])
    without = masked_rank(values.iloc[:, :3], frame([[True] * 4] * 3).iloc[:, :3], 2)
    mask = frame([[True, True, True, False]] * 3)
    with_outsider = masked_rank(values, mask, 2)
    pd.testing.assert_frame_equal(with_outsider.iloc[:, :3], without)
    assert with_outsider[4].isna().all()


def test_invalid_values_and_small_cross_sections_give_nan():
    values = frame([[1, np.nan, 3, 4], [np.nan, np.nan, 3, 4], [1, 2, 3, 4]])
    mask = frame([[True] * 4] * 3)
    r = masked_rank(values, mask, min_count=3)
    assert r.iloc[0, [0, 2, 3]].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0]) and np.isnan(r.iloc[0, 1])   # NaN excluded from the count
    assert r.iloc[1].isna().all()                                                                          # only 2 valid < min_count
    assert r.iloc[2].notna().all()


def test_ranks_use_the_same_date_only():
    """Changing another date's values must not change this date's ranks."""
    a = frame([[1, 2, 3, 4], [4, 3, 2, 1], [1, 3, 2, 4]])
    b = a.copy(); b.iloc[2] = [9, 9, 1, 0]
    m = frame([[True] * 4] * 3)
    pd.testing.assert_frame_equal(masked_rank(a, m, 2).iloc[:2], masked_rank(b, m, 2).iloc[:2])


def test_cs_rank_feature_names_and_output():
    f = get_feature("cs_rank", 1)(inputs=["a", "b"], min_count=2)
    assert f.columns() == ["a_csrank", "b_csrank"] and f.inputs() == ["a", "b"]
    out = f.compute({"a": frame([[1, 2, 3, 4]] * 3), "b": frame([[4, 3, 2, 1]] * 3)}, frame([[True] * 4] * 3))
    assert out["a_csrank"].iloc[0].tolist() == [0.25, 0.5, 0.75, 1.0] and out["b_csrank"].iloc[0].tolist() == [1.0, 0.75, 0.5, 0.25]
