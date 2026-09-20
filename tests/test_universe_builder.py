from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal

from conftest import FakeClient, make_bars
from predict_stock.data import universe_builder as ub

START, END = date(2026, 6, 1), date(2026, 9, 18)


def bars_with_value(n, close, volume, start=date(2026, 6, 1)):
    return [replace(b, close=Decimal(close), volume=volume) for b in make_bars(start, n)]


def test_ranking_uses_median_traded_value_over_last_window():
    spike = bars_with_value(80, 10000, 1_000)
    spike[-1] = replace(spike[-1], volume=10**9)            # a single huge day must not lift the median
    steady = bars_with_value(80, 10000, 5_000)
    ranked = ub.rank([ub.summarize("SPIKE", spike, 60), ub.summarize("STEADY", steady, 60)])
    assert [c.symbol for c in ranked] == ["STEADY", "SPIKE"]
    assert ranked[0].median_value_vnd == Decimal(10000) * 5_000


def test_window_only_counts_recent_bars():
    old_big = bars_with_value(100, 10000, 1_000_000)
    for i in range(40, 100):                                 # last 60 bars are tiny
        old_big[i] = replace(old_big[i], volume=1)
    assert ub.summarize("X", old_big, 60).median_value_vnd == Decimal(10000)


def test_too_short_history_is_not_rankable():
    assert ub.summarize("X", bars_with_value(59, 10000, 1), 60) is None


def test_ties_break_by_symbol():
    a, b = ub.summarize("B", bars_with_value(60, 1, 1), 60), ub.summarize("A", bars_with_value(60, 1, 1), 60)
    assert [c.symbol for c in ub.rank([a, b])] == ["A", "B"]


def test_build_selects_top_n_and_reports_every_exclusion():
    series = {
        "BIG": make_bars(date(2026, 3, 2), 140, base=90000),
        "MID": bars_with_value(140, 20000, 100_000, date(2026, 3, 2)),
        "SMALL": bars_with_value(140, 5000, 1_000, date(2026, 3, 2)),
        "SHORT": make_bars(date(2026, 9, 1), 10),                       # < window
        "DEAD": make_bars(date(2026, 1, 5), 100),                       # last bar long before END
    }
    sel, ranked, excluded = ub.build(FakeClient(series), ["BIG", "MID", "SMALL", "SHORT", "DEAD", "NOPE"],
                                     start=date(2026, 1, 1), end=END, window=60, top_n=2)
    assert [c.symbol for c in sel] == ["BIG", "MID"]
    assert [c.symbol for c in ranked] == ["BIG", "MID", "SMALL"]
    assert "invalid symbol" in excluded["NOPE"]
    assert "bars" in excluded["SHORT"] and "stale" in excluded["DEAD"]


def test_snapshot_csv_starts_at_first_bar_and_can_be_applied(engine, tmp_path):
    from conftest import FLOOR
    from predict_stock.db.session import session_scope
    from predict_stock.universe import get_members
    from predict_stock.universe_sync import apply_snapshot, read_snapshot_csv
    early = ub.Candidate("EARLY", date(2017, 5, 2), date(2026, 9, 18), 2500, Decimal(1))
    late = ub.Candidate("LATE", date(2020, 3, 9), date(2026, 9, 18), 1700, Decimal(1))
    out = tmp_path / "u.csv"
    ub.write_snapshot_csv(out, [late, early], date(2018, 1, 1), "a note, with, commas")
    with session_scope(engine) as s:
        apply_snapshot(s, "LG", read_snapshot_csv(out), date(2026, 9, 18), source="t", floor=FLOOR, create=True)
    with session_scope(engine) as s:
        assert get_members(s, "LG", date(2018, 1, 1)) == ["EARLY"]      # start is clipped to history_start
        assert get_members(s, "LG", date(2020, 3, 8)) == ["EARLY"]      # LATE not a member before its first bar
        assert get_members(s, "LG", date(2020, 3, 9)) == ["EARLY", "LATE"]


def test_read_candidates_ignores_comments_blanks_and_duplicates(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("# header\nvcb\n\nFPT  # inline\nVCB\n")
    assert ub.read_candidates(p) == ["VCB", "FPT"]
