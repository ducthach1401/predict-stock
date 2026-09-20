from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select

from predict_stock.db.models import UniverseMembership
from predict_stock.db.session import session_scope
from predict_stock.universe import MembershipError, get_members, get_symbols_between, load_membership_csv

HEADER = "universe_code,symbol,effective_from,effective_to\n"


def write(tmp_path, body, name="m.csv"):
    p = tmp_path / name
    p.write_text(HEADER + body, encoding="utf-8")
    return p


def test_membership_is_point_in_time(engine, tmp_path):
    # AAA leaves on 2024-07-01 (exclusive end), CCC joins on that day, BBB is always in.
    f = write(tmp_path, "U1,AAA,2023-01-01,2024-07-01\nU1,BBB,2023-01-01,\nU1,CCC,2024-07-01,\n")
    with session_scope(engine) as s:
        load_membership_csv(s, f)
        assert get_members(s, "U1", date(2022, 12, 31)) == []
        assert get_members(s, "U1", date(2023, 1, 1)) == ["AAA", "BBB"]            # from is inclusive
        assert get_members(s, "U1", date(2024, 6, 30)) == ["AAA", "BBB"]
        assert get_members(s, "U1", date(2024, 7, 1)) == ["BBB", "CCC"]            # to is exclusive
        assert get_members(s, "U1", date(2030, 1, 1)) == ["BBB", "CCC"]
        assert get_members(s, "OTHER", date(2024, 1, 1)) == []


def test_symbols_between_covers_leavers_and_joiners(engine, tmp_path):
    f = write(tmp_path, "U1,AAA,2023-01-01,2024-07-01\nU1,BBB,2023-01-01,\nU1,CCC,2024-07-01,\nU1,DDD,2020-01-01,2022-01-01\n")
    with session_scope(engine) as s:
        load_membership_csv(s, f)
        assert get_symbols_between(s, "U1", date(2023, 6, 1), date(2024, 6, 30)) == ["AAA", "BBB"]
        assert get_symbols_between(s, "U1", date(2024, 6, 1), date(2024, 7, 1)) == ["AAA", "BBB", "CCC"]
        assert get_symbols_between(s, "U1", date(2021, 6, 1), date(2021, 6, 2)) == ["DDD"]
        assert get_symbols_between(s, "U1", date(2022, 1, 1), date(2022, 6, 1)) == []  # DDD ended exactly 2022-01-01


def test_symbol_can_leave_and_rejoin(engine, tmp_path):
    f = write(tmp_path, "U1,AAA,2020-01-01,2021-01-01\nU1,AAA,2022-01-01,\n")
    with session_scope(engine) as s:
        load_membership_csv(s, f)
        assert get_members(s, "U1", date(2020, 6, 1)) == ["AAA"]
        assert get_members(s, "U1", date(2021, 6, 1)) == []
        assert get_members(s, "U1", date(2022, 6, 1)) == ["AAA"]


def test_universe_size_is_not_assumed(engine, tmp_path):
    f = write(tmp_path, "".join(f"BIG,S{i:03d},2024-01-01,\n" for i in range(57)))
    with session_scope(engine) as s:
        load_membership_csv(s, f)
        assert len(get_members(s, "BIG", date(2024, 6, 1))) == 57


def test_reload_is_idempotent(engine, tmp_path):
    f = write(tmp_path, "U1,AAA,2023-01-01,\nU1,BBB,2023-01-01,\n")
    with session_scope(engine) as s:
        load_membership_csv(s, f)
    with session_scope(engine) as s:
        load_membership_csv(s, f)
        assert len(s.scalars(select(UniverseMembership)).all()) == 2


def test_reload_can_close_an_open_interval(engine, tmp_path):
    with session_scope(engine) as s:
        load_membership_csv(s, write(tmp_path, "U1,AAA,2023-01-01,\n", "a.csv"))
    with session_scope(engine) as s:
        load_membership_csv(s, write(tmp_path, "U1,AAA,2023-01-01,2024-01-01\n", "b.csv"))
        rows = s.scalars(select(UniverseMembership)).all()
        assert len(rows) == 1 and rows[0].effective_to == date(2024, 1, 1)
        assert get_members(s, "U1", date(2024, 1, 1)) == []


@pytest.mark.parametrize("body,msg", [
    ("U1,AAA,2023-01-01,2024-01-01\nU1,AAA,2023-06-01,\n", "overlaps"),      # overlap inside file
    ("U1,AAA,2023-01-01,\nU1,AAA,2024-01-01,\n", "overlaps"),                # open interval then new one
    ("U1,AAA,2024-01-01,2023-01-01\n", "must be after"),
    ("U1,AAA,2024-01-01,2024-01-01\n", "must be after"),
    ("U1,AAA,not-a-date,\n", "bad date"),
    ("U1,,2024-01-01,\n", "empty"),
])
def test_invalid_membership_rejected_before_writing(engine, tmp_path, body, msg):
    with session_scope(engine) as s:
        with pytest.raises(MembershipError, match=msg):
            load_membership_csv(s, write(tmp_path, body))
        assert s.scalars(select(UniverseMembership)).all() == []


def test_overlap_with_rows_already_in_db_is_rejected(engine, tmp_path):
    with session_scope(engine) as s:
        load_membership_csv(s, write(tmp_path, "U1,AAA,2023-01-01,2024-01-01\n", "a.csv"))
    with session_scope(engine) as s:
        with pytest.raises(MembershipError, match="overlaps"):
            load_membership_csv(s, write(tmp_path, "U1,AAA,2023-06-01,\n", "b.csv"))


def test_missing_columns_and_empty_file(engine, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("universe_code,symbol\nU1,AAA\n")
    empty = tmp_path / "empty.csv"
    empty.write_text(HEADER)
    with session_scope(engine) as s:
        with pytest.raises(MembershipError, match="missing columns"):
            load_membership_csv(s, bad)
        with pytest.raises(MembershipError, match="no data rows"):
            load_membership_csv(s, empty)


def test_comments_blank_lines_and_case_are_handled(engine, tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("# provenance note\n" + HEADER + "\nU1, aaa ,2023-01-01,\n")
    with session_scope(engine) as s:
        load_membership_csv(s, p)
        assert get_members(s, "U1", date(2023, 2, 1)) == ["AAA"]
