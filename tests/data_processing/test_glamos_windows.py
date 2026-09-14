"""Tests of the GLAMOS window selection over a set of allowed years, and of the year
helpers it rests on.

A split by year gives each side a set of years, and a geodetic window may only be used
by the side that owns the years it covers, up to a tolerance of a year or two. None of
this needs the GLAMOS table: the candidate windows are synthetic.
"""

import pandas as pd
import pytest

from data_processing.glamos import select_glamos_windows, window_years_outside
from data_processing.utils.years import (
    contiguous_year_runs,
    years_from_time_range,
    years_outside,
)


def _windows(rows):
    """Candidate windows in the shape `load_glamos_volume_change` returns."""
    df = pd.DataFrame(rows, columns=["SGI-ID", "y0", "y1", "sigma"])
    df["dur"] = df.y1 - df.y0
    df["covered"] = 100.0
    return df


def test_contiguous_year_runs():
    assert contiguous_year_runs([1953, 1951, 1952, 1960]) == [
        (1951, 1953),
        (1960, 1960),
    ]
    assert contiguous_year_runs([]) == []


def test_years_of_a_window_follow_the_convention_of_its_bounds():
    # GLAMOS: the 1st of January of y1 closes the window, so y1 is not covered
    assert list(years_from_time_range("1956-01-01", "1959-01-01")) == [1956, 1957, 1958]
    # Rabatel16: the window runs from October to October and touches both years
    assert list(years_from_time_range("1983-10-01", "1985-10-01")) == [1983, 1984, 1985]
    assert years_outside("1956-01-01", "1959-01-01", {1956, 1957}) == [1958]
    assert window_years_outside(1956, 1959, {1956, 1957, 1958}) == []


def test_only_the_windows_inside_the_allowed_years_are_kept():
    windows = _windows(
        [
            ("A", 1960, 1975, 0.1),  # entirely in the early years
            ("B", 1985, 1999, 0.1),  # entirely in the late years
            ("C", 1970, 1990, 0.1),  # straddles the boundary by a lot
        ]
    )
    early = set(range(1951, 1980))

    chosen = select_glamos_windows(
        windows, min_year=1950, min_window_years=10, allowed_years=early
    )
    assert list(chosen.index) == ["A"]


def test_a_window_may_cross_by_the_tolerance_given():
    windows = _windows(
        [
            ("A", 1960, 1975, 0.1),  # inside the early years
            ("B", 1960, 1982, 0.1),  # covers 1980 and 1981, two late years
            ("C", 1960, 1984, 0.1),  # covers four late years
        ]
    )
    early = set(range(1951, 1980))
    options = dict(min_year=1950, min_window_years=10, allowed_years=early)

    assert set(select_glamos_windows(windows, **options).index) == {"A"}
    kept = select_glamos_windows(windows, max_outside_years=2, **options)
    assert set(kept.index) == {"A", "B"}
    # B is longer than A, so it is the one an entity would keep
    assert window_years_outside(1960, 1982, early) == [1980, 1981]


def test_the_longest_window_of_an_entity_still_wins_inside_the_allowed_years():
    windows = _windows(
        [
            ("A", 1960, 1972, 0.1),
            ("A", 1960, 1978, 0.5),  # longer, higher sigma
            ("A", 1960, 1995, 0.1),  # longer still, but outside the allowed years
        ]
    )
    chosen = select_glamos_windows(
        windows,
        min_year=1950,
        min_window_years=10,
        allowed_years=set(range(1951, 1980)),
    )
    assert len(chosen) == 1
    assert (chosen.iloc[0].y0, chosen.iloc[0].y1) == (1960, 1978)


def test_allowed_years_are_ignored_when_not_given():
    windows = _windows([("A", 1960, 1995, 0.1)])
    assert len(select_glamos_windows(windows, min_year=1950, min_window_years=10)) == 1
