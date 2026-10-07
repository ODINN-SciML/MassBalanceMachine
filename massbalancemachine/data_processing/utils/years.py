"""Helpers about the calendar years a geodetic window spans, used both to generate its
gridded products and to keep it inside the years of one side of a train/validation
split.

They live here, in a module that imports nothing from `data_processing`, because the
geodetic sources (`data_processing.glamos`, `data_processing.rabatel16`) need them and
are themselves imported by `data_processing.gridded_utils`.
"""

import pandas as pd


def contiguous_year_runs(years) -> "list[tuple[int, int]]":
    """Maximal runs of consecutive years in `years`, as (first, last) pairs, included.

    `contiguous_year_runs([1953, 1951, 1952, 1960])` gives `[(1951, 1953), (1960, 1960)]`.
    """
    runs = []
    for year in sorted({int(y) for y in years}):
        if runs and year == runs[-1][1] + 1:
            runs[-1][1] = year
        else:
            runs.append([year, year])
    return [(first, last) for first, last in runs]


def years_from_time_range(start, end) -> range:
    """Calendar years a geodetic window spans.

    The year of `end` is left out when the window stops on the 1st of January,
    since no month of that year is inside the window. The sources do not share a
    convention: a GLAMOS window runs from the 1st of January of `y0` to the 1st of
    January of `y1` and spans `y0 .. y1 - 1`, while a Rabatel16 one runs from the 1st
    of October to the 1st of October and touches the year of each bound.
    """
    year_start = pd.Timestamp(start).year
    offset = 1 if (pd.Timestamp(end).month == 1) and (pd.Timestamp(end).day == 1) else 0
    year_end = pd.Timestamp(end).year - offset
    return range(year_start, year_end + 1)


def years_outside(start, end, allowed_years) -> "list[int]":
    """The years of a window that `allowed_years` does not hold."""
    allowed = set(allowed_years)
    return [y for y in years_from_time_range(start, end) if y not in allowed]
