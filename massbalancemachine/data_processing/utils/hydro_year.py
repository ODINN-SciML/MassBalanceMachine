import numpy as np
import pandas as pd
from calendar import month_abbr
from typing import Union, Callable, Dict, List, Optional, Tuple

months_hydro_year = [
    month.lower() for month in month_abbr[10:] + month_abbr[1:10]
]  # Standard hydro year (oct..sep)

# ---------------- Flexible month mapping ----------------


def hydro_year_bounds(hy):
    """Return start and end timestamps of hydrological year hy."""
    start = pd.Timestamp(year=hy, month=10, day=1)
    end = pd.Timestamp(year=hy + 1, month=9, day=30)
    return start, end


def overlap_days(a_start, a_end, b_start, b_end):
    """Inclusive overlap in days between two date intervals."""
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if start > end:
        return 0
    return (end - start).days + 1


def assign_hydro_year(row):
    from_date = row["FROM_DATE"]
    to_date = row["TO_DATE"]

    if pd.isna(from_date) or pd.isna(to_date):
        return pd.NA
    if to_date < from_date:
        return pd.NA

    # Candidate hydro years to inspect.
    # Wide enough to capture:
    # - fully covered hydro year
    # - nearest hydro year for partial overlap cases
    min_hy = from_date.year - 1
    max_hy = to_date.year

    covered_hys = []
    overlaps = {}

    for hy in range(min_hy, max_hy + 1):
        hy_start, hy_end = hydro_year_bounds(hy)

        # Rule 1/2: fully covered hydro years
        if from_date <= hy_start and to_date >= hy_end:
            covered_hys.append(hy)

        # Rule 3: overlap amount
        overlaps[hy] = overlap_days(from_date, to_date, hy_start, hy_end)

    # Rule 1: exactly one fully covered hydro year
    if len(covered_hys) == 1:
        return covered_hys[0]

    # Rule 2: more than one fully covered hydro year -> take the first one
    if len(covered_hys) > 1:
        return min(covered_hys)

    # Rule 3: no fully covered hydro year -> take the hydro year with maximum overlap
    # In case of tie, min(...) picks the first hydro year
    max_overlap = max(overlaps.values())
    best_hys = [hy for hy, days in overlaps.items() if days == max_overlap]
    return min(best_hys)


def count_touched_months(start, end):
    """Number of calendar months touched by [start, end], inclusive."""
    if pd.isna(start) or pd.isna(end) or end < start:
        return 0
    return (end.year - start.year) * 12 + (end.month - start.month) + 1


def months_before(row):
    # extra interval before hydrological year
    extra_start = row["FROM_DATE"]
    extra_end = row["HY_START"] - pd.Timedelta(days=1)

    if extra_start > extra_end:
        return 0
    return count_touched_months(extra_start, extra_end)


def months_after(row):
    # extra interval after hydrological year
    extra_start = row["HY_END"] + pd.Timedelta(days=1)
    extra_end = row["TO_DATE"]

    if extra_start > extra_end:
        return 0
    return count_touched_months(extra_start, extra_end)


def _compute_head_tail_pads_from_df(df):
    df = df.copy()
    df["FROM_DATE"] = pd.to_datetime(df["FROM_DATE"])
    df["TO_DATE"] = pd.to_datetime(df["TO_DATE"])
    df["HY_YEAR"] = df.apply(assign_hydro_year, axis=1)
    # Hydrological year bounds
    df["HY_START"] = pd.to_datetime(df["HY_YEAR"].astype(str) + "-10-01")
    df["HY_END"] = pd.to_datetime((df["HY_YEAR"] + 1).astype(str) + "-09-30")
    # Determine padding needed for each row
    df["MONTHS_BEFORE"] = df.apply(months_before, axis=1)
    df["MONTHS_AFTER"] = df.apply(months_after, axis=1)

    assert (
        df.MONTHS_BEFORE.max() + df.MONTHS_AFTER.max() <= 12
    ), "Impossible to pad with less than 12 months. Please review the raw data and filter out entries with a too large time window."

    fromDateMin = 10 - df.MONTHS_BEFORE.max()
    assert (
        fromDateMin > 0
    ), "The padding before the hydrological year cannot go beyond the month of january."
    toDateMax = 9 + df.MONTHS_AFTER.max()
    month_abbr_from_zero = month_abbr[1:]
    months_tail_pad = (
        [month_abbr_from_zero[i - 1].lower() + "_" for i in range(fromDateMin, 10)]
        if fromDateMin <= 9
        else []
    )
    months_head_pad = (
        [
            month_abbr_from_zero[(i - 1) % 12].lower() + "_"
            for i in range(10, toDateMax + 1)
        ]
        if toDateMax >= 10
        else []
    )
    return months_head_pad, months_tail_pad


def _rebuild_month_index(months_head_pad, months_tail_pad) -> None:
    """
    Recompute month list and index mappings given current tail/head pads.
    """
    month_list = _make_month_abbr_hydr(
        months_tail_pad, months_head_pad
    )  # returns ordered list of tokens

    month_pos = {m: i + 1 for i, m in enumerate(month_list)}

    return month_list, month_pos


def _make_month_abbr_hydr(
    months_tail_pad: List[str],
    months_head_pad: List[str],
) -> List[str]:
    """
    Create a flexible hydrological month token list depending on tail/head padding.

    Returns
    -------
    list[str] : ordered tokens, e.g.
        ['aug_', 'sep_', 'oct','nov','dec','jan','feb','mar','apr','may','jun','jul','aug','sep','oct_']
    """

    full = months_tail_pad + months_hydro_year + months_head_pad
    return full


def build_head_tail_pads_from_monthly_df(data):
    assert (
        "MONTHS" in data.keys()
    ), "The dataframe must be in monthly format but the provided dataframe has no 'MONTHS' field."
    hydro_months = [m.lower() for m in month_abbr[10:] + month_abbr[1:10]]
    headMonths = [m.lower() for m in month_abbr[10:] + month_abbr[1:4]]
    tailMonths = list(set(hydro_months).difference(headMonths))
    uniqueMonths = list(data.MONTHS.unique())
    for m in hydro_months:
        uniqueMonths.remove(m)
    uniqueMonths = [m.strip("_") for m in uniqueMonths]
    tail = sorted(
        set(uniqueMonths).intersection(tailMonths),
        key=lambda x: np.argwhere(np.array(hydro_months) == x)[0, 0],
    )
    tail = [m + "_" for m in tail]
    head = sorted(
        set(uniqueMonths).intersection(headMonths),
        key=lambda x: np.argwhere(np.array(hydro_months) == x)[0, 0],
    )
    head = [m + "_" for m in head]
    return head, tail
