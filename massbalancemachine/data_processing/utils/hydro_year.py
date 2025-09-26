import numpy as np
from calendar import month_abbr
from typing import Union, Callable, Dict, List, Optional, Tuple

months_hydro_year = [
    month.lower() for month in month_abbr[10:] + month_abbr[1:10]
]  # Standard hydro year (oct..sep)

# ---------------- Flexible month mapping ----------------


def _compute_head_tail_pads_from_df(df):
    fromDate = np.unique([int(str(date)[4:6]) for date in df.FROM_DATE])
    toDate = np.unique([int(str(date)[4:6]) for date in df.TO_DATE])
    months_tail_pad = (
        [month_abbr[i].lower() + "_" for i in range(min(fromDate), 10)]
        if min(fromDate) <= 9
        else []
    )
    months_head_pad = (
        [month_abbr[i].lower() + "_" for i in range(10, max(toDate) + 1)]
        if max(toDate) >= 10
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
