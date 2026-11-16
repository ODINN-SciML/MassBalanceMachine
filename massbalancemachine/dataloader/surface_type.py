"""Time sequences for the surface type (snow or ice) of the TIlike model, see the
`sfc_type` block of the parameters, `models.TorchNeuralNetworkRegressor.TILikeSfcModel`.

The snow fraction of a row depends on the cumulative mass balance of the previous
months of the same point within the same hydrological year (1 October - 30
September). This module builds, from the metadata of the rows, the information the
model needs to compute these cumulative sums on rows that are not ordered in time:

- `perm`: the order of the rows sorted by sequence, then by month;
- `start`: for each row in that order, the position (in that order) of the first row
  of its sequence;
- `spinup`: whether the row is a spin-up row, whose mass balance the model sets to
  zero.

A sequence is one point and one hydrological year. A point is a stake measurement
(its `ID`) for the stakes, and a grid cell (its `POINT_LAT`/`POINT_LON`) for the
geodetic grids, whose IDs are per point and calendar year. The stakes and the grids
both carry spin-up rows, the months before a measurement or a geodetic window since
the start of its hydrological year, with the climate and topography of their own
location (`spinup_month_list` of `transform_to_monthly.transform_to_monthly`, `spinup=True` in
`gridded_utils.load_grid_multi_years`).
"""

from calendar import month_abbr

import numpy as np
import pandas as pd
import torch

from data_processing.gridded_utils import GRID_TREE_COLUMN, SPINUP_COLUMN
from data_processing.utils.hydro_year import stake_calendar_months

# The months of a geodetic grid after September, in its calendar YEAR
GRID_PADDED_MONTHS = ["oct_", "nov_", "dec_"]


def calendar_months(years, months, months_head_pad=None):
    """Calendar year and month number (1-12) of rows given their `YEAR` and `MONTHS`,
    see `utils.hydro_year.stake_calendar_months`, which also covers the geodetic grids
    (calendar years: `jan` to `sep`, then `oct_`, `nov_`, `dec_`)."""
    return stake_calendar_months(years, months, months_head_pad)


def hydro_year_sequences(point_key, years, months, months_head_pad=None):
    """Order of the rows by sequence (point and hydrological year) and month.

    Args:
        point_key (np.ndarray): Integer key of the point of every row.
        years, months: The `YEAR` and `MONTHS` columns, see `calendar_months`.
        months_head_pad (list): The months padded after the hydrological year of the
            whole table the rows are taken from, see `calendar_months`.

    Returns a dict with the np.ndarray of int64 `perm` and `start` described in the
    module docstring.
    """
    point_key = np.asarray(point_key, dtype=np.int64)
    if len(point_key) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return {"perm": empty, "start": empty.copy()}
    cal_year, number = calendar_months(years, months, months_head_pad)
    hydro_year = cal_year + (number >= 10)
    time = cal_year * 12 + number - 1
    hy_min = hydro_year.min()
    seq = point_key * (hydro_year.max() - hy_min + 1) + (hydro_year - hy_min)
    perm = np.lexsort((time, seq))
    seq_sorted, time_sorted = seq[perm], time[perm]
    first = np.ones(len(perm), dtype=bool)
    first[1:] = seq_sorted[1:] != seq_sorted[:-1]
    duplicated = ~first & np.r_[False, time_sorted[1:] == time_sorted[:-1]]
    assert (
        not duplicated.any()
    ), f"{duplicated.sum()} rows repeat the month of another row of the same point."
    start = np.maximum.accumulate(np.where(first, np.arange(len(perm)), 0))
    return {"perm": perm.astype(np.int64), "start": start.astype(np.int64)}


def sequences_to_torch(seq, spinup=None):
    """The output of `hydro_year_sequences` as the `sfc` argument of the model."""
    sfc = {k: torch.from_numpy(v) for k, v in seq.items()}
    sfc["spinup"] = (
        torch.from_numpy(np.asarray(spinup, dtype=bool)) if spinup is not None else None
    )
    return sfc


def stake_sequences(df, months_head_pad=None):
    """`sfc` argument of the model for a stake table, one sequence per measurement
    and hydrological year. `months_head_pad`: the months padded after the
    hydrological year of the whole table, when `df` is a subset of it (see
    `calendar_months`)."""
    point_key, _ = pd.factorize(df["ID"].values)
    seq = hydro_year_sequences(
        point_key, df["YEAR"].values, df["MONTHS"].values, months_head_pad
    )
    spinup = df[SPINUP_COLUMN].values if SPINUP_COLUMN in df.columns else None
    return sequences_to_torch(seq, spinup)


def grid_sequences(df):
    """`sfc` argument of the model for a geodetic grid, one sequence per grid point
    and hydrological year.

    A grid with the spin-up months of the surface type (`spinup=True` in
    `gridded_utils.load_grid_multi_years`) flags them in `SFC_SPINUP`, and keys its
    points by `GRID_TREE` too: the windows of a glacier on outlines of different
    epochs are read from different trees, whose points must not be chained."""
    keys = ["POINT_LAT", "POINT_LON"]
    if GRID_TREE_COLUMN in df.columns:
        keys = [GRID_TREE_COLUMN] + keys
    point_key = df.groupby(keys, sort=False).ngroup().values
    seq = hydro_year_sequences(
        point_key, df["YEAR"].values, df["MONTHS"].values, GRID_PADDED_MONTHS
    )
    spinup = df[SPINUP_COLUMN].values if SPINUP_COLUMN in df.columns else None
    return sequences_to_torch(seq, spinup)
