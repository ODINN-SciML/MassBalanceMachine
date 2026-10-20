"""
GSWP3-W5E5 temperature and precipitation in place of those of ERA5-Land.

GSWP3-W5E5 (GSWP3 before 1979, W5E5 after; 1901-2019) is the observational climate of
ISIMIP3a and the forcing of OGGM's glacier projections. With `climate_source:
gswp3_w5e5` in the `training` block of the params, the stake tables and the geodetic
grids keep everything they were built with except five columns, replaced when they are
read (`apply_climate_source`), each month with the GSWP3-W5E5 value of the month whose
ERA5-Land value it replaces (`calendar_months`):

    t2m                    monthly mean temperature of the GSWP3-W5E5 cell, degC
    tp                     precipitation, m per day (the ERA5-Land convention)
    tp_sum                 tp times the days of the month, m
    ALTITUDE_CLIMATE       surface altitude of the cell, m
    ELEVATION_DIFFERENCE   POINT_ELEVATION - ALTITUDE_CLIMATE, m

The other climate variables (fal, slhf, sshf, ssrd, str, u10, v10) stay those of
ERA5-Land: OGGM's GSWP3-W5E5 files only have temperature and precipitation.

The cell of a glacier is the one OGGM gives its GSWP3-W5E5 glacier directory: the point
of OGGM's flattened files closest to the centre of its RGI 6.2 outline (CenLon, CenLat),
as `oggm.shop.w5e5.process_gswp3_w5e5_data` chooses it. A glacier that is not an RGI 6.2
glacier (the entities of the custom-outline sources) is placed at the mean position of
its rows. The stakes and the grid of a glacier therefore share one cell.

The files are OGGM's own (`oggm.shop.w5e5.get_gswp3_w5e5_file`), downloaded once to its
download cache, so the climate is exactly the one of the OGGM runs.
"""

import functools
from calendar import month_abbr

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from data_processing.glacier_utils import get_region_shape_file
from data_processing.gridded_utils import MONTH_TO_ID
from data_processing.utils.hydro_year import (
    build_head_tail_pads_from_monthly_df,
    _rebuild_month_index,
)

CLIMATE_SOURCES = ("era5land", "gswp3_w5e5")
DEFAULT_CLIMATE_SOURCE = "era5land"
REPLACED_COLUMNS = ["t2m", "tp", "tp_sum", "ALTITUDE_CLIMATE", "ELEVATION_DIFFERENCE"]

FIRST_YEAR = 1901


@functools.lru_cache(maxsize=1)
def _flattened():
    """OGGM's flattened GSWP3-W5E5 monthly files: (temperature degC and precipitation
    m per day, both time x points, the point longitudes in 0-360, latitudes and surface
    altitudes)."""
    from oggm.shop.w5e5 import get_gswp3_w5e5_file

    with xr.open_dataset(get_gswp3_w5e5_file("GSWP3_W5E5", "inv")) as ds:
        lon = ds.longitude.values.astype(float)
        lat = ds.latitude.values.astype(float)
        altitude = np.asarray(ds.ASurf.values, dtype=float).reshape(-1)
    with xr.open_dataset(get_gswp3_w5e5_file("GSWP3_W5E5", "tmp")) as ds:
        time = pd.DatetimeIndex(ds.time.values)
        temp = ds.tas.values.astype(float) - 273.15
    with xr.open_dataset(get_gswp3_w5e5_file("GSWP3_W5E5", "prcp")) as ds:
        assert (pd.DatetimeIndex(ds.time.values) == time).all()
        prcp = ds.pr.values.astype(float) * 86400.0 / 1000.0  # kg m-2 s-1 -> m per day
    assert (
        time[0] == pd.Timestamp(f"{FIRST_YEAR}-01-01")
        and (
            (time.year - FIRST_YEAR) * 12 + time.month - 1 == np.arange(len(time))
        ).all()
    ), "the GSWP3-W5E5 files are not consecutive months from January 1901"
    return temp, prcp, lon, lat, altitude


def nearest_points(lons, lats):
    """Index of the point of the flattened files closest to each (lon, lat), with the
    metric of `oggm.shop.w5e5.process_gswp3_w5e5_data`."""
    _, _, plon, plat, _ = _flattened()
    lons = np.asarray(lons, dtype=float)
    lons = np.where(lons < 0, lons + 360.0, lons)
    lats = np.asarray(lats, dtype=float)
    d = (plon[None, :] - lons[:, None]) ** 2 + (plat[None, :] - lats[:, None]) ** 2
    return np.argmin(d, axis=1)


def altitude_climate_at(lats, lons):
    """Surface altitude of the GSWP3-W5E5 cell of each (lat, lon), m."""
    return _flattened()[4][nearest_points(lons, lats)]


@functools.lru_cache(maxsize=None)
def _rgi_centres(rgi_region):
    """CenLon, CenLat of the RGI 6.2 glaciers of a region, indexed by RGIId."""
    return gpd.read_file(
        get_region_shape_file(f"{int(rgi_region):02d}"),
        ignore_geometry=True,
        columns=["RGIId", "CenLon", "CenLat"],
    ).set_index("RGIId")


def glacier_points(df, glacier_column="RGIId"):
    """The GSWP3-W5E5 point of each glacier of `df` ({glacier id: point index})."""
    glaciers = pd.unique(df[glacier_column])
    lon, lat = {}, {}
    for g in glaciers:
        g = str(g)
        if g.startswith("RGI60-"):
            centres = _rgi_centres(g.split("-")[1].split(".")[0])
            if g in centres.index:
                lon[g], lat[g] = centres.CenLon[g], centres.CenLat[g]
                continue
        rows = df[glacier_column].astype(str) == g
        lon[g], lat[g] = df.POINT_LON[rows].mean(), df.POINT_LAT[rows].mean()
    ids = list(lon)
    return dict(zip(ids, nearest_points([lon[g] for g in ids], [lat[g] for g in ids])))


def calendar_months(df, grid):
    """Calendar year and month of each row of a monthly table, from its `YEAR` and its
    month token `MONTHS`, with the conventions the stored ERA5-Land values were
    attached with:

    - a stake table (`grid=False`) runs over hydrological years padded on both sides:
      its tokens are those of `utils.hydro_year._rebuild_month_index` for the padding of
      the table (`build_head_tail_pads_from_monthly_df`), consecutive months from the
      first token, in YEAR - 1, as `get_climate_data._create_date_range` dates them;
    - a geodetic grid (`grid=True`) runs over calendar years: every token of a row is in
      its YEAR, numbered by `gridded_utils.MONTH_TO_ID`, as `geodetic_window_weights`
      reads them."""
    year = df.YEAR.astype(int)
    if grid:
        months = year * 12 + df.MONTHS.map(MONTH_TO_ID) - 1
    else:
        head, tail = build_head_tail_pads_from_monthly_df(df)
        month_list, month_pos = _rebuild_month_index(head, tail)
        first = list(month_abbr).index(month_list[0].strip("_").capitalize())
        months = (year - 1) * 12 + first - 1 + df.MONTHS.map(month_pos) - 1
    assert months.notna().all(), f"unknown month tokens {set(df.MONTHS[months.isna()])}"
    months = months.astype(int).to_numpy()
    return months // 12, months % 12 + 1


def apply_climate_source(
    df, climate_source, grid, glacier_column="RGIId", missing="raise"
):
    """`df` with the columns of `REPLACED_COLUMNS` taken from `climate_source`.

    `df` is a monthly table, a stake table or a geodetic grid (`grid`, see
    `calendar_months`), with `glacier_column`, `YEAR`, `MONTHS`, `POINT_ELEVATION`,
    `POINT_LAT` and `POINT_LON`. With "era5land" it is returned unchanged. Rows outside 1901-2019 have no GSWP3-W5E5 climate: with
    `missing="raise"` they raise, with `missing="drop"` every measurement (`ID`) with
    such a month is dropped, and the number dropped is printed."""
    assert (
        climate_source in CLIMATE_SOURCES
    ), f"climate_source must be one of {CLIMATE_SOURCES}, got {climate_source}"
    if climate_source == "era5land" or len(df) == 0:
        return df
    temp, prcp, _, _, altitude = _flattened()
    points = glacier_points(df, glacier_column)
    point = df[glacier_column].astype(str).map(points).to_numpy()
    cal_year, month = calendar_months(df, grid)
    t = (cal_year - FIRST_YEAR) * 12 + month - 1
    inside = (t >= 0) & (t < temp.shape[0])
    if not inside.all():
        if missing == "raise":
            raise ValueError(
                f"{(~inside).sum()} rows fall outside GSWP3-W5E5 ({FIRST_YEAR}-2019), "
                f"years {sorted(set(cal_year[~inside]))}"
            )
        assert missing == "drop", f"missing must be 'raise' or 'drop', got {missing}"
        bad_ids = set(df.ID[~inside])
        keep = ~df.ID.isin(bad_ids).to_numpy()
        print(
            f"GSWP3-W5E5 ends in 2019: {len(bad_ids)} measurements "
            f"({(~keep).sum()} monthly rows) after it are dropped"
        )
        df, point, cal_year, month, t = (
            df[keep],
            point[keep],
            cal_year[keep],
            month[keep],
            t[keep],
        )
    df = df.copy()
    days = pd.to_datetime(
        pd.DataFrame({"year": cal_year, "month": month, "day": 1})
    ).dt.days_in_month.to_numpy()
    df["t2m"] = temp[t, point]
    df["tp"] = prcp[t, point]
    if "tp_sum" in df.columns:
        df["tp_sum"] = df["tp"].to_numpy() * days
    df["ALTITUDE_CLIMATE"] = altitude[point]
    df["ELEVATION_DIFFERENCE"] = df["POINT_ELEVATION"].to_numpy() - altitude[point]
    return df


def replace_metadata_elevation_difference(precomputed_meta, df):
    """Bring the `ELEVATION_DIFFERENCE` of precomputed geodetic metadata (see
    `gridded_utils.load_precomputed_metadata`), whose rows are those of the multi-year
    grid, in line with the grid `df` after `apply_climate_source`."""
    metadata = precomputed_meta["metadata"]
    if "ELEVATION_DIFFERENCE" not in metadata.columns:
        return precomputed_meta
    assert len(metadata) == len(
        df
    ), "the precomputed metadata and the grid have different rows"
    metadata = metadata.assign(
        ELEVATION_DIFFERENCE=df["ELEVATION_DIFFERENCE"].to_numpy()
    )
    out = dict(precomputed_meta, metadata=metadata)
    for key, by in [("grouped_ids", "ID_int"), ("grouped_glwd_m_ids", "GLWD_M_ID_int")]:
        if key in out and "ELEVATION_DIFFERENCE" in out[key].columns:
            first = metadata.groupby(by).ELEVATION_DIFFERENCE.first()
            out[key] = out[key].assign(
                ELEVATION_DIFFERENCE=first.reindex(out[key].index).to_numpy()
            )
    return out
