"""Tests of the GSWP3-W5E5 climate source: the calendar months of the month tokens, the
replacement of t2m, tp, tp_sum, ALTITUDE_CLIMATE and ELEVATION_DIFFERENCE, the rows
outside 1901-2019, and the precomputed metadata.

They read OGGM's flattened GSWP3-W5E5 files and the RGI 6.2 outlines of region 11
from OGGM's download cache (downloaded on first use), and are skipped when these cannot
be obtained.
"""

import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data_processing import gswp3_w5e5 as gw

RHONE = "RGI60-11.01238"


@pytest.fixture(scope="module")
def flattened():
    try:
        return gw._flattened()
    except Exception as exc:  # no network and no cached files
        pytest.skip(f"GSWP3-W5E5 files not available: {exc}")


@pytest.fixture(scope="module")
def rhone_point(flattened):
    try:
        centres = gw._rgi_centres(11)
    except Exception as exc:
        pytest.skip(f"RGI 6.2 outlines not available: {exc}")
    return int(gw.nearest_points([centres.CenLon[RHONE]], [centres.CenLat[RHONE]])[0])


HYDRO_YEAR = [
    "oct",
    "nov",
    "dec",
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
]
CALENDAR_YEAR = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct_",
    "nov_",
    "dec_",
]


def _rows(tokens, year=2005, glacier=RHONE, elevation=2500.0, ids=None):
    n = len(tokens)
    return pd.DataFrame(
        {
            "RGIId": [glacier] * n,
            "ID": ids if ids is not None else [0] * n,
            "YEAR": [year] * n,
            "MONTHS": tokens,
            "POINT_ELEVATION": [elevation] * n,
            "POINT_LAT": [46.6] * n,
            "POINT_LON": [8.4] * n,
            "t2m": [-99.0] * n,
            "tp": [-99.0] * n,
            "tp_sum": [-99.0] * n,
            "ALTITUDE_CLIMATE": [-99.0] * n,
            "ELEVATION_DIFFERENCE": [-99.0] * n,
            "ssrd": [1.0] * n,
        }
    )


def _stakes(year=2005, **kw):
    """A stake table: one annual measurement over a hydrological year padded with
    August-September before it (`aug_`, `sep_`) and October after it (`oct_`)."""
    return _rows(["aug_", "sep_"] + HYDRO_YEAR + ["oct_"], year=year, **kw)


def test_calendar_months_of_a_stake_table():
    year, month = gw.calendar_months(_stakes(2005), grid=False)
    expected = [(2004, 8), (2004, 9), (2004, 10), (2004, 11), (2004, 12)] + [
        (2005, m) for m in range(1, 11)
    ]
    assert list(zip(year, month)) == expected


def test_calendar_months_of_a_grid():
    year, month = gw.calendar_months(_rows(CALENDAR_YEAR, year=2005), grid=True)
    assert list(zip(year, month)) == [(2005, m) for m in range(1, 13)]


def test_era5land_leaves_the_data_unchanged():
    df = _stakes()
    assert gw.apply_climate_source(df, "era5land", grid=False) is df


def test_unknown_source_raises():
    with pytest.raises(AssertionError):
        gw.apply_climate_source(_stakes(), "era5", grid=False)


def test_values_are_those_of_the_cell_of_oggm(flattened, rhone_point):
    from oggm.shop.w5e5 import get_gswp3_w5e5_file

    df = _rows(CALENDAR_YEAR, year=2004, elevation=2700.0)
    out = gw.apply_climate_source(df, "gswp3_w5e5", grid=True)
    times = pd.date_range("2004-01-01", "2004-12-01", freq="MS")
    with xr.open_dataset(get_gswp3_w5e5_file("GSWP3_W5E5", "tmp")) as ds:
        tas = ds.tas.isel(points=rhone_point).sel(time=times).values
    with xr.open_dataset(get_gswp3_w5e5_file("GSWP3_W5E5", "prcp")) as ds:
        pr = ds.pr.isel(points=rhone_point).sel(time=times).values
    with xr.open_dataset(get_gswp3_w5e5_file("GSWP3_W5E5", "inv")) as ds:
        asurf = float(np.asarray(ds.ASurf.values).reshape(-1)[rhone_point])

    np.testing.assert_allclose(out.t2m, tas.astype(float) - 273.15, atol=1e-4)
    np.testing.assert_allclose(out.tp, pr * 86400 / 1000, rtol=1e-6)  # m per day
    np.testing.assert_allclose(
        out.tp_sum, out.tp * times.days_in_month, rtol=1e-12
    )  # 2004: leap
    assert (out.ALTITUDE_CLIMATE == asurf).all()
    np.testing.assert_allclose(out.ELEVATION_DIFFERENCE, 2700.0 - asurf)
    # The other variables are kept
    assert (out.ssrd == 1.0).all()


def test_stakes_and_grid_of_a_glacier_share_its_cell(flattened, rhone_point):
    stake = gw.apply_climate_source(
        _stakes(2005, elevation=2300.0), "gswp3_w5e5", grid=False
    )
    grid = gw.apply_climate_source(
        _rows(CALENDAR_YEAR, year=2005, elevation=3000.0).assign(
            POINT_LAT=46.0, POINT_LON=7.0
        ),
        "gswp3_w5e5",
        grid=True,
    )
    # An RGI 6.2 glacier is placed at its RGI centre, whatever the rows' positions,
    # and the same month gets the same value: January 2005, row 5 of the stake table
    assert stake.ALTITUDE_CLIMATE.iloc[0] == grid.ALTITUDE_CLIMATE.iloc[0]
    assert stake.t2m.iloc[5] == grid.t2m.iloc[0]


def test_months_after_2019_raise_or_drop_their_measurement(flattened, rhone_point):
    # Hydrological year 2020 runs past December 2019; 2019 is covered
    df = pd.concat(
        [_stakes(2020, ids=[1] * 15), _stakes(2019, ids=[2] * 15)], ignore_index=True
    )
    with pytest.raises(ValueError):
        gw.apply_climate_source(df, "gswp3_w5e5", grid=False, missing="raise")
    out = gw.apply_climate_source(df, "gswp3_w5e5", grid=False, missing="drop")
    assert set(out.ID) == {2}
    assert len(out) == 15


def test_metadata_elevation_difference_follows_the_grid():
    metadata = pd.DataFrame(
        {
            "ID_int": [0, 0, 1],
            "GLWD_M_ID_int": [0, 1, 2],
            "ELEVATION_DIFFERENCE": [1.0, 2.0, 3.0],
        }
    )
    meta = {
        "metadata": metadata,
        "grouped_ids": pd.DataFrame(
            {"ELEVATION_DIFFERENCE": [1.0, 3.0]}, index=pd.Index([0, 1], name="ID_int")
        ),
        "grouped_glwd_m_ids": pd.DataFrame(
            {"ELEVATION_DIFFERENCE": [1.0, 2.0, 3.0]},
            index=pd.Index([0, 1, 2], name="GLWD_M_ID_int"),
        ),
    }
    grid = pd.DataFrame({"ELEVATION_DIFFERENCE": [10.0, 20.0, 30.0]})
    out = gw.replace_metadata_elevation_difference(meta, grid)
    assert list(out["metadata"].ELEVATION_DIFFERENCE) == [10.0, 20.0, 30.0]
    assert list(out["grouped_ids"].ELEVATION_DIFFERENCE) == [10.0, 30.0]
    assert list(out["grouped_glwd_m_ids"].ELEVATION_DIFFERENCE) == [10.0, 20.0, 30.0]
    # The input is not modified
    assert list(metadata.ELEVATION_DIFFERENCE) == [1.0, 2.0, 3.0]


def test_set_climate_source_from_params():
    from cli.common import setClimateSource

    cfg = types.SimpleNamespace(climate_source="era5land")
    setClimateSource(cfg, {"training": {}})
    assert cfg.climate_source == "era5land"
    setClimateSource(cfg, {"training": {"climate_source": "gswp3_w5e5"}})
    assert cfg.climate_source == "gswp3_w5e5"
    with pytest.raises(AssertionError):
        setClimateSource(cfg, {"training": {"climate_source": "cru"}})
