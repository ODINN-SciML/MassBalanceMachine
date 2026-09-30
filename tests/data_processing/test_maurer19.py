"""Tests of the Maurer19 source: the dates of the 1975-2000 period, the target built
from the averaged product, the polygonization of the 1975 masks and the identifiers.

The tests reading the NSIDC products are skipped when they have not been downloaded
to `.data/Maurer19`.
"""

import glob
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point

from data_processing.maurer19 import (
    END_DATE,
    geodetic_target_Maurer19,
    gridded_file_name,
    hexagon_start_date,
    mask_to_polygon,
    maurer19_avg_folder,
    maurer19_periods,
    parse_dem_years,
)
from dataloader.GeoDataLoader import buildGlacierMappingMultiSource, isNativeGlacierId


def _table():
    return pd.DataFrame(
        {
            "RGIId": ["RGI50-14.00001", "RGI50-15.00002", "RGI50-15.00003"],
            "mwe_per_year": [-0.2, -0.3, -0.1],
            "sigma_mwe_per_year": [0.1, 0.2, 0.15],
            "pctCov": [90.0, 40.0, 70.0],
            "pctDeb": [10.0, 50.0, 0.0],
            "dem_years": [[1974.0, 2000.0], [1975.5, 1976.0], [1973.9]],
            "area_mean_km2": [5.0, 12.0, 4.0],
        }
    )


def test_dem_years_are_parsed_whatever_the_separator():
    assert parse_dem_years("1974, 1975.83 2000") == [1974.0, 1975.83, 2000.0]
    assert parse_dem_years("[1976.0;2000.0]") == [1976.0, 2000.0]


def test_start_is_the_earliest_hexagon_dem():
    # A whole year is the 1st of January, a fraction the nearest first of a month
    assert hexagon_start_date([1974.0, 2000.0]) == pd.Timestamp("1974-01-01")
    assert hexagon_start_date([1976.0, 1975.5]) == pd.Timestamp("1975-07-01")
    assert hexagon_start_date([1973.99]) == pd.Timestamp("1974-01-01")
    with pytest.raises(AssertionError):
        hexagon_start_date([2000.0, 2005.0])


def test_periods_run_to_2000_with_their_length_in_months():
    periods = maurer19_periods(_table()).set_index("RGIId")
    row = periods.loc["RGI50-15.00002"]
    assert row.FROM_DATE == pd.Timestamp("1975-07-01")
    assert row.TO_DATE == END_DATE
    assert row.n_years == pytest.approx(24.5)
    assert row.cumulative_mwe == pytest.approx(-0.3 * 24.5)
    assert (periods.y1 == 2000).all()
    assert periods.index.is_unique


def test_target_filters():
    target = geodetic_target_Maurer19(
        table=_table(), require_1975_outline=False, min_coverage=50
    )
    assert list(target.RGIId) == ["RGI50-14.00001", "RGI50-15.00003"]
    target = geodetic_target_Maurer19(
        table=_table(),
        require_1975_outline=False,
        glacier_ids_to_keep=["RGI50-15.00002"],
    )
    assert list(target.RGIId) == ["RGI50-15.00002"]
    # 1975-2000 does not fit inside the years 1980 to 2000
    assert geodetic_target_Maurer19(
        table=_table(), require_1975_outline=False, allowed_years=range(1980, 2001)
    ).empty
    assert (
        len(
            geodetic_target_Maurer19(
                table=_table(), require_1975_outline=False, min_year=1974
            )
        )
        == 2
    )


def test_glaciers_without_a_1975_outline_are_left_out(monkeypatch):
    import data_processing.maurer19 as maurer19

    catalog = {
        gridded_file_name(g): "link" for g in ["RGI50-14.00001", "RGI50-15.00003"]
    }
    monkeypatch.setattr(maurer19, "gridded_catalog", lambda: catalog)
    target = geodetic_target_Maurer19(table=_table())
    assert list(target.RGIId) == ["RGI50-14.00001", "RGI50-15.00003"]


def test_gridded_file_name():
    assert (
        gridded_file_name("RGI50-15.02201")
        == "HMA_Glacier_dH_1975-2000_RGI50_15_02201.nc"
    )


@pytest.mark.parametrize("lat_ascending", [True, False])
def test_mask_is_polygonized_on_its_cells(tmp_path, lat_ascending):
    step = 0.001
    lat = 28.0 + step * np.arange(6)
    lon = 86.0 + step * np.arange(5)
    mask = np.zeros((6, 5))
    mask[1:4, 1:3] = 1  # 6 cells
    mask[5, 4] = 1  # one isolated cell
    if not lat_ascending:
        lat, mask = lat[::-1], mask[::-1]
    path = str(tmp_path / "glacier.nc")
    xr.Dataset(
        {"glacierStartMask": (("lat", "lon"), mask)},
        coords={"latitude": ("lat", lat), "longitude": ("lon", lon)},
    ).to_netcdf(path)

    polygon = mask_to_polygon(path)
    assert polygon.geom_type == "MultiPolygon"
    assert polygon.area == pytest.approx(7 * step**2)
    # Cell centres of the mask lie inside, the others outside
    assert polygon.contains(Point(86.0 + step, 28.0 + 2 * step))
    assert polygon.contains(Point(86.0 + 4 * step, 28.0 + 5 * step))
    assert not polygon.contains(Point(86.0, 28.0))


def test_rgi50_ids_are_native_to_maurer19_only():
    assert isNativeGlacierId("RGI50-15.02201", "Maurer19")
    assert not isNativeGlacierId("RGI60-15.02201", "Maurer19")
    assert not isNativeGlacierId("RGI50-15.02201", "GLAMOS")
    mapping = buildGlacierMappingMultiSource(
        ["RGI50-15.02201", "B36-26"],
        {"Maurer19": {"RGI50-15.02201"}, "GLAMOS": {"B36-26"}},
    )
    assert mapping == {
        "Maurer19": {"RGI50-15.02201": "RGI50-15.02201"},
        "GLAMOS": {"B36-26": "B36-26"},
    }


@pytest.mark.skipif(
    not glob.glob(os.path.join(maurer19_avg_folder(), "*1975-2000*.shp")),
    reason="HMA_GlacierAvg_dH not downloaded",
)
def test_averaged_product_reproduces_the_paper():
    periods = maurer19_periods()
    # About 650 glaciers over 1975-2000, larger than 3 km²
    assert 600 <= len(periods) <= 700
    assert periods.FROM_DATE.dt.year.between(1973, 1980).all()
    # Fig. 2A: mean -0.21 m w.e. per year; the regional mean of Table 1 is -0.22
    assert periods.mwe_per_year.mean() == pytest.approx(-0.21, abs=0.02)
    assert (periods.sigma_mwe_per_year > 0).all()
