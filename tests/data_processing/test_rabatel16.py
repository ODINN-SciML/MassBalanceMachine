"""Tests of the Rabatel16 source: the preparation of its DEM, and the sum of the annual
mass balances over the period of the geodetic target.

None of them needs the Rabatel16 files or a download: the DEM and the tables are
synthetic.
"""

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from data_processing.gridded_utils import (
    _window_month_bounds,
    years_from_time_range,
)
from data_processing.oggm_utils import check_elevation_raster
from data_processing.rabatel16 import (
    DEM_NODATA,
    assert_rows_describe_same_glaciers,
    keep_rabatel16_series,
    period_sigma_mwe_per_year,
    period_smb_to_window,
    write_dem_with_nodata,
)

TRANSFORM = from_origin(300000, 5000000, 44.5, 53.1)


def _raw_dem(tmp_path):
    """A 6x6 DEM whose western column is the 0 m fill, and whose next two columns are
    the blended edge found along the border of the IGN DEM, repeated as it is where
    the raster duplicates a column."""
    raw = np.full((6, 6), 3200, dtype="float32")
    raw[:, 0] = 0
    raw[:, 1:3] = 2100
    path = str(tmp_path / "raw.tif")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=6,
        height=6,
        count=1,
        dtype="float32",
        crs="EPSG:32632",
        transform=TRANSFORM,
    ) as dst:
        dst.write(raw, 1)
    return raw, path


def _read(path):
    with rasterio.open(path) as dem:
        assert dem.nodata == DEM_NODATA
        assert dem.crs.to_epsg() == 32632
        # the non-square pixels of the raw grid are kept as they are
        assert dem.transform == TRANSFORM
        return dem.read(1, masked=True)


def test_zero_elevations_and_their_edge_become_missing(tmp_path):
    raw, src_path = _raw_dem(tmp_path)
    dst_path = str(tmp_path / "dem.tif")
    write_dem_with_nodata(src_path, dst_path)

    values = _read(dst_path)
    expected_missing = np.zeros_like(raw, dtype=bool)
    expected_missing[:, :3] = True
    np.testing.assert_array_equal(values.mask, expected_missing)
    np.testing.assert_array_equal(values.compressed(), raw[~expected_missing])
    check_elevation_raster(dst_path)


def test_edge_can_be_kept(tmp_path):
    raw, src_path = _raw_dem(tmp_path)
    dst_path = str(tmp_path / "dem.tif")
    write_dem_with_nodata(src_path, dst_path, edge_cells=0)

    values = _read(dst_path)
    np.testing.assert_array_equal(values.mask, raw == 0)
    np.testing.assert_array_equal(values.compressed(), raw[raw != 0])


def _annual_smb():
    """G1 has the whole of 1984-1987, G2 misses 1986, G3 has 1985 only."""
    rows = [
        ("G1", "a", y, s) for y, s in zip(range(1984, 1988), [0.4, -0.5, 0.2, -1.3])
    ]
    rows += [("G2", "b", y, s) for y, s in [(1984, -1.0), (1985, -0.8), (1987, -2.0)]]
    rows += [("G3", "c", 1985, -0.1)]
    return pd.DataFrame(rows, columns=["GLIMS_ID", "name", "year", "smb"])


def test_the_period_is_the_sum_of_its_hydrological_years():
    target = period_smb_to_window(_annual_smb(), 1985, 1987)

    assert list(target.RGIId) == ["G1"]
    g1 = target.iloc[0]
    assert g1.cumulative_mwe == pytest.approx(-0.5 + 0.2 - 1.3)
    assert g1.mwe_per_year == pytest.approx((-0.5 + 0.2 - 1.3) / 3)
    assert g1.sigma_mwe_per_year == pytest.approx(period_sigma_mwe_per_year(3))
    assert g1.n_years == 3

    # from the October before the first year to the October ending the last one
    window = (g1.FROM_DATE, g1.TO_DATE)
    assert window == (pd.Timestamp("1984-10-01"), pd.Timestamp("1987-10-01"))
    lo, hi = _window_month_bounds(window)
    assert hi - lo == 36
    assert list(years_from_time_range(*window)) == [1984, 1985, 1986, 1987]


def test_glaciers_missing_a_year_of_the_period_are_left_out():
    target = period_smb_to_window(_annual_smb(), 1984, 1985)
    assert list(target.RGIId) == ["G1", "G2"]
    assert list(target.cumulative_mwe) == pytest.approx([-0.1, -1.8])

    assert list(period_smb_to_window(_annual_smb(), 1985, 1985).RGIId) == [
        "G1",
        "G2",
        "G3",
    ]


def test_uncertainty_is_carried_from_the_reference_period():
    # the published value over the reference period, 1983-84 to 2013-14
    assert period_sigma_mwe_per_year(31) == pytest.approx(0.3)
    # only the year-to-year part of the annual error, sqrt(0.22² - 0.12²), averages
    # out: fewer years add a little of it on top of the shared part
    sigma_year_squared = 0.22**2 - 0.12**2
    for n in (1, 5, 16):
        expected = np.sqrt(0.3**2 + sigma_year_squared * (1 / n - 1 / 31))
        assert period_sigma_mwe_per_year(n) == pytest.approx(expected)
    assert period_sigma_mwe_per_year(16) == pytest.approx(0.3017, abs=1e-4)
    assert period_sigma_mwe_per_year(40) < 0.3 < period_sigma_mwe_per_year(30)


def test_only_series_reproducing_table_1_are_kept():
    table1 = {"RS": ("remote sensing", -1.0), "FIELD": ("field series", -1.0)}
    # 1984-2013 alternate around -1.0 and 2014 is -1.0: the mean over 1984-2014 is -1.0
    rows = [("RS", "rs", y, -1.0 + (0.5 if y % 2 else -0.5)) for y in range(1984, 2014)]
    rows += [("RS", "rs", 2014, -1.0)]
    # the field series of a glacier of Table 1 has another mean, and extends further
    rows += [("FIELD", "field", y, -1.2) for y in range(1959, 2016)]
    # a glacier the paper does not study
    rows += [("OTHER", "other", y, -1.0) for y in range(1984, 2015)]
    smb = pd.DataFrame(rows, columns=["GLIMS_ID", "name", "year", "smb"])

    kept = keep_rabatel16_series(smb, table1=table1)
    assert set(kept.GLIMS_ID) == {"RS"}
    assert len(kept) == 31


def test_a_glacier_of_table_1_cannot_go_missing():
    smb = pd.DataFrame(
        [("RS", "rs", 1990, -1.0)], columns=["GLIMS_ID", "name", "year", "smb"]
    )
    with pytest.raises(AssertionError, match="missing"):
        keep_rabatel16_series(smb, table1={"ABSENT": ("absent", -1.0)})


def test_a_period_cannot_end_before_it_starts():
    with pytest.raises(AssertionError):
        period_smb_to_window(_annual_smb(), 1987, 1985)


def test_rows_in_the_same_order_are_accepted_despite_spelling():
    assert_rows_describe_same_glaciers(
        ["Saint Sorlin ", "Glacier de la Selle_1", "Sarennes", "Argentière"],
        ["de Saint Sorlin", "de la Selle 1", "de Sarenne 1", "d'Argentiere"],
    )


def test_shifted_rows_are_refused():
    with pytest.raises(AssertionError, match="same order"):
        assert_rows_describe_same_glaciers(
            ["Saint Sorlin ", "Glacier de la Selle_1", "Sarennes", "Argentière"],
            ["de la Selle 1", "de Sarenne 1", "d'Argentiere", "de Saint Sorlin"],
        )
