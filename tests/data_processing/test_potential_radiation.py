"""Tests of the potential clear-sky direct radiation (`data_processing.potential_radiation`).

None of them downloads data: the DEMs, the horizon DEM and the `pcsr.nc` are synthetic.
"""

import numpy as np
import pandas as pd
import pyproj
import pytest
import xarray as xr

import data_processing.product_utils as product_utils
from data_processing.Product import Product
from data_processing.product_utils import rgi_id_to_folders
from data_processing.potential_radiation import (
    N_AZIMUTHS,
    PCSR_FILE,
    _Raster,
    calendar_month_of_tokens,
    compute_pcsr,
    direct_radiation_monthly,
    horizon_angles,
    pcsr_for_rows,
    pcsr_of_stakes,
    solar_declination_and_eccentricity,
    solar_position,
    surface_normals,
)

LAT = np.radians(46.0)
SRS = "+proj=utm +zone=32 +datum=WGS84 +units=m +no_defs"
DEC, JUN = 11, 5  # indices of December and June


def _normal(slope_deg, aspect_deg):
    s, a = np.radians(slope_deg), np.radians(aspect_deg)
    return np.array([[np.sin(s) * np.sin(a), np.sin(s) * np.cos(a), np.cos(s)]])


def _pcsr(slope_deg, aspect_deg, horizon=None, elevation=2500.0):
    if horizon is None:
        horizon = np.zeros((1, N_AZIMUTHS), dtype=np.float32)
    return direct_radiation_monthly(
        _normal(slope_deg, aspect_deg), np.array([elevation]), horizon, LAT
    )[:, 0]


def test_declination_and_eccentricity():
    decl, ecc = solar_declination_and_eccentricity([80, 172, 355])
    assert np.degrees(decl) == pytest.approx([0.0, 23.44, -23.44], abs=0.6)
    assert ecc[2] > 1 > ecc[1]  # Earth closest to the sun in early January


def test_solar_position_at_noon_and_morning():
    decl = np.radians(23.44)
    cos_z, az = solar_position(LAT, decl, 0.0)
    assert np.degrees(np.arccos(cos_z)) == pytest.approx(46.0 - 23.44, abs=1e-6)
    assert np.degrees(az) == pytest.approx(180.0)
    _, az_morning = solar_position(LAT, decl, np.radians(-45))
    assert 0 < np.degrees(az_morning) < 180


@pytest.mark.parametrize("y_order", [1, -1])
def test_surface_normals_of_a_north_facing_plane(y_order):
    x = np.arange(5) * 10.0
    y = np.arange(5)[::y_order] * 10.0
    topo = -np.tan(np.radians(30)) * y[:, None] + 0 * x[None, :]  # lower to the north
    n = surface_normals(topo, x, y)
    assert n[2, 2] == pytest.approx(_normal(30, 0)[0])


def test_flat_surface_bounds_and_seasonal_cycle():
    flat = _pcsr(0, 0)
    assert (flat > 0).all()
    assert flat[JUN] > flat[DEC]
    # Below the daily mean at the top of the atmosphere at the June solstice
    assert flat.max() < 1368 * 1.05 / np.pi * 1.2


def test_slope_and_aspect_in_december():
    south, flat, north = _pcsr(30, 180), _pcsr(0, 0), _pcsr(15, 0)
    assert south[DEC] > flat[DEC] > north[DEC] > 0
    # A slope facing north steeper than the noon sun (20.6 deg) gets nothing in December
    assert _pcsr(30, 0)[DEC] == pytest.approx(0.0, abs=1e-3)
    # East and west are symmetric
    assert _pcsr(30, 90) == pytest.approx(_pcsr(30, 270), rel=1e-4)


def test_wall_to_the_south_shades_in_winter_only():
    x = np.arange(-2000, 2001, 20.0)
    y = np.arange(2000, -2001, -20.0)
    topo = np.zeros((len(y), len(x)))
    topo[(y > -600) & (y < -500)] = 1000.0  # ridge 500 m south of the origin
    fine = _Raster(topo, x, y)
    horizon = horizon_angles(
        np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        fine,
        None,
    )
    south = horizon[0, N_AZIMUTHS // 2]
    assert np.degrees(south) == pytest.approx(np.degrees(np.arctan(1000 / 500)), abs=2)
    assert horizon[0, 0] == pytest.approx(0.0)  # north is open

    shaded = _pcsr(0, 0, horizon=horizon, elevation=0.0)
    open_sky = _pcsr(0, 0, elevation=0.0)
    assert shaded[DEC] == pytest.approx(0.0, abs=1e-3)
    assert shaded[JUN] > 0.5 * open_sky[JUN]


def _dem_file(folder, topo, x, y):
    ds = xr.Dataset(
        {"topo": (("y", "x"), topo)}, coords={"x": x, "y": y}, attrs={"pyproj_srs": SRS}
    )
    path = folder / "dem.nc"
    ds.to_netcdf(path)
    return path


def test_compute_pcsr_flat_dem_matches_flat_point(tmp_path):
    x = 400000 + np.arange(10) * 30.0
    y = 5100000 - np.arange(8) * 30.0
    path = _dem_file(tmp_path, np.full((8, 10), 2500.0), x, y)
    cx, cy = np.meshgrid(
        np.arange(x[0] - 12000, x[-1] + 12000, 90.0),
        y[0] + 12000 - np.arange(0, 24300, 90.0),
    )
    coarse = _Raster(np.full(cx.shape, 2500.0), cx[0], cy[:, 0])
    ds = compute_pcsr(path, coarse=coarse)
    assert ds.pcsr.dims == ("month", "y", "x")
    _, lat = pyproj.Transformer.from_crs(SRS, "EPSG:4326", always_xy=True).transform(
        x.mean(), y.mean()
    )
    expected = direct_radiation_monthly(
        _normal(0, 0), np.array([2500.0]), np.zeros((1, N_AZIMUTHS)), np.radians(lat)
    )[:, 0]
    # The curvature of the Earth only lowers the far terrain: no shading on a plain
    np.testing.assert_allclose(
        ds.pcsr.values, expected[:, None, None] * np.ones((1, 8, 10)), rtol=1e-5
    )


def test_calendar_month_of_tokens():
    tokens = ["jan", "sep", "oct_", "_aug_", "aug_", "_oct", "dec"]
    assert list(calendar_month_of_tokens(tokens)) == [1, 9, 10, 8, 8, 10, 12]
    with pytest.raises(AssertionError):
        calendar_month_of_tokens(["foo"])


X = 400000 + np.arange(4) * 50.0
Y = 5100000 - np.arange(3) * 50.0


def _pcsr_file(folder, offset=0):
    """A `pcsr.nc` whose value encodes month, row and column, plus `offset`."""
    folder.mkdir(parents=True, exist_ok=True)
    month = np.arange(1, 13)
    values = (
        month[:, None, None] * 1000
        + np.arange(3)[None, :, None] * 10
        + np.arange(4)[None, None, :]
        + offset
    ).astype(np.float32)
    xr.Dataset(
        {"pcsr": (("month", "y", "x"), values)},
        coords={"month": month, "y": Y, "x": X},
        attrs={"pyproj_srs": SRS},
    ).to_netcdf(folder / PCSR_FILE)
    Product(str(folder / PCSR_FILE)).gen_chk()


def test_pcsr_for_rows_picks_pixel_and_month(tmp_path):
    x, y = X, Y
    _pcsr_file(tmp_path)

    to_lonlat = pyproj.Transformer.from_crs(SRS, "EPSG:4326", always_xy=True)
    lon, lat = to_lonlat.transform([x[1] + 10, x[3], x[0]], [y[2], y[0] - 20, y[1]])
    df = pd.DataFrame(
        {"POINT_LON": lon, "POINT_LAT": lat, "MONTHS": ["jan", "oct_", "_jul"]}
    )
    out = pcsr_for_rows(df, str(tmp_path))
    np.testing.assert_array_equal(out, [1000 + 21, 10000 + 3, 7000 + 10])


def test_pcsr_of_stakes_reads_the_grid_of_each_glacier(tmp_path, monkeypatch):
    monkeypatch.setattr(product_utils, "_data_path", str(tmp_path))
    ids = ["RGI60-11.00001", "RGI60-11.01450"]
    for k, rgi_id in enumerate(ids):
        _pcsr_file(
            tmp_path.joinpath("grids", "Hugonnet21", *rgi_id_to_folders(rgi_id)),
            offset=100 * k,
        )
    lon, lat = pyproj.Transformer.from_crs(SRS, "EPSG:4326", always_xy=True).transform(
        X[[0, 2, 1]], Y[[0, 1, 2]]
    )
    # Rows of the two glaciers interleaved, on a non-default index
    df = pd.DataFrame(
        {
            "RGIId": [ids[1], ids[0], ids[1]],
            "POINT_LON": lon,
            "POINT_LAT": lat,
            "MONTHS": ["jun", "dec", "_oct"],
        },
        index=[7, 3, 5],
    )
    np.testing.assert_array_equal(
        pcsr_of_stakes(df), [6000 + 100, 12000 + 12, 10000 + 100 + 21]
    )
