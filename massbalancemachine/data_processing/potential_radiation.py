"""Potential clear-sky direct radiation (`pcsr`) of the glacier grids.

The shortwave term of the TIlike model scales ERA5-Land `ssrd`, the radiation reaching
a flat and unshaded surface averaged over a ~9 km cell. What a pixel of a glacier
actually receives also depends on its slope, its aspect and the shadow of the
surrounding terrain, in a way that changes with the month and the latitude. `pcsr` is
that geometric signal: the daily mean of the potential clear-sky direct radiation on
the tilted pixel, in W m-2, for every calendar month, following Hock (1999):

    I = I0 (Rm/R)^2 psi^(P / (P0 cos Z)) cos(theta)

- I0 the solar constant, (Rm/R)^2 the eccentricity correction of the Earth's orbit;
- psi the clear-sky transmissivity at sea level, P/P0 = exp(-z / 8434.5 m) the ratio
  of the pressure at the pixel's elevation to sea level, Z the solar zenith angle;
- theta the angle between the sun and the normal of the pixel.

I is zero when the sun is behind the pixel's slope (cos theta <= 0) or below the
horizon of the pixel in the sun's direction. Only the direct beam is included: the
diffuse part is left to the sky view factor `svf`, already an input of the model.

The slope and aspect of a pixel come from the DEM of its grid tree (`dem.nc`, the OGGM
grid the products are built on). The horizon also needs the terrain far beyond that
grid, which only extends ~10 cells past the outline: it is taken from Copernicus
DEM 90 m tiles within `HORIZON_RADIUS` of the grid, downloaded and cached by OGGM
(`oggm.utils.get_topo_file`), reprojected to the grid's projection. Inside the extent
of `dem.nc`, the horizon is taken from `dem.nc` itself.

The result is written to `pcsr.nc` next to `svf.nc` in the folder of the glacier,
`pcsr(month, y, x)` on the x/y grid of `dem.nc`, every pixel of the grid included.
`pcsr_for_rows` attaches it to stake or grid rows from their coordinates and month
token.

Hock, R. (1999). A distributed temperature-index ice- and snowmelt model including
potential direct solar radiation. Journal of Glaciology, 45(149), 101-111.
"""

import calendar
import os
from calendar import month_abbr
from functools import lru_cache

import numpy as np
import pandas as pd
import pyproj
import tqdm
import rasterio
import rasterio.merge
import rasterio.transform
import rasterio.warp
import xarray as xr
from oggm import cfg as oggmCfg
from oggm import utils as oggmUtils
from scipy.ndimage import map_coordinates

from data_processing.Product import Product
from data_processing.product_utils import get_data_path, rgi_id_to_folders

PCSR_FILE = "pcsr.nc"
PCSR_COLUMN = "pcsr"

SOLAR_CONSTANT = 1368.0  # W m-2
TRANSMISSIVITY = 0.75  # psi, clear-sky transmissivity at sea level (Hock, 1999)
PRESSURE_SCALE_HEIGHT = 8434.5  # m, P/P0 = exp(-z / H)
EARTH_RADIUS = 6371000.0  # m, for the curvature of the horizon

HORIZON_RADIUS = 10000.0  # m around the grid
HORIZON_DEM_SOURCE = "COPDEM90"
HORIZON_DEM_RESOLUTION = 90.0  # m, the grid the tiles are reprojected to
N_AZIMUTHS = 72  # horizon directions, every 5 degrees

DAY_STEP = 5  # every 5th day of each month is integrated
TIME_STEPS_PER_DAY = 144  # 10 minutes
REFERENCE_YEAR = 2001  # non-leap; the geometry hardly changes from year to year

# Pixels processed at once, bounding the memory of the horizon search
PIXEL_CHUNK = 500

_MONTH_NAMES = [m.lower() for m in month_abbr]


# ---------------------------------------------------------------- solar geometry
def solar_declination_and_eccentricity(day_of_year):
    """Declination (rad) and eccentricity correction (Rm/R)^2 of the day of the year
    (1-365), Spencer (1971)."""
    g = 2 * np.pi * (np.asarray(day_of_year, dtype=float) - 1) / 365
    decl = (
        0.006918
        - 0.399912 * np.cos(g)
        + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g)
        + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g)
        + 0.00148 * np.sin(3 * g)
    )
    ecc = (
        1.000110
        + 0.034221 * np.cos(g)
        + 0.001280 * np.sin(g)
        + 0.000719 * np.cos(2 * g)
        + 0.000077 * np.sin(2 * g)
    )
    return decl, ecc


def solar_position(lat, decl, hour_angle):
    """Cosine of the zenith angle and azimuth (rad, clockwise from north) of the sun
    at latitude `lat` (rad), declination `decl` (rad) and `hour_angle` (rad, 0 at
    solar noon, negative in the morning)."""
    cos_z = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    # Azimuth from the south, positive towards the west, shifted to the north
    az_south = np.arctan2(
        np.sin(hour_angle),
        np.cos(hour_angle) * np.sin(lat) - np.tan(decl) * np.cos(lat),
    )
    return cos_z, np.mod(az_south + np.pi, 2 * np.pi)


def _sun_samples(lat, month):
    """Sun positions integrated for `month`: cos Z, azimuth, eccentricity correction,
    for the daytime steps of every `DAY_STEP`-th day, and the number of steps (day
    and night) they stand for."""
    first = pd.Timestamp(REFERENCE_YEAR, month, 1).dayofyear
    n_days = calendar.monthrange(REFERENCE_YEAR, month)[1]
    days = first + np.arange(0, n_days, DAY_STEP)
    hour_angle = (
        np.arange(TIME_STEPS_PER_DAY) + 0.5
    ) / TIME_STEPS_PER_DAY * 2 * np.pi - np.pi
    decl, ecc = solar_declination_and_eccentricity(days)
    cos_z, azimuth = solar_position(lat, decl[:, None], hour_angle[None, :])
    ecc = np.broadcast_to(ecc[:, None], cos_z.shape)
    day = cos_z > 0
    return cos_z[day], azimuth[day], ecc[day], cos_z.size


# ---------------------------------------------------------------- terrain
def surface_normals(topo, x, y):
    """Unit normals (east, north, up) of the pixels of a DEM, shape (ny, nx, 3).
    `x` and `y` are the coordinates of the columns and rows, `y` increasing to the
    north whatever its order in the array."""
    dz_dy, dz_dx = np.gradient(np.asarray(topo, dtype=float), y, x)
    n = np.stack([-dz_dx, -dz_dy, np.ones_like(dz_dx)], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


class _Raster:
    """A DEM on a regular grid, sampled bilinearly at map coordinates (NaN outside)."""

    def __init__(self, z, x, y):
        self.z = np.asarray(z, dtype=np.float32)
        self.x0, self.dx = float(x[0]), float(x[1] - x[0])
        self.y0, self.dy = float(y[0]), float(y[1] - y[0])
        self.nx, self.ny = len(x), len(y)

    def contains(self, xs, ys):
        col = (xs - self.x0) / self.dx
        row = (ys - self.y0) / self.dy
        return (col >= 0) & (col <= self.nx - 1) & (row >= 0) & (row <= self.ny - 1)

    def sample(self, xs, ys):
        col = (xs - self.x0) / self.dx
        row = (ys - self.y0) / self.dy
        return map_coordinates(
            self.z, [row.ravel(), col.ravel()], order=1, mode="constant", cval=np.nan
        ).reshape(np.shape(xs))


def horizon_angles(px, py, pz, fine, coarse, radius=HORIZON_RADIUS):
    """Elevation angle (rad) of the horizon of points (`px`, `py`, `pz`) in
    `N_AZIMUTHS` directions clockwise from north, shape (n_points, N_AZIMUTHS).

    The terrain is read from the `_Raster` `fine` where it covers the sample, else
    from `coarse` (None: `fine` only), up to `radius` metres, and lowered by the
    curvature of the Earth. A direction without any terrain has a horizon of 0.
    """
    step_fine = min(abs(fine.dx), abs(fine.dy))
    extent_fine = max(abs(fine.dx) * fine.nx, abs(fine.dy) * fine.ny)
    dist = np.arange(step_fine, min(extent_fine, radius) + step_fine, step_fine)
    if coarse is not None and dist[-1] < radius:
        step = min(abs(coarse.dx), abs(coarse.dy))
        dist = np.concatenate([dist, np.arange(dist[-1] + step, radius + step, step)])
    dist = dist.astype(np.float32)
    az = np.arange(N_AZIMUTHS) * 2 * np.pi / N_AZIMUTHS
    ex = (np.sin(az)[:, None] * dist[None, :]).astype(np.float32)
    ey = (np.cos(az)[:, None] * dist[None, :]).astype(np.float32)
    drop = (dist**2 / (2 * EARTH_RADIUS)).astype(np.float32)

    out = np.zeros((len(px), N_AZIMUTHS), dtype=np.float32)
    for s in range(0, len(px), PIXEL_CHUNK):
        sl = slice(s, s + PIXEL_CHUNK)
        xs = px[sl, None, None] + ex[None]
        ys = py[sl, None, None] + ey[None]
        z = fine.sample(xs, ys)
        if coarse is not None:
            outside = ~fine.contains(xs, ys)
            z[outside] = coarse.sample(xs[outside], ys[outside])
        tan = (z - drop - pz[sl, None, None]) / dist
        tan = np.where(np.isnan(tan), -np.inf, tan).max(axis=-1)
        out[sl] = np.arctan(np.maximum(tan, 0.0))
    return out


def _horizon_at(horizon, azimuth):
    """Horizon of every point in the direction(s) `azimuth`, linearly interpolated
    between the `N_AZIMUTHS` directions: shape (n_points, len(azimuth))."""
    pos = np.mod(azimuth, 2 * np.pi) / (2 * np.pi) * N_AZIMUTHS
    i0 = np.floor(pos).astype(int) % N_AZIMUTHS
    i1 = (i0 + 1) % N_AZIMUTHS
    w = (pos - np.floor(pos)).astype(np.float32)
    return horizon[:, i0] * (1 - w) + horizon[:, i1] * w


def direct_radiation_monthly(normals, elevation, horizon, lat):
    """Daily mean potential clear-sky direct radiation (W m-2) of every calendar month
    for points with unit `normals` (n, 3), `elevation` (n,), `horizon` (n,
    N_AZIMUTHS) at latitude `lat` (rad): shape (12, n)."""
    pressure_ratio = np.exp(-np.asarray(elevation, dtype=float) / PRESSURE_SCALE_HEIGHT)
    out = np.zeros((12, len(elevation)), dtype=np.float32)
    for month in range(1, 13):
        cos_z, azimuth, ecc, n_steps = _sun_samples(lat, month)
        sin_z = np.sqrt(np.clip(1 - cos_z**2, 0, None))
        sun = np.stack(
            [sin_z * np.sin(azimuth), sin_z * np.cos(azimuth), cos_z], axis=-1
        )
        sun_elevation = np.arcsin(np.clip(cos_z, -1, 1))
        for s in range(0, len(elevation), PIXEL_CHUNK):
            sl = slice(s, s + PIXEL_CHUNK)
            cos_theta = np.clip(normals[sl] @ sun.T, 0, None)
            lit = sun_elevation[None, :] > _horizon_at(horizon[sl], azimuth)
            beam = (
                SOLAR_CONSTANT
                * ecc[None, :]
                * TRANSMISSIVITY ** (pressure_ratio[sl, None] / cos_z[None, :])
            )
            out[month - 1, sl] = (beam * cos_theta * lit).sum(axis=1) / n_steps
    return out


# ---------------------------------------------------------------- horizon DEM
def _initialize_oggm():
    if not oggmCfg.PARAMS:
        oggmCfg.initialize(logging_level="WARNING")


def _horizon_bounds(x, y, radius):
    """Bounds (xmin, xmax, ymin, ymax) of the horizon DEM of a grid."""
    return (
        min(x[0], x[-1]) - radius,
        max(x[0], x[-1]) + radius,
        min(y[0], y[-1]) - radius,
        max(y[0], y[-1]) + radius,
    )


def horizon_tiles(srs, x, y, radius=HORIZON_RADIUS):
    """Paths of the `HORIZON_DEM_SOURCE` tiles covering `radius` around a grid of
    projection `srs` and coordinates `x`, `y`, downloaded to the OGGM cache if
    needed."""
    xmin, xmax, ymin, ymax = _horizon_bounds(x, y, radius)
    to_lonlat = pyproj.Transformer.from_crs(
        pyproj.CRS.from_user_input(srs), "EPSG:4326", always_xy=True
    )
    # The edges of the box, densely: a straight edge of the projection is curved in
    # longitude and latitude
    edge = np.linspace(0, 1, 50)
    bx = np.concatenate(
        [
            xmin + edge * (xmax - xmin),
            np.full(50, xmax),
            xmin + edge * (xmax - xmin),
            np.full(50, xmin),
        ]
    )
    by = np.concatenate(
        [
            np.full(50, ymin),
            ymin + edge * (ymax - ymin),
            np.full(50, ymax),
            ymin + edge * (ymax - ymin),
        ]
    )
    lon, lat = to_lonlat.transform(bx, by)
    _initialize_oggm()
    files, _ = oggmUtils.get_topo_file(
        [lon.min(), lon.max()], [lat.min(), lat.max()], source=HORIZON_DEM_SOURCE
    )
    return files


def horizon_tiles_of_dem(dem_path, radius=HORIZON_RADIUS):
    """`horizon_tiles` of the grid of a `dem.nc`."""
    with xr.open_dataset(dem_path) as dem:
        return horizon_tiles(
            dem.attrs["pyproj_srs"], dem.x.values, dem.y.values, radius
        )


def horizon_dem(srs, x, y, radius=HORIZON_RADIUS):
    """The `HORIZON_DEM_SOURCE` tiles around a grid of projection `srs` and
    coordinates `x`, `y`, reprojected to a `HORIZON_DEM_RESOLUTION` grid of the same
    projection extending `radius` past it: a `_Raster` (NaN where there is no tile)."""
    res = HORIZON_DEM_RESOLUTION
    xmin, xmax, ymin, ymax = _horizon_bounds(x, y, radius)
    files = horizon_tiles(srs, x, y, radius)

    xs = np.arange(xmin, xmax + res, res)
    ys = np.arange(ymax, ymin - res, -res)
    dst = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
    src = [rasterio.open(f) for f in files]
    try:
        mosaic, transform = rasterio.merge.merge(src, nodata=np.nan, dtype="float32")
        src_crs = src[0].crs
    finally:
        for s in src:
            s.close()
    rasterio.warp.reproject(
        mosaic[0],
        dst,
        src_transform=transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=rasterio.transform.from_origin(
            xs[0] - res / 2, ys[0] + res / 2, res, res
        ),
        dst_crs=pyproj.CRS.from_user_input(srs).to_wkt(),
        dst_nodata=np.nan,
        resampling=rasterio.warp.Resampling.bilinear,
    )
    return _Raster(dst, xs, ys)


# ---------------------------------------------------------------- per glacier
def compute_pcsr(dem_path, coarse=None):
    """The monthly potential clear-sky direct radiation of every pixel of `dem_path`
    (a `dem.nc`: `topo` on an OGGM x/y grid with `pyproj_srs`), as a Dataset with
    `pcsr(month, y, x)` in W m-2.

    `coarse` is the `_Raster` the horizon beyond `dem.nc` is read from, by default
    downloaded with `horizon_dem`; pass it to work offline or in tests.
    """
    with xr.open_dataset(dem_path) as dem:
        dem = dem.load()
    srs = dem.attrs["pyproj_srs"]
    x, y = dem.x.values.astype(float), dem.y.values.astype(float)
    topo = dem.topo.values.astype(float)
    assert np.isfinite(topo).all(), f"{dem_path} has missing elevations."
    if coarse is None:
        coarse = horizon_dem(srs, x, y)

    fine = _Raster(topo, x, y)
    xx, yy = np.meshgrid(x, y)
    normals = surface_normals(topo, x, y).reshape(-1, 3)
    horizon = horizon_angles(
        xx.ravel().astype(np.float32),
        yy.ravel().astype(np.float32),
        topo.ravel().astype(np.float32),
        fine,
        coarse,
    )

    to_lonlat = pyproj.Transformer.from_crs(
        pyproj.CRS.from_user_input(srs), "EPSG:4326", always_xy=True
    )
    _, lat_center = to_lonlat.transform(float(x.mean()), float(y.mean()))
    pcsr = direct_radiation_monthly(
        normals, topo.ravel(), horizon, np.radians(lat_center)
    )

    return xr.Dataset(
        {
            PCSR_COLUMN: (
                ("month", "y", "x"),
                pcsr.reshape(12, len(y), len(x)),
                {
                    "units": "W m-2",
                    "long_name": "Potential clear-sky direct radiation, daily mean of "
                    "the calendar month (Hock, 1999)",
                },
            )
        },
        coords={"month": np.arange(1, 13), "y": y, "x": x},
        attrs={
            "pyproj_srs": srs,
            "solar_constant": SOLAR_CONSTANT,
            "transmissivity": TRANSMISSIVITY,
            "horizon_radius_m": HORIZON_RADIUS,
            "horizon_dem": HORIZON_DEM_SOURCE,
            "n_azimuths": N_AZIMUTHS,
            "day_step": DAY_STEP,
            "time_steps_per_day": TIME_STEPS_PER_DAY,
        },
    )


def ensure_pcsr_file(glacier_folder):
    """Write `pcsr.nc` in the folder of a glacier from its `dem.nc`, unless it is
    already there; returns its path."""
    path = os.path.join(glacier_folder, PCSR_FILE)
    p = Product(path)
    if not p.is_up_to_date():
        dem_path = os.path.join(glacier_folder, "dem.nc")
        assert os.path.isfile(dem_path), (
            f"No DEM to compute the potential clear-sky radiation from: {dem_path} is "
            "missing. It is written with the sky view factor when the grids are built."
        )
        compute_pcsr(dem_path).to_netcdf(path)
        p.gen_chk()
    return path


@lru_cache(maxsize=64)
def _open_pcsr(path, mtime):
    with xr.open_dataset(path) as ds:
        return ds.load()


def calendar_month_of_tokens(months):
    """Calendar month (1-12) of month tokens of stakes or grids: 'jan', 'oct_',
    '_aug_' or the spin-up '_oct' are all the month they name."""
    codes, tokens = pd.factorize(np.asarray(months, dtype=object))
    assert (codes >= 0).all(), "Missing month tokens."
    names = pd.Series(tokens, dtype="object").str.strip("_")
    unknown = sorted(set(tokens[~names.isin(_MONTH_NAMES[1:]).to_numpy()]))
    assert not unknown, f"Unknown month tokens {unknown}."
    number = names.map(_MONTH_NAMES.index).to_numpy(dtype=np.int64)
    return number[codes]


def pcsr_for_rows(df, glacier_folder):
    """`pcsr` of rows with `POINT_LAT`, `POINT_LON` and a month token `MONTHS`: the
    value of their calendar month at the nearest pixel of the `pcsr.nc` of
    `glacier_folder` (written first if missing). Returns a float32 array."""
    path = ensure_pcsr_file(glacier_folder)
    ds = _open_pcsr(path, os.path.getmtime(path))
    to_grid = pyproj.Transformer.from_crs(
        "EPSG:4326", pyproj.CRS.from_user_input(ds.attrs["pyproj_srs"]), always_xy=True
    )
    lon = df["POINT_LON"].to_numpy(dtype=float)
    lat = df["POINT_LAT"].to_numpy(dtype=float)
    if len(df) == 1:
        # pyproj takes a 1-element array for a point and converts it to a scalar,
        # which NumPy deprecates
        x, y = (np.array([v]) for v in to_grid.transform(lon[0], lat[0]))
    else:
        x, y = to_grid.transform(lon, lat)
    gx, gy = ds.x.values, ds.y.values
    col = np.clip(np.rint((x - gx[0]) / (gx[1] - gx[0])).astype(int), 0, len(gx) - 1)
    row = np.clip(np.rint((y - gy[0]) / (gy[1] - gy[0])).astype(int), 0, len(gy) - 1)
    month = calendar_month_of_tokens(df["MONTHS"])
    return ds[PCSR_COLUMN].values[month - 1, row, col].astype(np.float32)


def pcsr_of_stakes(df):
    """`pcsr` of the rows of a monthly stake table (`RGIId`, `POINT_LAT`,
    `POINT_LON`, `MONTHS`): their month at their nearest pixel of the RGI grid of their
    glacier, in the `Hugonnet21` tree where their sky view factor comes from. Returns
    a float32 array aligned with the rows of `df`."""
    grid_path = os.path.join(get_data_path(), "grids", "Hugonnet21")
    groups = df.groupby("RGIId", sort=False).indices
    folders = {
        rgi_id: os.path.join(grid_path, *rgi_id_to_folders(rgi_id)) for rgi_id in groups
    }
    missing = sum(
        not Product(os.path.join(f, PCSR_FILE)).is_up_to_date()
        for f in folders.values()
    )
    print(
        f"Potential clear-sky radiation of the stakes of {len(groups)} glaciers, "
        f"{missing} of which need their {PCSR_FILE} computed (a few seconds each; "
        "scripts/compute_pcsr.py computes them in parallel beforehand)."
    )
    out = np.full(len(df), np.nan, dtype=np.float32)
    with tqdm.tqdm(groups.items(), total=len(groups), desc="pcsr of the stakes") as bar:
        for rgi_id, pos in bar:
            bar.set_postfix_str(rgi_id)
            out[pos] = pcsr_for_rows(df.iloc[pos], folders[rgi_id])
    return out
