import xarray as xr
import pyproj
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree

import data_processing


def _point_cells_on_grid(x, y, values, grid):
    """Raster on the pixels of `grid` where each point fills its own grid cell.

    The points form a regular lattice whose spacing and orientation are inferred from
    the nearest neighbours, so that a lattice coarser than `grid` or rotated with
    respect to it leaves no gap. Pixels outside every cell are NaN.

    Args:
        x, y (np.ndarray): Point coordinates in the projection of `grid`.
        values (np.ndarray): Value of each point.
        grid (salem.Grid): Grid of the pixels, finer than the lattice.
    """
    xg, yg = grid.xy_coordinates
    out = np.full(xg.size, np.nan)
    tree = cKDTree(np.c_[x, y])
    if len(x) > 1:
        dist, nn = tree.query(np.c_[x, y], k=2)
        spacing = np.median(dist[:, 1])
        # Orientation of the lattice, defined modulo 90 degrees
        angle = np.arctan2(y[nn[:, 1]] - y, x[nn[:, 1]] - x)
        theta = np.angle(np.mean(np.exp(4j * angle))) / 4
    else:
        spacing, theta = abs(grid.dx), 0.0

    # The nearest point of a pixel is the one whose square cell can contain it
    pixels = np.c_[xg.ravel(), yg.ravel()]
    dist, near = tree.query(pixels, distance_upper_bound=spacing)
    found = np.flatnonzero(np.isfinite(dist))
    ex = pixels[found, 0] - x[near[found]]
    ey = pixels[found, 1] - y[near[found]]
    u = np.cos(theta) * ex + np.sin(theta) * ey
    v = -np.sin(theta) * ex + np.cos(theta) * ey
    half = spacing / 2 * (1 + 1e-6)
    inside = (np.abs(u) <= half) & (np.abs(v) <= half)
    out[found[inside]] = values[near[found[inside]]]
    return out.reshape(xg.shape)


def mapGlacier(
    df,
    rgi_id,
    cfg,
    year=None,
    ax=None,
    max_abs=None,
    title=None,
    gdir=None,
    mapOnly=False,
    reverse_cb=False,
    label_cb="Annual MB (m.w.e.)",
):

    if year is not None:
        df_glacier_year = df[(df.RGIId == rgi_id) & (df.YEAR == year)]
    else:
        df_glacier_year = df[(df.RGIId == rgi_id)]
    assert len(df_glacier_year) > 0, (
        f"No point of {rgi_id}"
        + (f" in {year}" if year is not None else "")
        + " to map."
    )

    if gdir is None:
        # Initialize the OGGM Config
        data_processing.oggm_utils._initialize_oggm_config("")
        gdir = data_processing.oggm_utils._initialize_glacier_directories(
            [rgi_id], cfg
        )[0]

    with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
        ds = ds.load()

    # Get background topography
    smap = ds.salem.get_map(countries=False)
    if not mapOnly:
        smap.set_shapefile(gdir.read_shapefile("outlines"))
        smap.set_topography(ds.topo.data)

    # The points are drawn on their own grid, which is not the OGGM one of `gdir` for
    # the sources gridded on other outlines (GLAMOS for instance)
    transf = pyproj.Transformer.from_crs(
        "EPSG:4326", smap.grid.proj.srs, always_xy=True
    )
    x, y = transf.transform(
        df_glacier_year["POINT_LON"].to_numpy(), df_glacier_year["POINT_LAT"].to_numpy()
    )
    heat = _point_cells_on_grid(x, y, df_glacier_year["pred"].to_numpy(), smap.grid)
    assert np.isfinite(heat).any(), f"The points of {rgi_id} fall outside of the map."

    # Build color normalization (white is MB=0)
    max_abs = max_abs or df_glacier_year["pred"].abs().max()
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
    else:
        fig = None

    # Plot annual MB
    smap.set_cmap("RdBu_r" if reverse_cb else "RdBu")
    smap.set_norm(norm)
    smap.set_data(heat)
    smap.plot(ax=ax)
    smap.append_colorbar(ax=ax, label=label_cb)
    ax.set_title(title or (f"{rgi_id} year {year}" if year is not None else rgi_id))

    plt.tight_layout()

    return fig


def mapGlacierArray(
    values,
    metadata,
    rgi_id=None,
    year=None,
    ax=None,
    max_abs=None,
    title=None,
):
    """Plot a 2D LV95 raster using its ESRI ASCII-grid metadata.

    The input array must use ESRI ASCII-grid row order, with the northernmost
    row first. No OGGM data or coordinate transformation is required.
    """
    values = np.asarray(values, dtype=float)
    ncols = int(metadata["ncols"])
    nrows = int(metadata["nrows"])
    if values.shape != (nrows, ncols):
        raise ValueError(f"values has shape {values.shape}; expected {(nrows, ncols)}")

    nodata_value = metadata.get("nodata_value", metadata.get("NODATA_value"))
    if nodata_value is not None:
        values[values == float(nodata_value)] = np.nan

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("values contains no finite data")
    max_abs = (
        float(np.max(np.abs(finite_values))) if max_abs is None else float(max_abs)
    )
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
    else:
        fig = None

    cellsize = float(metadata["cellsize"])
    xllcorner = float(metadata["xllcorner"])
    yllcorner = float(metadata["yllcorner"])
    x = xllcorner + (np.arange(ncols) + 0.5) * cellsize
    y = yllcorner + (np.arange(nrows) + 0.5) * cellsize
    custom_grid = xr.Dataset(
        {
            "heat": (("y", "x"), np.flip(values, axis=0)),
        },
        coords={"x": x, "y": y},
    )
    custom_grid.attrs["pyproj_srs"] = "EPSG:2056"

    smap = custom_grid.salem.get_map(countries=False)
    smap.set_cmap("RdBu")
    smap.set_norm(norm)
    smap.set_data(custom_grid.heat.values)
    smap.plot(ax=ax)
    smap.append_colorbar(ax=ax, label="Annual MB (m.w.e.)")
    default_title = "Annual MB"
    if rgi_id is not None and year is not None:
        default_title = f"{rgi_id} year {year}"
    ax.set_title(title or default_title)
    plt.tight_layout()

    return fig
