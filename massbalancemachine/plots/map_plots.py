import xarray as xr
import pyproj
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import data_processing


def mapGlacier(df, rgi_id, year, cfg, ax=None, max_abs=None, title=None):

    df_glacier_year = df[(df.RGIId == rgi_id) & (df.YEAR == year)]

    # Initialize the OGGM Config
    data_processing.oggm_utils._initialize_oggm_config("")
    gdir = data_processing.oggm_utils._initialize_glacier_directories([rgi_id], cfg)[0]

    with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
        ds = ds.load()

    # Coordinate transformation from WGS84 to the projection of OGGM data
    transf = pyproj.Transformer.from_proj(
        pyproj.CRS.from_user_input("EPSG:4326"),
        pyproj.CRS.from_user_input(ds.pyproj_srs),
        always_xy=True,
    )
    lon = df_glacier_year["POINT_LON"].to_numpy()
    lat = df_glacier_year["POINT_LAT"].to_numpy()
    x, y = transf.transform(lon, lat)

    # Convert projected coordinates to nearest grid indices
    col = np.round((x - ds.x.values[0]) / (ds.x.values[1] - ds.x.values[0])).astype(int)
    row = np.round((y - ds.y.values[0]) / (ds.y.values[1] - ds.y.values[0])).astype(int)

    # Make an empty grid matching OGGM's gridded_data
    heat = xr.full_like(ds.topo, np.nan)
    valid = (row >= 0) & (row < ds.sizes["y"]) & (col >= 0) & (col < ds.sizes["x"])
    assert all(valid), "Some of the projected points fall outside of the OGGM grid"
    heat.values[row[valid], col[valid]] = df_glacier_year.loc[valid, "pred"].to_numpy()

    # Get background topography
    smap = ds.salem.get_map(countries=False)
    smap.set_shapefile(gdir.read_shapefile("outlines"))
    smap.set_topography(ds.topo.data)

    # Build color normalization (white is MB=0)
    max_abs = max_abs or df_glacier_year["pred"].abs().max()
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
    else:
        fig = None

    # Plot annual MB
    smap.set_cmap("RdBu")
    smap.set_norm(norm)
    smap.set_data(heat)
    smap.plot(ax=ax)
    smap.append_colorbar(ax=ax, label="Annual MB (m.w.e.)")
    ax.set_title(title or f"{rgi_id} year {year}")

    plt.tight_layout()

    return fig
