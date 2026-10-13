# --- standard library ---
import os

# --- third-party ---
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# ---------------- Sky-view factor (SVF) at stake points ---------------- #
def add_svf_from_rgi_zarr(
    df_pmb_topo,
    path_masked_grids,
    rgi_col="RGIId",
    lon_col="POINT_LON",
    lat_col="POINT_LAT",
    svf_var="svf",
    out_col="svf",
):
    """
    Add sky-view factor (SVF) to a PMB point dataframe by nearest-neighbor sampling
    from per-glacier Zarr grids named <RGIId>.zarr.

    Parameters
    ----------
    df_pmb_topo : pandas.DataFrame
        Must contain columns [rgi_col, lon_col, lat_col].
    path_masked_grids : str
        Directory containing <RGIId>.zarr datasets.
    rgi_col, lon_col, lat_col : str
        Column names in df_pmb_topo.
    svf_var : str
        Variable name in the Zarr dataset (default "svf").
    out_col : str
        Output column name in the dataframe.

    Returns
    -------
    pandas.DataFrame
        Copy of df_pmb_topo (filtered to glaciers that exist on disk) with added `out_col`.
    """

    # Which zarr files exist?
    zarr_ids = {f[:-5] for f in os.listdir(path_masked_grids) if f.endswith(".zarr")}

    # Keep only rows whose RGIId exists on disk
    df = df_pmb_topo[df_pmb_topo[rgi_col].isin(zarr_ids)].copy()

    # Initialize output
    df[out_col] = np.nan

    # Group by glacier (RGIId) and sample in bulk
    for rgi_id, group in df.groupby(rgi_col):
        file_path = os.path.join(path_masked_grids, f"{rgi_id}.zarr")
        try:
            # Zarr: open with open_zarr (your example), works well here
            ds = xr.open_zarr(file_path)

            # nearest-neighbor sample
            lon = group[lon_col].to_numpy()
            lat = group[lat_col].to_numpy()

            stake = ds[svf_var].sel(
                lon=xr.DataArray(lon, dims="points"),
                lat=xr.DataArray(lat, dims="points"),
                method="nearest",
            )

            # write back
            df.loc[group.index, out_col] = stake.to_numpy()

        except Exception as e:
            # keep going; column stays NaN for that glacier
            print(f"[add_svf_from_rgi_zarr] {rgi_id}: {type(e).__name__}: {e}")
            continue

    return df


def plot_missing_svf_for_all_glaciers(
    df_with_svf,
    path_masked_xr,
    rgi_col="RGIId",
    lon_col="POINT_LON",
    lat_col="POINT_LAT",
    svf_col="svf",  # column in df
    svf_var="svf",  # var name in zarr
    plot_valid_points=True,
    save_dir=None,
):
    """
    Plot SVF rasters and overlay stake points with missing SVF values for each glacier.

    For each glacier (grouped by `rgi_col`) where at least one point has missing SVF
    (`df_with_svf[svf_col]` is NaN), the function:
    - opens the corresponding Zarr dataset `<RGIId>.zarr` in `path_masked_xr`
    - plots the SVF raster (`svf_var`) on its lon/lat grid
    - overlays the missing-SVF points in red
    - optionally overlays valid points (non-NaN SVF) in white for context
    - either shows each figure interactively or saves PNGs to `save_dir`

    Parameters
    ----------
    df_with_svf : pandas.DataFrame
        Point dataset containing at least columns:
        - `rgi_col` (e.g., 'RGIId')
        - `lon_col`, `lat_col` (point coordinates)
        - `svf_col` (SVF values, may contain NaNs)
    path_masked_xr : str
        Directory containing per-glacier Zarr datasets named `<RGIId>.zarr`.
    rgi_col, lon_col, lat_col : str, optional
        Column names in `df_with_svf` for glacier id and coordinates.
    svf_col : str, optional
        Column name in `df_with_svf` holding SVF values.
    svf_var : str, optional
        Variable name inside the Zarr datasets (default 'svf').
    plot_valid_points : bool, optional
        If True, plot non-missing SVF points as well (white markers).
    save_dir : str or None, optional
        If provided, figures are saved as `<RGIId>_svf_missing.png` into this folder.
        If None, figures are displayed via `plt.show()`.

    Returns
    -------
    None

    Notes
    -----
    This is a diagnostic plot. Missing SVF often indicates that points fall outside
    the valid/masked SVF area (e.g., off-glacier pixels) or that glacier IDs / CRS
    do not match the raster grids.
    """
    df_missing = df_with_svf[df_with_svf[svf_col].isna()].copy()
    glaciers = sorted(df_missing[rgi_col].unique())

    print(f"Plotting {len(glaciers)} glaciers with missing SVF points.")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for rgi_id in glaciers:
        zpath = os.path.join(path_masked_xr, f"{rgi_id}.zarr")
        if not os.path.exists(zpath):
            print(f"[skip] missing zarr: {rgi_id}")
            continue

        ds = xr.open_zarr(zpath)
        da = ds[svf_var]

        miss = df_missing[df_missing[rgi_col] == rgi_id]
        lon_m = miss[lon_col].to_numpy()
        lat_m = miss[lat_col].to_numpy()

        if plot_valid_points:
            ok = df_with_svf[
                (df_with_svf[rgi_col] == rgi_id) & (~df_with_svf[svf_col].isna())
            ]
            lon_ok = ok[lon_col].to_numpy()
            lat_ok = ok[lat_col].to_numpy()

        fig, ax = plt.subplots(figsize=(7, 6))

        # SVF raster
        da.plot(ax=ax, x="lon", y="lat", robust=True, add_colorbar=True)

        # Missing points (red)
        ax.scatter(
            lon_m,
            lat_m,
            s=22,
            c="red",
            edgecolor="k",
            linewidth=0.3,
            label=f"missing svf (n={len(miss)})",
        )

        # Valid points (white) for context
        if plot_valid_points and len(ok) > 0:
            ax.scatter(
                lon_ok,
                lat_ok,
                s=12,
                c="white",
                edgecolor="k",
                linewidth=0.2,
                alpha=0.7,
                label=f"valid svf (n={len(ok)})",
            )

        ax.set_title(f"{rgi_id} — SVF with missing points")
        ax.legend(loc="upper right")

        # Zoom to raster extent
        ax.set_xlim(float(da["lon"].min()), float(da["lon"].max()))
        ax.set_ylim(float(da["lat"].min()), float(da["lat"].max()))

        plt.tight_layout()

        if save_dir is not None:
            out = os.path.join(save_dir, f"{rgi_id}_svf_missing.png")
            plt.savefig(out, dpi=200)
            plt.close(fig)
        else:
            plt.show()


def _nearest_index_1d(coord_vals, query_vals):
    """
    Return nearest indices in a sorted 1D coord array for query values.
    Works for increasing or decreasing coords.
    """
    coord_vals = np.asarray(coord_vals)
    q = np.asarray(query_vals)

    # ensure increasing for searchsorted
    increasing = coord_vals[0] < coord_vals[-1]
    if not increasing:
        coord_inc = coord_vals[::-1]
        idx = np.searchsorted(coord_inc, q)
        idx = np.clip(idx, 1, len(coord_inc) - 1)
        left = coord_inc[idx - 1]
        right = coord_inc[idx]
        choose_right = (q - left) > (right - q)
        out = idx - 1 + choose_right
        # map back to original (reversed) indices
        out = (len(coord_vals) - 1) - out
        return out.astype(int)

    idx = np.searchsorted(coord_vals, q)
    idx = np.clip(idx, 1, len(coord_vals) - 1)
    left = coord_vals[idx - 1]
    right = coord_vals[idx]
    choose_right = (q - left) > (right - q)
    out = idx - 1 + choose_right
    return out.astype(int)


def _fill_nearest_valid(values2d, iy, ix, max_radius=30):
    """
    For a single point index (iy, ix), if values2d[iy,ix] is NaN,
    search outward in square windows until a non-NaN is found.
    Return that value (nearest by Euclidean distance in index space).
    """
    if not np.isnan(values2d[iy, ix]):
        return values2d[iy, ix]

    ny, nx = values2d.shape

    for r in range(1, max_radius + 1):
        y0 = max(0, iy - r)
        y1 = min(ny, iy + r + 1)
        x0 = max(0, ix - r)
        x1 = min(nx, ix + r + 1)

        window = values2d[y0:y1, x0:x1]
        mask = ~np.isnan(window)
        if not mask.any():
            continue

        # indices of valid cells in the window
        vy, vx = np.where(mask)

        # convert to full-array indices
        vy_full = vy + y0
        vx_full = vx + x0

        # choose nearest valid by distance in index space
        dy = vy_full - iy
        dx = vx_full - ix
        k = np.argmin(dy * dy + dx * dx)

        return values2d[vy_full[k], vx_full[k]]

    return np.nan


def add_svf_nearest_valid(
    df_pmb_topo,
    path_masked_grids,
    rgi_col="RGIId",
    lon_col="POINT_LON",
    lat_col="POINT_LAT",
    svf_var="svf",
    out_col="svf",
    max_radius=30,
):
    """
    Add SVF from per-glacier zarr. For points that land on NaN (masked),
    fill using the nearest non-NaN SVF pixel within `max_radius` cells.
    """
    zarr_ids = {f[:-5] for f in os.listdir(path_masked_grids) if f.endswith(".zarr")}
    df = df_pmb_topo[df_pmb_topo[rgi_col].isin(zarr_ids)].copy()
    df[out_col] = np.nan

    for rgi_id, group in df.groupby(rgi_col):
        zpath = os.path.join(path_masked_grids, f"{rgi_id}.zarr")
        try:
            ds = xr.open_zarr(zpath)
            da = ds[svf_var]

            # assume 2D field on (lat, lon) with 1D coords lat/lon
            lons = da["lon"].values
            lats = da["lat"].values
            arr = da.values  # (lat, lon) typically

            qlon = group[lon_col].to_numpy()
            qlat = group[lat_col].to_numpy()

            ix = _nearest_index_1d(lons, qlon)
            iy = _nearest_index_1d(lats, qlat)

            sampled = arr[iy, ix].astype(float)

            # fill NaNs by nearest valid pixel
            nan_idx = np.where(np.isnan(sampled))[0]
            if len(nan_idx) > 0:
                for j in nan_idx:
                    sampled[j] = _fill_nearest_valid(
                        arr, int(iy[j]), int(ix[j]), max_radius=max_radius
                    )

            df.loc[group.index, out_col] = sampled

        except Exception as e:
            print(f"[add_svf_nearest_valid] {rgi_id}: {type(e).__name__}: {e}")
            continue

    return df


def plot_glacier_svf_with_points(
    df_with_svf,
    path_masked_xr,
    rgi_col="RGIId",
    lon_col="POINT_LON",
    lat_col="POINT_LAT",
    svf_col="svf",  # column in df
    svf_var="svf",  # variable in zarr
    save_dir=None,
    dpi=200,
):
    """
    Plot SVF raster maps and overlay point SVF values for each glacier.

    For each glacier found in `df_with_svf[rgi_col]`, the function:
    - opens the per-glacier Zarr dataset `<RGIId>.zarr` from `path_masked_xr`
    - plots the SVF raster (`svf_var`) on its lon/lat grid
    - overlays stake points colored by their assigned SVF values in `df_with_svf[svf_col]`
    - optionally marks points still missing SVF (NaNs) as red 'x'
    - either displays figures interactively or saves them to disk

    Parameters
    ----------
    df_with_svf : pandas.DataFrame
        Point dataset with columns:
        - `rgi_col` : glacier id matching Zarr filenames (e.g., 'RGIId')
        - `lon_col`, `lat_col` : point coordinates (WGS84 lon/lat)
        - `svf_col` : assigned SVF values (float; may include NaNs)
    path_masked_xr : str
        Directory containing per-glacier Zarr datasets named `<RGIId>.zarr`.
    rgi_col, lon_col, lat_col : str, optional
        Column names in `df_with_svf`.
    svf_col : str, optional
        Column holding SVF values to color the points.
    svf_var : str, optional
        Variable name inside the Zarr datasets (default 'svf').
    save_dir : str or None, optional
        If provided, save figures as `<RGIId>_svf_points.png` in this folder.
        If None, figures are displayed via `plt.show()`.
    dpi : int, optional
        Resolution for saved figures.

    Returns
    -------
    None

    Notes
    -----
    The raster colorbar (background) and point colorbar (assigned SVF) may not match
    exactly if the raster is masked or uses different scaling; the point colorbar
    reflects values actually assigned to the stake locations.
    """
    glaciers = sorted(df_with_svf[rgi_col].dropna().unique())

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    print(f"Plotting {len(glaciers)} glaciers.")

    for rgi_id in glaciers:
        gl_name = df_with_svf[df_with_svf[rgi_col] == rgi_id]["GLACIER"].iloc[0]
        zpath = os.path.join(path_masked_xr, f"{rgi_id}.zarr")
        if not os.path.exists(zpath):
            print(f"[skip] missing zarr: {rgi_id}")
            continue

        try:
            ds = xr.open_zarr(zpath)
            da = ds[svf_var]

            g = df_with_svf[df_with_svf[rgi_col] == rgi_id]
            lon = g[lon_col].to_numpy()
            lat = g[lat_col].to_numpy()
            vals = g[svf_col].to_numpy()

            ok = ~np.isnan(vals)
            miss = np.isnan(vals)

            fig, ax = plt.subplots(figsize=(7.5, 6.2))

            # Background raster
            im = da.plot(ax=ax, x="lon", y="lat", robust=True, add_colorbar=True)

            # Overlay: points colored by assigned SVF
            if ok.any():
                sc = ax.scatter(
                    lon[ok],
                    lat[ok],
                    c=vals[ok],
                    s=28,
                    edgecolor="k",
                    linewidth=0.3,
                )
                # colorbar for point values (separate from raster)
                cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("SVF (assigned to points)")

            # If anything still missing, show as red x
            if miss.any():
                ax.scatter(
                    lon[miss],
                    lat[miss],
                    s=50,
                    marker="x",
                    c="red",
                    linewidth=1.2,
                    label=f"still missing (n={miss.sum()})",
                )
                ax.legend(loc="upper right")

            ax.set_title(f"{gl_name}-{rgi_id} — SVF raster + stake-point SVF")
            ax.set_xlim(float(da["lon"].min()), float(da["lon"].max()))
            ax.set_ylim(float(da["lat"].min()), float(da["lat"].max()))
            plt.tight_layout()

            if save_dir is not None:
                out = os.path.join(save_dir, f"{rgi_id}_svf_points.png")
                plt.savefig(out, dpi=dpi)
                plt.close(fig)
            else:
                plt.show()

        except Exception as e:
            print(f"[error] {gl_name}-{rgi_id}: {type(e).__name__}: {e}")
            continue
