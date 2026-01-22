import os
import re
from tqdm.notebook import tqdm
import xarray as xr
import pandas as pd

from regions.Switzerland.scripts.config_CH import *
from regions.Switzerland.scripts.utils import *
from regions.Switzerland.scripts.geo_data import *


def load_glamos_grids(cfg, glacier_years, path_glamos):
    """
    Load gridded GLAMOS winter and annual mass-balance fields for multiple
    glaciers and years, interpolate DEM elevation onto the mass-balance grids,
    and return two combined pandas DataFrames.

    For each glacier and year, the function:
      1. Loads a masked DEM grid from a Zarr archive.
      2. Loads GLAMOS winter and annual mass-balance grids in WGS84 coordinates.
      3. Interpolates the DEM elevation onto the mass-balance grid using
         nearest-neighbour interpolation.
      4. Filters to valid grid cells with non-NaN mass balance and elevation.
      5. Aggregates all glaciers and years into separate DataFrames for
         winter and annual mass balance.

    Parameters
    ----------
    cfg : object
        Configuration object containing data paths and settings. Must provide
        ``cfg.dataPath`` and be compatible with ``load_glamos_wgs84``.
    glacier_years : dict
        Dictionary mapping glacier names (str) to iterable collections of
        years (int) for which GLAMOS data should be loaded.
        Example: ``{'Aletsch': [2000, 2001, 2002], ...}``
    path_glamos : str
        Base path to the directory containing GLAMOS glacier subdirectories.

    Returns
    -------
    df_GLAMOS_w : pandas.DataFrame
        DataFrame containing winter mass-balance data for all glaciers and
        years. Columns include:
            - 'year' (int)
            - 'glacier' (str)
            - 'apr' (float): winter mass balance
            - 'elevation' (float, meters)
    df_GLAMOS_a : pandas.DataFrame
        DataFrame containing annual mass-balance data for all glaciers and
        years. Columns include:
            - 'year' (int)
            - 'glacier' (str)
            - 'sep' (float): annual mass balance
            - 'elevation' (float, meters)

    Notes
    -----
    - GLAMOS mass-balance grids are loaded via ``load_glamos_wgs84`` and are
      assumed to be in WGS84 coordinates.
    - DEM elevation grids are loaded from Zarr files and interpolated onto
      the mass-balance grids using nearest-neighbour interpolation.
    - Only grid cells with valid (non-NaN) mass balance and elevation values
      are retained.
    - If no valid data are found for a given glacier or year, it is silently
      skipped.
    - The returned DataFrames do not retain spatial coordinates (x, y).

    """
    all_glacier_data_w, all_glacier_data_a = [], []

    for glacier_name in tqdm(glacier_years.keys(), desc="Loading GLAMOS data"):
        glacier_path = os.path.join(path_glamos, glacier_name)
        if not os.path.isdir(glacier_path):
            continue

        years = glacier_years[glacier_name]
        all_years_w, all_years_a = [], []

        for year in years:
            # --- Load DEM ---
            dem_path = (
                cfg.dataPath
                + path_GLAMOS_topo
                + f"xr_masked_grids/{glacier_name}_{year}.zarr"
            )
            if not os.path.exists(dem_path):
                continue
            ds_dem = xr.open_zarr(dem_path)

            # --- Load Winter MB ---
            ds_w = load_glamos_wgs84(cfg, glacier_name, year, period="winter")
            if ds_w is not None:
                masked_elev_interp = ds_dem["masked_elev"].interp_like(
                    ds_w, method="nearest"
                )
                masked_elev_interp = masked_elev_interp.assign_coords(
                    x=ds_w.x, y=ds_w.y
                )
                ds_merged_w = xr.merge(
                    [
                        ds_w.to_dataset(name="mb"),
                        masked_elev_interp.to_dataset(name="masked_elev"),
                    ],
                    compat="override",
                )

                df_w = ds_merged_w.to_dataframe().reset_index()
                df_w = df_w[df_w["mb"].notna() & df_w["masked_elev"].notna()]
                df_w = df_w[["x", "y", "mb", "masked_elev"]]
                df_w["year"] = year
                df_w["glacier"] = glacier_name
                df_w["period"] = "winter"
                all_years_w.append(df_w)

            # --- Load Annual MB ---
            ds_a = load_glamos_wgs84(cfg, glacier_name, year, period="annual")
            if ds_a is not None:
                masked_elev_interp = ds_dem["masked_elev"].interp_like(
                    ds_a, method="nearest"
                )
                masked_elev_interp = masked_elev_interp.assign_coords(
                    x=ds_a.x, y=ds_a.y
                )
                ds_merged_a = xr.merge(
                    [
                        ds_a.to_dataset(name="mb"),
                        masked_elev_interp.to_dataset(name="masked_elev"),
                    ],
                    compat="override",
                )

                df_a = ds_merged_a.to_dataframe().reset_index()
                df_a = df_a[df_a["mb"].notna() & df_a["masked_elev"].notna()]
                df_a = df_a[["x", "y", "mb", "masked_elev"]]
                df_a["year"] = year
                df_a["glacier"] = glacier_name
                df_a["period"] = "annual"
                all_years_a.append(df_a)

        # --- Concatenate per glacier ---
        if all_years_w:
            df_glacier_w = pd.concat(all_years_w, ignore_index=True)
            all_glacier_data_w.append(df_glacier_w)
        if all_years_a:
            df_glacier_a = pd.concat(all_years_a, ignore_index=True)
            all_glacier_data_a.append(df_glacier_a)

    # --- Final combined DataFrames ---
    df_GLAMOS_w = (
        pd.concat(all_glacier_data_w, ignore_index=True)
        if all_glacier_data_w
        else pd.DataFrame()
    )
    df_GLAMOS_a = (
        pd.concat(all_glacier_data_a, ignore_index=True)
        if all_glacier_data_a
        else pd.DataFrame()
    )

    # --- Drop x/y and rename elevation column ---
    if not df_GLAMOS_w.empty:
        df_GLAMOS_w = df_GLAMOS_w.drop(["x", "y", "period"], axis=1).rename(
            columns={"masked_elev": "elevation", "mb": "apr"}
        )
    if not df_GLAMOS_a.empty:
        df_GLAMOS_a = df_GLAMOS_a.drop(["x", "y", "period"], axis=1).rename(
            columns={"masked_elev": "elevation", "mb": "sep"}
        )

    return df_GLAMOS_w, df_GLAMOS_a


def load_glamos_wgs84(cfg, glacier, year, period):
    """
    Load a single GLAMOS distributed mass-balance grid and return it as an
    xarray.DataArray in WGS84 coordinates.

    The function locates the appropriate GLAMOS ``.grid`` file for the given
    glacier, year, and period (winter or annual), loads the gridded data,
    converts it to an ``xarray.DataArray``, and transforms the coordinates
    from the native Swiss coordinate system (LV03 or LV95) to WGS84.

    Parameters
    ----------
    cfg : object
        Configuration object containing data paths. Must provide
        ``cfg.dataPath`` and be compatible with ``pick_file_glamos``.
    glacier : str
        GLAMOS glacier identifier or name.
    year : int
        Hydrological year of the mass-balance grid.
    period : {"winter", "annual"}
        Mass-balance period to load. Use ``"winter"`` for winter balance
        and ``"annual"`` for annual balance.

    Returns
    -------
    da : xarray.DataArray or None
        Mass-balance grid as an ``xarray.DataArray`` in WGS84 coordinates.
        Returns ``None`` if no suitable grid file is found or if the coordinate
        system is unsupported.

    Notes
    -----
    - Supported source coordinate systems are:
        - LV03 (EPSG:21781)
        - LV95 (EPSG:2056)
    - Coordinate transformation is performed using nearest-neighbour
      reprojection via helper functions.
    - The returned DataArray contains mass-balance values only and does not
      include elevation or masking information.

    """
    path, cs = pick_file_glamos(cfg, glacier, year, period)
    if path is None:
        return None
    meta, arr = load_grid_file(path)
    da = convert_to_xarray_geodata(arr, meta)
    if cs == "lv03":
        return transform_xarray_coords_lv03_to_wgs84(da)
    elif cs == "lv95":
        return transform_xarray_coords_lv95_to_wgs84(da)
    else:
        return None


def pick_file_glamos(cfg, glacier, year, period="winter"):
    """
    Select the appropriate GLAMOS distributed mass-balance ``.grid`` file
    for a given glacier, year, and period.

    The function searches for fixed GLAMOS grid files in the standard
    directory structure and prioritizes LV95 grids over LV03 grids when
    both are available.

    Parameters
    ----------
    cfg : object
        Configuration object containing data paths. Must provide
        ``cfg.dataPath``.
    glacier : str
        GLAMOS glacier identifier or name.
    year : int
        Hydrological year of the mass-balance grid.
    period : {"winter", "annual"}, optional
        Mass-balance period to load. Default is ``"winter"``.

    Returns
    -------
    path : str or None
        Full path to the selected ``.grid`` file, or ``None`` if no file
        is found.
    cs : {"lv95", "lv03"} or None
        Coordinate system identifier of the selected grid file. Returns
        ``None`` if no file is found.

    Notes
    -----
    - File naming conventions are assumed to follow:
        ``<year>_<ann|win>_fix_<lv95|lv03>.grid``
    - LV95 grids are preferred over LV03 grids if both exist for the same
      glacier, year, and period.

    """
    suffix = "ann" if period == "annual" else "win"
    base = os.path.join(cfg.dataPath, path_distributed_MB_glamos, "GLAMOS", glacier)
    cand_lv95 = os.path.join(base, f"{year}_{suffix}_fix_lv95.grid")
    cand_lv03 = os.path.join(base, f"{year}_{suffix}_fix_lv03.grid")
    if os.path.exists(cand_lv95):
        return cand_lv95, "lv95"
    if os.path.exists(cand_lv03):
        return cand_lv03, "lv03"
    return None, None
