import os
import sys

# Disable CUDA for all worker processes at import time
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import xarray as xr
import massbalancemachine as mbm

# ── helpers ──────────────────────────────────────────────────────────────────


def create_glacier_grid_RGI(year, rgi_id, ds, start_month="10"):
    """
    Generate a WGMS-style point grid dataframe from an RGI masked topography dataset.

    Parameters
    ----------
    year : int
    rgi_id : str
    ds : xarray.Dataset  (must contain glacier_mask, masked_elev, masked_aspect,
                          masked_slope, svf, lon, lat)
    start_month : str   e.g. "10"

    Returns
    -------
    pd.DataFrame
    """
    # Single source of truth for the mask
    gl_mask_bool = ds["glacier_mask"].values.astype(bool)
    glacier_indices = np.where(gl_mask_bool)

    # Build 2-D coordinate grids
    lon_2d, lat_2d = np.meshgrid(ds["lon"].values, ds["lat"].values)
    lon = lon_2d[glacier_indices]
    lat = lat_2d[glacier_indices]

    data_grid = {
        "RGIId": [rgi_id] * int(gl_mask_bool.sum()),
        "POINT_LAT": lat,
        "POINT_LON": lon,
        "aspect": ds["masked_aspect"].values[gl_mask_bool],
        "slope": ds["masked_slope"].values[gl_mask_bool],
        "topo": ds["masked_elev"].values[gl_mask_bool],
        "svf": ds["svf"].values[gl_mask_bool],
    }
    df_grid = pd.DataFrame(data_grid)

    df_grid["POINT_ID"] = np.arange(1, len(df_grid) + 1)
    df_grid["N_MONTHS"] = 12
    df_grid["POINT_ELEVATION"] = df_grid["topo"]
    df_grid["POINT_BALANCE"] = 0
    df_grid["PERIOD"] = "annual"
    df_grid["YEAR"] = year
    df_grid["FROM_DATE"] = f"{year}{start_month}01"
    df_grid["TO_DATE"] = f"{year + 1}0930"

    return df_grid


# ── worker initializer ────────────────────────────────────────────────────────


def init_worker_rgi():
    """Runs once per worker process at pool startup."""
    pass  # extend here if you need to load shared read-only resources


# ── main worker function ──────────────────────────────────────────────────────


def process_monthly_grids_rgi(
    region_id: str,
    rgi_id: str,
    year: int,
    *,
    zarr_path: str,
    basepath: str,
    out_folder_root: str,
    vois_climate: list,
    vois_topo: list,
    meta_cols: list,
    era5_monthly_path: str,
    era5_geopot_path: str,
    start_month: str,
    cfg,
) -> str:
    """
    Process one (rgi_id, year) pair:
      - loads zarr
      - builds glacier grid
      - adds climate features
      - converts to monthly
      - writes parquet

    Returns a status string: "OK ...", "SKIP ...", or "ERROR ..."
    """
    try:
        out_folder = os.path.join(out_folder_root, rgi_id)
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(out_folder, f"{rgi_id}_grid_{year}.parquet")

        if os.path.exists(out_path):
            return f"SKIP {rgi_id} {year}: already exists"

        if not os.path.exists(zarr_path):
            return f"SKIP {rgi_id} {year}: zarr not found"

        # Open zarr inside the worker — never pass xr.Dataset across the boundary
        ds = xr.open_zarr(zarr_path)

        df_grid = create_glacier_grid_RGI(
            year, rgi_id, ds, start_month=start_month
        ).reset_index(drop=True)

        dataset_grid = mbm.data_processing.Dataset(
            cfg=cfg,
            data=df_grid,
            region_name=region_id,
            region_id=region_id,
            data_path=basepath,
        )
        dataset_grid.get_climate_features(
            climate_data=era5_monthly_path,
            geopotential_data=era5_geopot_path,
            change_units=True,
            smoothing_vois={
                "vois_climate": vois_climate,
                "vois_other": ["ALTITUDE_CLIMATE"],
            },
        )

        df_y_gl = dataset_grid.data
        df_y_gl["GLWD_ID"] = df_y_gl.apply(
            lambda x: mbm.data_processing.utils.get_hash(f"{x.RGIId}_{x.YEAR}"),
            axis=1,
        ).astype(str)
        df_y_gl = df_y_gl.dropna(subset=["RGIId"])
        df_y_gl["GLACIER"] = df_y_gl["RGIId"]

        dataset_grid_topo = mbm.data_processing.Dataset(
            cfg=cfg,
            data=df_y_gl,
            region_name=region_id,
            region_id=region_id,
            data_path=basepath,
        )
        dataset_grid_topo.convert_to_monthly(
            meta_data_columns=meta_cols,
            vois_climate=vois_climate,
            vois_topographical=vois_topo,
        )
        # remove points that fall outside of glacier mask
        df_grid = dataset_grid_topo.data.dropna(subset=vois_topo)
        df_grid.to_parquet(out_path, engine="pyarrow", compression="snappy")
        return f"OK {rgi_id} {year} -> {out_path}"

    except Exception as e:
        return f"ERROR {rgi_id} {year}: {type(e).__name__}: {e}"
