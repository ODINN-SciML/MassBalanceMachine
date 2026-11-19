import os
import sys
import joblib

# Disable CUDA for all worker processes at import time
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import xarray as xr
import massbalancemachine as mbm

from papers.Transformer_TF.scripts.dataset import build_combined_LSTM_dataset


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
    build_ds: bool = False,
    ds_cache_root: str = None,
    monthly_cols: list = None,
    static_cols: list = None,
    months_head_pad: list = None,
    months_tail_pad: list = None,
) -> str:
    try:
        out_folder = os.path.join(out_folder_root, rgi_id)
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(out_folder, f"{rgi_id}_grid_{year}.parquet")

        ds_path = None
        if build_ds:
            if ds_cache_root is None or monthly_cols is None or static_cols is None:
                return (
                    f"ERROR {rgi_id} {year}: build_ds=True requires "
                    f"ds_cache_root, monthly_cols, static_cols"
                )
            ds_out_folder = os.path.join(ds_cache_root, rgi_id, str(year))
            os.makedirs(ds_out_folder, exist_ok=True)
            ds_path = os.path.join(ds_out_folder, "ds.joblib")

        parquet_exists = os.path.exists(out_path)
        ds_exists = (ds_path is not None) and os.path.exists(ds_path)

        if parquet_exists and (not build_ds or ds_exists):
            return f"SKIP {rgi_id} {year}: already exists"

        # --- FAST PATH: parquet already exists, only the dataset is missing ---
        # Read it back instead of rebuilding from zarr + ERA5 + monthly conversion.
        if parquet_exists and build_ds and not ds_exists:
            df_grid_monthly = pd.read_parquet(out_path)

            if len(df_grid_monthly) == 0:
                return f"SKIP {rgi_id} {year}: parquet empty (ds not built)"

            mbm.utils.seed_all(cfg.seed)
            seq_ds = build_combined_LSTM_dataset(
                df_loss=df_grid_monthly,
                df_full=df_grid_monthly,
                monthly_cols=monthly_cols,
                static_cols=static_cols,
                months_head_pad=months_head_pad,
                months_tail_pad=months_tail_pad,
                normalize_target=True,
                expect_target=True,
                show_progress=False,
            )

            if (
                (seq_ds.month_mean is not None)
                or (seq_ds.static_mean is not None)
                or (seq_ds.y_mean is not None)
            ):
                return f"ERROR {rgi_id} {year}: dataset unexpectedly has scalers set before caching"

            joblib.dump(seq_ds, ds_path, compress=3)
            return f"OK {rgi_id} {year} -> (from existing parquet) {ds_path}"

        # --- FULL PATH: build everything from zarr (parquet missing, or both missing) ---
        if not os.path.exists(zarr_path):
            return f"SKIP {rgi_id} {year}: zarr not found"

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
        df_grid_monthly = dataset_grid_topo.data.dropna(subset=vois_topo)

        if not parquet_exists:
            df_grid_monthly.to_parquet(out_path, engine="pyarrow", compression="snappy")

        if build_ds and not ds_exists:
            if len(df_grid_monthly) == 0:
                return f"SKIP {rgi_id} {year}: no grid rows after topo filtering (ds not built)"

            mbm.utils.seed_all(cfg.seed)
            seq_ds = build_combined_LSTM_dataset(
                df_loss=df_grid_monthly,
                df_full=df_grid_monthly,
                monthly_cols=monthly_cols,
                static_cols=static_cols,
                months_head_pad=months_head_pad,
                months_tail_pad=months_tail_pad,
                normalize_target=True,
                expect_target=True,
                show_progress=False,
            )

            if (
                (seq_ds.month_mean is not None)
                or (seq_ds.static_mean is not None)
                or (seq_ds.y_mean is not None)
            ):
                return f"ERROR {rgi_id} {year}: dataset unexpectedly has scalers set before caching"

            joblib.dump(seq_ds, ds_path, compress=3)

        return f"OK {rgi_id} {year} -> {out_path}" + (
            f" + {ds_path}" if build_ds else ""
        )

    except Exception as e:
        return f"ERROR {rgi_id} {year}: {type(e).__name__}: {e}"
