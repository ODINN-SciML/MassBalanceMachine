# --- standard library ---
import os
import re
import tempfile
import shutil
import logging
from calendar import monthrange
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- third-party ---
import numpy as np
import pandas as pd
import xarray as xr
import pyproj
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial.distance import cdist
from tqdm import tqdm

# --- project/local ---
import massbalancemachine as mbm
from oggm import utils, workflow, tasks
from oggm import cfg as oggmCfg
from regions.Switzerland.scripts.wgs84_ch1903 import *
from regions.Switzerland.scripts.config_CH import *
from regions.Switzerland.scripts.helpers import *
from regions.Switzerland.scripts.geodata import (
    LV03toWGS84,
    xr_SGI_masked_topo,
    coarsenDS,
    get_rgi_sgi_ids,
    transformDates,
    load_grid_file,
)


# --- per-process initializer (caps threads, seeds if you want) ---
def _worker_init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


# --- single task executed in a worker ---
def _process_one_item_sgi(item, cfg, type_, path_save, path_SGI_topo, path_xr_svf):
    """
    Returns tuple: (status, item, msg)
    status in {"ok","skip","err"}
    """
    try:
        # lazily import heavy deps inside worker to reduce parent footprint
        import geopandas as gpd

        # read shapefile once per worker process
        shp_path = os.path.join(
            cfg.dataPath, path_SGI_topo, "inventory_sgi2016_r2020/SGI_2016_glaciers.shp"
        )
        glacier_outline_sgi = gpd.read_file(shp_path)

        # resolve SGI id
        if type_ == "glacier_name":
            sgi_id, rgi_id, rgi_shp = get_rgi_sgi_ids(cfg, item)
            if not sgi_id:
                return ("skip", item, "Missing SGI ID")
        elif type_ == "sgi_id":
            sgi_id = item
        else:
            return ("err", item, f"Unknown type '{type_}'")

        # build dataset (in wgs84)
        ds = xr_SGI_masked_topo(glacier_outline_sgi, sgi_id, cfg)
        if ds is None:
            return ("skip", item, "xr_SGI_masked_topo returned None")

        # resample
        ds_latlon = coarsenDS(ds)
        if ds_latlon is None:
            return ("skip", item, "coarsenDS returned None")

        # Merge with SVF
        # load corresponding SVF (already in lat/lon) and merge

        svf_path = os.path.join(path_xr_svf, f"{sgi_id}_svf_latlon.nc")
        if not os.path.exists(svf_path):
            print(f"SVF not found for {sgi_id}: {svf_path}")
        else:
            ds_svf = xr.open_dataset(svf_path)

        # Sort ascending for interpolation stability
        if ds_latlon.lon[0] > ds_latlon.lon[-1]:
            ds_latlon = ds_latlon.sortby("lon")
        if ds_latlon.lat[0] > ds_latlon.lat[-1]:
            ds_latlon = ds_latlon.sortby("lat")
        if ds_svf.lon[0] > ds_svf.lon[-1]:
            ds_svf = ds_svf.sortby("lon")
        if ds_svf.lat[0] > ds_svf.lat[-1]:
            ds_svf = ds_svf.sortby("lat")

        svf_vars = [v for v in ["svf", "asvf", "opns"] if v in ds_svf.data_vars]

        # If grids match, merge; else interpolate SVF to ds_latlon grid
        if np.array_equal(ds_latlon.lon.values, ds_svf.lon.values) and np.array_equal(
            ds_latlon.lat.values, ds_svf.lat.values
        ):
            ds_latlon = xr.merge([ds_latlon, ds_svf[svf_vars]])
        else:
            svf_on_grid = ds_svf[svf_vars].interp(
                lon=ds_latlon.lon, lat=ds_latlon.lat, method="linear"
            )
            for v in svf_vars:
                svf_on_grid[v] = svf_on_grid[v].astype("float32")
            ds_latlon = ds_latlon.assign(**{v: svf_on_grid[v] for v in svf_vars})

        # Add masked versions using glacier_mask already in ds_latlon
        if "glacier_mask" in ds_latlon:
            gmask = xr.where(ds_latlon["glacier_mask"] == 1, 1.0, np.nan)
            for v in svf_vars:
                ds_latlon[f"masked_{v}"] = gmask * ds_latlon[v]

        # atomic save: write to temp then replace
        final_path = os.path.join(path_save, f"{item}.zarr")
        tmp_dir = tempfile.mkdtemp(prefix=f".tmp_{item}_", dir=path_save)
        try:
            ds_latlon.to_zarr(tmp_dir, mode="w")
            # remove existing if present, then atomic move
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            os.replace(tmp_dir, final_path)
        except Exception as e:
            # cleanup tmp on failure
            try:
                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir)
            except Exception:
                pass
            return ("err", item, f"Save error: {e}")

        return ("ok", item, final_path)

    except Exception as e:
        return ("err", item, str(e))


def create_sgi_topo_masks_parallel(
    cfg, path_xr_svf, iterator, type="glacier_name", path_save=None, max_workers=None
):
    """
    Parallel version of create_sgi_topo_masks (CPU-only).
    Each item writes <item>.zarr into path_save.
    """
    if path_save is None:
        # assumes 'path_SGI_topo' is defined/importable in this scope
        path_save = os.path.join(cfg.dataPath, path_SGI_topo, "xr_masked_grids/")

    os.makedirs(path_save, exist_ok=True)

    # IMPORTANT: do NOT empty the folder *after* we start; clear it up front if desired:
    # emptyfolder(path_save)  # <- only if you truly want to wipe existing outputs

    iterator = list(iterator)
    n = len(iterator)
    if n == 0:
        print("No items to process.")
        return

    if max_workers is None:
        max_workers = min(max(1, (os.cpu_count() or 2) - 1), 32)

    # Linux: use 'fork' to avoid pickling helpers; great for notebooks too
    ctx = mp.get_context("fork")

    ok = skip = err = 0
    results = {}

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_init,
        mp_context=ctx,
    ) as ex:
        futures = [
            ex.submit(
                _process_one_item_sgi,
                item,
                cfg,
                type,
                path_save,
                path_SGI_topo,
                path_xr_svf,
            )
            for item in iterator
        ]

        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Processing ({max_workers} workers)",
        ):
            status, item, msg = fut.result()
            results[item] = (status, msg)
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
                # optional: print(f"[SKIP] {item}: {msg}")
            else:
                err += 1
                print(f"[ERR]  {item}: {msg}")

    print(f"Done. ok={ok}  skip={skip}  err={err}  total={n}")
    return results
