"""
grid_utils.py
=============
Grid creation and monthly parquet generation utilities, extracted from
1_4_Create_grids.ipynb.

Covers:
  - OGGM grid stats (resolution, extent, lat/lon bounds)
  - Masked xarray grid creation (process_one_glacier, run_all_regions)
  - Monthly ERA5-aligned parquet generation (run_one_region, run_all_glacier_grids)
  - Specific-glacier helpers (run_for_glaciers)
  - Glacier area computation
"""

import os
import sys
import gc
import glob
import re
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm.notebook import tqdm

import massbalancemachine as mbm

from papers.Transformer_TF.scripts.geodata import grid_inputs_rgi
from papers.Transformer_TF.scripts.geodata import grid_inputs


# ── OGGM grid stats ───────────────────────────────────────────────────────────


def get_region_grid_stats(cfg, region_id, cache_dir=None, force_recompute=False):
    """
    Compute per-glacier grid stats (resolution, extent, lat/lon bounds) for a
    region. Caches to CSV so repeated calls don't re-scan every zarr file.
    """
    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"grid_stats_rgi{region_id}.csv")
        if (not force_recompute) and os.path.exists(cache_path):
            print(f"  Loaded cached grid stats: {cache_path}")
            return pd.read_csv(cache_path)

    path_RGIs = os.path.join(cfg.dataPath, f"OGGM/rgi_region_{region_id}", "xr_grids")
    zarr_paths = sorted(glob.glob(os.path.join(path_RGIs, "*.zarr")))
    records = []
    for p in tqdm(zarr_paths, desc=f"RGI {region_id}", unit="glacier"):
        rgi_id = os.path.splitext(os.path.basename(p))[0]
        try:
            ds = xr.open_zarr(p)
            dx = float(np.abs(np.diff(ds["x"].values)).mean())
            dy = float(np.abs(np.diff(ds["y"].values)).mean())
            nx, ny = ds.sizes.get("x"), ds.sizes.get("y")
            # cheap corner-only reprojection to get lat/lon bounds
            ds_crs = ds.rio.write_crs(ds.pyproj_srs)
            lon_min, lat_min, lon_max, lat_max = ds_crs.rio.transform_bounds(
                "EPSG:4326"
            )
            records.append(
                (
                    rgi_id,
                    dx,
                    dy,
                    nx,
                    ny,
                    nx * dx,
                    ny * dy,
                    lon_min,
                    lon_max,
                    lat_min,
                    lat_max,
                    None,
                )
            )
            ds.close()
        except Exception as e:
            records.append(
                (
                    rgi_id,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    f"{type(e).__name__}: {e}",
                )
            )

    df = pd.DataFrame(
        records,
        columns=[
            "rgi_id",
            "dx_m",
            "dy_m",
            "nx",
            "ny",
            "extent_x_m",
            "extent_y_m",
            "lon_min",
            "lon_max",
            "lat_min",
            "lat_max",
            "error",
        ],
    )
    df["max_extent_m"] = df[["extent_x_m", "extent_y_m"]].max(axis=1)
    df["n_pixels"] = df["nx"] * df["ny"]
    df["region"] = region_id

    if cache_path is not None:
        df.to_csv(cache_path, index=False)
        print(f"  Cached grid stats: {cache_path}")

    return df


def compute_region_bbox(df, buffer_deg=0.5):
    """Empirical bbox from actual glacier corner coordinates, with a small buffer."""
    valid = df.dropna(subset=["lon_min", "lon_max", "lat_min", "lat_max"])
    return {
        "lat": (
            float(valid["lat_min"].min()) - buffer_deg,
            float(valid["lat_max"].max()) + buffer_deg,
        ),
        "lon": (
            float(valid["lon_min"].min()) - buffer_deg,
            float(valid["lon_max"].max()) + buffer_deg,
        ),
    }


# ── Masked xarray grid creation ───────────────────────────────────────────────


def get_target_resolution(ds, dx_m, dy_m):
    """
    Pick a target resolution based on glacier physical extent, not native dx_m
    (since OGGM's own dx_m caps around 200m regardless of glacier size).

    Extent thresholds → target resolution:
      < 500 m   → None (keep native)
      < 5 km    → 50 m
      < 20 km   → 100 m
      >= 20 km  → 200 m
    """
    nx = ds.sizes.get("x")
    ny = ds.sizes.get("y")
    max_extent_m = max(nx * dx_m, ny * dy_m)

    if max_extent_m < 500:
        return None
    elif max_extent_m < 5_000:
        return 50
    elif max_extent_m < 20_000:
        return 100
    else:
        return 200


def process_one_glacier(
    rgi_gl: str,
    path_RGIs: str,
    path_xr_svf: str,
    path_xr_grids: str,
    target_res_m: int = None,
):
    """
    Worker: load OGGM grid, mask, optional coarsen (resolution chosen by
    glacier extent), reproject to lat/lon, merge SVF, write per-glacier zarr.
    Returns a small status tuple (rgi_gl, status, message).
    """
    try:
        # 1) Masked OGGM grid in projected coords
        ds, _ = create_masked_glacier_grid(path_RGIs, rgi_gl)

        # 2) Optional coarsen based on glacier extent
        dx_m, dy_m = get_res_from_projected(ds)
        chosen_target_res_m = target_res_m or get_target_resolution(ds, dx_m, dy_m)

        if chosen_target_res_m is not None and dx_m < chosen_target_res_m:
            ds = coarsenDS_mercator(ds, target_res_m=chosen_target_res_m)

        # 3) Reproject to WGS84 lat/lon
        original_proj = ds.pyproj_srs
        ds = ds.rio.write_crs(original_proj)
        ds_latlon = ds.rio.reproject("EPSG:4326").rename({"x": "lon", "y": "lat"})

        # 4) Load SVF + merge (if exists)
        svf_path = os.path.join(path_xr_svf, f"{rgi_gl}_svf_latlon.nc")
        if os.path.exists(svf_path):
            with xr.open_dataset(svf_path) as ds_svf:
                ds_svf = ds_svf.load()

                if "x" in ds_svf.dims or "y" in ds_svf.dims:
                    ds_svf = ds_svf.rename({"x": "lon", "y": "lat"})
                if "longitude" in ds_svf.dims or "latitude" in ds_svf.dims:
                    ds_svf = ds_svf.rename({"longitude": "lon", "latitude": "lat"})

            for coord in ["lon", "lat"]:
                if ds_latlon[coord][0] > ds_latlon[coord][-1]:
                    ds_latlon = ds_latlon.sortby(coord)
                if ds_svf[coord][0] > ds_svf[coord][-1]:
                    ds_svf = ds_svf.sortby(coord)

            svf_vars = [v for v in ("svf", "asvf", "opns") if v in ds_svf.data_vars]

            if svf_vars:
                if np.array_equal(
                    ds_latlon.lon.values, ds_svf.lon.values
                ) and np.array_equal(ds_latlon.lat.values, ds_svf.lat.values):
                    ds_latlon = xr.merge([ds_latlon, ds_svf[svf_vars]])
                else:
                    svf_on_grid = ds_svf[svf_vars].interp(
                        lon=ds_latlon.lon, lat=ds_latlon.lat, method="linear"
                    )
                    for v in svf_vars:
                        svf_on_grid[v] = svf_on_grid[v].astype("float32")
                    ds_latlon = ds_latlon.assign(
                        **{v: svf_on_grid[v] for v in svf_vars}
                    )

                if "glacier_mask" in ds_latlon:
                    gmask = xr.where(
                        ds_latlon["glacier_mask"].values.astype(bool), 1.0, np.nan
                    )
                    for v in svf_vars:
                        ds_latlon[f"masked_{v}"] = gmask * ds_latlon[v]

        # 5) Save final lat/lon grid
        save_path = os.path.join(path_xr_grids, f"{rgi_gl}.zarr")
        ds_latlon.to_zarr(save_path, mode="w")

        return (rgi_gl, "ok", "")

    except Exception as e:
        return (rgi_gl, "error", f"{type(e).__name__}: {e}")


def glacier_ids_from_oggm_grids(path_RGIs: str):
    """Return RGI IDs from per-glacier OGGM zarr grids in path_RGIs."""
    zarrs = sorted(glob.glob(os.path.join(path_RGIs, "*.zarr")))
    return [os.path.splitext(os.path.basename(p))[0] for p in zarrs]


def run_parallel_processing_region(
    rgi_ids,
    path_RGIs,
    path_xr_svf,
    path_xr_grids,
    n_workers=6,
    clear_out=False,
    target_res_m=None,
):
    """Run process_one_glacier in parallel for a list of RGI IDs."""
    if clear_out:
        import shutil

        shutil.rmtree(path_xr_grids, ignore_errors=True)
    os.makedirs(path_xr_grids, exist_ok=True)

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(
                process_one_glacier,
                rgi_id,
                path_RGIs,
                path_xr_svf,
                path_xr_grids,
                target_res_m,
            ): rgi_id
            for rgi_id in rgi_ids
        }
        for fut in tqdm(as_completed(futures), total=len(futures)):
            results.append(fut.result())

    n_ok = sum(r[1] == "ok" for r in results)
    n_err = sum(r[1] == "error" for r in results)
    print(f"Done. ok={n_ok}, error={n_err}")
    if n_err:
        for rgi_id, status, msg in results:
            if status == "error":
                print(f"  [{rgi_id}] {msg}")

    return results


def run_all_regions(
    cfg,
    RGI_REGIONS,
    rgi_per_region=None,
    stake_only_regions=None,
    n_workers=6,
    clear_out=False,
    target_res_m=None,
):
    """
    Create masked lat/lon xarray grids for all RGI regions.

    Parameters
    ----------
    stake_only_regions : set of str, optional
        Region IDs for which to restrict to stake RGIs only (requires rgi_per_region).
    target_res_m : int or None
        Manual resolution override; None = auto-pick per glacier by extent.
    """
    all_results = {}

    for rgi_region, spec in RGI_REGIONS.items():
        rgi_region_str = str(rgi_region).zfill(2)
        region_folder = spec["folder"]
        print(f"\n========== RGI {rgi_region_str} ({spec['name']}) ==========")

        path_xr_svf = os.path.join(
            cfg.dataPath, "RGI_v6", region_folder, "svf_nc_latlon"
        )
        path_xr_grids = os.path.join(
            cfg.dataPath, "RGI_v6", region_folder, "xr_masked_grids"
        )
        path_RGIs = os.path.join(
            cfg.dataPath, f"OGGM/rgi_region_{rgi_region_str}", "xr_grids"
        )

        if not os.path.isdir(path_RGIs):
            print(f"  Skipping: missing xr_grids folder: {path_RGIs}")
            continue
        if not os.path.isdir(path_xr_svf):
            print(f"  Skipping: missing svf folder: {path_xr_svf}")
            continue

        rgi_ids = glacier_ids_from_oggm_grids(path_RGIs)
        print(f"  Found {len(rgi_ids)} glacier grids")

        if stake_only_regions and rgi_region_str in stake_only_regions:
            if rgi_per_region and rgi_region_str in rgi_per_region:
                allowed = set(rgi_per_region[rgi_region_str])
                rgi_ids = [r for r in rgi_ids if r in allowed]
                print(f"  Filtered to {len(rgi_ids)} stake RGIs")
            else:
                print(
                    f"  ⚠️  stake_only requested for {rgi_region_str} "
                    f"but no rgi_per_region entry found — using all"
                )

        if not rgi_ids:
            continue

        os.makedirs(path_xr_grids, exist_ok=True)
        results = run_parallel_processing_region(
            rgi_ids=rgi_ids,
            path_RGIs=path_RGIs,
            path_xr_svf=path_xr_svf,
            path_xr_grids=path_xr_grids,
            n_workers=n_workers,
            clear_out=clear_out,
            target_res_m=target_res_m,
        )
        all_results[rgi_region] = results

    return all_results


# ── Glacier area computation ──────────────────────────────────────────────────


def compute_glacier_areas_km2(
    zarr_dir, rgi_ids, cache_path=None, force_recompute=False
):
    """
    Compute true glacier area (km²) from glacier_mask pixel count × pixel area,
    using the already-reprojected lat/lon masked grids in zarr_dir.
    Caches results to avoid recomputing on every run.
    """
    if cache_path and os.path.exists(cache_path) and not force_recompute:
        print(f"  Loading cached areas: {cache_path}")
        return pd.read_csv(cache_path)

    records = []
    for rgi_id in tqdm(rgi_ids, desc="Computing areas", unit="glacier"):
        try:
            ds = xr.open_zarr(os.path.join(zarr_dir, f"{rgi_id}.zarr"))
            gl_mask = ds["glacier_mask"].values.astype(bool)
            n_glacier_px = int(gl_mask.sum())

            lat_res = float(np.abs(np.diff(ds["lat"].values)).mean())
            lon_res = float(np.abs(np.diff(ds["lon"].values)).mean())
            mean_lat = float(ds["lat"].values.mean())
            m_per_deg_lat = 111_320
            m_per_deg_lon = 111_320 * np.cos(np.deg2rad(mean_lat))
            px_area_m2 = (lat_res * m_per_deg_lat) * (lon_res * m_per_deg_lon)

            area_km2 = n_glacier_px * px_area_m2 / 1e6
            records.append((rgi_id, area_km2, None))
            ds.close()
        except Exception as e:
            records.append((rgi_id, None, f"{type(e).__name__}: {e}"))

    df = pd.DataFrame(records, columns=["rgi_id", "area_km2", "error"])
    if cache_path:
        df.to_csv(cache_path, index=False)
        print(f"  Cached areas: {cache_path}")
    return df


# ── Monthly parquet generation ────────────────────────────────────────────────


def glacier_ids_from_xr_grids(path):
    """Return RGI IDs from per-glacier masked zarr grids."""
    zarrs = sorted(glob.glob(os.path.join(path, "*.zarr")))
    return [os.path.splitext(os.path.basename(p))[0] for p in zarrs]


def clip_era5_for_region(era5_path, geopot_path, region_id, out_dir, RGI_BBOX=None):
    """Clip ERA5 files to region bounding box. Skips if already exists."""
    bbox = RGI_BBOX[region_id]
    os.makedirs(out_dir, exist_ok=True)

    clipped = {}
    for path, name in [(era5_path, "era5"), (geopot_path, "geopot")]:
        out_path = os.path.join(out_dir, f"{name}_{region_id}.nc")
        if os.path.exists(out_path):
            print(f"  Already clipped: {out_path}")
        else:
            print(f"  Clipping {name} for region {region_id}...")
            ds = xr.open_dataset(path)
            ds.sel(
                latitude=slice(bbox["lat"][1], bbox["lat"][0]),  # ERA5 lat descending
                longitude=slice(bbox["lon"][0], bbox["lon"][1]),
            ).to_netcdf(out_path)
            print(f"  Saved ({os.path.getsize(out_path)/1e9:.2f} GB): {out_path}")
        clipped[name] = out_path

    return {
        "era5_climate_data": clipped["era5"],
        "geopotential_data": clipped["geopot"],
    }


def check_glaciers_in_bbox(rgi_ids, zarr_dir, bbox):
    """Warn if any glacier centroids fall outside the clipped ERA5 bbox."""
    outside = []
    for rgi_id in rgi_ids:
        try:
            ds = xr.open_zarr(os.path.join(zarr_dir, f"{rgi_id}.zarr"))
            gl_mask = ds["glacier_mask"].values.astype(bool)
            lon_2d, lat_2d = np.meshgrid(ds["lon"].values, ds["lat"].values)
            mean_lon = float(lon_2d[gl_mask].mean())
            mean_lat = float(lat_2d[gl_mask].mean())
            if not (
                bbox["lat"][0] <= mean_lat <= bbox["lat"][1]
                and bbox["lon"][0] <= mean_lon <= bbox["lon"][1]
            ):
                outside.append((rgi_id, mean_lat, mean_lon))
        except Exception as e:
            print(f"  Could not check {rgi_id}: {e}")
    return outside


def run_one_region(
    cfg,
    region_id,
    spec,
    rgi_per_region,
    paths,
    vois_climate,
    vois_topographical,
    meta_cols,
    years=range(2010, 2025),
    start_month="10",
    n_workers=6,
    force_recompute=False,
    min_area_km2=1.0,
    build_ds=False,
    monthly_cols=None,
    static_cols=None,
    months_head_pad=None,
    months_tail_pad=None,
    RGI_BBOX=None,
    RGI_ERA5_SOURCE=None,
):
    """Process all glaciers for a single RGI region (monthly parquets + optional ds cache)."""
    region_id_str = str(region_id).zfill(2)
    print(f"\n========== RGI {region_id_str} ({spec['name']}) ==========")

    basepath = os.path.join(cfg.dataPath, "RGI_v6", spec["folder"])
    zarr_dir = os.path.join(basepath, "xr_masked_grids")
    out_dir = os.path.join(basepath, "monthly_parquets")
    os.makedirs(out_dir, exist_ok=True)

    if build_ds and (monthly_cols is None or static_cols is None):
        raise ValueError("build_ds=True requires monthly_cols and static_cols")

    ds_cache_root = os.path.join(basepath, "grid_ds_cache") if build_ds else None

    # --- optionally wipe existing parquets ---
    if force_recompute:
        existing = glob.glob(os.path.join(out_dir, "**", "*.parquet"), recursive=True)
        if existing:
            print(f"  force_recompute=True: deleting {len(existing)} parquet files...")
            for f in existing:
                os.remove(f)
        else:
            print("  force_recompute=True: no existing parquet files found.")

    if not os.path.isdir(zarr_dir):
        print(f"  Skipping: missing xr_masked_grids: {zarr_dir}")
        return {"ok": 0, "skip": 0, "err": 0}

    # --- resolve ERA5 source ---
    era5_source = RGI_ERA5_SOURCE.get(region_id_str, "EU_US_CANADA")
    suffix = f"_{era5_source}" if era5_source != "EU_US_CANADA" else ""
    base_paths = {
        "era5_climate_data": paths[f"era5_climate_data{suffix}"],
        "geopotential_data": paths[f"geopotential_data{suffix}"],
    }
    print(f"  Using ERA5 source: {era5_source}")

    # --- clip ERA5 ---
    era5_clip_dir = os.path.join(cfg.dataPath, "ERA5Land/clipped_to_RGI")
    if RGI_BBOX and region_id_str in RGI_BBOX:
        region_paths = clip_era5_for_region(
            base_paths["era5_climate_data"],
            base_paths["geopotential_data"],
            region_id_str,
            out_dir=era5_clip_dir,
            RGI_BBOX=RGI_BBOX,
        )
    else:
        print(f"  No bounding box defined for {region_id_str}, using full ERA5.")
        region_paths = base_paths

    # --- build task list ---
    rgi_ids = glacier_ids_from_xr_grids(zarr_dir)

    if rgi_per_region is not None and region_id_str in rgi_per_region:
        requested = set(rgi_per_region[region_id_str])
        rgi_ids = [r for r in rgi_ids if r in requested]
        print(f"  Glacier filter: {len(rgi_ids)} requested glaciers")

    if min_area_km2 is not None:
        area_cache = os.path.join(basepath, "glacier_areas_km2.csv")
        areas_df = compute_glacier_areas_km2(zarr_dir, rgi_ids, cache_path=area_cache)
        valid = areas_df.dropna(subset=["area_km2"])
        kept = set(valid.loc[valid["area_km2"] >= min_area_km2, "rgi_id"])
        n_before = len(rgi_ids)
        rgi_ids = [r for r in rgi_ids if r in kept]
        print(
            f"  Area filter (>= {min_area_km2} km²): {n_before} -> {len(rgi_ids)} glaciers"
        )

    if RGI_BBOX and region_id_str in RGI_BBOX:
        outside = check_glaciers_in_bbox(rgi_ids, zarr_dir, RGI_BBOX[region_id_str])
        if outside:
            print(f"  ⚠️  {len(outside)} glaciers outside clipped bbox:")
            for rgi_id, lat, lon in outside:
                print(f"    {rgi_id}: lat={lat:.2f}, lon={lon:.2f}")
        else:
            print(f"  ✓ All {len(rgi_ids)} glaciers inside clipped bbox")

    tasks = [(r, y) for r in rgi_ids for y in years]
    print(f"  {len(rgi_ids)} glaciers × {len(list(years))} years = {len(tasks)} tasks")
    if not tasks:
        return {"ok": 0, "skip": 0, "err": 0}

    # --- logging ---
    n_ok = n_err = n_skip = 0
    log_dir = Path("logs/errors")
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"rgi_{region_id_str}_{ts}.log"
    err_path = log_dir / f"rgi_{region_id_str}_{ts}_ERRORS.log"

    with open(log_path, "w") as f:
        f.write(f"Run started: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"max_workers={n_workers}, n_tasks={len(tasks)}\n")
        f.write("-" * 80 + "\n")

    # --- parallel execution ---
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=grid_inputs_rgi.init_worker_rgi,
    ) as ex:
        futs = [
            ex.submit(
                grid_inputs_rgi.process_monthly_grids_rgi,
                region_id_str,
                rgi_id,
                year,
                zarr_path=os.path.join(zarr_dir, f"{rgi_id}.zarr"),
                basepath=basepath,
                out_folder_root=out_dir,
                vois_climate=vois_climate,
                vois_topo=vois_topographical,
                meta_cols=meta_cols,
                era5_monthly_path=region_paths["era5_climate_data"],
                era5_geopot_path=region_paths["geopotential_data"],
                start_month=start_month,
                cfg=cfg,
                build_ds=build_ds,
                ds_cache_root=ds_cache_root,
                monthly_cols=monthly_cols,
                static_cols=static_cols,
                months_head_pad=months_head_pad,
                months_tail_pad=months_tail_pad,
            )
            for rgi_id, year in tasks
        ]

        with tqdm(total=len(futs), desc=f"RGI {region_id_str}", unit="task") as pbar:
            for fut in as_completed(futs):
                try:
                    res = fut.result()
                except Exception as e:
                    res = f"ERROR <future>: {type(e).__name__}: {e}"

                if res.startswith("OK"):
                    n_ok += 1
                elif res.startswith("ERROR"):
                    n_err += 1
                elif res.startswith("SKIP"):
                    n_skip += 1

                if res.startswith(("SKIP", "ERROR")):
                    line = f"[{datetime.now().isoformat(timespec='seconds')}] {res}\n"
                    with open(log_path, "a") as f:
                        f.write(line)
                    if res.startswith("ERROR"):
                        with open(err_path, "a") as f:
                            f.write(line)

                pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_err)
                pbar.update(1)

    with open(log_path, "a") as f:
        f.write("-" * 80 + "\n")
        f.write(f"Run finished: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"OK={n_ok}, SKIP={n_skip}, ERROR={n_err}\n")

    print(f"  Done. OK={n_ok}, SKIP={n_skip}, ERROR={n_err}")
    if n_err:
        print(f"  Errors: {err_path}")

    gc.collect()
    return {"ok": n_ok, "skip": n_skip, "err": n_err}


def run_all_glacier_grids(
    cfg,
    RGI_REGIONS,
    rgi_per_region,
    paths,
    vois_climate,
    vois_topographical,
    meta_cols,
    years=range(2010, 2025),
    start_month="10",
    n_workers=6,
    force_recompute=False,
    min_area_km2=1.0,
    build_ds=False,
    monthly_cols=None,
    static_cols=None,
    months_head_pad=None,
    months_tail_pad=None,
    RGI_BBOX=None,
    RGI_ERA5_SOURCE=None,
):
    """Process all regions sequentially, each with clipped ERA5."""
    return {
        region_id: run_one_region(
            cfg=cfg,
            region_id=region_id,
            spec=spec,
            rgi_per_region=rgi_per_region,
            paths=paths,
            vois_climate=vois_climate,
            vois_topographical=vois_topographical,
            meta_cols=meta_cols,
            years=years,
            start_month=start_month,
            n_workers=n_workers,
            force_recompute=force_recompute,
            min_area_km2=min_area_km2,
            build_ds=build_ds,
            monthly_cols=monthly_cols,
            static_cols=static_cols,
            months_head_pad=months_head_pad,
            months_tail_pad=months_tail_pad,
            RGI_BBOX=RGI_BBOX,
            RGI_ERA5_SOURCE=RGI_ERA5_SOURCE,
        )
        for region_id, spec in RGI_REGIONS.items()
    }


def run_for_glaciers(
    cfg,
    glacier_rgi_ids,  # {region_id_str: [rgi_id, ...]}
    RGI_REGIONS,
    paths,
    vois_climate,
    vois_topographical,
    meta_cols,
    years=range(2000, 2025),
    start_month="10",
    n_workers=6,
    force_recompute=False,
    build_ds=False,
    monthly_cols=None,
    static_cols=None,
    months_head_pad=None,
    months_tail_pad=None,
    min_area_km2=None,
    RGI_BBOX=None,
    RGI_ERA5_SOURCE=None,
):
    """
    Run grid processing for specific glaciers only.

    glacier_rgi_ids maps region_id_str -> list of RGI IDs to process,
    e.g. {"11": ["RGI60-11.01450"], "08": ["RGI60-08.00001"]}.
    Only regions present in glacier_rgi_ids are processed.
    min_area_km2 defaults to None so explicitly requested glaciers are never
    area-filtered away.
    """
    regions_to_run = {
        r: spec for r, spec in RGI_REGIONS.items() if r in glacier_rgi_ids
    }
    if not regions_to_run:
        raise ValueError(f"None of {list(glacier_rgi_ids)} found in RGI_REGIONS")

    return run_all_glacier_grids(
        cfg=cfg,
        RGI_REGIONS=regions_to_run,
        rgi_per_region=glacier_rgi_ids,
        paths=paths,
        vois_climate=vois_climate,
        vois_topographical=vois_topographical,
        meta_cols=meta_cols,
        years=years,
        start_month=start_month,
        n_workers=n_workers,
        force_recompute=force_recompute,
        min_area_km2=min_area_km2,
        build_ds=build_ds,
        monthly_cols=monthly_cols,
        static_cols=static_cols,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
        RGI_BBOX=RGI_BBOX,
        RGI_ERA5_SOURCE=RGI_ERA5_SOURCE,
    )


def run_for_stake_glaciers(
    cfg,
    monthly_cache,  # the monthly_cache dict already in scope
    region_code_to_rgi,  # dict: {monthly_cache_code: rgi_id_str}
    # e.g. {"CH": "11", "NOR": "08", "ALA": "01"}
    RGI_REGIONS,
    paths,
    vois_climate,
    vois_topographical,
    meta_cols,
    years=range(2000, 2025),
    start_month="10",
    n_workers=6,
    force_recompute=False,
    build_ds=False,
    monthly_cols=None,
    static_cols=None,
    months_head_pad=None,
    months_tail_pad=None,
    RGI_BBOX=None,
    RGI_ERA5_SOURCE=None,
):
    """
    Run grid processing for all glaciers with stakes in the given regions.
    Stakes are identified from monthly_cache — any glacier with at least one
    stake measurement in the cache is included.

    region_code_to_rgi maps monthly_cache keys to RGI region ID strings:
        {"CH": "11", "NOR": "08", "SJM": "07", "ALA": "01", ...}
    """
    glacier_rgi_ids = {}

    for code, rgi_id_str in region_code_to_rgi.items():
        if code not in monthly_cache:
            print(f"  WARNING: {code} not in monthly_cache, skipping")
            continue

        rgi_ids = monthly_cache[code]["data_monthly"]["RGIId"].unique().tolist()
        glacier_rgi_ids[rgi_id_str] = rgi_ids
        print(f"  {code} (RGI {rgi_id_str}): {len(rgi_ids)} stake glaciers")

    if not glacier_rgi_ids:
        raise ValueError("No valid regions found in monthly_cache")

    return run_for_glaciers(
        cfg=cfg,
        glacier_rgi_ids=glacier_rgi_ids,
        RGI_REGIONS=RGI_REGIONS,
        paths=paths,
        vois_climate=vois_climate,
        vois_topographical=vois_topographical,
        meta_cols=meta_cols,
        years=years,
        start_month=start_month,
        n_workers=n_workers,
        force_recompute=force_recompute,
        build_ds=build_ds,
        monthly_cols=monthly_cols,
        static_cols=static_cols,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
        min_area_km2=None,  # don't area-filter stake glaciers
        RGI_BBOX=RGI_BBOX,
        RGI_ERA5_SOURCE=RGI_ERA5_SOURCE,
    )
