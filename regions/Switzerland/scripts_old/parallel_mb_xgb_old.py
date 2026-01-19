"""
Parallel mass-balance extrapolation using XGBoost
(analogous to parallel_mb_nn.py but calling XGB model instead of NN)
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from scripts.utils import emptyfolder
from mbm.geodata import GeoData


class XGBJobConfig:
    """Configuration for parallel glacier XGBoost mass-balance extrapolation."""

    def __init__(
        self,
        cfg,
        model_filename,
        all_columns,
        months_head_pad,
        months_tail_pad,
        data_path,
        path_glacier_grid_glamos,
        path_xr_grids,
        path_save_glw,
        cpu_only=True,
        ONLY_GEODETIC=False,
        save_monthly_pred=True,
        args=None,
        param_init=None,
        max_workers=1,
    ):
        self.cfg = cfg
        self.model_filename = model_filename
        self.all_columns = all_columns
        self.months_head_pad = months_head_pad
        self.months_tail_pad = months_tail_pad
        self.data_path = data_path
        self.path_glacier_grid_glamos = path_glacier_grid_glamos
        self.path_xr_grids = path_xr_grids
        self.path_save_glw = path_save_glw
        self.cpu_only = cpu_only
        self.ONLY_GEODETIC = ONLY_GEODETIC
        self.save_monthly_pred = save_monthly_pred
        self.args = args
        self.param_init = param_init
        self.max_workers = max_workers

        # XGBoost fixed flag
        self.type_model = "XGBoost"


def _process_one_glacier(job, glacier_name, periods_per_glacier):
    """
    Extrapolate mass balance for a single glacier across all years.
    Called in parallel from run_glacier_mb_xgb().
    """

    glacier_path = os.path.join(
        job.data_path, job.path_glacier_grid_glamos, glacier_name
    )
    if not os.path.exists(glacier_path):
        return {"glacier": glacier_name, "status": "missing input folder"}

    glacier_files = sorted(
        [f for f in os.listdir(glacier_path) if f.endswith(".parquet")]
    )
    if not glacier_files:
        return {"glacier": glacier_name, "status": "no parquet files"}

    geodetic_range = range(
        np.min(periods_per_glacier[glacier_name]),
        np.max(periods_per_glacier[glacier_name]) + 1,
    )
    years = [int(f.split("_")[2].split(".")[0]) for f in glacier_files]
    if job.ONLY_GEODETIC:
        years = [y for y in years if y in geodetic_range]

    for year in tqdm(years, desc=f"Processing {glacier_name}", leave=False):
        parquet_path = os.path.join(glacier_path, f"{glacier_name}_grid_{year}.parquet")
        if not os.path.exists(parquet_path):
            print("skip", glacier_name, year, "parquet missing")
            continue

        df = pd.read_parquet(parquet_path)
        df = df.drop_duplicates()
        df = df[[c for c in job.all_columns if c in df.columns]]
        df = df.dropna()

        path_glacier_dem = os.path.join(
            job.path_xr_grids, f"{glacier_name}_{year}.zarr"
        )
        if not os.path.exists(path_glacier_dem):
            print("skip", glacier_name, year, "DEM zarr missing")
            continue

        geoData = GeoData(
            df,
            months_head_pad=job.months_head_pad,
            months_tail_pad=job.months_tail_pad,
        )

        # Core call — identical to NN version except we use type_model="XGBoost"
        geoData.gridded_MB_pred(
            df,
            job.model_filename,
            glacier_name,
            year,
            job.all_columns,
            path_glacier_dem,
            job.path_save_glw,
            save_monthly_pred=job.save_monthly_pred,
            type_model=job.type_model,
        )

    return {"glacier": glacier_name, "status": "OK"}


def run_glacier_mb_xgb(job, glacier_list, periods_per_glacier):
    """
    Run XGBoost mass-balance extrapolation across glaciers in parallel.
    """

    emptyfolder(job.path_save_glw)

    from concurrent.futures import ProcessPoolExecutor

    results = []
    with ProcessPoolExecutor(max_workers=job.max_workers) as executor:
        futures = {
            executor.submit(
                _process_one_glacier, job, glacier_name, periods_per_glacier
            ): glacier_name
            for glacier_name in glacier_list
        }

        for future in tqdm(futures, desc="Glaciers", leave=True):
            try:
                results.append(future.result())
            except Exception as e:
                results.append({"glacier": futures[future], "status": f"ERROR: {e}"})

    return results
