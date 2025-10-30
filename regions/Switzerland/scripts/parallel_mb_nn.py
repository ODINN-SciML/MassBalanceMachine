# scripts/parallel_mb_nn.py
from __future__ import annotations
import os, sys, io, warnings, logging, multiprocessing as mp
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from functools import partial
from typing import Iterable, List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

import massbalancemachine as mbm
from regions.Switzerland.scripts.helpers import *

try:
    from tqdm.auto import tqdm as _tqdm_default
except Exception:

    def _tqdm_default(x, **k):
        return x


# ----------------- Config -----------------
@dataclass(frozen=True)
class NNJobConfig:
    cfg: Any
    model_filename: str
    args: Dict[str, Any]
    param_init: Dict[str, Any]
    all_columns: List[str]
    months_head_pad: int
    months_tail_pad: int
    data_path: str
    path_glacier_grid_glamos: str
    path_xr_grids: str
    path_save_glw: str
    seed: int = 0
    max_workers: Optional[int] = None
    cpu_only: bool = True
    ONLY_GEODETIC: bool = False
    save_monthly_pred: bool = True


# ----------------- Worker Init -----------------
def worker_init_quiet(cpu_only: bool = True):
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    if cpu_only:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


# ----------------- Model cache -----------------
_MODEL = None


def get_nn_model(model_filename, args, param_init, cfg):
    """Load the NN model once per worker and keep it cached in memory."""
    global _MODEL
    if _MODEL is None:
        loaded_model = mbm.models.CustomNeuralNetRegressor.load_model(
            cfg,
            model_filename,
            **{**args, **param_init},
        )
        loaded_model = loaded_model.set_params(device="cpu")
        loaded_model = loaded_model.to("cpu")
        _MODEL = loaded_model
    return _MODEL


# ----------------- Per glacier-year -----------------
def process_glacier_year(
    args: Tuple[str, int],
    job: NNJobConfig,
) -> Tuple[str, str, int, str]:
    glacier_name, year = args
    try:
        glacier_path = os.path.join(
            job.data_path + job.path_glacier_grid_glamos, glacier_name
        )
        parquet_path = os.path.join(glacier_path, f"{glacier_name}_grid_{year}.parquet")
        if not os.path.exists(parquet_path):
            return ("skip", glacier_name, year, "parquet missing")

        df_grid_monthly = pd.read_parquet(parquet_path)
        df_grid_monthly.drop_duplicates(inplace=True)
        df_grid_monthly = df_grid_monthly[
            [c for c in job.all_columns if c in df_grid_monthly.columns]
        ]
        df_grid_monthly = df_grid_monthly.dropna()

        path_glacier_dem = os.path.join(
            job.path_xr_grids, f"{glacier_name}_{year}.zarr"
        )
        if not os.path.exists(path_glacier_dem):
            return ("skip", glacier_name, year, "DEM zarr missing")

        model = get_nn_model(job.model_filename, job.args, job.param_init, job.cfg)

        geoData = mbm.geodata.GeoData(
            df_grid_monthly,
            months_head_pad=job.months_head_pad,
            months_tail_pad=job.months_tail_pad,
        )

        geoData.gridded_MB_pred(
            df_grid_monthly,
            model,
            glacier_name,
            year,
            job.all_columns,
            path_glacier_dem,
            job.path_save_glw,
            save_monthly_pred=job.save_monthly_pred,
            type_model="NN",
        )

        return ("ok", glacier_name, year, "")
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        return ("err", glacier_name, year, err_msg)


# ----------------- Task Builder -----------------
def build_tasks(
    glacier_list: List[str],
    periods_per_glacier: Dict[str, Iterable[int]],
    data_path: str,
    path_glacier_grid_glamos: str,
    ONLY_GEODETIC: bool,
) -> List[Tuple[str, int]]:
    tasks: List[Tuple[str, int]] = []
    for glacier_name in glacier_list:
        glacier_path = os.path.join(data_path + path_glacier_grid_glamos, glacier_name)
        if not os.path.exists(glacier_path):
            continue
        glacier_files = sorted(
            [f for f in os.listdir(glacier_path) if f.endswith(".parquet")]
        )
        if not glacier_files:
            continue

        geodetic_range = range(
            np.min(periods_per_glacier[glacier_name]),
            np.max(periods_per_glacier[glacier_name]) + 1,
        )
        years = [int(f.split("_")[2].split(".")[0]) for f in glacier_files]
        if ONLY_GEODETIC:
            years = [y for y in years if y in geodetic_range]
        for y in years:
            tasks.append((glacier_name, y))
    return tasks


# ----------------- Main -----------------
def run_glacier_mb_nn(
    job: NNJobConfig,
    glacier_list: List[str],
    periods_per_glacier: Dict[str, Iterable[int]],
    tqdm=_tqdm_default,
) -> Dict[str, int]:
    os.makedirs(job.path_save_glw, exist_ok=True)
    emptyfolder(job.path_save_glw)

    tasks = build_tasks(
        glacier_list,
        periods_per_glacier,
        job.data_path,
        job.path_glacier_grid_glamos,
        job.ONLY_GEODETIC,
    )
    if len(tasks) == 0:
        return dict(ok=0, skip=0, err=0, total=0)

    ctx = mp.get_context("fork")
    max_workers = job.max_workers or min(max(1, (os.cpu_count() or 2) - 1), 32)

    ok = skip = err = 0
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=lambda: worker_init_quiet(job.cpu_only),
        mp_context=ctx,
    ) as ex:
        fn = partial(process_glacier_year, job=job)
        futures = [ex.submit(fn, t) for t in tasks]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"NN Predict ({max_workers} workers)",
        ):
            status, g, y, msg = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
                print(f"[SKIP] {g} {y}: {msg}")
            else:
                err += 1
                print(f"[ERROR] {g} {y}: {msg}")

    return dict(ok=ok, skip=skip, err=err, total=len(tasks))
