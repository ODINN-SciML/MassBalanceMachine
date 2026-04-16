"""
Parallel neural-network-based glacier mass balance prediction.

This script runs gridded mass balance predictions using a trained neural
network model for multiple glacier–year combinations in parallel. It
handles model loading, per-glacier/year processing, and multiprocessing
execution with controlled resource usage.
"""

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
from regions.TF_Europe.scripts.utils import *

try:
    from tqdm.auto import tqdm as _tqdm_default
except Exception:

    def _tqdm_default(x, **k):
        return x


# ----------------- Config -----------------
@dataclass(frozen=True)
class NNMbJobConfig:
    """
    Configuration container for parallel NN glacier mass balance jobs.

    Attributes
    ----------
    cfg : Any
        Global configuration object used to load the trained model.
    model_filename : str
        Filename of the trained neural network model.
    args : dict
        Model initialization arguments.
    param_init : dict
        Initial model parameters (e.g. weights, architecture settings).
    all_columns : list of str
        List of feature columns required for NN prediction.
    months_head_pad : int
        Number of months to ignore at the beginning of each time series.
    months_tail_pad : int
        Number of months to ignore at the end of each time series.
    data_path : str
        Base path to input data.
    path_glacier_grid_glamos : str
        Relative path to glacier gridded parquet files.
    path_xr_grids : str
        Path to xarray/zarr DEM grids.
    path_save_glw : str
        Output directory for predicted glacier-wide mass balance.
    seed : int, optional
        Random seed for reproducibility.
    max_workers : int, optional
        Maximum number of parallel worker processes.
    cpu_only : bool, optional
        If True, force CPU-only execution.
    ONLY_GEODETIC : bool, optional
        If True, restrict processing to geodetic periods only.
    save_monthly_pred : bool, optional
        If True, save monthly gridded predictions.
    """

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
def worker_init_quiet_nn(cpu_only: bool = True):
    """
    Initialize a worker process for NN inference with minimal output and
    restricted resources.

    This function:
    - Suppresses stdout and stderr
    - Optionally disables CUDA
    - Limits the number of threads used by BLAS, MKL, OpenMP, and PyTorch

    Parameters
    ----------
    cpu_only : bool, optional
        If True, disable GPU usage by setting CUDA_VISIBLE_DEVICES.
    """
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
_NN_MODEL = None


def load_nn_model_cached(model_filename, args, param_init, cfg):
    """
    Load and cache the neural network model once per worker.

    The model is loaded on first call within a worker process and then
    reused for subsequent glacier-year evaluations to avoid redundant
    disk I/O and initialization overhead.

    Parameters
    ----------
    model_filename : str
        Filename of the trained NN model.
    args : dict
        Model loading arguments.
    param_init : dict
        Model initialization parameters.
    cfg : Any
        Global configuration object.

    Returns
    -------
    CustomNeuralNetRegressor
        Loaded neural network model on CPU.
    """
    global _NN_MODEL
    if _NN_MODEL is None:
        loaded_model = mbm.models.CustomNeuralNetRegressor.load_model(
            cfg,
            model_filename,
            **{**args, **param_init},
        )
        loaded_model = loaded_model.set_params(device="cpu")
        loaded_model = loaded_model.to("cpu")
        _NN_MODEL = loaded_model
    return _NN_MODEL


# ----------------- Per glacier-year -----------------
def process_glacier_year_nn(
    args: Tuple[str, int],
    job: NNMbJobConfig,
) -> Tuple[str, str, int, str]:
    """
    Run NN mass balance prediction for a single glacier and year.

    Parameters
    ----------
    args : tuple (str, int)
        Glacier name and year.
    job : NNMbJobConfig
        Job configuration object.

    Returns
    -------
    tuple
        (status, glacier_name, year, message), where status is one of
        {"ok", "skip", "err"}.
    """
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

        model = load_nn_model_cached(
            job.model_filename, job.args, job.param_init, job.cfg
        )

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
def build_nn_tasks(
    glacier_list: List[str],
    periods_per_glacier: Dict[str, Iterable[int]],
    data_path: str,
    path_glacier_grid_glamos: str,
    ONLY_GEODETIC: bool,
) -> List[Tuple[str, int]]:
    """
    Build a list of (glacier, year) tasks for NN inference.

    Parameters
    ----------
    glacier_list : list of str
        List of glacier identifiers.
    periods_per_glacier : dict
        Mapping of glacier name to iterable of geodetic years.
    data_path : str
        Base data directory.
    path_glacier_grid_glamos : str
        Relative path to glacier parquet data.
    ONLY_GEODETIC : bool
        If True, restrict years to geodetic range.

    Returns
    -------
    list of tuple
        List of (glacier_name, year) tasks.
    """
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
def run_glacier_mb_nn_parallel(
    job: NNMbJobConfig,
    glacier_list: List[str],
    periods_per_glacier: Dict[str, Iterable[int]],
    tqdm=_tqdm_default,
) -> Dict[str, int]:
    """
    Run parallel NN-based mass balance predictions for multiple glaciers.

    Parameters
    ----------
    job : NNMbJobConfig
        Job configuration object.
    glacier_list : list of str
        List of glaciers to process.
    periods_per_glacier : dict
        Mapping of glacier name to iterable of geodetic years.
    tqdm : callable, optional
        Progress bar wrapper (default: tqdm.auto.tqdm).

    Returns
    -------
    dict
        Dictionary with counts of successful, skipped, and failed tasks:
        {"ok", "skip", "err", "total"}.
    """
    os.makedirs(job.path_save_glw, exist_ok=True)
    emptyfolder(job.path_save_glw)

    tasks = build_nn_tasks(
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
        initializer=lambda: worker_init_quiet_nn(job.cpu_only),
        mp_context=ctx,
    ) as ex:
        fn = partial(process_glacier_year_nn, job=job)
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
