# scripts/parallel_mb.py

"""
Parallel LSTM-based glacier mass balance inference.

This module runs gridded glacier mass balance (MB) inference using a trained
LSTM model for many glacier–year combinations in parallel.

High-level workflow
-------------------
1. Build (glacier, year) tasks from available parquet grid files, optionally
   restricted to geodetic years.
2. For each worker process:
   - Initialize with quiet stdout/stderr and restricted thread counts.
   - Load the LSTM model once (cached per process).
3. For each glacier-year:
   - Read monthly grid parquet, prepare annual and winter subsets.
   - Fit scalers using the provided training dataset indices.
   - Run annual prediction, save to per-year DEM grid (zarr).
   - Optionally compute and save cumulative monthly products.
   - Run winter prediction (if winter months exist), save outputs.

All functions are suffixed with `_lstm` (or contain `lstm`) to avoid name
collisions with other parallel inference scripts (e.g., NN, XGB).
"""

from __future__ import annotations

import os
import sys
import io
import multiprocessing as mp
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
from regions.Switzerland.scripts.utils import *

# 3rd party progress bar is optional; caller can pass a shim
try:
    from tqdm.auto import tqdm as _tqdm_default
except Exception:  # pragma: no cover

    def _tqdm_default(x, **k):
        return x


# ----------------- dataclass config -----------------
@dataclass(frozen=True)
class MBJobConfig_lstm:
    """
    Configuration container for parallel LSTM glacier MB inference.

    Attributes
    ----------
    cfg : Any
        Configuration object used by massbalancemachine (e.g., SwitzerlandConfig).
    MONTHLY_COLS : list of str
        Monthly (time-varying) feature column names.
    STATIC_COLS : list of str
        Static (time-invariant) feature column names.
    fields_not_features : list of str
        Extra metadata fields that must be kept but are not model features
        (e.g., cfg.fieldsNotFeatures).
    model_filename : str
        Path to a saved LSTM model state dict.
    custom_params : dict
        Model hyperparameters used by `mbm.models.LSTM_MB.build_model_from_params`.
    ds_train : Any
        Untransformed training dataset used to fit scalers.
    train_idx : Iterable[int]
        Indices of training samples used when fitting scalers.
    months_head_pad : int
        Number of padded months at the beginning of each sequence.
    months_tail_pad : int
        Number of padded months at the end of each sequence.
    data_path : str
        Base data path (often cfg.dataPath).
    path_glacier_grid_glamos : str
        Relative path to glacier parquet grid files.
    path_xr_grids : str
        Path to per-glacier per-year DEM grids (zarr).
    path_save_glw : str
        Output directory where glacier-wide products are written.
    seed : int, optional
        Random seed for loader reproducibility.
    max_workers : int, optional
        Maximum number of parallel worker processes.
    cpu_only : bool, optional
        If True, force CPU-only execution.
    ONLY_GEODETIC : bool, optional
        If True, only process years within the geodetic range for each glacier.
    save_monthly : bool, optional
        If True, also save cumulative monthly products.
    denorm : bool, optional
        If True, denormalize monthly predictions when supported by the model helper.
    """

    # Required: external objects / constants
    cfg: Any
    MONTHLY_COLS: List[str]
    STATIC_COLS: List[str]
    fields_not_features: List[str]

    # Model/artifacts
    model_filename: str
    custom_params: Dict[str, Any]
    ds_train: Any
    train_idx: Iterable[int]
    months_head_pad: int
    months_tail_pad: int

    # Data sources / destinations
    data_path: str
    path_glacier_grid_glamos: str
    path_xr_grids: str
    path_save_glw: str

    # Misc
    seed: int = 0
    max_workers: Optional[int] = None
    cpu_only: bool = True
    ONLY_GEODETIC: bool = True
    save_monthly: bool = True
    denorm: bool = True


# ----------------- worker init (quiet + CPU threads cap) -----------------
def worker_init_quiet_lstm(cpu_only: bool = True) -> None:
    """
    Initialize a worker process for LSTM inference.

    Suppresses stdout/stderr and caps thread usage to reduce oversubscription.
    Optionally disables CUDA to enforce CPU-only inference.

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


# ----------------- per-process model cache -----------------
_LSTM_MODEL = None  # one per process


def get_lstm_model_cpu_cached(
    cfg: Any, params_used: Dict[str, Any], model_filename: str
):
    """
    Build, load, and cache the LSTM model once per worker (CPU).

    The first call in a worker process:
    - builds the model from params
    - loads a state_dict from `model_filename`
    - puts it into eval() mode
    Subsequent calls reuse the cached model.

    Parameters
    ----------
    cfg : Any
        massbalancemachine configuration object.
    params_used : dict
        Hyperparameter dictionary for model construction.
    model_filename : str
        Path to a serialized PyTorch state_dict.

    Returns
    -------
    torch.nn.Module
        The cached LSTM model on CPU in eval mode.
    """
    global _LSTM_MODEL
    if _LSTM_MODEL is None:
        device = torch.device("cpu")
        model = mbm.models.LSTM_MB.build_model_from_params(
            cfg, params_used, device, verbose=False
        )
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state)
        model.eval()
        _LSTM_MODEL = model
    return _LSTM_MODEL


# ----------------- one glacier-year task -----------------
def process_glacier_year_lstm(
    args: Tuple[str, int],
    job: MBJobConfig_lstm,
) -> Tuple[str, str, int, str]:
    """
    Run LSTM gridded MB prediction for a single (glacier, year).

    The function loads the parquet grid for the glacier-year, constructs
    annual and winter sequence datasets, fits scalers using the provided
    training dataset, runs predictions, and writes outputs.

    Parameters
    ----------
    args : tuple (str, int)
        (glacier_name, year) task identifier.
    job : MBJobConfig_lstm
        Job configuration object.

    Returns
    -------
    tuple
        (status, glacier_name, year, message), where status is one of:
        - "ok"   : success
        - "skip" : missing input data (parquet/DEM/etc.)
        - "err"  : an exception occurred
    """
    glacier_name, year = args
    try:
        # paths
        glacier_path = os.path.join(
            job.data_path + job.path_glacier_grid_glamos, glacier_name
        )
        if not os.path.exists(glacier_path):
            return ("skip", glacier_name, year, "glacier folder missing")

        parquet_path = os.path.join(glacier_path, f"{glacier_name}_grid_{year}.parquet")
        if not os.path.exists(parquet_path):
            return ("skip", glacier_name, year, "parquet missing")

        df_grid_monthly = pd.read_parquet(parquet_path).copy()
        df_grid_monthly.drop_duplicates(inplace=True)

        REQUIRED = ["GLACIER", "YEAR", "ID", "PERIOD", "MONTHS"]
        all_columns = job.MONTHLY_COLS + job.STATIC_COLS + job.fields_not_features
        needed = set(all_columns) | set(REQUIRED)
        keep = [c for c in df_grid_monthly.columns if c in needed]
        df_grid_monthly = df_grid_monthly[keep]

        if "POINT_BALANCE" not in df_grid_monthly.columns:
            df_grid_monthly["POINT_BALANCE"] = 0.0  # dummy target

        # If the grid starts at aug_ need to set PMB at NaN for aug_ and sep_
        extrapolate_months = ["aug_", "sep_"]
        df_grid_monthly.loc[
            df_grid_monthly["MONTHS"].str.lower().isin(extrapolate_months),
            "POINT_BALANCE",
        ] = np.nan

        winter_months = ["sep", "oct", "nov", "dec", "jan", "feb", "mar", "apr"]
        df_grid_monthly_w = (
            df_grid_monthly[df_grid_monthly["MONTHS"].str.lower().isin(winter_months)]
            .copy()
            .dropna(subset=["ID", "MONTHS"])
        )
        df_grid_monthly_w["PERIOD"] = "winter"

        df_grid_monthly_a = df_grid_monthly.dropna(subset=["ID", "MONTHS"])

        # Fit scalers on TRAIN only
        ds_train_copy = (
            mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
                job.ds_train
            )
        )
        ds_train_copy.fit_scalers(job.train_idx)

        # Annual ds/loader
        ds_gl_a = mbm.data_processing.MBSequenceDataset.from_dataframe(
            df_grid_monthly_a,
            job.MONTHLY_COLS,
            job.STATIC_COLS,
            months_tail_pad=job.months_tail_pad,
            months_head_pad=job.months_head_pad,
            expect_target=True,
            show_progress=False,
        )
        test_gl_dl_a = mbm.data_processing.MBSequenceDataset.make_test_loader(
            ds_gl_a, ds_train_copy, seed=job.seed, batch_size=128
        )

        # Model
        model = get_lstm_model_cpu_cached(
            job.cfg, job.custom_params, job.model_filename
        )
        device = torch.device("cpu")

        # Predict annual
        df_preds_a = model.predict_with_keys(device, test_gl_dl_a, ds_gl_a)
        data_a = df_preds_a[["ID", "pred"]].set_index("ID")
        meta_cols = [
            c
            for c in ["YEAR", "POINT_LAT", "POINT_LON", "GLWD_ID"]
            if c in df_grid_monthly_a.columns
        ]
        grouped_ids_a = (
            df_grid_monthly_a.groupby("ID")[meta_cols]
            .first()
            .merge(data_a, left_index=True, right_index=True, how="left")
        )
        months_per_id_a = df_grid_monthly_a.groupby("ID")["MONTHS"].unique()
        grouped_ids_a = grouped_ids_a.merge(
            months_per_id_a, left_index=True, right_index=True
        )
        grouped_ids_a.reset_index(inplace=True)
        grouped_ids_a.sort_values(by="ID", inplace=True)

        pred_y_annual = grouped_ids_a.copy()
        pred_y_annual["PERIOD"] = "annual"
        pred_y_annual = pred_y_annual.drop(columns=["YEAR"], errors="ignore")

        # Load per-year DEM grid and save annual
        path_glacier_dem = os.path.join(
            job.path_xr_grids, f"{glacier_name}_{year}.zarr"
        )
        if not os.path.exists(path_glacier_dem):
            return ("skip", glacier_name, year, "DEM zarr missing")

        ds = xr.open_dataset(path_glacier_dem)

        geoData = mbm.geodata.GeoData(
            df_grid_monthly_a,
            months_head_pad=job.months_head_pad,
            months_tail_pad=job.months_tail_pad,
        )
        os.makedirs(job.path_save_glw, exist_ok=True)
        geoData._save_prediction(
            ds, pred_y_annual, glacier_name, year, job.path_save_glw, "annual"
        )

        if job.save_monthly:
            # --- Compute and save cumulative monthly predictions ---
            hydro_months = [
                "oct",
                "nov",
                "dec",
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
            ]

            df_preds_monthly = model.predict_monthly_with_keys(
                device,
                test_gl_dl_a,
                ds_gl_a,
                month_names=None,
                denorm=job.denorm,
                consistent_denorm=job.denorm,
            )

            df_wide = (
                df_preds_monthly.pivot_table(
                    index="ID",
                    columns="MONTH",
                    values="pred_consistent",
                    aggfunc="first",
                )
                .reindex(
                    columns=[
                        m
                        for m in hydro_months
                        if m in df_preds_monthly["MONTH"].unique()
                    ]
                )
                .reset_index()
            )

            df_wide = df_wide.merge(
                df_grid_monthly_a[["ID", "POINT_LON", "POINT_LAT"]].drop_duplicates(),
                on="ID",
                how="left",
            )

            df_cumulative = df_wide.copy()
            df_cumulative[hydro_months] = df_cumulative[hydro_months].cumsum(axis=1)

            df_cumulative = df_cumulative[
                ["ID", "POINT_LON", "POINT_LAT"] + hydro_months
            ]

            for month in hydro_months:
                df_month = df_cumulative[["ID", "POINT_LON", "POINT_LAT"]].copy()
                df_month["pred"] = df_wide[month].values
                df_month["cum_pred"] = df_cumulative[month].values

                geoData_m = mbm.geodata.GeoData(
                    df_grid_monthly_a,
                    months_head_pad=job.months_head_pad,
                    months_tail_pad=job.months_tail_pad,
                )
                geoData_m.data = df_month
                geoData_m.pred_to_xr(ds, pred_var="cum_pred", source_type="sgi")

                save_path = os.path.join(job.path_save_glw, glacier_name)
                os.makedirs(save_path, exist_ok=True)
                geoData_m.save_arrays(
                    f"{glacier_name}_{year}_{month}.zarr",
                    path=save_path + "/",
                    proj_type="wgs84",
                )

        # Winter branch
        if len(df_grid_monthly_w) == 0:
            return ("ok", glacier_name, year, "no winter months")

        ds_gl_w = mbm.data_processing.MBSequenceDataset.from_dataframe(
            df_grid_monthly_w,
            job.MONTHLY_COLS,
            job.STATIC_COLS,
            months_tail_pad=job.months_tail_pad,
            months_head_pad=job.months_head_pad,
            expect_target=False,
            show_progress=False,
        )
        test_gl_dl_w = mbm.data_processing.MBSequenceDataset.make_test_loader(
            ds_gl_w, ds_train_copy, seed=job.seed, batch_size=128
        )

        df_preds_w = model.predict_with_keys(device, test_gl_dl_w, ds_gl_w)

        data_w = df_preds_w[["ID", "pred"]].set_index("ID")
        grouped_ids_w = (
            df_grid_monthly_w.groupby("ID")[meta_cols]
            .first()
            .merge(data_w, left_index=True, right_index=True, how="left")
        )
        months_per_id_w = df_grid_monthly_w.groupby("ID")["MONTHS"].unique()
        grouped_ids_w = grouped_ids_w.merge(
            months_per_id_w, left_index=True, right_index=True
        )
        grouped_ids_w.reset_index(inplace=True)
        grouped_ids_w.sort_values(by="ID", inplace=True)

        pred_y_winter = grouped_ids_w.copy()
        pred_y_winter["PERIOD"] = "winter"
        pred_y_winter = pred_y_winter.drop(columns=["YEAR"], errors="ignore")

        geoData_w = mbm.geodata.GeoData(
            df_grid_monthly_w,
            months_head_pad=job.months_head_pad,
            months_tail_pad=job.months_tail_pad,
        )
        geoData_w._save_prediction(
            ds, pred_y_winter, glacier_name, year, job.path_save_glw, "winter"
        )

        return ("ok", glacier_name, year, "")

    except Exception as e:
        return ("err", glacier_name, year, str(e))


# ----------------- task builder -----------------
def build_tasks_lstm(
    glacier_list: List[str],
    periods_per_glacier: Dict[str, Iterable[int]],
    data_path: str,
    path_glacier_grid_glamos: str,
    ONLY_GEODETIC: bool,
) -> List[Tuple[str, int]]:
    """
    Build a list of (glacier, year) tasks for LSTM inference.

    Tasks are determined from available parquet files and optionally
    restricted to each glacier's geodetic year range.

    Parameters
    ----------
    glacier_list : list of str
        Glaciers to consider.
    periods_per_glacier : dict
        Mapping glacier -> iterable of years defining the geodetic range.
    data_path : str
        Base data directory.
    path_glacier_grid_glamos : str
        Relative path to glacier parquet grids.
    ONLY_GEODETIC : bool
        If True, filter years to the geodetic range.

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
            [
                f
                for f in os.listdir(glacier_path)
                if glacier_name in f and f.endswith(".parquet")
            ]
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


# ----------------- main entry point -----------------
def run_glacier_mb_lstm_parallel(
    job: MBJobConfig_lstm,
    glacier_list: List[str],
    periods_per_glacier: Dict[str, Iterable[int]],
    tqdm=_tqdm_default,
) -> Dict[str, int]:
    """
    Run parallel LSTM glacier-year MB inference and save outputs.

    Parameters
    ----------
    job : MBJobConfig_lstm
        LSTM inference job configuration.
    glacier_list : list of str
        Glaciers to process.
    periods_per_glacier : dict
        Mapping glacier -> iterable of years defining geodetic range.
    tqdm : callable, optional
        Progress wrapper (defaults to tqdm.auto.tqdm if available).

    Returns
    -------
    dict
        Summary counts: {"ok": int, "skip": int, "err": int, "total": int}.
    """
    os.makedirs(job.path_save_glw, exist_ok=True)
    emptyfolder(job.path_save_glw)

    tasks = build_tasks_lstm(
        glacier_list,
        periods_per_glacier,
        job.data_path,
        job.path_glacier_grid_glamos,
        job.ONLY_GEODETIC,
    )
    if len(tasks) == 0:
        return dict(ok=0, skip=0, err=0, total=0)

    ctx = mp.get_context("fork")  # Linux
    max_workers = job.max_workers or min(max(1, (os.cpu_count() or 2) - 1), 32)

    class _Devnull(io.StringIO):
        def write(self, *args, **kwargs):
            return 0

    ok = skip = err = 0
    errors = []

    with redirect_stdout(_Devnull()):  # silence workers and tqdm stdout
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=lambda: worker_init_quiet_lstm(job.cpu_only),
            mp_context=ctx,
        ) as ex:
            fn = partial(process_glacier_year_lstm, job=job)
            futures = [ex.submit(fn, t) for t in tasks]

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"LSTM Predict ({max_workers} workers)",
            ):
                status, g, y, msg = fut.result()
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skip += 1
                else:
                    err += 1
                    errors.append((g, y, msg))

    # ----------- OUTSIDE redirect_stdout, printing now works -----------
    if errors:
        print("\nErrors:")
        for g, y, msg in errors[:25]:
            print(f"{g} {y} → {msg}")
        if len(errors) > 25:
            print(f"... ({len(errors) - 25} more errors not shown)")

    print("\nSUMMARY:", dict(ok=ok, skip=skip, err=err, total=len(tasks)))
    return dict(ok=ok, skip=skip, err=err, total=len(tasks))
