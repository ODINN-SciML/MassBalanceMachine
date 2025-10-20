# scripts/parallel_mb.py

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

# add near the other imports
import massbalancemachine as mbm
from regions.Switzerland.scripts.helpers import *

# 3rd party progress bar is optional; caller can pass a shim
try:
    from tqdm.auto import tqdm as _tqdm_default
except Exception:  # pragma: no cover

    def _tqdm_default(x, **k):
        return x


# ----------------- dataclass config -----------------
@dataclass(frozen=True)
class MBJobConfig:
    # Required: external objects / constants
    cfg: Any  # mbm.SwitzerlandConfig or similar
    MONTHLY_COLS: List[str]
    STATIC_COLS: List[str]
    fields_not_features: List[str]  # cfg.fieldsNotFeatures in your setup

    # Model/artifacts
    model_filename: str
    custom_params: Dict[str, Any]
    ds_train: Any  # Untransformed training dataset
    train_idx: Iterable[int]  # indices for scaler fitting
    months_head_pad: int
    months_tail_pad: int

    # Data sources / destinations
    data_path: str  # cfg.dataPath
    path_glacier_grid_glamos: str
    path_xr_grids: str  # e.g. .../xr_masked_grids
    path_save_glw: str  # output folder

    # Misc
    seed: int = 0
    max_workers: Optional[int] = None
    cpu_only: bool = True
    ONLY_GEODETIC: bool = True


# ----------------- worker init (quiet + CPU threads cap) -----------------
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


# ----------------- per-process model cache -----------------
_MODEL = None  # one per process


def get_model_cpu(cfg, params_used, model_filename):
    """Build+load the model once per worker (cached)."""
    global _MODEL
    if _MODEL is None:
        device = torch.device("cpu")
        model = mbm.models.LSTM_MB.build_model_from_params(
            cfg, params_used, device, verbose=False
        )
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state)
        model.eval()
        _MODEL = model
    return _MODEL


# ----------------- one glacier-year task -----------------
def process_glacier_year(
    args: Tuple[str, int],
    job: MBJobConfig,
) -> Tuple[str, str, int, str]:
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
            expect_target=False,
            show_progress=False,
        )
        test_gl_dl_a = mbm.data_processing.MBSequenceDataset.make_test_loader(
            ds_gl_a, ds_train_copy, seed=job.seed, batch_size=128
        )

        # Model
        model = get_model_cpu(job.cfg, job.custom_params, job.model_filename)
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
        print(e)
        return ("err", glacier_name, year, str(e))


# ----------------- task builder -----------------
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

        # Process only geodetic years if specified
        if ONLY_GEODETIC:
            years = [y for y in years if y in geodetic_range]
        for y in years:
            tasks.append((glacier_name, y))
    return tasks


# ----------------- main entry point -----------------
def run_glacier_mb(
    job: MBJobConfig,
    glacier_list: List[str],
    periods_per_glacier: Dict[str, Iterable[int]],
    tqdm=_tqdm_default,
) -> Dict[str, int]:
    """Run parallel glacier-year MB inference & save. Returns summary counts."""
    # ensure output folder exists & is empty if desired
    os.makedirs(job.path_save_glw, exist_ok=True)
    emptyfolder(job.path_save_glw)
    # caller can empty it beforehand if they want

    tasks = build_tasks(
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

    with redirect_stdout(_Devnull()):  # keep stderr so tqdm is visible
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
                desc=f"Predicting ({max_workers} workers)",
            ):
                status, g, y, msg = fut.result()
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skip += 1
                else:
                    err += 1

    return dict(ok=ok, skip=skip, err=err, total=len(tasks))
