"""Fine-tuned models and distributed predictions of the multi-region model
(notebook `3.3 Pooled_model_pred`; figures in `pooled_pred_figures.py`).

- `train_or_load_ft_models_for_region`: fine-tuned models of one region (ID split, all repeats).
- `predict_grid_from_cached_datasets`: predict MB on the cached glacier grid datasets.
- `aggregate_gridded_transformer_by_glacier`: predicted MB per 100 m elevation band.
- `_stake_overlay_transformer`: stakes on top of a predicted MB map.
"""

import gc
import glob
import os
import re

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import xarray as xr
import massbalancemachine as mbm
from sklearn.metrics import root_mean_squared_error

from papers.Transformer_TF.PMB_sources.RGI_11_Switzerland.scripts.geo_data import (
    apply_gaussian_filter,
)
from papers.Transformer_TF.scripts.config_TF_Europe import MONTHLY_COLS, STATIC_COLS
from papers.Transformer_TF.scripts.config_models import PARAMS_FT
from papers.Transformer_TF.scripts.dataset.xreg_lstm_dataset import (
    build_or_load_lstm_dataset_only,
)
from papers.Transformer_TF.scripts.models.transformer_helpers import (
    finetune_transformer_on_target,
)


def train_or_load_ft_models_for_region(
    region_code,
    monthly_cache,
    model_pooled,
    models_dir_ft,
    strategy,
    n_repeats,
    model_date,
    cfg,
    device,
    best_params,
    ds_pooled_scaler,
    ft_frac=0.10,
    force_retrain=False,
    CACHE_DIR="",
):
    """
    Train (or load from disk if already trained) N_REPEATS fine-tuned models
    for a single region using ID-level splitting, same pattern as CEU.
    """
    if region_code not in monthly_cache:
        raise KeyError(f"{region_code} not in monthly_cache")

    df_monthly = monthly_cache[region_code]["data_monthly"]
    df_monthly_aug = monthly_cache[region_code]["data_monthly_aug"]
    head_pad = monthly_cache[region_code]["months_head_pad"]
    tail_pad = monthly_cache[region_code]["months_tail_pad"]

    winter_ids_full = (
        df_monthly[df_monthly["PERIOD"] == "winter"]["ID"].unique().tolist()
    )
    annual_ids_full = (
        df_monthly[df_monthly["PERIOD"] == "annual"]["ID"].unique().tolist()
    )

    print(
        f"  {region_code}: {df_monthly['GLACIER'].nunique()} glaciers | "
        f"winter IDs: {len(winter_ids_full)} | annual IDs: {len(annual_ids_full)}"
    )

    ft_models = []
    for rep in range(n_repeats):
        model_path = str(
            models_dir_ft
            / f"transformer_ft_{strategy}_{region_code}_IDsplit_rep{rep}_{model_date}.pt"
        )

        winter_ids = list(winter_ids_full)
        annual_ids = list(annual_ids_full)
        rep_seed = cfg.seed + rep
        rng = np.random.default_rng(rep_seed)
        rng.shuffle(winter_ids)
        rng.shuffle(annual_ids)

        n_ft_w = max(1, int(len(winter_ids) * ft_frac)) if winter_ids else 0
        n_ft_a = max(1, int(len(annual_ids) * ft_frac))
        ft_ids = winter_ids[:n_ft_w] + annual_ids[:n_ft_a]

        ds_ft = build_or_load_lstm_dataset_only(
            cfg=cfg,
            key=f"FD_to_{region_code}_IDsplit_rep{rep}_FT",
            df_loss=df_monthly[df_monthly["ID"].isin(ft_ids)],
            df_full=df_monthly_aug[df_monthly_aug["ID"].isin(ft_ids)],
            months_head_pad=head_pad,
            months_tail_pad=tail_pad,
            MONTHLY_COLS=MONTHLY_COLS,
            STATIC_COLS=STATIC_COLS,
            cache_dir=str(CACHE_DIR),
            force_recompute=False,
            kind="ft",
            show_progress=False,
        )

        print(f"  [rep {rep}] ds_ft: {len(ds_ft)} seqs | model: {model_path}")

        ft_model = finetune_transformer_on_target(
            cfg=cfg,
            model_pooled=model_pooled,
            ds_ft=ds_ft,
            ds_pooled_scaler=ds_pooled_scaler,
            device=device,
            best_params=best_params,
            model_filename=model_path,
            strategy=strategy,
            epochs=PARAMS_FT["epochs"],
            lr_factor=PARAMS_FT["lr_factor"],
            force_retrain=force_retrain,  # False = load if exists, train if not
        )
        ft_models.append((rep, ft_model))
        print(f"  [rep {rep}] done")

    return ft_models


def predict_grid_from_cached_datasets(
    cfg,
    ds_cache_dir,
    model,
    ds_pooled_scaler,
    device,
    region_folder,
    path_save_glw,
    months_head_pad,
    months_tail_pad,
    batch_size=512,
    skip_existing=True,
    glacier_filter=None,
    period="annual",
    empty_cache_every=1,  # bump to 5-10 if empty_cache()'s sync overhead slows things down
):
    basepath = os.path.join(cfg.dataPath, "RGI_v6", region_folder)
    zarr_dir = os.path.join(basepath, "xr_masked_grids")
    monthly_parquet_dir = os.path.join(basepath, "monthly_parquets")
    os.makedirs(path_save_glw, exist_ok=True)

    ds_paths = sorted(glob.glob(os.path.join(ds_cache_dir, "*", "*", "ds.joblib")))
    if not ds_paths:
        raise FileNotFoundError(f"No cached datasets found under {ds_cache_dir}")

    if glacier_filter is not None:
        glacier_filter = set(glacier_filter)
        ds_paths = [
            p
            for p in ds_paths
            if os.path.basename(os.path.dirname(os.path.dirname(p))) in glacier_filter
        ]

    # print(ds_paths)
    n_rows_total = 0
    written_paths = []
    skipped_empty = 0

    for i, ds_path in enumerate(ds_paths):
        rgi_id = os.path.basename(os.path.dirname(os.path.dirname(ds_path)))
        year = int(os.path.basename(os.path.dirname(ds_path)))
        # print(year)
        # if year == 2008:
        #     continue
        out_zarr = os.path.join(path_save_glw, rgi_id, f"{rgi_id}_{year}_{period}.zarr")
        if skip_existing and os.path.exists(out_zarr):
            written_paths.append(out_zarr)
            continue

        xr_zarr_path = os.path.join(zarr_dir, f"{rgi_id}.zarr")
        if not os.path.exists(xr_zarr_path):
            print(f"  SKIP {rgi_id} {year}: masked zarr not found at {xr_zarr_path}")
            continue
        ds_latlon = xr.open_zarr(xr_zarr_path)

        monthly_path = os.path.join(
            monthly_parquet_dir, rgi_id, f"{rgi_id}_grid_{year}.parquet"
        )
        if not os.path.exists(monthly_path):
            print(f"  SKIP {rgi_id} {year}: monthly parquet not found")
            ds_latlon.close()
            continue

        ds = joblib.load(ds_path)
        if len(ds) == 0:
            skipped_empty += 1
            ds_latlon.close()
            continue

        ds_copy = mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(ds)
        dl = mbm.data_processing.MBSequenceDataset.make_test_loader(
            ds_copy, ds_pooled_scaler, batch_size=batch_size, seed=cfg.seed
        )

        # force inference mode explicitly — if predict_with_keys doesn't
        # already wrap this in no_grad()/eval(), every forward pass keeps
        # building an autograd graph that never gets freed until backward(),
        # which is a classic silent GPU-memory leak over a long loop
        with torch.no_grad():
            df_preds = model.predict_with_keys(device, dl, ds_copy, denorm=True)
        df_preds = df_preds[df_preds["PERIOD"] == period]

        if len(df_preds) == 0:
            skipped_empty += 1
            ds_latlon.close()
            continue

        df_monthly = pd.read_parquet(
            monthly_path, columns=["ID", "POINT_LAT", "POINT_LON"]
        ).drop_duplicates(subset=["ID"])

        df_merged = df_preds[["ID", "pred"]].merge(df_monthly, on="ID", how="left")
        df_merged = df_merged.dropna(subset=["POINT_LAT", "POINT_LON", "pred"])

        if len(df_merged) == 0:
            skipped_empty += 1
            ds_latlon.close()
            continue

        geoData = mbm.geodata.GeoData(
            df_merged,
            months_head_pad=months_head_pad,
            months_tail_pad=months_tail_pad,
        )
        geoData._save_prediction(
            ds_latlon, df_merged, rgi_id, year, path_save_glw, period
        )

        n_rows_total += len(df_merged)
        written_paths.append(out_zarr)

        # explicitly close the zarr store instead of relying on del+gc —
        # xr.open_zarr keeps a lazy/dask-backed handle open until .close()
        ds_latlon.close()
        del ds, ds_copy, dl, df_preds, df_merged, ds_latlon
        gc.collect()

        # return PyTorch's cached-but-unused GPU memory to the driver;
        # gc.collect() alone frees Python refs but not the CUDA allocator pool
        if torch.cuda.is_available() and (i % empty_cache_every == 0):
            torch.cuda.empty_cache()

    if skipped_empty:
        print(f"  ({skipped_empty} skipped: empty or missing data)")
    return written_paths


def aggregate_gridded_transformer_by_glacier(rgi_id, path_pred, period="annual"):
    lstm_tag = "annual" if period in ("annual", "ann", "yearly") else "winter"
    base = os.path.join(path_pred, rgi_id)

    years = []
    if os.path.isdir(base):
        for f in glob.glob(os.path.join(base, f"*_{lstm_tag}.zarr")):
            m = re.match(r".*[\\/].+?_(\d{4})_" + re.escape(lstm_tag) + r"\.zarr$", f)
            if m:
                years.append(int(m.group(1)))
    years = sorted(set(years))

    if not years:
        print(f"  No predictions found for {rgi_id} in {base}")
        return pd.DataFrame()

    dfs = []
    for year in years:
        zarr_path = os.path.join(base, f"{rgi_id}_{year}_{lstm_tag}.zarr")
        if not os.path.exists(zarr_path):
            continue
        try:
            ds = apply_gaussian_filter(xr.open_dataset(zarr_path, engine="zarr"))
            lon_name = "lon" if "lon" in ds.coords else "longitude"
            lat_name = "lat" if "lat" in ds.coords else "latitude"

            df = (
                ds["pred_masked"]
                .to_dataframe()
                .reset_index()
                .drop(["x", "y"], axis=1, errors="ignore")
                .merge(
                    ds["masked_elev"]
                    .to_dataframe()
                    .reset_index()
                    .drop(["x", "y"], axis=1, errors="ignore"),
                    on=[lat_name, lon_name],
                    how="left",
                )
                .dropna()
                .rename(
                    columns={
                        "pred_masked": "pred",
                        "masked_elev": "POINT_ELEVATION",
                        lat_name: "lat",
                        lon_name: "lon",
                    }
                )
            )
            df["YEAR"] = year
            df["PERIOD"] = lstm_tag

            min_alt = np.floor(df["POINT_ELEVATION"].min() / 100) * 100
            max_alt = np.ceil(df["POINT_ELEVATION"].max() / 100) * 100
            bins = np.arange(min_alt, max_alt + 100, 100)
            df["altitude_interval"] = pd.cut(
                df["POINT_ELEVATION"], bins=bins, right=False
            )
            centers = {
                iv: round((iv.left + iv.right) / 2)
                for iv in df["altitude_interval"].cat.categories
            }
            df["altitude_interval"] = df["altitude_interval"].map(centers)
            dfs.append(df)
        except Exception as e:
            print(f"  [error] {rgi_id} {year}: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _stake_overlay_transformer(
    ax, glacier, year, cmap, norm, ds_pred, df_stakes, period
):
    """Overlay stakes on transformer prediction map, return RMSE if possible."""
    if df_stakes is None or ds_pred is None:
        return None

    sub = df_stakes[
        (df_stakes["GLACIER"] == glacier)
        & (df_stakes["YEAR"] == year)
        & (df_stakes["PERIOD"] == period)
    ].copy()

    if sub.empty:
        return None

    def _sample(row):
        try:
            return ds_pred.sel(
                lat=row["POINT_LAT"], lon=row["POINT_LON"], method="nearest"
            )["pred_masked"].item()
        except Exception:
            return np.nan

    sub["pred_at_stake"] = sub.apply(_sample, axis=1)
    sub = sub.dropna(subset=["pred_at_stake"])

    if not sub.empty:
        sns.scatterplot(
            data=sub,
            x="POINT_LON",
            y="POINT_LAT",
            hue="POINT_BALANCE",
            palette=cmap,
            hue_norm=norm,
            ax=ax,
            s=25,
            legend=False,
        )

    if "POINT_BALANCE" in sub.columns and len(sub) > 0:
        return root_mean_squared_error(sub["POINT_BALANCE"], sub["pred_at_stake"])
    return None
