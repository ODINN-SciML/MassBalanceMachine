"""Domain shift between the pooled source data and each target region (notebook `3.4 Distance`)."""

import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from papers.Transformer_TF.scripts.dataset.topoclimatic_distance import (
    build_global_scalers_multi_source_simple,
    compute_domain_shift,
    estimate_global_bandwidths_simple,
)

# unit-sphere coordinates used as the "geographic" channel of the domain shift
COORD_COLS = ["COORD_X", "COORD_Y", "COORD_Z"]


def add_cartesian_coords(df, lon_col="POINT_LON", lat_col="POINT_LAT"):
    """Add unit-sphere Cartesian coordinates to avoid longitude wraparound issues."""
    df = df.copy()
    lat_rad = np.radians(df[lat_col])
    lon_rad = np.radians(df[lon_col])
    df["COORD_X"] = np.cos(lat_rad) * np.cos(lon_rad)
    df["COORD_Y"] = np.cos(lat_rad) * np.sin(lon_rad)
    df["COORD_Z"] = np.sin(lat_rad)
    return df


def compute_pooled_domain_shifts(
    cfg,
    res_xreg_by_source,
    monthly_cache,
    source_regions,
    target_regions,
    monthly_cols,
    static_cols,
    cache_path,
    recompute=False,
):
    """
    Climate, topographic, geographic and joint Sinkhorn distances (plus MMD² and
    energy distances) between the pooled training data of all source regions and
    the full data of each target region. Scalers and blurs are fitted on the
    source training data; three bandwidths (0.5x, 1x, 2x the blur) are used for
    the kernel distances.

    Loaded from cache_path if it exists (unless recompute=True), else computed and
    saved there. Returns {target region: shift dict}.
    """
    if recompute and os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"Deleted existing cache: {cache_path}")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            all_shifts_pooled = pickle.load(f)["all_shifts_pooled"]
        print(f"Loaded pooled shifts from cache: {cache_path}")
        return all_shifts_pooled

    # scalers and blurs from the source training data
    res_source_only_coords = {
        region: {
            "df_train": add_cartesian_coords(res_xreg_by_source[region]["df_train"])
        }
        for region in source_regions
    }
    scaler_m, scaler_s, scaler_joint, scaler_geo = (
        build_global_scalers_multi_source_simple(
            res_source_only_coords,
            monthly_cols=monthly_cols,
            static_cols=static_cols,
            geo_cols=COORD_COLS,
        )
    )
    blur_m, blur_s, blur_joint, blur_geo = estimate_global_bandwidths_simple(
        res_source_only_coords,
        monthly_cols=monthly_cols,
        static_cols=static_cols,
        geo_cols=COORD_COLS,
        scaler_m=scaler_m,
        scaler_s=scaler_s,
        scaler_geo=scaler_geo,
        seed=cfg.seed,
    )
    bandwidths_m = [blur_m * 0.5, blur_m, blur_m * 2.0]
    bandwidths_s = [blur_s * 0.5, blur_s, blur_s * 2.0]
    bandwidths_geo = [blur_geo * 0.5, blur_geo, blur_geo * 2.0]

    # source = full pooled training data (all source regions combined)
    df_src_pooled = add_cartesian_coords(
        pd.concat(
            [res_xreg_by_source[r]["df_train"] for r in source_regions],
            ignore_index=True,
        )
    )

    all_shifts_pooled = {}
    for region in tqdm(target_regions, desc="Pooled domain shifts"):
        df_tgt = add_cartesian_coords(monthly_cache[region]["data_monthly"])
        shift = compute_domain_shift(
            df_src=df_src_pooled,
            df_tgt=df_tgt,
            monthly_cols=monthly_cols,
            static_cols=static_cols,
            geo_cols=COORD_COLS,
            scaler_m=scaler_m,
            scaler_s=scaler_s,
            scaler_geo=scaler_geo,
            scaler_joint=scaler_joint,
            blur_m=blur_m,
            blur_s=blur_s,
            blur_geo=blur_geo,
            blur_joint=blur_joint,
            bandwidths_m=bandwidths_m,
            bandwidths_s=bandwidths_s,
            bandwidths_geo=bandwidths_geo,
            seed=cfg.seed,
        )
        all_shifts_pooled[region] = shift
        print(
            f"  {region}: joint={shift['D_sinkhorn_joint']:.4f} "
            f"clim={shift['D_sinkhorn_climate']:.4f} "
            f"topo={shift['D_sinkhorn_topo']:.4f} "
            f"geo={shift['D_sinkhorn_geo']:.4f}"
        )

    with open(cache_path, "wb") as f:
        pickle.dump({"all_shifts_pooled": all_shifts_pooled}, f)
    print(f"Saved pooled shifts to cache: {cache_path}")
    return all_shifts_pooled


def compute_ft_gain(
    df_ft_metrics_id,
    strategy="full",
    metric_keys=("RMSE_annual", "NRMSE_annual", "RMSE_winter", "NRMSE_winter"),
):
    """
    Fine-tuning gain per region: zero-shot minus fine-tuned error (positive = FT
    reduces the error), from the repeat means in df_ft_metrics_id. The std combines
    the two repeat stds in quadrature (an approximation, not a paired difference).

    Returns {region: {f"ft_gain_{strategy}": {f"{k}_gain": ..., f"{k}_gain_std": ...}}}.
    """
    gains = {}
    for region, results in df_ft_metrics_id.items():
        if "no_ft" not in results or strategy not in results:
            continue
        no_ft, ft = results["no_ft"], results[strategy]
        gain = {}
        for k in metric_keys:
            gain[f"{k}_gain"] = no_ft[k] - ft[k]
            gain[f"{k}_gain_std"] = float(
                np.sqrt(no_ft.get(f"{k}_std", 0.0) ** 2 + ft.get(f"{k}_std", 0.0) ** 2)
            )
        gains[region] = {f"ft_gain_{strategy}": gain}
    return gains
