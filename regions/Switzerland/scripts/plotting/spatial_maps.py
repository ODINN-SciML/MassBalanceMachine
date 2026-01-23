import seaborn as sns
from cmcrameri import cm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
import numpy as np
import os
from matplotlib.patches import Rectangle
from typing import Sequence, Optional, Tuple
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr
import colormaps as cmaps

from sklearn.metrics import (
    root_mean_squared_error,
)

from regions.Switzerland.scripts.config_CH import *
from regions.Switzerland.scripts.geo_data import *

from regions.Switzerland.scripts.plotting.palettes import get_color_maps


def plot_2glaciers_2years_glamos_vs_lstm(
    glacier_names,
    years_by_glacier,
    cfg,
    df_stakes=None,
    path_distributed_mb=None,
    path_pred_lstm=None,
    period="annual",
    apply_smoothing_fn=apply_gaussian_filter,
    add_panel_labels=True,  # whether to show labels
    panel_label_start="a",  # starting letter
):
    """
    Layout (2 rows × 4 panels) with one colorbar per glacier–year (outside maps):

      Row 1 (glacier_names[0]): G1-GLAMOS(y1a), G1-LSTM(y1a), |cbar|, G1-GLAMOS(y1b), G1-LSTM(y1b), |cbar|
      Row 2 (glacier_names[1]): G2-GLAMOS(y2a), G2-LSTM(y2a), |cbar|, G2-GLAMOS(y2b), G2-LSTM(y2b), |cbar|
    """
    assert len(glacier_names) == 2, "glacier_names must have length 2"
    assert len(years_by_glacier) == 2 and all(
        len(p) == 2 for p in years_by_glacier
    ), "years_by_glacier must be ((y1a,y1b),(y2a,y2b))"
    assert path_distributed_mb and path_pred_lstm

    # ---------- helpers ----------
    def pick_file_glamos(glacier, year, period="annual"):
        suffix = "ann" if period == "annual" else "win"
        base = os.path.join(cfg.dataPath, path_distributed_mb, "GLAMOS", glacier)
        cand_lv95 = os.path.join(base, f"{year}_{suffix}_fix_lv95.grid")
        cand_lv03 = os.path.join(base, f"{year}_{suffix}_fix_lv03.grid")
        if os.path.exists(cand_lv95):
            return cand_lv95, "lv95"
        if os.path.exists(cand_lv03):
            return cand_lv03, "lv03"
        return None, None

    def load_glamos_wgs84(glacier, year):
        path, cs = pick_file_glamos(glacier, year, period)
        if path is None:
            return None
        meta, arr = load_grid_file(path)
        da = convert_to_xarray_geodata(arr, meta)
        if cs == "lv03":
            return transform_xarray_coords_lv03_to_wgs84(da)
        if cs == "lv95":
            return transform_xarray_coords_lv95_to_wgs84(da)
        return None

    def load_lstm_ds(glacier, year):
        zpath = os.path.join(path_pred_lstm, glacier, f"{glacier}_{year}_{period}.zarr")
        if not os.path.exists(zpath):
            return None
        ds = xr.open_zarr(zpath)
        if apply_smoothing_fn is not None:
            ds = apply_smoothing_fn(ds)
        return ds

    def lonlat_names(obj):
        coords = getattr(obj, "coords", {})
        if "lon" in coords and "lat" in coords:
            return "lon", "lat"
        if "longitude" in coords and "latitude" in coords:
            return "longitude", "latitude"
        return "lon", "lat"

    def stake_overlay_rmse(ax, glacier, year, cmap, norm, da_glamos, ds_lstm, which):
        if df_stakes is None:
            return None
        sub = df_stakes[
            (df_stakes.GLACIER == glacier) & (df_stakes.YEAR == year)
        ].copy()
        if period == "annual" and "PERIOD" in sub.columns:
            sub = sub[sub.PERIOD == "annual"].copy()
        if sub.empty:
            return None

        lx, ly = lonlat_names(
            ds_lstm if which == "LSTM" and ds_lstm is not None else da_glamos
        )

        # Function to extract mass balance for each stake
        def _get_predicted_mb(lon_name, lat_name, row, ds):
            try:
                return ds.sel(
                    {lon_name: row.POINT_LON, lat_name: row.POINT_LAT}, method="nearest"
                ).pred_masked.item()  # Convert to scalar
            except Exception:
                print(
                    f"Warning: Stake at ({row.POINT_LON}, {row.POINT_LAT}) is out of bounds."
                )
                return np.nan

        def _get_predicted_mb_glamos(lon_name, lat_name, row, ds):
            try:
                return ds.sel(
                    {lon_name: row.POINT_LON, lat_name: row.POINT_LAT}, method="nearest"
                ).item()  # Convert to scalar
            except Exception:
                print(
                    f"Warning: Stake at ({row.POINT_LON}, {row.POINT_LAT}) is out of bounds."
                )
                return np.nan

        def _safe_pred(ds, row):
            try:
                return _get_predicted_mb(lx, ly, row, ds)
            except Exception:
                return np.nan

        def _safe_glamos(row):
            try:
                return _get_predicted_mb_glamos(lx, ly, row, da_glamos)
            except Exception:
                return np.nan

        if which == "GLAMOS":
            sub["FIELD"] = sub.apply(_safe_glamos, axis=1)
        else:
            sub["FIELD"] = (
                sub.apply(lambda r: _safe_pred(ds_lstm, r), axis=1)
                if ds_lstm is not None
                else np.nan
            )

        # ---------- PRINT STAKE VALUES ----------
        if "POINT_BALANCE" in sub.columns:
            print("\n" + "-" * 70)
            print(
                f"{which} STAKES | Glacier: {glacier} | Year: {year} | Period: {period}"
            )
            print("-" * 70)
            print(
                sub[["POINT_LON", "POINT_LAT", "POINT_BALANCE", "FIELD"]]
                .rename(
                    columns={
                        "POINT_LON": "lon",
                        "POINT_LAT": "lat",
                        "POINT_BALANCE": "obs_MB",
                    }
                )
                .round(3)
                .to_string(index=False)
            )
            print("-" * 70)

        hue_col = "POINT_BALANCE" if "POINT_BALANCE" in sub.columns else "FIELD"
        sns.scatterplot(
            data=sub,
            x="POINT_LON",
            y="POINT_LAT",
            hue=hue_col,
            palette=cmap,
            hue_norm=norm,
            ax=ax,
            s=18,
            legend=False,
        )

        if "POINT_BALANCE" in sub.columns and not np.all(np.isnan(sub["FIELD"])):
            return root_mean_squared_error(sub["POINT_BALANCE"], sub["FIELD"])
        return None

    # ---------- figure & gridspec (2 rows × 6 columns with CB slots) ----------
    fig = plt.figure(figsize=(28, 15))
    gs = gridspec.GridSpec(
        nrows=2,
        ncols=6,
        figure=fig,
        width_ratios=[1, 1, 0.045, 1, 1, 0.045],
        wspace=0.30,
        hspace=0.12,
    )

    first_ax_in_row = [None, None]
    map_axes = []  # store map panels only (exclude colorbars)

    for r, glacier in enumerate(glacier_names):
        row_years = years_by_glacier[r]
        for j, year in enumerate(row_years):
            col_base = 3 * j
            ax_g = fig.add_subplot(gs[r, col_base + 0], sharey=first_ax_in_row[r])
            if first_ax_in_row[r] is None:
                first_ax_in_row[r] = ax_g
            ax_m = fig.add_subplot(gs[r, col_base + 1], sharey=first_ax_in_row[r])
            ax_cb = fig.add_subplot(gs[r, col_base + 2])

            da_g = load_glamos_wgs84(glacier, year)
            ds_m = load_lstm_ds(glacier, year)

            vals = []
            if da_g is not None:
                vals += [float(da_g.min().item()), float(da_g.max().item())]
            if ds_m is not None and "pred_masked" in ds_m:
                vals += [
                    float(ds_m["pred_masked"].min().item()),
                    float(ds_m["pred_masked"].max().item()),
                ]
            if not vals:
                for ax in (ax_g, ax_m):
                    ax.text(
                        0.5, 0.5, f"No data\n{glacier} {year}", ha="center", va="center"
                    )
                    ax.set_axis_off()
                ax_cb.set_axis_off()
                continue

            vmin, vmax = min(vals), max(vals)
            (
                cmap,
                norm,
            ) = get_color_maps(
                vmin,
                vmax,
            )

            # --- GLAMOS panel ---
            mappable_g = None
            if da_g is None:
                ax_g.text(
                    0.5, 0.5, f"No GLAMOS\n{glacier} {year}", ha="center", va="center"
                )
                ax_g.set_axis_off()
            else:
                mappable_g = da_g.plot.imshow(
                    ax=ax_g, cmap=cmap, norm=norm, add_colorbar=False
                )
                ax_g.set_title(f"{glacier.capitalize()} – GLAMOS ({year})", fontsize=16)
                mean_g = float(da_g.mean().item())
                var_g = float(da_g.var().item())
                rmse_g = stake_overlay_rmse(
                    ax_g, glacier, year, cmap, norm, da_g, ds_m, which="GLAMOS"
                )
                text_g = (
                    f"RMSE: {rmse_g:.2f}\n" if rmse_g is not None else ""
                ) + f"mean MB: {mean_g:.2f}\nvar: {var_g:.2f}"
                ax_g.text(
                    0.03,
                    0.03,
                    text_g,
                    transform=ax_g.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )
                map_axes.append(ax_g)

            # --- LSTM panel ---
            mappable_m = None
            if ds_m is None or "pred_masked" not in ds_m:
                ax_m.text(
                    0.5, 0.5, f"No LSTM\n{glacier} {year}", ha="center", va="center"
                )
                ax_m.set_axis_off()
            else:
                mappable_m = ds_m["pred_masked"].plot.imshow(
                    ax=ax_m, cmap=cmap, norm=norm, add_colorbar=False
                )
                ax_m.set_title(f"{glacier.capitalize()} – MBM ({year})", fontsize=16)
                mean_m = float(ds_m["pred_masked"].mean().item())
                var_m = float(ds_m["pred_masked"].var().item())
                rmse_m = stake_overlay_rmse(
                    ax_m, glacier, year, cmap, norm, da_g, ds_m, which="LSTM"
                )
                text_m = (
                    f"RMSE: {rmse_m:.2f}\n" if rmse_m is not None else ""
                ) + f"mean MB: {mean_m:.2f}\nvar: {var_m:.2f}"
                ax_m.text(
                    0.03,
                    0.03,
                    text_m,
                    transform=ax_m.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )
                map_axes.append(ax_m)

            # --- shared colorbar ---
            pair_mappable = mappable_m or mappable_g
            if pair_mappable is not None:
                cb = fig.colorbar(pair_mappable, cax=ax_cb)
                cb.set_label("Mass Balance [m w.e.]", fontsize=16)
                cb.ax.set_ylim(vmin, vmax)

            # tidy labels
            if j == 0:
                ax_g.set_ylabel("Latitude")
                ax_m.tick_params(labelleft=False)
            else:
                ax_g.tick_params(labelleft=False)
                ax_m.tick_params(labelleft=False)
                ax_g.set_ylabel("")
                ax_m.set_ylabel("")
            if r == 0:
                ax_g.tick_params(labelbottom=False)
                ax_m.tick_params(labelbottom=False)
            else:
                ax_g.set_xlabel("Longitude")
                ax_m.set_xlabel("Longitude")

    # ---- Optionally add subplot labels ----
    if add_panel_labels:
        import string

        # Create a sequence of letters starting from the user-specified one
        all_letters = list(string.ascii_lowercase)
        try:
            start_idx = all_letters.index(panel_label_start.lower())
        except ValueError:
            start_idx = 0  # default to 'a' if invalid input
        labels = all_letters[start_idx : start_idx + len(map_axes)]

        for idx, (ax, label) in enumerate(zip(map_axes, labels)):
            ax.text(
                0.02,
                0.98,
                f"({label})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=18,
                color="black",
                bbox=dict(
                    facecolor="white",
                    alpha=0.5,
                    edgecolor="none",
                    boxstyle="round,pad=0.2",
                ),
                zorder=10,
            )

    plt.tight_layout()
    plt.show()

    return fig, map_axes


def plot_mass_balance_comparison(
    glacier_name,
    year,
    cfg,
    df_stakes,
    fig,
    axes,
    path_distributed_mb,  # base for GLAMOS grids
    path_pred_lstm,  # base for LSTM/XGB zarrs
    period="annual",
):
    """
    Plot mass balance comparison (GLAMOS vs MBM/LSTM) for a glacier and year,
    with RMSE, mean, and variance text annotations. Works for both annual and winter periods.
    """

    # ---- Filter stake data for glacier, year, and period ----
    stakes_data = df_stakes[
        (df_stakes.GLACIER == glacier_name)
        & (df_stakes.YEAR == year)
        & (df_stakes.PERIOD == period)
    ].copy()

    # ---- Locate GLAMOS grid (ann/win, LV03/LV95) ----
    def pick_file(cfg, glacier_name, year, period="annual"):
        suffix = "ann" if period == "annual" else "win"
        base = os.path.join(cfg.dataPath, path_distributed_mb, "GLAMOS", glacier_name)
        cand_lv95 = os.path.join(base, f"{year}_{suffix}_fix_lv95.grid")
        cand_lv03 = os.path.join(base, f"{year}_{suffix}_fix_lv03.grid")
        if os.path.exists(cand_lv95):
            return cand_lv95, "lv95"
        if os.path.exists(cand_lv03):
            return cand_lv03, "lv03"
        return None, None

    grid_path, coord_system = pick_file(cfg, glacier_name, year, period)
    if grid_path is None:
        raise FileNotFoundError(
            f"No GLAMOS {period} grid found for {glacier_name} {year}"
        )

    # ---- Load and transform GLAMOS data ----
    metadata, grid_data = load_grid_file(grid_path)
    da_glamos = convert_to_xarray_geodata(grid_data, metadata)
    if coord_system == "lv03":
        da_glamos_wgs84 = transform_xarray_coords_lv03_to_wgs84(da_glamos)
    elif coord_system == "lv95":
        da_glamos_wgs84 = transform_xarray_coords_lv95_to_wgs84(da_glamos)
    else:
        raise ValueError(f"Unknown coordinate system: {coord_system}")

    da_glamos_wgs84 = apply_gaussian_filter(da_glamos_wgs84, variable_name=None)

    # ---- Load LSTM predictions ----
    mbm_file_lstm = os.path.join(
        path_pred_lstm, glacier_name, f"{glacier_name}_{year}_{period}.zarr"
    )
    if not os.path.exists(mbm_file_lstm):
        raise FileNotFoundError(f"Missing MBM zarr: {mbm_file_lstm}")

    ds_mbm = apply_gaussian_filter(xr.open_zarr(mbm_file_lstm))

    # ---- Determine coordinate names ----
    lon_name = "lon" if "lon" in ds_mbm.coords else "longitude"
    lat_name = "lat" if "lat" in ds_mbm.coords else "latitude"

    # ---- Sample model and GLAMOS values at stake points ----
    def _get_predicted_mb(lon_name, lat_name, row, ds):
        try:
            return ds.sel(
                {lon_name: row.POINT_LON, lat_name: row.POINT_LAT}, method="nearest"
            ).pred_masked.item()  # Convert to scalar
        except Exception:
            print(
                f"Warning: Stake at ({row.POINT_LON}, {row.POINT_LAT}) is out of bounds."
            )
            return np.nan

    def _get_predicted_mb_glamos(lon_name, lat_name, row, ds):
        try:
            return ds.sel(
                {lon_name: row.POINT_LON, lat_name: row.POINT_LAT}, method="nearest"
            ).item()  # Convert to scalar
        except Exception:
            print(
                f"Warning: Stake at ({row.POINT_LON}, {row.POINT_LAT}) is out of bounds."
            )
            return np.nan

    if not stakes_data.empty:
        stakes_data["Predicted_MB_LSTM"] = stakes_data.apply(
            lambda row: _get_predicted_mb(lon_name, lat_name, row, ds_mbm), axis=1
        )
        stakes_data["GLAMOS_MB"] = stakes_data.apply(
            lambda row: _get_predicted_mb_glamos(
                lon_name, lat_name, row, da_glamos_wgs84
            ),
            axis=1,
        )
        stakes_data.dropna(subset=["Predicted_MB_LSTM", "GLAMOS_MB"], inplace=True)

    # ---- Color range ----
    vmin = min(
        float(da_glamos_wgs84.min().item()),
        float(ds_mbm["pred_masked"].min().item()),
    )
    vmax = max(
        float(da_glamos_wgs84.max().item()),
        float(ds_mbm["pred_masked"].max().item()),
    )

    cmap, norm = get_color_maps(
        vmin,
        vmax,
    )

    # ---- Plot setup ----
    # fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fontsize_text = 16

    # ======================
    # GLAMOS panel
    # ======================
    mappable = da_glamos_wgs84.plot.imshow(
        ax=axes[0],
        cmap=cmap,
        norm=norm,
        cbar_kwargs={"label": "Mass Balance [m w.e.]"},
    )
    axes[0].set_title(f"GLAMOS ({period.capitalize()})")
    mappable.colorbar.ax.set_ylim(vmin, vmax)

    if not stakes_data.empty:
        sns.scatterplot(
            data=stakes_data,
            x="POINT_LON",
            y="POINT_LAT",
            hue="POINT_BALANCE",
            palette=cmap,
            hue_norm=norm,
            ax=axes[0],
            s=25,
            legend=False,
        )

    mean_glamos = float(da_glamos_wgs84.mean().item())
    var_glamos = float(da_glamos_wgs84.var().item())
    rmse_glamos = (
        root_mean_squared_error(stakes_data.POINT_BALANCE, stakes_data.GLAMOS_MB)
        if not stakes_data.empty
        else None
    )

    text_glamos = ""
    if rmse_glamos is not None:
        text_glamos += f"RMSE: {rmse_glamos:.2f}\n"
    text_glamos += f"mean MB: {mean_glamos:.2f}\nvar: {var_glamos:.2f}"

    axes[0].text(
        0.05,
        0.2,
        text_glamos,
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=fontsize_text,
    )

    # ======================
    # MBM/LSTM panel
    # ======================
    mappable = ds_mbm["pred_masked"].plot.imshow(
        ax=axes[1],
        cmap=cmap,
        norm=norm,
        cbar_kwargs={"label": "Mass Balance [m w.e.]"},
    )
    mappable.colorbar.ax.set_ylim(vmin, vmax)
    axes[1].set_title(f"MBM LSTM ({period.capitalize()})")

    if not stakes_data.empty:
        sns.scatterplot(
            data=stakes_data,
            x="POINT_LON",
            y="POINT_LAT",
            hue="POINT_BALANCE",
            palette=cmap,
            hue_norm=norm,
            ax=axes[1],
            s=30,
            legend=False,
        )

    mean_mbm = float(ds_mbm["pred_masked"].mean().item())
    var_mbm = float(ds_mbm["pred_masked"].var().item())
    rmse_mbm = (
        root_mean_squared_error(
            stakes_data.POINT_BALANCE, stakes_data.Predicted_MB_LSTM
        )
        if not stakes_data.empty
        else None
    )

    text_mbm = ""
    if rmse_mbm is not None:
        text_mbm += f"RMSE: {rmse_mbm:.2f}\n"
    text_mbm += f"mean MB: {mean_mbm:.2f}\nvar: {var_mbm:.2f}"

    axes[1].text(
        0.05,
        0.2,
        text_mbm,
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=fontsize_text,
    )


def plot_mass_balance_comparison_cropped(
    glacier_name,
    year,
    cfg,
    df_stakes,
    fig,
    axes,
    path_distributed_mb,  # base for GLAMOS grids
    path_pred_lstm,  # base for LSTM/XGB zarrs
    period="annual",
    crop_pad_px=2,
    crop_source="mbm",  # "mbm" (default) or "glamos"
):
    """
    Same as plot_mass_balance_comparison, but crops the plotted map extent to the
    glacier footprint (non-NaN region) to reduce empty space.

    Cropping is done in index-space (pixels) using a valid-data mask and applied
    consistently to the plotted rasters and stake-point axis limits.
    """

    # ---- Filter stake data for glacier, year, and period ----
    stakes_data = df_stakes[
        (df_stakes.GLACIER == glacier_name)
        & (df_stakes.YEAR == year)
        & (df_stakes.PERIOD == period)
    ].copy()

    # ---- Locate GLAMOS grid (ann/win, LV03/LV95) ----
    def pick_file(cfg, glacier_name, year, period="annual"):
        suffix = "ann" if period == "annual" else "win"
        base = os.path.join(cfg.dataPath, path_distributed_mb, "GLAMOS", glacier_name)
        cand_lv95 = os.path.join(base, f"{year}_{suffix}_fix_lv95.grid")
        cand_lv03 = os.path.join(base, f"{year}_{suffix}_fix_lv03.grid")
        if os.path.exists(cand_lv95):
            return cand_lv95, "lv95"
        if os.path.exists(cand_lv03):
            return cand_lv03, "lv03"
        return None, None

    grid_path, coord_system = pick_file(cfg, glacier_name, year, period)
    if grid_path is None:
        raise FileNotFoundError(
            f"No GLAMOS {period} grid found for {glacier_name} {year}"
        )

    # ---- Load and transform GLAMOS data ----
    metadata, grid_data = load_grid_file(grid_path)
    da_glamos = convert_to_xarray_geodata(grid_data, metadata)
    if coord_system == "lv03":
        da_glamos_wgs84 = transform_xarray_coords_lv03_to_wgs84(da_glamos)
    elif coord_system == "lv95":
        da_glamos_wgs84 = transform_xarray_coords_lv95_to_wgs84(da_glamos)
    else:
        raise ValueError(f"Unknown coordinate system: {coord_system}")

    da_glamos_wgs84 = apply_gaussian_filter(da_glamos_wgs84, variable_name=None)

    # ---- Load LSTM predictions ----
    mbm_file_lstm = os.path.join(
        path_pred_lstm, glacier_name, f"{glacier_name}_{year}_{period}.zarr"
    )
    if not os.path.exists(mbm_file_lstm):
        raise FileNotFoundError(f"Missing MBM zarr: {mbm_file_lstm}")

    ds_mbm = apply_gaussian_filter(xr.open_zarr(mbm_file_lstm))

    # ---- Determine coordinate names ----
    lon_name = "lon" if "lon" in ds_mbm.coords else "longitude"
    lat_name = "lat" if "lat" in ds_mbm.coords else "latitude"

    # ---- Sample model and GLAMOS values at stake points ----
    # Function to extract mass balance for each stake
    def _get_predicted_mb(lon_name, lat_name, row, ds):
        try:
            return ds.sel(
                {lon_name: row.POINT_LON, lat_name: row.POINT_LAT}, method="nearest"
            ).pred_masked.item()  # Convert to scalar
        except Exception:
            print(
                f"Warning: Stake at ({row.POINT_LON}, {row.POINT_LAT}) is out of bounds."
            )
            return np.nan

    def _get_predicted_mb_glamos(lon_name, lat_name, row, ds):
        try:
            return ds.sel(
                {lon_name: row.POINT_LON, lat_name: row.POINT_LAT}, method="nearest"
            ).item()  # Convert to scalar
        except Exception:
            print(
                f"Warning: Stake at ({row.POINT_LON}, {row.POINT_LAT}) is out of bounds."
            )
            return np.nan

    if not stakes_data.empty:
        stakes_data["Predicted_MB_LSTM"] = stakes_data.apply(
            lambda row: _get_predicted_mb(lon_name, lat_name, row, ds_mbm), axis=1
        )
        stakes_data["GLAMOS_MB"] = stakes_data.apply(
            lambda row: _get_predicted_mb_glamos(
                lon_name, lat_name, row, da_glamos_wgs84
            ),
            axis=1,
        )
        stakes_data.dropna(subset=["Predicted_MB_LSTM", "GLAMOS_MB"], inplace=True)

    # ---- Color range ----
    vmin = min(
        float(da_glamos_wgs84.min().item()), float(ds_mbm["pred_masked"].min().item())
    )
    vmax = max(
        float(da_glamos_wgs84.max().item()), float(ds_mbm["pred_masked"].max().item())
    )
    cmap, norm = get_color_maps(vmin, vmax)

    fontsize_text = 16

    # ------------------------------------------------------------------
    # Cropping helper: find bounding box of valid pixels, then slice
    # ------------------------------------------------------------------
    def crop_da(da, pad_px=2):
        """
        Crop a 2D DataArray to the bounding box of non-NaN values.
        Returns (da_cropped, (xmin, xmax, ymin, ymax)) in coordinate space.
        """
        # Ensure we target the last two dims (assumed spatial)
        if da.ndim != 2:
            raise ValueError(f"Expected 2D DataArray for cropping, got {da.ndim}D")

        valid = np.isfinite(da.values)
        if not np.any(valid):
            # no valid data: return as-is
            xmin, xmax = float(da[da.dims[1]].min()), float(da[da.dims[1]].max())
            ymin, ymax = float(da[da.dims[0]].min()), float(da[da.dims[0]].max())
            return da, (xmin, xmax, ymin, ymax)

        iy, ix = np.where(valid)
        y0, y1 = iy.min(), iy.max()
        x0, x1 = ix.min(), ix.max()

        y0 = max(y0 - pad_px, 0)
        x0 = max(x0 - pad_px, 0)
        y1 = min(y1 + pad_px, valid.shape[0] - 1)
        x1 = min(x1 + pad_px, valid.shape[1] - 1)

        ydim, xdim = da.dims[0], da.dims[1]
        da_c = da.isel({ydim: slice(y0, y1 + 1), xdim: slice(x0, x1 + 1)})

        # Coordinate-space bounds for consistent axis limits
        xcoord = da_c[xdim].values
        ycoord = da_c[ydim].values
        xmin, xmax = float(np.min(xcoord)), float(np.max(xcoord))
        ymin, ymax = float(np.min(ycoord)), float(np.max(ycoord))

        return da_c, (xmin, xmax, ymin, ymax)

    # Choose cropping mask source
    if crop_source.lower() == "glamos":
        da_for_crop = da_glamos_wgs84
    else:
        da_for_crop = ds_mbm["pred_masked"]

    da_crop_ref, bounds = crop_da(da_for_crop, pad_px=crop_pad_px)
    xmin, xmax, ymin, ymax = bounds

    # Crop both rasters with the same bounds by slicing their coords
    # (more robust than reusing pixel indices when grids match but names differ)
    def crop_to_bounds(da, bounds):
        xmin, xmax, ymin, ymax = bounds
        ydim, xdim = da.dims[0], da.dims[1]
        # handle ascending/descending coords
        y = da[ydim].values
        x = da[xdim].values
        y_slice = slice(ymin, ymax) if y[0] < y[-1] else slice(ymax, ymin)
        x_slice = slice(xmin, xmax) if x[0] < x[-1] else slice(xmax, xmin)
        return da.sel({ydim: y_slice, xdim: x_slice})

    da_glamos_plot = crop_to_bounds(da_glamos_wgs84, bounds)
    da_mbm_plot = crop_to_bounds(ds_mbm["pred_masked"], bounds)

    # ======================
    # GLAMOS panel
    # ======================
    mappable = da_glamos_plot.plot.imshow(
        ax=axes[0],
        cmap=cmap,
        norm=norm,
        cbar_kwargs={"label": "Mass Balance [m w.e.]"},
    )
    axes[0].set_title(f"GLAMOS ({period.capitalize()})")
    mappable.colorbar.ax.set_ylim(vmin, vmax)

    if not stakes_data.empty:
        sns.scatterplot(
            data=stakes_data,
            x="POINT_LON",
            y="POINT_LAT",
            hue="POINT_BALANCE",
            palette=cmap,
            hue_norm=norm,
            ax=axes[0],
            s=25,
            legend=False,
        )

    # Apply consistent map extent (also trims point view)
    axes[0].set_xlim(xmin, xmax)
    axes[0].set_ylim(ymin, ymax)

    mean_glamos = float(da_glamos_wgs84.mean().item())
    var_glamos = float(da_glamos_wgs84.var().item())
    rmse_glamos = (
        root_mean_squared_error(stakes_data.POINT_BALANCE, stakes_data.GLAMOS_MB)
        if not stakes_data.empty
        else None
    )

    text_glamos = ""
    if rmse_glamos is not None:
        text_glamos += f"RMSE: {rmse_glamos:.2f}\n"
    text_glamos += f"mean MB: {mean_glamos:.2f}\nvar: {var_glamos:.2f}"

    axes[0].text(
        0.05,
        0.2,
        text_glamos,
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=fontsize_text,
    )

    # ======================
    # MBM/LSTM panel
    # ======================
    mappable = da_mbm_plot.plot.imshow(
        ax=axes[1],
        cmap=cmap,
        norm=norm,
        cbar_kwargs={"label": "Mass Balance [m w.e.]"},
    )
    mappable.colorbar.ax.set_ylim(vmin, vmax)
    axes[1].set_title(f"MBM LSTM ({period.capitalize()})")

    if not stakes_data.empty:
        sns.scatterplot(
            data=stakes_data,
            x="POINT_LON",
            y="POINT_LAT",
            hue="POINT_BALANCE",
            palette=cmap,
            hue_norm=norm,
            ax=axes[1],
            s=30,
            legend=False,
        )

    axes[1].set_xlim(xmin, xmax)
    axes[1].set_ylim(ymin, ymax)

    mean_mbm = float(ds_mbm["pred_masked"].mean().item())
    var_mbm = float(ds_mbm["pred_masked"].var().item())
    rmse_mbm = (
        root_mean_squared_error(
            stakes_data.POINT_BALANCE, stakes_data.Predicted_MB_LSTM
        )
        if not stakes_data.empty
        else None
    )

    text_mbm = ""
    if rmse_mbm is not None:
        text_mbm += f"RMSE: {rmse_mbm:.2f}\n"
    text_mbm += f"mean MB: {mean_mbm:.2f}\nvar: {var_mbm:.2f}"

    axes[1].text(
        0.05,
        0.2,
        text_mbm,
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=fontsize_text,
    )

    return da_glamos_plot, da_mbm_plot, bounds
