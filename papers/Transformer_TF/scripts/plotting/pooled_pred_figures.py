"""Distributed predictions of the multi-region model (notebook `3.3 Pooled_model_pred`).

- `build_ft_models_for_pred` / `build_all_region_configs`: which fine-tuned models
  and RGI folders belong to each region and stake glacier.
- `run_region`: zero-shot and fine-tuned prediction grids for all glaciers of a region.
- `plot_application_figure`: ZS vs FT maps and elevation gradients (Fig. 7).
- `plot_gradients_grid`: ZS vs FT elevation gradients of many glaciers (Fig. A11–A16).
"""

import gc
import os
import math
import string
import itertools

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import root_mean_squared_error
from tqdm.auto import tqdm
from massbalancemachine.plots.style import COLOR_ANNUAL, COLOR_WINTER

from papers.Transformer_TF.PMB_sources.RGI_11_Switzerland.scripts.plotting.palettes import (
    get_color_maps,
)
from papers.Transformer_TF.PMB_sources.RGI_11_Switzerland.scripts.geo_data import (
    apply_gaussian_filter,
)

# reads {rgi_id}/{rgi_id}_{year}_{period}.zarr
from papers.Transformer_TF.PMB_sources.RGI_11_Switzerland.scripts.plotting.spatial_maps import (
    _load_lstm_ds,
)
from papers.Transformer_TF.scripts.plotting.pooled_pred import (
    _stake_overlay_transformer,
    aggregate_gridded_transformer_by_glacier,
    predict_grid_from_cached_datasets,
)

# RGI regions of the prediction grids, with the region codes of their stake data
RGI_REGIONS_PRED = {
    "01": {
        "name": "Alaska",
        "folder": "RGI_01_Alaska",
        "file": "01_rgi60_Alaska.shp",
        "code": "ALA",
        "era5_source": "EU_US_CANADA",
    },
    "06": {
        "name": "Iceland",
        "folder": "RGI_06_Iceland",
        "file": "06_rgi60_Iceland.shp",
        "code": "ISL",
        "era5_source": "EU_US_CANADA",
    },
    "07": {
        "name": "Svalbard",
        "folder": "RGI_07_Svalbard",
        "file": "07_rgi60_Svalbard.shp",
        "code": "SJM",
        "era5_source": "EU_US_CANADA",
    },
    "08": {
        "name": "Scandinavia",
        "folder": "RGI_08_Scandinavia",
        "file": "08_rgi60_Scandinavia.shp",
        "code": "SCA",
        "subregions": ["Norway"],
        "subregions_codes": ["NOR"],
        "era5_source": "EU_US_CANADA",
    },
    "11": {
        "name": "CentralEurope",
        "folder": "RGI_11_CentralEurope",
        "file": "11_rgi60_CentralEurope.shp",
        "code": "CEU",
        "subregions": ["France", "Switzerland", "Italy_Austria"],
        "subregions_codes": ["FR", "CH", "IT", "AT"],
        "era5_source": "EU_US_CANADA",
    },
    "13": {
        "name": "CentralAsia",
        "folder": "RGI_13_CentralAsia",
        "file": "13_rgi60_CentralAsia.shp",
        "code": "CENTRALASIA",
        "subregions_codes": ["CA_3", "CA_4"],
        "era5_source": "HMA",
    },
}


def resolve_rgi_region(region_code, rgi_regions=RGI_REGIONS_PRED):
    """
    Map a monthly_cache region code (e.g. 'CH', 'NOR', 'CA_3') to its
    (rgi_id_str, region_info) entry in rgi_regions, checking both top-level
    'code' and 'subregions_codes'.
    """
    for rgi_id_str, info in rgi_regions.items():
        if info["code"] == region_code:
            return rgi_id_str, info
        if region_code in info.get("subregions_codes", []):
            return rgi_id_str, info
    raise KeyError(f"{region_code!r} not found in rgi_regions")


def build_ft_models_for_pred(ft_models_by_rgi, rgi_to_ft_region):
    """
    Fine-tuned models per prediction key '<rgi_id>_<region code>'. All Central
    European regions (CH, FR, IT, AT) use the model fine-tuned on Switzerland.
    """
    ft_models_for_pred = {
        "11_CH": ft_models_by_rgi["11"],
        "08_NOR": ft_models_by_rgi["08"],
        "01_ALA": ft_models_by_rgi["01"],
        "13_CA_3": ft_models_by_rgi["13_CA_3"],
        "13_CA_4": ft_models_by_rgi["13_CA_4"],
        "07_SJM": ft_models_by_rgi["07"],
        "11_FR": ft_models_by_rgi["11"],
        "11_IT": ft_models_by_rgi["11"],
        "11_AT": ft_models_by_rgi["11"],
    }
    # add any other region fine-tuned via rgi_to_ft_region
    for rgi_id_str, ft_code in rgi_to_ft_region.items():
        key = f"{rgi_id_str}_{ft_code}"
        if key not in ft_models_for_pred and rgi_id_str in ft_models_by_rgi:
            ft_models_for_pred[key] = ft_models_by_rgi[rgi_id_str]
    return ft_models_for_pred


def make_glacier_config_for_region(region_code, monthly_cache, dfs, ft_models_for_pred):
    """
    One entry per stake glacier of a region: its RGI IDs, RGI folder and number, the
    key of its fine-tuned model in ft_models_for_pred (None if there is none), and
    the stake data of its RGI region.
    """
    if region_code not in monthly_cache:
        raise KeyError(f"{region_code!r} not in monthly_cache")

    rgi_id_str, info = resolve_rgi_region(region_code)
    ft_local_key = f"{rgi_id_str}_{region_code}"
    df_stakes = dfs.get(rgi_id_str)
    data_monthly = monthly_cache[region_code]["data_monthly"]

    return {
        glacier: {
            "rgi_ids": data_monthly.query(f"GLACIER == '{glacier}'")["RGIId"]
            .unique()
            .tolist(),
            "region_folder": info["folder"],
            "rgi_id_str": rgi_id_str,
            "ft_local_key": (
                ft_local_key if ft_local_key in ft_models_for_pred else None
            ),
            "df_stakes": df_stakes,
        }
        for glacier in data_monthly["GLACIER"].unique()
    }


def build_all_region_configs(region_codes, monthly_cache, dfs, ft_models_for_pred):
    """
    Glacier configs of all regions. Returns (all_region_configs, all_glacier_config):
    {region code: {glacier: config}} and {glacier: config}.
    """
    all_region_configs = {}
    for region_code in region_codes:
        try:
            cfg_dict = make_glacier_config_for_region(
                region_code, monthly_cache, dfs, ft_models_for_pred
            )
        except KeyError as e:
            print(f"  [skip] {region_code}: {e}")
            continue
        all_region_configs[region_code] = cfg_dict
        n_ft = sum(1 for g in cfg_dict.values() if g["ft_local_key"] is not None)
        print(
            f"{len(cfg_dict)} {region_code} stake glaciers "
            f"({n_ft} with FT models available): {list(cfg_dict.keys())}"
        )

    all_glacier_config = {
        name: gcfg
        for cfg_dict in all_region_configs.values()
        for name, gcfg in cfg_dict.items()
    }
    return all_region_configs, all_glacier_config


def run_region(
    region_code,
    all_region_configs,
    cfg,
    device,
    model_pooled,
    ds_pooled_scaler,
    ft_models_for_pred,
    pred_base,
    months_head_pad,
    months_tail_pad,
    glacier_names=None,
    empty_cache_every=1,
):
    """
    Annual prediction grids of the multi-region zero-shot model and of the fine-tuned
    model (repeat 0) for all stake glaciers of one region (or only glacier_names),
    from the cached grid datasets in `RGI_v6/<folder>/grid_ds_cache`. Grids are saved
    to `<pred_base>/RGI_<rgi>_ZS` and `<pred_base>/RGI_<rgi>_FT_<key>_rep0`.

    Run one region at a time (restart the kernel in between for a clean GPU state).
    """
    if region_code not in all_region_configs:
        raise KeyError(
            f"{region_code!r} not in all_region_configs. "
            f"Available: {list(all_region_configs)}"
        )

    cfg_dict = all_region_configs[region_code]
    if glacier_names is not None:
        missing = [g for g in glacier_names if g not in cfg_dict]
        if missing:
            raise KeyError(
                f"{missing} not found in region {region_code!r}. "
                f"Available: {list(cfg_dict)}"
            )
        cfg_dict = {g: cfg_dict[g] for g in glacier_names}

    pbar = tqdm(cfg_dict.items(), total=len(cfg_dict), desc=region_code)
    for glacier_name, gcfg in pbar:
        pbar.set_description(f"{region_code}:{glacier_name}")
        folder = gcfg["region_folder"]
        rgi_id_str = gcfg["rgi_id_str"]
        common = dict(
            cfg=cfg,
            ds_cache_dir=os.path.join(cfg.dataPath, "RGI_v6", folder, "grid_ds_cache"),
            ds_pooled_scaler=ds_pooled_scaler,
            device=device,
            region_folder=folder,
            months_head_pad=months_head_pad,
            months_tail_pad=months_tail_pad,
            glacier_filter=gcfg["rgi_ids"],
            period="annual",
            empty_cache_every=empty_cache_every,
        )

        # multi-region zero-shot
        predict_grid_from_cached_datasets(
            model=model_pooled,
            path_save_glw=os.path.join(pred_base, f"RGI_{rgi_id_str}_ZS"),
            **common,
        )

        # multi-region fine-tuned (first repeat only)
        ft_local_key = gcfg["ft_local_key"]
        if ft_local_key and ft_local_key in ft_models_for_pred:
            for rep, ft_model in ft_models_for_pred[ft_local_key]:
                if rep == 0:
                    predict_grid_from_cached_datasets(
                        model=ft_model,
                        path_save_glw=os.path.join(
                            pred_base, f"RGI_{rgi_id_str}_FT_{ft_local_key}_rep{rep}"
                        ),
                        **common,
                    )
        else:
            pbar.write(
                f"  [no FT] {glacier_name}: no fine-tuned model for {ft_local_key}"
            )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print(f"Done with region {region_code} ({len(cfg_dict)} glaciers).")


def lookup_glacier_config(glacier_name, all_glacier_config):
    """
    Case-insensitive lookup into all_glacier_config. Returns the canonical key (the
    casing used in df_stakes['GLACIER']) and its config.
    """
    if glacier_name in all_glacier_config:
        return glacier_name, all_glacier_config[glacier_name]
    for key in all_glacier_config:
        if key.lower() == glacier_name.lower():
            return key, all_glacier_config[key]
    raise KeyError(f"{glacier_name!r} not found in ALL_GLACIER_CONFIG")


def zs_ft_paths(gcfg, pred_base, rep=0):
    """Folders of the zero-shot and fine-tuned (repeat rep) grids of a glacier."""
    rgi_id_str = gcfg["rgi_id_str"]
    ft_local_key = gcfg["ft_local_key"]
    zs_path = os.path.join(pred_base, f"RGI_{rgi_id_str}_ZS")
    ft_path = (
        os.path.join(pred_base, f"RGI_{rgi_id_str}_FT_{ft_local_key}_rep{rep}")
        if ft_local_key
        else None
    )
    return zs_path, ft_path


def plot_pixel_scatter_zs_vs_ft(ds_zs, ds_ft, ax, fontsize=8):
    """
    Pixel-wise scatter of multi-region zero-shot vs fine-tuned predictions. Both come
    from _load_lstm_ds off the same masked WGS84 grid, so no reprojection is needed.
    """
    if (
        ds_zs is None
        or ds_ft is None
        or "pred_masked" not in ds_zs
        or "pred_masked" not in ds_ft
    ):
        ax.set_axis_off()
        return

    zs = ds_zs["pred_masked"]
    ft = ds_ft["pred_masked"]

    if zs.sizes != ft.sizes:
        ft = ft.interp(lat=zs["lat"], lon=zs["lon"], method="nearest")

    mask = np.isfinite(zs.values) & np.isfinite(ft.values)
    x = zs.values[mask]
    y = ft.values[mask]

    if x.size == 0:
        ax.text(0.5, 0.5, "No overlap", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.scatter(x, y, s=4, alpha=0.3, color="#bababa")

    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, "k--", lw=0.8)

    r = np.corrcoef(x, y)[0, 1]
    rmse = root_mean_squared_error(x, y)

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.text(
        0.05,
        0.05,
        f"N = {x.size}\nr = {r:.2f}\nRMSE = {rmse:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
    )

    ax.set_xlabel("Zero-shot MB", fontsize=fontsize)
    ax.set_ylabel("Fine-tuned MB", fontsize=fontsize)
    ax.yaxis.set_label_coords(-0.28, 0.63)
    ax.tick_params(axis="both", which="both", labelsize=8)
    ax.set_title("")


def _plot_elevation_gradient(ax, df_zs, df_ft, df_stakes_gl, glacier_name, period):
    """ZS and FT mean ± std per elevation band, and stake means in 40 m bins."""
    for df, label, color in (
        (df_zs, "Zero-shot", COLOR_ANNUAL),
        (df_ft, "Fine-tuned", COLOR_WINTER),
    ):
        if df.empty:
            continue
        grouped = df.groupby("altitude_interval")["pred"]
        mean, std = grouped.mean(), grouped.std().fillna(0)
        ax.plot(mean.values, mean.index, color=color, lw=1.2, label=label)
        ax.fill_betweenx(
            mean.index,
            mean.values - std.values,
            mean.values + std.values,
            alpha=0.15,
            color=color,
        )

    sub = df_stakes_gl[
        (df_stakes_gl["GLACIER"] == glacier_name) & (df_stakes_gl["PERIOD"] == period)
    ].copy()
    if not sub.empty and "POINT_ELEVATION" in sub.columns:
        bin_size = 40
        bins = np.arange(
            np.floor(sub["POINT_ELEVATION"].min() / bin_size) * bin_size,
            np.ceil(sub["POINT_ELEVATION"].max() / bin_size) * bin_size + bin_size,
            bin_size,
        )
        sub["alt_bin"] = pd.cut(sub["POINT_ELEVATION"], bins=bins, right=False)
        centers = {
            iv: round((iv.left + iv.right) / 2) for iv in sub["alt_bin"].cat.categories
        }
        sub["alt_bin"] = sub["alt_bin"].map(centers)
        stake_mean = sub.groupby("alt_bin")["POINT_BALANCE"].mean()
        ax.scatter(
            stake_mean.values,
            stake_mean.index,
            marker="o",
            s=18,
            facecolors="none",
            edgecolors="black",
            lw=0.8,
            label="Stakes",
            zorder=5,
        )


def _gradient_legend_handles():
    return [
        Patch(facecolor=COLOR_ANNUAL, alpha=0.25, label="Zero-shot band"),
        Line2D([0], [0], color=COLOR_ANNUAL, lw=1.2, label="Zero-shot mean"),
        Patch(facecolor=COLOR_WINTER, alpha=0.25, label="Fine-tuned band"),
        Line2D([0], [0], color=COLOR_WINTER, lw=1.2, label="Fine-tuned mean"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.2,
            label="Stakes mean",
        ),
    ]


def plot_application_figure(
    map_rows, all_glacier_config, pred_base, ft_rep=0, period="annual"
):
    """
    Fig. 7: for each (glacier, year) cell of map_rows (up to 2 per row), the
    zero-shot and fine-tuned MB maps with stakes overlaid, and a pixel scatter of ZS
    vs FT; one colour scale per row. Bottom row: ZS vs FT elevation gradients of all
    glaciers, with stake means.

    map_rows : list of rows, each a list of {"glacier_name", "year", "text_pos"
        ('left' or 'right', position of the annotation), "ft_rep" (optional)}.
    """
    grad_glaciers = [cell["glacier_name"] for row in map_rows for cell in row]
    cm = 1 / 2.54

    ylabel_size = 8
    title_size = 9
    label_size = 9
    text_size = 8
    tick_size = 8

    n_map_rows = len(map_rows)
    n_grad_cols = len(grad_glaciers)

    # --- Figure + grid: map rows on top, gradient row at the bottom ---
    fig = plt.figure(figsize=(40 * cm, (n_map_rows * 11 + 9) * cm), dpi=300)

    gs_global = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[n_map_rows * (3.2 / 3), 1],
        hspace=0.1,
    )
    # per row: ZS, FT, scatter for cell 1; ZS, FT, scatter for cell 2; colorbar
    gs_maps = gs_global[0].subgridspec(
        n_map_rows,
        7,
        width_ratios=[1, 1, 0.7, 1, 1, 0.7, 0.05],
        wspace=0.4,
        hspace=0.3,
    )
    gs_grad = gs_global[1].subgridspec(
        1, n_grad_cols, width_ratios=[1] * n_grad_cols, wspace=0.35
    )

    label_iter = iter(
        string.ascii_lowercase + string.ascii_uppercase
    )  # 52 panel labels

    # --- Map rows: ZS vs FT ---
    for r, row_cells in enumerate(map_rows):
        # resolve each (glacier, year) cell into its config and prediction folders
        resolved_cells = []
        for cell_cfg in row_cells:
            glacier_name, gcfg = lookup_glacier_config(
                cell_cfg["glacier_name"], all_glacier_config
            )
            zs_path, ft_path = zs_ft_paths(
                gcfg, pred_base, rep=cell_cfg.get("ft_rep", ft_rep)
            )
            resolved_cells.append(
                {
                    "glacier_name": glacier_name,
                    "year": cell_cfg["year"],
                    "rgi_id": gcfg["rgi_ids"][0],
                    "df_stakes": gcfg["df_stakes"],
                    "zs_path": zs_path,
                    "ft_path": ft_path,
                    "text_pos": cell_cfg.get("text_pos", "left"),
                }
            )

        # 1) colour range across every cell of the row, both models
        row_vals = []
        for cell in resolved_cells:
            if cell["ft_path"] is None:
                continue
            for path in (cell["zs_path"], cell["ft_path"]):
                ds_tmp = _load_lstm_ds(
                    glacier=cell["rgi_id"],
                    year=cell["year"],
                    path_pred_lstm=path,
                    period=period,
                    apply_smoothing_fn=apply_gaussian_filter,
                )
                if ds_tmp is not None and "pred_masked" in ds_tmp:
                    row_vals += [
                        float(ds_tmp["pred_masked"].min()),
                        float(ds_tmp["pred_masked"].max()),
                    ]

        if len(row_vals) == 0:
            print(f"  [skip] row {r}: no predictions found")
            continue

        row_vmin = max(min(row_vals), -12)
        row_vmax = max(row_vals)
        vmax_display = min(row_vmax, 4.0)
        vmin_display = max(row_vmin, -12)
        cmap, norm = get_color_maps(row_vmin, row_vmax)

        ax_cb = fig.add_subplot(gs_maps[r, 6])
        last_mappable = None

        # 2) plot each cell (up to 2 per row)
        for j, cell in enumerate(resolved_cells):
            if cell["ft_path"] is None:
                print(
                    f"  [skip] {cell['glacier_name']}: no fine-tuned model for this region"
                )
                continue

            glacier_name = cell["glacier_name"]
            year = cell["year"]
            rgi_id = cell["rgi_id"]
            df_stakes = cell["df_stakes"]
            text_pos = cell["text_pos"]

            ax_zs = fig.add_subplot(gs_maps[r, 3 * j])
            ax_ft = fig.add_subplot(gs_maps[r, 3 * j + 1])
            ax_s = fig.add_subplot(gs_maps[r, 3 * j + 2])

            ds_zs = _load_lstm_ds(
                glacier=rgi_id,
                year=year,
                path_pred_lstm=cell["zs_path"],
                period=period,
                apply_smoothing_fn=apply_gaussian_filter,
            )
            ds_ft = _load_lstm_ds(
                glacier=rgi_id,
                year=year,
                path_pred_lstm=cell["ft_path"],
                period=period,
                apply_smoothing_fn=apply_gaussian_filter,
            )

            for ds, ax, label in (
                (ds_zs, ax_zs, "Zero-shot"),
                (ds_ft, ax_ft, "Fine-tuned"),
            ):
                if ds is None or "pred_masked" not in ds:
                    continue
                img = ds["pred_masked"].plot.imshow(
                    ax=ax, cmap=cmap, norm=norm, add_colorbar=False
                )
                ax.set_title(
                    f"{glacier_name.capitalize()} ({year})\n{label}",
                    fontsize=title_size,
                )

                mean_mb = float(ds["pred_masked"].mean())
                var_mb = float(ds["pred_masked"].var())
                rmse = _stake_overlay_transformer(
                    ax=ax,
                    glacier=glacier_name,
                    year=year,
                    cmap=cmap,
                    norm=norm,
                    ds_pred=ds,
                    df_stakes=df_stakes,
                    period=period,
                )
                txt = (
                    f"RMSE: {rmse:.2f}\n" if rmse else ""
                ) + f"mean MB: {mean_mb:.2f}\nvar: {var_mb:.2f}"
                x_txt, y_txt, ha_txt = (
                    (0.96, 0.04, "right")
                    if text_pos == "right"
                    else (0.04, 0.03, "left")
                )
                ax.text(
                    x_txt,
                    y_txt,
                    txt,
                    transform=ax.transAxes,
                    ha=ha_txt,
                    va="bottom",
                    fontsize=text_size,
                )
                ax.grid(alpha=0.2)
                last_mappable = img

            plot_pixel_scatter_zs_vs_ft(
                ds_zs=ds_zs, ds_ft=ds_ft, ax=ax_s, fontsize=text_size
            )
            ax_s.set_title("ZS vs FT", fontsize=title_size)

            ax_zs.set_ylabel("Latitude", fontsize=ylabel_size)
            ax_zs.yaxis.set_label_coords(-0.35 if j == 0 else -0.08, 0.63)
            ax_ft.set_ylabel("")
            ax_zs.set_xlabel("Longitude", fontsize=ylabel_size)
            ax_ft.set_xlabel("Longitude", fontsize=ylabel_size)
            ax_zs.tick_params(axis="x", labelsize=tick_size)
            ax_ft.tick_params(axis="x", labelsize=tick_size)

            if j == 0:
                ax_zs.tick_params(
                    axis="y", left=True, labelleft=True, labelsize=tick_size
                )
            else:
                ax_zs.tick_params(axis="y", left=False, labelleft=False)
            ax_ft.tick_params(labelleft=False, labelsize=tick_size)

            for ax in (ax_zs, ax_ft, ax_s):
                ax.text(
                    0.04,
                    0.98,
                    f"({next(label_iter)})",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=label_size,
                )

        # one colorbar per row; arrows mark values beyond the displayed range
        if last_mappable is not None:
            cb = fig.colorbar(last_mappable, cax=ax_cb)
            cb.ax.set_ylim(vmin_display, vmax_display)
            if min(row_vals) < vmin_display:
                ax_cb.plot(
                    [0.5],
                    [-0.02],
                    marker="v",
                    color="black",
                    markersize=4,
                    clip_on=False,
                    transform=ax_cb.transAxes,
                )
            if max(row_vals) > vmax_display:
                ax_cb.plot(
                    [0.5],
                    [1.02],
                    marker="^",
                    color="black",
                    markersize=4,
                    clip_on=False,
                    transform=ax_cb.transAxes,
                )
            cb.set_label("Mass Balance [m w.e.]", fontsize=ylabel_size)
            cb.ax.tick_params(labelsize=tick_size)
        else:
            ax_cb.set_axis_off()

    # --- Gradient row ---
    for c, gl in enumerate(grad_glaciers):
        ax = fig.add_subplot(gs_grad[0, c])
        glacier_name, gcfg = lookup_glacier_config(gl, all_glacier_config)
        rgi_id = gcfg["rgi_ids"][0]
        zs_path, ft_path = zs_ft_paths(gcfg, pred_base, rep=ft_rep)

        if ft_path is None:
            ax.set_title(f"{gl} — N/A", fontsize=title_size)
            ax.set_axis_off()
            continue

        df_zs = aggregate_gridded_transformer_by_glacier(rgi_id, zs_path, period=period)
        df_ft = aggregate_gridded_transformer_by_glacier(rgi_id, ft_path, period=period)
        _plot_elevation_gradient(
            ax, df_zs, df_ft, gcfg["df_stakes"], glacier_name, period
        )

        ax.axvline(0, color="k", lw=0.6, ls="--", alpha=0.5)
        ax.set_title(glacier_name.capitalize(), fontsize=title_size, pad=2)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=tick_size, pad=2)
        ax.text(
            0.04,
            0.98,
            f"({next(label_iter)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=label_size,
        )
        ax.set_ylabel("Elevation (m a.s.l.)" if c == 0 else "", fontsize=ylabel_size)
        ax.set_xlabel("Mass balance (m w.e.)", fontsize=ylabel_size)

    # --- Shared legend ---
    fig.legend(
        handles=_gradient_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=3,
        fontsize=10,
        frameon=True,
    )
    plt.subplots_adjust(bottom=0.10)
    return fig


def _panel_label_generator():
    """a, b, ..., z, aa, ab, ..., az, ba, ... (never runs out)."""
    n = 1
    while True:
        for combo in itertools.product(string.ascii_lowercase, repeat=n):
            yield "".join(combo)
        n += 1


def plot_gradients_grid(
    glacier_names,
    all_glacier_config,
    pred_base,
    ncols=4,
    period="annual",
    ft_rep=0,
    figsize_col=6,
    figsize_row=6,
    cm=1 / 2.54,
    title_size=9,
    label_size=9,
    tick_size=8,
    ylabel_size=8,
    legend_height_cm=2.2,
):
    """
    Fig. A11–A16: ZS vs FT elevation gradients (with stake means), one glacier per
    panel, ncols per row. Glaciers without a config, a fine-tuned model or
    predictions on disk are left out.
    """
    # resolve and load first, so only glaciers with something to plot get a panel
    entries = []
    for gl in glacier_names:
        try:
            glacier_name, gcfg = lookup_glacier_config(gl, all_glacier_config)
        except KeyError as e:
            print(f"  [skip] {gl}: {e}")
            continue

        rgi_id = gcfg["rgi_ids"][0]
        zs_path, ft_path = zs_ft_paths(gcfg, pred_base, rep=ft_rep)
        if ft_path is None:
            print(f"  [skip] {glacier_name}: no fine-tuned model for this region")
            continue

        df_zs = aggregate_gridded_transformer_by_glacier(rgi_id, zs_path, period=period)
        df_ft = aggregate_gridded_transformer_by_glacier(rgi_id, ft_path, period=period)
        if df_zs.empty and df_ft.empty:
            print(f"  [skip] {glacier_name}: no predictions found")
            continue

        entries.append(
            {
                "glacier_name": glacier_name,
                "df_stakes": gcfg["df_stakes"],
                "df_zs": df_zs,
                "df_ft": df_ft,
            }
        )

    if not entries:
        raise RuntimeError("No glaciers with predictions found to plot.")

    nrows = math.ceil(len(entries) / ncols)
    fig_w = ncols * figsize_col * cm
    fig_h = nrows * figsize_row * cm + legend_height_cm * cm

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_w, fig_h), dpi=300, squeeze=False
    )

    label_iter = _panel_label_generator()
    for idx, entry in enumerate(entries):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        glacier_name = entry["glacier_name"]

        _plot_elevation_gradient(
            ax, entry["df_zs"], entry["df_ft"], entry["df_stakes"], glacier_name, period
        )

        ax.axvline(0, color="k", lw=0.6, ls="--", alpha=0.5)
        ax.set_title(glacier_name.capitalize(), fontsize=title_size, pad=2)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=tick_size, pad=2)
        ax.text(
            0.04,
            0.98,
            f"({next(label_iter)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=label_size,
        )
        ax.set_ylabel("Elevation (m a.s.l.)" if c == 0 else "", fontsize=ylabel_size)
        ax.set_xlabel("Mass balance (m w.e.)", fontsize=ylabel_size)

    for j in range(len(entries), nrows * ncols):
        axes[j // ncols, j % ncols].set_axis_off()

    bottom_frac = (legend_height_cm * cm) / fig_h
    fig.tight_layout(rect=[0, bottom_frac, 1, 1])
    fig.legend(
        handles=_gradient_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        fontsize=10,
        frameon=True,
    )
    return fig
