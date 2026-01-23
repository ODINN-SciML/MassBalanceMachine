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

from regions.Switzerland.scripts.plotting.palettes import get_cmap_hex


def plot_mbm_vs_geodetic_by_area_bin(
    df,
    bins=[0, 1, 5, 10, 100, np.inf],
    labels=["<1", "1–5", "5–10", "10–100", ">100"],
    max_bins=4,
    figsize=(25, 10),
    annotate_rmse=True,
    rmse_box=True,
    geodetic_sigma_col="Geodetic MB sigma",  # <-- NEW
    errorbar_alpha=0.5,  # <-- NEW
    errorbar_elinewidth=1.0,  # <-- NEW
    errorbar_capsize=2.0,  # <-- NEW
):
    """
    Scatter MBM MB vs observed geodetic MB by area bin, with x-error bars showing
    GLAMOS geodetic MB uncertainties (sigma). Expects a column named
    `geodetic_sigma_col` (default: "Geodetic MB sigma").
    """
    subplot_labels = ["(a)", "(b)", "(c)", "(d)"]
    df = df.copy().replace([np.inf, -np.inf], np.nan)

    # Ensure sigma column exists; if missing, use zeros (no errorbars)
    if geodetic_sigma_col not in df.columns:
        print(
            f"Warning: '{geodetic_sigma_col}' not found. Plotting without uncertainties."
        )
        df[geodetic_sigma_col] = 0.0

    df["Area_bin"] = pd.cut(
        df["Area"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
        ordered=True,
    )
    categories = list(df["Area_bin"].cat.categories)
    bins_in_use = [b for b in categories if (df["Area_bin"] == b).any()]
    if not bins_in_use:
        raise ValueError("No data fall into the specified area bins.")

    n_plots = min(max_bins, len(bins_in_use))
    fig, axs = plt.subplots(1, n_plots, figsize=figsize, sharex=True, sharey=True)
    if n_plots == 1:
        axs = np.array([axs])

    # Global limits – include x-uncertainty so error bars aren’t clipped
    mask_bins = df["Area_bin"].isin(bins_in_use[:n_plots])
    all_x = df.loc[mask_bins, "Geodetic MB"]
    all_y = df.loc[mask_bins, "MBM MB"]
    all_sig = df.loc[mask_bins, geodetic_sigma_col].fillna(0.0)

    valid = ~(all_x.isna() | all_y.isna())
    if valid.any():
        xmin = float((all_x[valid] - all_sig[valid]).min())
        xmax = float((all_x[valid] + all_sig[valid]).max())
        ymin = float(all_y[valid].min())
        ymax = float(all_y[valid].max())
        pad = 0.25
        vmin, vmax = min(xmin, ymin) - pad, max(xmax, ymax) + pad
    else:
        vmin, vmax = -1.0, 1.0

    for i, area_bin in enumerate(bins_in_use[:n_plots]):
        ax = axs[i]
        df_bin = (
            df[df["Area_bin"] == area_bin]
            .dropna(subset=["Geodetic MB", "MBM MB"])
            .copy()
        )
        df_bin["GLACIER"] = df_bin.get("GLACIER", pd.Series(index=df_bin.index)).apply(
            lambda x: x.capitalize() if isinstance(x, str) else x
        )

        # Scatter
        hue_kw = {}
        if "GLACIER" in df_bin.columns:
            hue_kw = dict(hue="GLACIER", style="GLACIER")

        palette = sns.color_palette(
            get_cmap_hex(cmaps.ice, 1 + df_bin.get("GLACIER", pd.Series()).nunique())
        )

        sns.scatterplot(
            data=df_bin,
            x="Geodetic MB",
            y="MBM MB",
            alpha=0.9,
            ax=ax,
            s=250,
            palette=palette,
            **hue_kw,
        )

        # X-error bars for geodetic uncertainties
        if geodetic_sigma_col in df_bin.columns:
            xvals = df_bin["Geodetic MB"].to_numpy(dtype=float)
            yvals = df_bin["MBM MB"].to_numpy(dtype=float)
            xerr = df_bin[geodetic_sigma_col].fillna(0.0).to_numpy(dtype=float)

            # Draw errorbars without adding legend entries
            ax.errorbar(
                xvals,
                yvals,
                xerr=xerr,
                yerr=None,
                fmt="none",
                ecolor="k",
                alpha=errorbar_alpha,
                elinewidth=errorbar_elinewidth,
                capsize=errorbar_capsize,
                capthick=errorbar_elinewidth,
                zorder=0,  # behind points
            )

        # Axes decorations
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.axvline(0, color="grey", linestyle="--", linewidth=1)
        ax.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax.axline((0, 0), slope=1, color="grey", linestyle="--", linewidth=1)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)

        ax.text(
            0.02,
            1,
            subplot_labels[i],
            transform=ax.transAxes,
            fontsize=24,
            va="top",
            ha="left",
        )

        ax.set_xlabel("")  # global x-label handled below
        ax.set_ylabel("Modelled MB [m w.e.]", fontsize=20)
        ax.set_title(f"Area: {area_bin} km²", fontsize=24)

        # RMSE + r
        n = len(df_bin)
        if annotate_rmse and n > 1:
            resid = df_bin["MBM MB"].to_numpy() - df_bin["Geodetic MB"].to_numpy()
            rmse = float(np.sqrt(np.nanmean(resid**2)))
            r, _ = pearsonr(df_bin["Geodetic MB"], df_bin["MBM MB"])
            box = (
                dict(facecolor="white", alpha=0.7, edgecolor="none")
                if rmse_box
                else None
            )
            ax.text(
                0.93,
                0.02,
                f"RMSE = {rmse:.2f}, r = {r:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=20,
                bbox=box,
            )
        elif annotate_rmse:
            box = (
                dict(facecolor="white", alpha=0.7, edgecolor="none")
                if rmse_box
                else None
            )
            ax.text(
                0.93,
                0.02,
                "No data",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=20,
                bbox=box,
            )

        # Legend handling
        if "GLACIER" in df_bin.columns:
            handles, labels_ = ax.get_legend_handles_labels()
            ax.legend(
                handles,
                labels_,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),  # <-- closer to x-axis
                borderaxespad=0,
                fontsize=14,
                ncol=2,
                frameon=True,
            )
        else:
            if ax.legend_:
                ax.legend_.remove()

    fig.canvas.draw()  # need a renderer for legend extents
    plt.show()
    return fig


def plot_scatter_comparison(
    ax, df, glacier_name, color_mbm, color_glamos, title_suffix=""
):
    sns.scatterplot(
        df,
        x="Geodetic MB",
        y="MBM MB",
        color=color_mbm,
        alpha=0.7,
        label="MBM MB",
        marker="o",
        ax=ax,
    )
    sns.scatterplot(
        df,
        x="Geodetic MB",
        y="GLAMOS MB",
        color=color_glamos,
        alpha=0.7,
        label="GLAMOS MB",
        marker="s",
        ax=ax,
    )

    ax.axline((0, 0), slope=1, color="grey", linestyle="--", linewidth=1)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.grid(True, linestyle="--", linewidth=0.5)

    ax.set_xlabel("Geodetic MB [m w.e.]", fontsize=12)
    ax.set_ylabel("Modeled MB [m w.e.]", fontsize=12)
    ax.set_title(f"{glacier_name.capitalize()} Glacier {title_suffix}", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
