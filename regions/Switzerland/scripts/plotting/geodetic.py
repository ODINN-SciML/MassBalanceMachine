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
    Compare MBM and geodetic glacier-wide mass balance across glacier area bins.

    This function creates a multi-panel scatter plot where each panel corresponds
    to one glacier area class (defined by ``bins``/``labels``). Within each panel,
    points show modelled MB (MBM) versus observed geodetic MB. If available, x-error
    bars represent geodetic uncertainty (sigma).

    Panels share x/y limits and include a 1:1 reference line and zero lines. Optionally,
    each panel is annotated with RMSE (MBM - Geodetic) and Pearson correlation.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing at least the following columns:
        - ``'Area'`` : float, glacier area (km²) used for binning
        - ``'Geodetic MB'`` : float, observed geodetic mass balance (m w.e.)
        - ``'MBM MB'`` : float, modelled mass balance from MBM (m w.e.)
        Optionally:
        - ``geodetic_sigma_col`` (default ``'Geodetic MB sigma'``): float, 1-sigma
          uncertainty on geodetic MB used for x-error bars
        - ``'GLACIER'`` : str, used to color/style points within each panel
    bins : list of float, optional
        Bin edges for glacier area classes (km²). Default bins create five classes.
    labels : list of str, optional
        Labels for the area bins. Must be one shorter than ``bins``.
    max_bins : int, optional
        Maximum number of area-bin panels to plot (starting from the smallest bins).
    figsize : tuple, optional
        Figure size passed to ``plt.subplots``.
    annotate_rmse : bool, optional
        If True, annotate each panel with RMSE and Pearson r when enough points exist.
    rmse_box : bool, optional
        If True, draw a semi-transparent text box behind the RMSE annotation.
    geodetic_sigma_col : str, optional
        Column name for geodetic uncertainties (used for x-error bars). If missing,
        the function falls back to plotting without uncertainties.
    errorbar_alpha : float, optional
        Alpha value for the x-error bars.
    errorbar_elinewidth : float, optional
        Line width for the x-error bars.
    errorbar_capsize : float, optional
        Capsize for the x-error bars.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure containing the area-bin panels.

    Raises
    ------
    ValueError
        If no rows fall into the specified area bins.

    Notes
    -----
    - Axis limits are computed globally across the plotted bins and expanded to
      include geodetic uncertainties so that error bars are not clipped.
    - If ``'GLACIER'`` is present, a per-panel legend is placed below the subplot.
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
    """
    Plot a per-glacier comparison of MBM and GLAMOS mass balance against geodetic MB.

    This helper overlays two scatter series on a provided axis:
      - MBM MB vs geodetic MB
      - GLAMOS MB vs geodetic MB

    It also adds a 1:1 reference line, horizontal/vertical zero lines, a grid,
    and axis labels/title.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw into.
    df : pandas.DataFrame
        Input data containing the following columns:
        - ``'Geodetic MB'`` : float, observed geodetic mass balance (m w.e.)
        - ``'MBM MB'`` : float, MBM modelled mass balance (m w.e.)
        - ``'GLAMOS MB'`` : float, GLAMOS mass balance (m w.e.)
    glacier_name : str
        Glacier name used in the plot title.
    color_mbm : str or tuple
        Color used for the MBM scatter series.
    color_glamos : str or tuple
        Color used for the GLAMOS scatter series.
    title_suffix : str, optional
        Extra string appended to the title (e.g., period or time span).

    Returns
    -------
    None
        Modifies the provided axis in place.

    Notes
    -----
    - This function always creates a legend and uses fixed marker styles:
      circles for MBM and squares for GLAMOS.
    """
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
