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

from regions.Switzerland.scripts.config_CH import *
from regions.Switzerland.scripts.plotting.palettes import _default_style

from massbalancemachine.plots.style import COLOR_ANNUAL, COLOR_WINTER


def plot_glamos_by_elevation_periods(
    df_all_a,
    df_all_w,
    ax=None,
    color_annual=COLOR_ANNUAL,
    color_winter=COLOR_WINTER,
    lw=1.3,
    mean_linestyle=":",
    label_prefix="GLAMOS",
    show_band=False,
    band_alpha=0.25,
):
    """
    Plot GLAMOS mass balance profiles by elevation for annual and winter periods.

    The function aggregates GLAMOS values by elevation band and year, then
    summarizes across years to produce a mean profile and (optionally) a
    min–max envelope. Annual and winter curves are plotted on the same axis.

    Parameters
    ----------
    df_all_a : pandas.DataFrame
        Data for the annual period. Must include columns:
        ['SOURCE', 'altitude_interval', 'YEAR', 'pred'].
    df_all_w : pandas.DataFrame
        Data for the winter period. Must include columns:
        ['SOURCE', 'altitude_interval', 'YEAR', 'pred'].
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If None, a new figure/axis is created.
    color_annual : str or tuple, optional
        Line/band color for the annual profile.
    color_winter : str or tuple, optional
        Line/band color for the winter profile.
    lw : float, optional
        Line width for the mean profile.
    mean_linestyle : str, optional
        Linestyle used for the mean profile.
    label_prefix : str, optional
        Prefix used in legend labels (default: 'GLAMOS').
    show_band : bool, optional
        If True, draw a min–max envelope across years for each elevation band.
    band_alpha : float, optional
        Transparency for the min–max envelope.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 6))

    def _aggregate_source(df, source_name):
        if df is None or df.empty:
            return pd.DataFrame(
                columns=["altitude_interval", "mean_Ba", "min_Ba", "max_Ba"]
            )
        df = df[df["SOURCE"].str.upper() == source_name.upper()]
        if df.empty:
            return pd.DataFrame(
                columns=["altitude_interval", "mean_Ba", "min_Ba", "max_Ba"]
            )
        per_year = (
            df.groupby(["SOURCE", "altitude_interval", "YEAR"])["pred"]
            .mean()
            .reset_index()
        )
        agg = (
            per_year.groupby(["SOURCE", "altitude_interval"])["pred"]
            .agg(mean_Ba="mean", min_Ba="min", max_Ba="max")
            .reset_index()
            .drop(columns=["SOURCE"])
        )
        return agg

    agg_glam_a = _aggregate_source(df_all_a, "GLAMOS")
    agg_glam_w = _aggregate_source(df_all_w, "GLAMOS")

    # --- annual ---
    if not agg_glam_a.empty:
        if show_band:
            ax.fill_betweenx(
                agg_glam_a["altitude_interval"],
                agg_glam_a["min_Ba"],
                agg_glam_a["max_Ba"],
                color=color_annual,
                alpha=band_alpha,
                label=f"{label_prefix} band (annual)",
            )
        ax.plot(
            agg_glam_a["mean_Ba"],
            agg_glam_a["altitude_interval"],
            color=color_annual,
            linestyle=mean_linestyle,
            linewidth=lw,
            label=f"{label_prefix} mean (annual)",
        )

    # --- winter ---
    if not agg_glam_w.empty:
        if show_band:
            ax.fill_betweenx(
                agg_glam_w["altitude_interval"],
                agg_glam_w["min_Ba"],
                agg_glam_w["max_Ba"],
                color=color_winter,
                alpha=band_alpha,
                label=f"{label_prefix} band (winter)",
            )
        ax.plot(
            agg_glam_w["mean_Ba"],
            agg_glam_w["altitude_interval"],
            color=color_winter,
            linestyle=mean_linestyle,
            linewidth=lw,
            label=f"{label_prefix} mean (winter)",
        )

    return ax


def plot_lstm_by_elevation_periods(
    df_all_a,
    df_all_w,
    ax=None,
    color_annual=COLOR_ANNUAL,
    color_winter=COLOR_WINTER,
    band_alpha=0.25,
    lw=1.2,
    mean_linestyle="-",
    label_prefix="LSTM",
    show_band=True,
):
    """
    Plot LSTM mass balance profiles by elevation for annual and winter periods.

    The function aggregates LSTM predictions by elevation band and year, then
    summarizes across years to produce a mean profile and (optionally) a
    min–max envelope. Annual and winter curves are plotted on the same axis.

    Parameters
    ----------
    df_all_a : pandas.DataFrame
        Data for the annual period. Must include columns:
        ['SOURCE', 'altitude_interval', 'YEAR', 'pred'].
    df_all_w : pandas.DataFrame
        Data for the winter period. Must include columns:
        ['SOURCE', 'altitude_interval', 'YEAR', 'pred'].
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If None, a new figure/axis is created.
    color_annual : str or tuple, optional
        Line/band color for the annual profile.
    color_winter : str or tuple, optional
        Line/band color for the winter profile.
    band_alpha : float, optional
        Transparency for the min–max envelope.
    lw : float, optional
        Line width for the mean profile.
    mean_linestyle : str, optional
        Linestyle used for the mean profile.
    label_prefix : str, optional
        Prefix used in legend labels (default: 'LSTM').
    show_band : bool, optional
        If True, draw a min–max envelope across years for each elevation band.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 6))

    def _aggregate_source(df, source_name):
        if df is None or df.empty:
            return pd.DataFrame(
                columns=["altitude_interval", "mean_Ba", "min_Ba", "max_Ba"]
            )
        df = df[df["SOURCE"].str.upper() == source_name.upper()]
        if df.empty:
            return pd.DataFrame(
                columns=["altitude_interval", "mean_Ba", "min_Ba", "max_Ba"]
            )
        per_year = (
            df.groupby(["SOURCE", "altitude_interval", "YEAR"])["pred"]
            .mean()
            .reset_index()
        )
        agg = (
            per_year.groupby(["SOURCE", "altitude_interval"])["pred"]
            .agg(mean_Ba="mean", min_Ba="min", max_Ba="max")
            .reset_index()
            .drop(columns=["SOURCE"])
        )
        return agg

    agg_lstm_a = _aggregate_source(df_all_a, "LSTM")
    agg_lstm_w = _aggregate_source(df_all_w, "LSTM")

    # --- annual ---
    if not agg_lstm_a.empty:
        if show_band:
            ax.fill_betweenx(
                agg_lstm_a["altitude_interval"],
                agg_lstm_a["min_Ba"],
                agg_lstm_a["max_Ba"],
                color=color_annual,
                alpha=band_alpha,
                label=f"{label_prefix} band (annual)",
            )
        ax.plot(
            agg_lstm_a["mean_Ba"],
            agg_lstm_a["altitude_interval"],
            color=color_annual,
            linestyle=mean_linestyle,
            linewidth=lw,
            label=f"{label_prefix} mean (annual)",
        )

    # --- winter ---
    if not agg_lstm_w.empty:
        if show_band:
            ax.fill_betweenx(
                agg_lstm_w["altitude_interval"],
                agg_lstm_w["min_Ba"],
                agg_lstm_w["max_Ba"],
                color=color_winter,
                alpha=band_alpha,
                label=f"{label_prefix} band (winter)",
            )
        ax.plot(
            agg_lstm_w["mean_Ba"],
            agg_lstm_w["altitude_interval"],
            color=color_winter,
            linestyle=mean_linestyle,
            linewidth=lw,
            label=f"{label_prefix} mean (winter)",
        )

    return ax


def plot_stakes_by_elevation_periods(
    df_stakes,
    glacier_name,
    valid_bins=None,
    ax=None,
    color_annual="#1f77b4",
    color_winter="#ff7f0e",
    marker_size=14,
):
    """
    Plot observed stake mass-balance means by elevation for annual and winter periods.

    Stake measurements are grouped into 100 m elevation bins, averaged across
    all available years, and plotted as elevation profiles. Annual and winter
    observations are shown using different marker styles and colors.

    Parameters
    ----------
    df_stakes : pandas.DataFrame
        Stake observations with required columns:
        ['GLACIER', 'PERIOD', 'YEAR', 'POINT_ELEVATION', 'POINT_BALANCE'].
        PERIOD must contain values compatible with 'annual' and 'winter'.
    glacier_name : str
        Name of the glacier to plot (used to filter `df_stakes`).
    valid_bins : set or array-like, optional
        If provided, only elevation bins present in `valid_bins` are plotted.
        This is useful for restricting stake observations to bins covered by
        model output (e.g. LSTM or GLAMOS).
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If None, a new figure and axis are created.
    color_annual : str or tuple, optional
        Edge color used for annual stake markers.
    color_winter : str or tuple, optional
        Edge color used for winter stake markers.
    marker_size : float, optional
        Marker size used for stake points.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the plotted stake elevation profiles.

    Notes
    -----
    - Elevation bins are defined as 100 m intervals; each stake is assigned to
      the center of its corresponding bin.
    - Markers are hollow to visually distinguish point observations from
      model-derived curves or envelopes.
    - No legend is created here; legends are typically handled by the caller
      to allow combined plots with models and references.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 6))

    style = _default_style(color_annual, color_winter)

    def bin_center_100m(z):
        left = np.floor(z / 100.0) * 100.0
        return left + 50.0

    stakes = df_stakes[df_stakes["GLACIER"] == glacier_name].copy()
    if stakes.empty:
        return ax

    stakes["altitude_interval"] = bin_center_100m(stakes["POINT_ELEVATION"])

    if valid_bins is not None:
        stakes = stakes[stakes["altitude_interval"].isin(valid_bins)]
        if stakes.empty:
            return ax

    for period_key, tag in (("annual", "annual"), ("winter", "winter")):
        ssub = stakes[stakes["PERIOD"].str.lower() == tag]
        if ssub.empty:
            continue
        sagg = (
            ssub.groupby("altitude_interval", as_index=False)["POINT_BALANCE"]
            .mean()
            .rename(columns={"POINT_BALANCE": "mean_obs_Ba"})
            .sort_values("altitude_interval")
        )
        ax.scatter(
            sagg["mean_obs_Ba"],
            sagg["altitude_interval"],
            s=marker_size,
            marker="o" if tag == "annual" else "s",
            facecolors="none",
            edgecolors=style[period_key]["color"],
            linewidths=0.9,
            label=f"Stakes mean ({tag})",
        )

    return ax


def plot_mb_by_elevation_periods_combined(
    df_all_a,
    df_all_w,
    df_stakes,
    glacier_name,
    ax=None,
    color_annual=COLOR_ANNUAL,
    color_winter=COLOR_WINTER,
):
    """
    Plot glacier mass balance versus elevation for annual and winter periods,
    combining model results (LSTM), reference data (GLAMOS), and stake observations.

    This function produces a single elevation-profile plot that overlays:
      - LSTM model results:
          * mean mass balance per elevation bin
          * min–max envelope across years (shaded band)
      - GLAMOS reference data:
          * mean mass balance per elevation bin (no envelope)
      - Stake observations:
          * mean observed balance per elevation bin, shown as markers

    Annual and winter periods are plotted simultaneously using distinct colors.
    Stake observations are restricted to elevation bins that are present in the
    model/reference outputs to ensure visual comparability.

    Parameters
    ----------
    df_all_a : pandas.DataFrame
        Aggregated annual-period data containing at least the columns:
        ``['SOURCE', 'altitude_interval', 'YEAR', 'PERIOD', 'pred']``.
        Must include rows for ``SOURCE == 'LSTM'`` and/or ``'GLAMOS'``.
    df_all_w : pandas.DataFrame
        Same as ``df_all_a`` but for the winter period.
    df_stakes : pandas.DataFrame
        Stake observation data with columns:
        ``['GLACIER', 'PERIOD', 'YEAR', 'POINT_ELEVATION', 'POINT_BALANCE']``.
    glacier_name : str
        Name of the glacier to select stake observations.
    ax : matplotlib.axes.Axes, optional
        Axes to plot into. If ``None``, a new figure and axes are created.
    color_annual : str or tuple, optional
        Color used for annual-period curves and markers.
    color_winter : str or tuple, optional
        Color used for winter-period curves and markers.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the combined elevation–mass balance plot.

    Notes
    -----
    - This function is a high-level wrapper that delegates plotting to:
      ``plot_lstm_by_elevation_periods``,
      ``plot_glamos_by_elevation_periods``,
      and ``plot_stakes_by_elevation_periods``.
    - The resulting plot is suitable for direct comparison between modeled,
      reference, and observed mass balances along the glacier elevation profile.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 6))

    # Draw LSTM + GLAMOS
    plot_lstm_by_elevation_periods(
        df_all_a, df_all_w, ax=ax, color_annual=color_annual, color_winter=color_winter
    )
    plot_glamos_by_elevation_periods(
        df_all_a, df_all_w, ax=ax, color_annual=color_annual, color_winter=color_winter
    )

    # Compute valid bins from whatever was plotted (union of bins present)
    bins = set()
    for df in (df_all_a, df_all_w):
        if df is None or df.empty:
            continue
        if "altitude_interval" in df.columns:
            bins.update(pd.unique(df["altitude_interval"].dropna()))

    plot_stakes_by_elevation_periods(
        df_stakes,
        glacier_name,
        valid_bins=bins,
        ax=ax,
        color_annual=color_annual,
        color_winter=color_winter,
    )

    # cosmetics (same as before)
    ax.set_ylabel("Elevation bin (m a.s.l.)")
    ax.set_xlabel("Mass balance (m w.e.)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="best", fontsize=8)

    return ax
