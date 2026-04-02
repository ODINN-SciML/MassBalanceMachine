import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch
import numpy as np
import math

import massbalancemachine as mbm

from regions.TF_Europe.scripts.config_TF_Europe import *
from regions.TF_Europe.scripts.plotting.style import alpha_labels

from massbalancemachine.plots.style import COLOR_ANNUAL, COLOR_WINTER


def plot_predictions_summary(
    grouped_ids,
    scores_annual,
    scores_winter,
    ax_xlim=(-8, 6),
    ax_ylim=(-8, 6),
    color_annual=COLOR_ANNUAL,
    color_winter=COLOR_WINTER,
):
    """
    Create a summary figure showing predicted vs observed PMB and mean time series.

    The figure includes:
      - Predicted vs observed scatter (annual + winter),
      - Mean annual PMB time series (predicted vs observed),
      - Mean winter PMB time series (predicted vs observed).

    Parameters
    ----------
    grouped_ids : pandas.DataFrame
        Table containing predictions and targets with at least columns
        ['target', 'pred', 'PERIOD', 'YEAR'].
    scores_annual : dict
        Annual performance metrics (e.g., keys like 'rmse', 'R2', 'Bias').
    scores_winter : dict
        Winter performance metrics (e.g., keys like 'rmse', 'R2', 'Bias').
    ax_xlim, ax_ylim : tuple, optional
        Axis limits for the predicted-vs-observed panel.
    color_annual, color_winter : str, optional
        Colors used for annual and winter points in the scatter plot.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure (useful for saving or further customization).
    """
    subplot_labels = ["(a)", "(b)", "(c)"]
    # Create figure
    fig = plt.figure(figsize=(20, 8))

    # Define grid: 2 rows x 2 columns
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2])

    # Left plot spans both rows
    ax1 = fig.add_subplot(gs[:, 0])

    # Right column has two plots
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    # Left panel: Predictions vs Truth
    ax1.set_title("", fontsize=24)
    pred_vs_truth(
        ax1,
        grouped_ids,
        scores_annual,
        # hue="PERIOD",
        add_legend=False,
        palette=[color_annual, color_winter],
        ax_xlim=ax_xlim,
        ax_ylim=ax_ylim,
    )
    ax1.text(
        0.02,
        0.98,
        subplot_labels[0],
        transform=ax1.transAxes,
        fontsize=24,
        verticalalignment="top",
        horizontalalignment="left",
    )

    legend_NN = "\n".join(
        [
            r"$\mathrm{RMSE_a}=%.2f$, $\mathrm{RMSE_w}=%.2f$"
            % (scores_annual["rmse"], scores_winter["rmse"]),
            r"$\mathrm{R^2_a}=%.2f$, $\mathrm{R^2_w}=%.2f$"
            % (scores_annual["R2"], scores_winter["R2"]),
            r"$\mathrm{Bias_a}=%.2f$, $\mathrm{Bias_w}=%.2f$"
            % (scores_annual["Bias"], scores_winter["Bias"]),
        ]
    )
    ax1.text(
        0.25,
        0.98,
        legend_NN,
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=20,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    # Top-right: Mean annual PMB
    color_pred = "#762a83"
    color_obs = "black"
    ax2.set_title("Mean yearly annual point mass balance", fontsize=24)
    grouped_ids_xgb_annual = grouped_ids[grouped_ids.PERIOD == "annual"].sort_values(
        by="YEAR"
    )
    plot_mean_pred(
        grouped_ids_xgb_annual,
        ax2,
        color_pred=color_pred,
        color_obs=color_obs,
        linestyle_pred="-",
        linestyle_obs="--",
    )
    ax2.set_ylabel("PMB [m w.e.]", fontsize=20)
    ax2.text(
        0.01,
        0.98,
        subplot_labels[1],
        transform=ax2.transAxes,
        fontsize=24,
        verticalalignment="top",
        horizontalalignment="left",
    )

    # Bottom-right: Mean winter PMB
    ax3.set_title("Mean yearly winter point mass balance", fontsize=24)
    grouped_ids_xgb_winter = grouped_ids[grouped_ids.PERIOD == "winter"].sort_values(
        by="YEAR"
    )
    plot_mean_pred(
        grouped_ids_xgb_winter,
        ax3,
        color_pred=color_pred,
        color_obs=color_obs,
        linestyle_pred="-",
        linestyle_obs="--",
    )
    ax3.set_ylabel("PMB [m w.e.]", fontsize=20)
    ax3.text(
        0.01,
        0.98,
        subplot_labels[2],
        transform=ax3.transAxes,
        fontsize=24,
        verticalalignment="top",
        horizontalalignment="left",
    )
    # Remove legend from ax3 if it exists
    if ax3.get_legend() is not None:
        ax3.get_legend().remove()

    plt.tight_layout()
    return fig  # return figure in case further customization or saving is needed


def pred_vs_truth(
    ax,
    grouped_ids,
    scores,
    palette=None,
    add_legend=True,
    ax_xlim=(-8, 6),
    ax_ylim=(-8, 6),
):
    df = grouped_ids.copy()
    df["PERIOD"] = df["PERIOD"].astype(str).str.strip().str.lower()

    # Split explicitly
    df_annual = df[df["PERIOD"] == "annual"]
    df_winter = df[df["PERIOD"] == "winter"]

    # Decide colors if user passed palette=[annual_color, winter_color]
    annual_color = (
        palette[0] if (palette is not None and len(palette) >= 2) else COLOR_ANNUAL
    )
    winter_color = (
        palette[1] if (palette is not None and len(palette) >= 2) else COLOR_WINTER
    )

    # --- Draw annual first (bottom) ---
    sns.scatterplot(
        data=df_annual,
        x="target",
        y="pred",
        ax=ax,
        color=annual_color,
        edgecolor=None,
        legend=False,
        zorder=1,
        marker="o",
        alpha=0.8,
    )

    # --- Draw winter second (top) ---
    sns.scatterplot(
        data=df_winter,
        x="target",
        y="pred",
        ax=ax,
        color=winter_color,
        edgecolor=None,
        legend=False,
        zorder=2,
        marker="o",
        alpha=0.8,
    )

    ax.set_ylabel("Modeled PMB [m w.e.]", fontsize=20)
    ax.set_xlabel("Observed PMB [m w.e.]", fontsize=20)

    if add_legend:
        legend_xgb = "\n".join(
            (
                (r"$\mathrm{RMSE}=%.2f$," % (scores["rmse"],)),
                (r"$\mathrm{\rho}=%.2f$" % (scores["pearson_corr"],)),
            )
        )
        ax.text(
            0.03,
            0.98,
            legend_xgb,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=20,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

    # Legend that matches draw order
    handles = [
        Patch(color=annual_color, label="Annual"),
        Patch(color=winter_color, label="Winter"),
    ]
    ax.legend(handles=handles, fontsize=20, loc="lower right", ncol=2)

    # reference lines
    ax.axline((0, 0), slope=1, color="grey", linestyle="-", linewidth=0.2)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.grid()

    ax.set_xlim(ax_xlim)
    ax.set_ylim(ax_ylim)


import matplotlib.colors as mcolors


def pred_vs_truth_density(
    ax,
    grouped_ids,
    scores,
    palette=None,
    add_legend=True,
    ax_xlim=(-8, 6),
    ax_ylim=(-8, 6),
    s=100,
    clim=None,  # e.g. (0, 0.4) like your example; applied per season
    show_colorbar=False,
    colorbar_label="Point density",
):
    """
    Like plot_prediction(): points are colored by KDE density.
    But keeps Annual/Winter visually distinct by using a single-hue colormap
    derived from the season color, and calls the sub-plotter twice.
    """

    from scipy.stats import gaussian_kde

    df = grouped_ids.copy()
    df["PERIOD"] = df["PERIOD"].astype(str).str.strip().str.lower()

    df_annual = df[df["PERIOD"] == "annual"]
    df_winter = df[df["PERIOD"] == "winter"]

    annual_color = (
        palette[0] if (palette is not None and len(palette) >= 2) else COLOR_ANNUAL
    )
    winter_color = (
        palette[1] if (palette is not None and len(palette) >= 2) else COLOR_WINTER
    )

    def _cmap_from_color(color, min_light=0.6, max_dark=1.0):
        """
        Build a colormap from light -> full color (darker) for density shading.
        """
        rgb = np.array(mcolors.to_rgb(color))
        white = np.array([1.0, 1.0, 1.0])
        light = white * (1 - min_light) + rgb * min_light  # very light tint
        dark = rgb * max_dark  # full color
        return mcolors.LinearSegmentedColormap.from_list("", [light, dark])

    def _plot_season_density(sub, base_color, zorder):
        if len(sub) < 2:
            return None

        x = sub["target"].to_numpy()
        y = sub["pred"].to_numpy()

        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)

        # sort so dense points are drawn last
        idx = np.argsort(z)
        x, y, z = x[idx], y[idx], z[idx]

        cmap = _cmap_from_color(base_color)

        sc = ax.scatter(
            x,
            y,
            c=z,
            cmap=cmap,
            s=s,
            edgecolors="none",
            zorder=zorder,
        )

        if clim is not None:
            sc.set_clim(*clim)

        return sc

    # Draw annual then winter (winter on top)
    sc_a = _plot_season_density(
        df_annual,
        annual_color,
        zorder=2,
    )
    sc_w = _plot_season_density(
        df_winter,
        winter_color,
        zorder=3,
    )

    # Colorbar: show one (use winter if available, else annual)
    sc_for_cb = sc_w if sc_w is not None else sc_a
    if show_colorbar and sc_for_cb is not None:
        cb = plt.colorbar(sc_for_cb, ax=ax)
        cb.set_label(colorbar_label)

    # Labels
    ax.set_ylabel("Modeled PMB [m w.e.]", fontsize=20)
    ax.set_xlabel("Observed PMB [m w.e.]", fontsize=20)

    # Metrics box (same as your original style)
    if add_legend and scores is not None:
        legend_text = "\n".join(
            (
                rf"$\mathrm{{RMSE}}={scores['rmse']:.2f}$",
                rf"$\rho={scores['pearson_corr']:.2f}$",
            )
        )
        ax.text(
            0.03,
            0.98,
            legend_text,
            transform=ax.transAxes,
            va="top",
            fontsize=20,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
        )

    # Legend for seasons
    handles = [
        Patch(color=annual_color, label="Annual"),
        Patch(color=winter_color, label="Winter"),
    ]
    ax.legend(handles=handles, fontsize=20, loc="lower right", ncol=2)

    # Reference lines
    ax.axline((0, 0), slope=1, color="grey", linestyle="-", linewidth=0.4)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.grid(alpha=0.3)

    ax.set_xlim(ax_xlim)
    ax.set_ylim(ax_ylim)


def plot_mean_pred(
    grouped_ids,
    ax,
    color_pred=COLOR_ANNUAL,
    color_obs="orange",
    linestyle_pred="--",
    linestyle_obs="-",
):
    """
    Plot mean observed and predicted PMB time series with ±1 std envelopes.

    Parameters
    ----------
    grouped_ids : pandas.DataFrame
        DataFrame with columns ['YEAR', 'target', 'pred'].
    ax : matplotlib.axes.Axes
        Axis to draw on.
    color_pred, color_obs : str, optional
        Line/fill colors for predicted and observed series.
    linestyle_pred, linestyle_obs : str, optional
        Line styles for predicted and observed series.

    Returns
    -------
    None
        Modifies the provided axis in place.
    """
    # Aggregate once
    g = grouped_ids.groupby("YEAR")
    years = np.sort(g.size().index.values)

    obs_mean = g["target"].mean().reindex(years).values
    obs_std = g["target"].std().reindex(years).values

    pred_mean = g["pred"].mean().reindex(years).values
    pred_std = g["pred"].std().reindex(years).values

    # Observations
    ax.fill_between(
        years, obs_mean - obs_std, obs_mean + obs_std, color=color_obs, alpha=0.3
    )
    ax.plot(
        years,
        obs_mean,
        color=color_obs,
        label="observed",
        linestyle=linestyle_obs,
    )

    # Predictions
    ax.plot(
        years,
        pred_mean,
        color=color_pred,
        label="predicted",
        linestyle=linestyle_pred,
        marker="v",
    )
    ax.fill_between(
        years, pred_mean - pred_std, pred_mean + pred_std, color=color_pred, alpha=0.3
    )

    # Rotate x labels (safer than set_xticklabels)
    ax.tick_params(axis="x", rotation=45)

    # Metrics
    scores = mbm.metrics.scores(obs_mean, pred_mean)
    mae = scores["mae"]
    rmse = scores["rmse"]
    pearson_corr = scores["pearson_corr"]
    legend_text = "\n".join((rf"$\mathrm{{RMSE}}={rmse:.2f}$",))
    ax.text(0.055, 0.96, legend_text, transform=ax.transAxes, va="top", fontsize=20)

    ax.legend(fontsize=20, loc="lower right")


def plot_predictions_three_models_side_by_side(
    grouped_ids_lstm,
    grouped_ids_nn,
    grouped_ids_xgb,
    scores_annual_lstm,
    scores_winter_lstm,
    scores_annual_nn,
    scores_winter_nn,
    scores_annual_xgb,
    scores_winter_xgb,
    ax_xlim=(-8, 6),
    ax_ylim=(-8, 6),
    color_annual=COLOR_ANNUAL,
    color_winter=COLOR_WINTER,
):
    """
    Plot side-by-side prediction–observation comparisons for three models.

    The function creates a three-panel figure comparing predicted versus
    observed point mass balance (PMB) for LSTM, NN (MLP), and XGBoost models.
    Each panel shows annual and winter predictions, includes model-specific
    performance metrics, and shares common axes. A single legend indicating
    annual and winter periods is placed below the panels.

    Parameters
    ----------
    grouped_ids_lstm : pandas.DataFrame
        Prediction results for the LSTM model, including columns such as
        ['target', 'pred', 'PERIOD'].
    grouped_ids_nn : pandas.DataFrame
        Prediction results for the NN (MLP) model with the same structure
        as `grouped_ids_lstm`.
    grouped_ids_xgb : pandas.DataFrame
        Prediction results for the XGBoost model with the same structure
        as `grouped_ids_lstm`.
    scores_annual_lstm : dict
        Annual performance metrics for the LSTM model (e.g. rmse, R2, Bias).
    scores_winter_lstm : dict
        Winter performance metrics for the LSTM model.
    scores_annual_nn : dict
        Annual performance metrics for the NN (MLP) model.
    scores_winter_nn : dict
        Winter performance metrics for the NN (MLP) model.
    scores_annual_xgb : dict
        Annual performance metrics for the XGBoost model.
    scores_winter_xgb : dict
        Winter performance metrics for the XGBoost model.
    ax_xlim : tuple of float, optional
        Limits for the x-axis (observed PMB).
    ax_ylim : tuple of float, optional
        Limits for the y-axis (predicted PMB).
    color_annual : str or tuple, optional
        Color used for annual-period points.
    color_winter : str or tuple, optional
        Color used for winter-period points.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure containing the three comparison panels.
    """

    model_labels = ["LSTM", "MLP", "XGBoost"]
    subplot_labels = ["(a)", "(b)", "(c)"]
    grouped_inputs = [grouped_ids_lstm, grouped_ids_nn, grouped_ids_xgb]
    scores_annuals = [scores_annual_lstm, scores_annual_nn, scores_annual_xgb]
    scores_winters = [scores_winter_lstm, scores_winter_nn, scores_winter_xgb]

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.20)

    for ax, grouped, label, sl, scores_annual, scores_winter in zip(
        axes,
        grouped_inputs,
        model_labels,
        subplot_labels,
        scores_annuals,
        scores_winters,
    ):
        # Predictions vs Truth panel
        pred_vs_truth(
            ax,
            grouped,
            scores_annual,
            # hue="PERIOD",
            add_legend=False,
            palette=[color_annual, color_winter],
            ax_xlim=ax_xlim,
            ax_ylim=ax_ylim,
        )

        # Title
        ax.set_title(label, fontsize=20)

        # Subplot label (a, b, c)
        ax.text(
            0.03,
            0.97,
            sl,
            transform=ax.transAxes,
            fontsize=20,
            verticalalignment="top",
            horizontalalignment="left",
        )

        # Score box
        legend_str = "\n".join(
            [
                r"$\mathrm{RMSE_a}=%.2f$, $\mathrm{RMSE_w}=%.2f$"
                % (scores_annual["rmse"], scores_winter["rmse"]),
                r"$\mathrm{R^2_a}=%.2f$, $\mathrm{R^2_w}=%.2f$"
                % (scores_annual["R2"], scores_winter["R2"]),
                r"$\mathrm{B_a}=%.2f$, $\mathrm{B_w}=%.2f$"
                % (scores_annual["Bias"], scores_winter["Bias"]),
            ]
        )
        ax.text(
            0.98,  # near the right edge
            0.03,  # near the bottom
            legend_str,
            transform=ax.transAxes,
            fontsize=16,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.60),
        )

        # remove legend
        ax.legend().remove()

    # Common axis label
    axes[0].set_ylabel("Predicted PMB (m w.e.)", fontsize=18)

    # ---------- Shared legend BELOW subplots ----------
    handles = [
        Patch(color=COLOR_ANNUAL, label="Annual"),
        Patch(color=COLOR_WINTER, label="Winter"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=16,
        frameon=True,
        bbox_to_anchor=(0.5, -0.04),  # move down if needed
    )
    # --------------------------------------------------

    fig.tight_layout(rect=(0, 0.05, 1, 1))  # leave room at bottom for legend
    return fig


def plot_individual_glacier_pred(
    grouped_ids,
    color_annual,
    color_winter,
    axs,
    subplot_labels=None,
    custom_order=None,
    add_text=True,
    ax_xlim=(-9, 6),
    ax_ylim=(-9, 6),
    gl_area={},
):
    """
    Plot predicted vs observed PMB for multiple glaciers in a panel grid.

    Winter points are always drawn on top of annual points by plotting in two layers.
    """
    df_all = grouped_ids.copy()
    df_all["PERIOD"] = df_all["PERIOD"].astype(str).str.strip().str.lower()

    if custom_order is None:
        custom_order = df_all["GLACIER"].unique()

    ax_flat = axs.flatten()
    n_plots = min(len(custom_order), len(ax_flat))

    # Auto-generate labels if none provided (a), (b), ...
    if subplot_labels is None:
        subplot_labels = alpha_labels(n_plots)
    else:
        if len(subplot_labels) < n_plots:
            subplot_labels = list(subplot_labels) + alpha_labels(
                n_plots - len(subplot_labels)
            )
        else:
            subplot_labels = list(subplot_labels)[:n_plots]

    for i, test_gl in enumerate(custom_order[:n_plots]):
        df_gl = df_all[df_all.GLACIER == test_gl]
        ax1 = ax_flat[i]

        # --- Two-layer scatter: annual first (bottom), winter second (top) ---
        df_gl_annual = df_gl[df_gl["PERIOD"] == "annual"]
        df_gl_winter = df_gl[df_gl["PERIOD"] == "winter"]

        if not df_gl_annual.empty:
            sns.scatterplot(
                data=df_gl_annual,
                x="target",
                y="pred",
                ax=ax1,
                color=color_annual,
                marker="o",
                legend=False,
                zorder=1,
                alpha=0.8,
            )

        if not df_gl_winter.empty:
            sns.scatterplot(
                data=df_gl_winter,
                x="target",
                y="pred",
                ax=ax1,
                color=color_winter,
                marker="o",
                legend=False,
                zorder=2,
                alpha=0.8,
            )

        # diagonal and axes zero lines
        ax1.axline((0, 0), slope=1, color="grey", linestyle="-", linewidth=0.2)
        ax1.axvline(0, color="grey", linestyle="--", linewidth=1)
        ax1.axhline(0, color="grey", linestyle="--", linewidth=1)

        # Set limits
        if ax_xlim is None:
            ymin = math.floor(min(df_gl.pred.min(), df_gl.target.min()))
            ymax = math.ceil(max(df_gl.pred.max(), df_gl.target.max()))
            ax1.set_xlim(ymin, ymax)
            ax1.set_ylim(ymin, ymax)
        else:
            ax1.set_xlim(ax_xlim)
            ax1.set_ylim(ax_ylim)

        ax1.grid(alpha=0.2)
        ax1.tick_params(labelsize=18, pad=2)
        ax1.set_ylabel("")
        ax1.set_xlabel("")

        # Subplot label
        ax1.text(
            0.02,
            0.98,
            subplot_labels[i],
            transform=ax1.transAxes,
            fontsize=24,
            va="top",
            ha="left",
        )

        # Glacier elevation (safe)
        if "gl_elv" in df_gl.columns and len(df_gl["gl_elv"].dropna()) > 0:
            gl_elv = int(np.round(df_gl["gl_elv"].dropna().values[0], 0))
        else:
            gl_elv = np.nan

        # Metrics text
        legend_lines = []

        if not df_gl_annual.empty:
            scores_annual = mbm.metrics.scores(
                df_gl_annual["target"], df_gl_annual["pred"]
            )
            legend_lines.append(
                rf"$\mathrm{{RMSE_a}}={scores_annual['rmse']:.2f},\ "
                rf"\mathrm{{R^2_a}}={scores_annual['r2']:.2f},\ "
                rf"\mathrm{{B_a}}={scores_annual['bias']:.2f}$"
            )

        if not df_gl_winter.empty:
            scores_winter = mbm.metrics.scores(
                df_gl_winter["target"], df_gl_winter["pred"]
            )
            legend_lines.append(
                rf"$\mathrm{{RMSE_w}}={scores_winter['rmse']:.2f},\ "
                rf"\mathrm{{R^2_w}}={scores_winter['r2']:.2f},\ "
                rf"\mathrm{{B_w}}={scores_winter['bias']:.2f}$"
            )

        if add_text and legend_lines:
            ax1.text(
                0.98,
                0.02,
                "\n".join(legend_lines),
                transform=ax1.transAxes,
                fontsize=18,
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.0),
            )

        # Title with area
        area = gl_area.get(test_gl.lower(), np.nan)
        if np.isfinite(area):
            area_disp = np.round(area, 3) if area < 0.1 else np.round(area, 1)
        else:
            area_disp = np.nan

        ax1.set_title(
            f"{test_gl.capitalize()} ({area_disp} km$^2$, {gl_elv} m a.s.l.)",
            fontsize=20,
        )

    return ax_flat
