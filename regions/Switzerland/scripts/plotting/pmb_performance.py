import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch
import numpy as np
import math

from regions.Switzerland.scripts.config_CH import *
from regions.Switzerland.scripts.plotting.style import alpha_labels

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
        hue="PERIOD",
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
    hue="GLACIER",
    palette=None,
    color=COLOR_ANNUAL,
    add_legend=True,
    ax_xlim=(-8, 6),
    ax_ylim=(-8, 6),
):
    """
    Scatter plot of predicted vs observed point mass balance with 1:1 reference.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    grouped_ids : pandas.DataFrame
        DataFrame with at least ['target', 'pred', 'PERIOD'] and optionally
        the column specified by `hue`.
    scores : dict
        Metrics to display in the annotation (expects keys like 'rmse',
        'pearson_corr' if `add_legend=True`).
    hue : str or None, optional
        Column used for coloring groups (e.g., 'GLACIER' or 'PERIOD').
        If None, no categorical legend is shown.
    palette : list or dict, optional
        Seaborn palette used when `hue` is provided.
    color : str, optional
        Fallback color when no palette/hue is used.
    add_legend : bool, optional
        If True, adds a metrics text box.
    ax_xlim, ax_ylim : tuple, optional
        Axis limits.

    Returns
    -------
    None
        Modifies the provided axis in place.
    """
    sns.scatterplot(
        grouped_ids,
        x="target",
        y="pred",
        palette=palette,
        hue=hue,
        ax=ax,
        color=color,
        style="PERIOD",
        markers={"annual": "o", "winter": "o"},
    )  # optional custom marker map)

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
    if hue is not None:
        ax.legend(fontsize=20, loc="lower right", ncol=2)
    else:
        ax.legend([], [], frameon=False)
    # diagonal line
    pt = (0, 0)
    ax.axline(pt, slope=1, color="grey", linestyle="-", linewidth=0.2)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.grid()
    leg = ax.get_legend()
    if leg is not None:
        for txt in leg.get_texts():
            t = txt.get_text().strip().lower()
            if t in ("annual", "winter"):
                txt.set_text(t.capitalize())
    # Set ylimits to be the same as xlimits
    ax.set_xlim(ax_xlim)
    ax.set_ylim(ax_ylim)
    plt.tight_layout()


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
            hue="PERIOD",
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
    subplot_labels=None,  # <— now optional
    custom_order=None,
    add_text=True,
    ax_xlim=(-9, 6),
    ax_ylim=(-9, 6),
    gl_area={},
):
    """
    Plot predicted vs observed PMB for multiple glaciers in a panel grid.

    Creates one scatter subplot per glacier (annual + winter), adds 1:1 line,
    optional per-glacier metric text, and a title including area and mean elevation.

    Parameters
    ----------
    grouped_ids : pandas.DataFrame
        DataFrame containing at least ['GLACIER', 'PERIOD', 'target', 'pred', 'gl_elv'].
    color_annual, color_winter : str
        Colors for annual and winter points.
    axs : numpy.ndarray of matplotlib.axes.Axes
        Array of axes (e.g., from plt.subplots) to populate.
    subplot_labels : list of str, optional
        Labels like ['(a)', '(b)', ...]. If None, labels are auto-generated.
    custom_order : list of str, optional
        Glacier plotting order. If None, uses unique values from grouped_ids['GLACIER'].
    add_text : bool, optional
        If True, annotates each subplot with per-period metrics when available.
    ax_xlim, ax_ylim : tuple or None, optional
        Axis limits. If None, limits are set from the glacier-specific data.
    gl_area : dict, optional
        Mapping from glacier name (lowercase) to area in km^2 for subplot titles.

    Returns
    -------
    numpy.ndarray
        Flattened array of axes used for plotting.
    """
    color_palette_period = [color_annual, color_winter]

    if custom_order is None:
        custom_order = grouped_ids["GLACIER"].unique()

    ax_flat = axs.flatten()
    n_plots = min(len(custom_order), len(ax_flat))

    # Auto-generate labels if none provided (a), (b), ...
    if subplot_labels is None:
        subplot_labels = alpha_labels(n_plots)
    else:
        # if provided shorter/longer, trim or extend deterministically
        if len(subplot_labels) < n_plots:
            subplot_labels = list(subplot_labels) + alpha_labels(
                n_plots - len(subplot_labels)
            )
        else:
            subplot_labels = list(subplot_labels)[:n_plots]

    for i, test_gl in enumerate(custom_order[:n_plots]):
        gl_elv = int(
            np.round(grouped_ids[grouped_ids.GLACIER == test_gl]["gl_elv"].values[0], 0)
        )
        df_gl = grouped_ids[grouped_ids.GLACIER == test_gl]

        ax1 = ax_flat[i]

        sns.scatterplot(
            df_gl,
            x="target",
            y="pred",
            palette=color_palette_period,
            hue="PERIOD",
            style="PERIOD",
            markers={"annual": "o", "winter": "o"},
            ax=ax1,
            hue_order=["annual", "winter"],
        )

        # diagonal and axes zero lines
        ax1.axline((0, 0), slope=1, color="grey", linestyle="-", linewidth=0.2)
        ax1.axvline(0, color="grey", linestyle="--", linewidth=1)
        ax1.axhline(0, color="grey", linestyle="--", linewidth=1)

        # Set symmetric limits or provided limits
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

        # remove legend (we’ll make a global one elsewhere if needed)
        leg = ax1.get_legend()
        if leg is not None:
            leg.remove()

        # Subplot label (auto or provided)
        ax1.text(
            0.02,
            0.98,
            subplot_labels[i],
            transform=ax1.transAxes,
            fontsize=24,
            va="top",
            ha="left",
        )

        # Metrics text
        legend_lines = []
        df_gl_annual = df_gl[df_gl["PERIOD"] == "annual"]
        if not df_gl_annual.empty:
            scores_annual = mbm.metrics.scores(
                df_gl_annual["target"], df_gl_annual["pred"]
            )
            legend_lines.append(
                rf"$\mathrm{{RMSE_a}}={scores_annual['rmse']:.2f},\ "
                rf"\mathrm{{R^2_a}}={scores_annual['r2']:.2f},\ "
                rf"\mathrm{{B_a}}={scores_annual['bias']:.2f}$"
            )

        df_gl_winter = df_gl[df_gl["PERIOD"] == "winter"]
        if not df_gl_winter.empty:
            scores_winter = mbm.metrics.scores(
                df_gl_winter["target"], df_gl_winter["pred"]
            )
            legend_lines.append(
                rf"$\mathrm{{RMSE_b}}={scores_winter['rmse']:.2f},\ "
                rf"\mathrm{{R^2_b}}={scores_winter['r2']:.2f},\ "
                rf"\mathrm{{B_b}}={scores_winter['bias']:.2f}$"
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

        area = gl_area.get(test_gl.lower(), np.nan)
        if area < 0.1:
            area = np.round(area, 3)
        else:
            area = np.round(area, 1)

        ax1.set_title(
            f"{test_gl.capitalize()} ({area} km$^2$, {gl_elv} m a.s.l.)", fontsize=20
        )

    return ax_flat
