import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import numpy as np
import math

import metrics

overwrite_legend = {"pearson_corr": "\\rho", "R2": "R^2", "r2": "R^2"}


def predVSTruth(
    grouped_ids,
    ax=None,
    scores={},
    title="",
    modelName=None,
    hue=None,
    xlabel="Observed PMB [m w.e.]",
    ylabel="Predicted PMB [m w.e.]",
    ax_xlim=None,
    ax_ylim=None,
    precLegend=3,
    **kwargs,
):
    """
    Plots predicted vs. observed values for point mass balance (PMB) using seaborn scatterplot,
    with options to add score annotations, custom axis labels/limits, and coloring.

    Parameters
    ----------
    grouped_ids : pandas.DataFrame
        DataFrame containing at least 'target' (observed values) and 'pred' (predicted values) columns.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto; if None, a new figure and axes are created.
    scores : dict, optional
        Dictionary of score names (str) and values (float) to display in the plot legend (e.g., {'mae': 0.1, 'rmse': 0.2}).
    title : str, optional
        Title for the plot.
    modelName : str, optional
        Model name to display in the legend (e.g., 'nn'); if None, no model name is shown.
    hue : str or None, optional
        Column name in grouped_ids used for coloring points by group.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    ax_xlim : tuple or None, optional
        x-axis limits as (min, max).
    ax_ylim : tuple or None, optional
        y-axis limits as (min, max).
    precLegend : int, optional
        Number of decimal places for rounding score values (default: 3).
    **kwargs
        Additional keyword arguments passed to seaborn.scatterplot.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if a new one was created, otherwise None.
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    else:
        fig = None

    marker_nn = "o"
    sns.scatterplot(grouped_ids, x="target", y="pred", ax=ax, hue=hue, **kwargs)

    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)

    if scores is not None:
        legend = []
        for k, v in scores.items():
            leg = overwrite_legend.get(k, k.upper())
            if isinstance(v, dict):
                line_legend = []
                for k2, v2 in v.items():
                    roundedVal = f"{v2:.{precLegend}f}"
                    if modelName is None:
                        line_legend.append(
                            r"$\mathrm{%s_{%s}}=%s$" % (leg, k2, roundedVal)
                        )
                    else:
                        line_legend.append(
                            r"$\mathrm{%s_{%s %s}}=%s$"
                            % (leg, modelName, k2, roundedVal)
                        )
                line_legend = ", ".join(line_legend)
                legend.append(line_legend)
            else:
                roundedVal = f"{v:.{precLegend}f}"
                if modelName is None:
                    legend.append(r"$\mathrm{%s}=%s$" % (leg, roundedVal))
                else:
                    legend.append(
                        r"$\mathrm{%s_{%s}}=%s$" % (leg, modelName, roundedVal)
                    )

        ax.text(
            0.03,
            0.98,
            "\n".join(legend),
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=20,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )
    if hue is not None:
        ax.legend(fontsize=20, loc="lower right", ncol=2)
    else:
        ax.legend([], [], frameon=False)

    # Diagonal line
    pt = (0, 0)
    ax.axline(pt, slope=1, color="grey", linestyle="-", linewidth=0.2)
    ax.axvline(0, color="grey", linestyle="-", linewidth=1)
    ax.axhline(0, color="grey", linestyle="-", linewidth=1)

    ax.grid()
    ax.set_title(title, fontsize=20)

    # Set axes limits
    if ax_xlim is not None:
        ax.set_xlim(ax_xlim)
    if ax_ylim is not None:
        ax.set_ylim(ax_ylim)

    plt.tight_layout()

    return fig  # To log figure during training


def predVSTruthPerGlacier(
    grouped_ids,
    axs=None,
    scores={},
    titles={},
    custom_order=None,
    xlabel="Observed PMB [m w.e.]",
    ylabel="Predicted PMB [m w.e.]",
    ax_xlim=None,
    ax_ylim=None,
    precLegend=3,
    **kwargs,
):
    """
    Plots predicted vs. observed values of point mass balance (PMB) for each glacier in separate subplots.

    This function generates scatter plots for each glacier present in `grouped_ids`,
    displaying predicted versus observed PMB values. The function supports custom subplot axes,
    control over axis labels and limits, per-glacier score annotations, and additional styling via kwargs.

    Parameters
    ----------
    grouped_ids : pandas.DataFrame
        DataFrame containing at least 'target' (observed), 'pred' (predicted), and a glacier identifier column ('GLACIER' or 'RGIId').
    axs : numpy.ndarray or list of matplotlib.axes.Axes, optional
        Array or list of Axes objects to draw the plots onto; should be at least as long as the number of glaciers.
    titles: dict, optional
        Dictionary mapping glacier names or RGI IDs to the title to use in each of the subplots (default: empty dict).
    scores : dict, optional
        Dictionary mapping glacier names or RGI IDs to their respective score dictionaries or floats for annotation (default: empty dict).
        For each glacier:
            - If the score is a float (e.g., 0.21), it is displayed as a single annotation, e.g. "$\\mathrm{RMSE}=0.21$".
            - If the score is a dict (e.g., {'annual': 0.2, 'winter': 0.3}), each key-value pair is annotated as
              "$\\mathrm{R^2_annual}=0.20, \\mathrm{R^2_winter}=0.30$".
    custom_order : list or None, optional
        Custom order of glacier names or RGI IDs for plotting. If None, uses the the 'GLACIER' column if it exists in the dataframe, and 'RGIId' otherwise.
    xlabel : str, optional
        Label for the x-axis (default: "Observed PMB [m w.e.]").
    ylabel : str, optional
        Label for the y-axis (default: "Predicted PMB [m w.e.]").
    ax_xlim : tuple or None, optional
        x-axis limits as (min, max). If None, calculated from the data for each subplot.
    ax_ylim : tuple or None, optional
        y-axis limits as (min, max). If None, calculated from the data for each subplot.
    precLegend : int, optional
        Number of decimal places for rounding score values (default: 3).
    **kwargs
        Additional keyword arguments passed to seaborn.scatterplot.

    Returns
    -------
    None
        The function modifies the provided axes in-place and does not return any value.

    Notes
    -----
    - For each glacier, draws a scatter plot of predicted vs. observed PMB, a 1:1 reference line,
      vertical and horizontal zero lines, and the provided scores as annotations.
    - If a score for a glacier is a float, it is shown as a single value in the annotation box.
    - If a score for a glacier is a dict, each sub-score is shown with its key as a subscript in the annotation box.
    - Ensures axis limits are consistent if not specified.
    """

    order_key = "GLACIER" if "GLACIER" in grouped_ids.keys() else "RGIId"
    custom_order = custom_order or grouped_ids[order_key].unique()

    if axs is None:
        N = len(custom_order)
        n = np.sqrt(N / 2.0)
        nRows = int(np.ceil(n))  # Scales as 2n
        nCols = int(np.floor(N / nRows))  # Scales as n
        if nCols * nRows < N:
            nCols += 1
        fig, axs = plt.subplots(
            nRows, nCols, figsize=(20 * nCols / 3, 30 * nRows / 8), sharex=False
        )
    else:
        fig = None

    for i, test_gl in enumerate(custom_order):
        df_gl = grouped_ids[grouped_ids[order_key] == test_gl]

        ax = axs.flatten()[i]

        sns.scatterplot(
            df_gl,
            x="target",
            y="pred",
            ax=ax,
            hue_order=["annual", "winter", "summer"],
            **kwargs,
        )

        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)

        # Diagonal line
        pt = (0, 0)
        ax.axline(pt, slope=1, color="grey", linestyle="-", linewidth=0.2)
        ax.axvline(0, color="grey", linestyle="-", linewidth=1)
        ax.axhline(0, color="grey", linestyle="-", linewidth=1)

        ax.grid()

        glacier_title = titles.get(test_gl) if titles is not None else None
        ax.set_title(glacier_title or test_gl.capitalize(), fontsize=28)

        # Set ylimits to be the same as xlimits
        if ax_xlim is None and ax_ylim is None:
            ymin = math.floor(min(df_gl.pred.min(), df_gl.target.min()))
            ymax = math.ceil(max(df_gl.pred.max(), df_gl.target.max()))
            ax.set_xlim(ymin, ymax)
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_xlim(ax_xlim)
            ax.set_ylim(ax_ylim)

        ax.legend(fontsize=18, loc="lower right", ncol=2)

        glacier_scores = scores.get(test_gl) if scores is not None else None
        if glacier_scores is not None:
            legend = []
            for k, v in glacier_scores.items():
                leg = overwrite_legend.get(k, k.upper())
                if isinstance(v, dict):
                    line_legend = []
                    for k2, v2 in v.items():
                        roundedVal = f"{v2:.{precLegend}f}"
                        line_legend.append(
                            r"$\mathrm{%s_{%s}}=%s$" % (leg, k2, roundedVal)
                        )
                    line_legend = ", ".join(line_legend)
                    legend.append(line_legend)
                else:
                    roundedVal = f"{v:.{precLegend}f}"
                    legend.append(r"$\mathrm{%s}=%s$" % (leg, roundedVal))

            ax.text(
                0.03,
                0.98,
                "\n".join(legend),
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=20,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
            )

    plt.tight_layout()

    return fig  # To log figure during training


def predVSTruthGlacierWide(
    geoTarget,
    geoPred,
    geoErr,
    ax=None,
    title="Glacier wide MB",
    ax_xlim=(-1.5, 0.5),
    ax_ylim=(-1.5, 1.0),
    color="orange",
):

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None

    for g in geoPred.keys():
        ax.errorbar(
            geoTarget[g], geoPred[g], xerr=2 * geoErr[g], label=g, fmt="o", color=color
        )
        plt.text(geoTarget[g] + 0.02, geoPred[g] + 0.02, g, fontsize=10)

    # Diagonal line
    pt = (0, 0)
    ax.axline(pt, slope=1, color="grey", linestyle="-", linewidth=0.3)
    ax.axvline(0, color="grey", linestyle="-", linewidth=1)
    ax.axhline(0, color="grey", linestyle="-", linewidth=1)

    ax.grid()
    ax.set_title(title, fontsize=20)

    # Set axes limits
    if ax_xlim is not None:
        ax.set_xlim(ax_xlim)
    if ax_ylim is not None:
        ax.set_ylim(ax_ylim)

    xlabel = "Observed mean SMB / year [m w.e.]"
    ylabel = "Predicted mean SMB / year [m w.e.]"

    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)

    plt.tight_layout()

    return fig


def plotMeanPred(
    grouped_ids,
    ax,
    color_pred="blue",
    color_obs="black",
    linestyle_pred="--",
    linestyle_obs="-",
):
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
    scores = metrics.scores(obs_mean, pred_mean)
    mae = scores["mae"]
    rmse = scores["rmse"]
    pearson_corr = scores["pearson_corr"]
    legend_text = "\n".join((rf"$\mathrm{{RMSE}}={rmse:.3f}$",))
    ax.text(0.03, 0.96, legend_text, transform=ax.transAxes, va="top", fontsize=20)

    ax.legend(fontsize=20, loc="lower right")


def predVSTruthTimeSeries(
    grouped_ids,
    scores_annual=None,
    scores_winter=None,
    scores_summer=None,
    color_annual="green",
    color_winter="blue",
    color_summer="red",
    modelName=None,
    xlabel="Observed PMB [m w.e.]",
    ylabel="Predicted PMB [m w.e.]",
    ax_xlim=None,
    ax_ylim=None,
    precLegend=3,
    **kwargs,
):
    """
    Plots predicted vs. observed values for point mass balance (PMB) along with time series over the year for each period.

    Parameters
    ----------
    grouped_ids : pandas.DataFrame
        DataFrame containing at least 'target' (observed values) and 'pred' (predicted values) columns.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto; if None, a new figure and axes are created.
    scores : dict, optional
        Dictionary of score names (str) and values (float) to display in the plot legend (e.g., {'mae': 0.1, 'rmse': 0.2}).
    title : str, optional
        Title for the plot.
    modelName : str, optional
        Model name to display in the legend (e.g., 'nn'); if None, no model name is shown.
    hue : str or None, optional
        Column name in grouped_ids used for coloring points by group.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    ax_xlim : tuple or None, optional
        x-axis limits as (min, max).
    ax_ylim : tuple or None, optional
        y-axis limits as (min, max).
    precLegend : int, optional
        Number of decimal places for rounding score values (default: 3).
    **kwargs
        Additional keyword arguments passed to seaborn.scatterplot.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if a new one was created, otherwise None.
    """

    hasAnnual = (
        scores_annual is not None
        and len(grouped_ids[grouped_ids.PERIOD == "annual"]) > 0
    )
    hasWinter = (
        scores_winter is not None
        and len(grouped_ids[grouped_ids.PERIOD == "winter"]) > 0
    )
    hasSummer = (
        scores_summer is not None
        and len(grouped_ids[grouped_ids.PERIOD == "summer"]) > 0
    )
    nRows = 0
    if hasAnnual:
        nRows += 1
    if hasWinter:
        nRows += 1
    if hasSummer:
        nRows += 1
    assert "At least scores for either annual, winter or summer must be provided."

    # Create figure
    fig = plt.figure(figsize=(20, 8))

    # Define grid: nRows x 2 columns
    gs = gridspec.GridSpec(nRows, 2, width_ratios=[1, 2])

    # Left plot spans both rows
    ax1 = fig.add_subplot(gs[:, 0])

    # Right column has multiple plots
    axAnnual = fig.add_subplot(gs[0, 1]) if hasAnnual else None
    axWinter = fig.add_subplot(gs[1 if hasAnnual else 0, 1]) if hasWinter else None
    axSummer = (
        fig.add_subplot(
            gs[
                2 if hasAnnual and hasWinter else (1 if hasAnnual != hasWinter else 0),
                1,
            ]
        )
        if hasSummer
        else None
    )

    keysScores = (
        scores_annual.keys()
        if scores_annual is not None
        else (
            scores_winter.keys() if scores_winter is not None else scores_summer.keys()
        )
    )
    scores_predVSTruth = {}
    for k in keysScores:
        scores_predVSTruth[k] = {}
        if hasAnnual:
            scores_predVSTruth[k]["annual"] = scores_annual[k]
        if hasWinter:
            scores_predVSTruth[k]["winter"] = scores_winter[k]
        if hasSummer:
            scores_predVSTruth[k]["summer"] = scores_summer[k]

    # Drop entries for which PERIOD is not included in ["annual", "winter", "summer"]
    grouped_ids_drop = grouped_ids[
        grouped_ids.PERIOD.isin(["annual", "winter", "summer"])
    ]

    predVSTruth(
        grouped_ids_drop,
        ax1,
        scores=scores_predVSTruth,
        hue="PERIOD",
        palette={
            "annual": color_annual,
            "winter": color_winter,
            "summer": color_summer,
        },
        ax_xlim=ax_xlim,
        ax_ylim=ax_ylim,
        xlabel=xlabel,
        ylabel=ylabel,
        precLegend=precLegend,
    )

    if hasAnnual:
        axAnnual.set_title("Mean yearly annual point mass balance", fontsize=24)
        grouped_ids_annual = grouped_ids[grouped_ids.PERIOD == "annual"].sort_values(
            by="YEAR"
        )
        plotMeanPred(
            grouped_ids_annual,
            axAnnual,
            linestyle_pred="-",
            linestyle_obs="--",
        )

    if hasWinter:
        axWinter.set_title("Mean yearly winter point mass balance", fontsize=24)
        grouped_ids_winter = grouped_ids[grouped_ids.PERIOD == "winter"].sort_values(
            by="YEAR"
        )
        plotMeanPred(
            grouped_ids_winter,
            axWinter,
            linestyle_pred="-",
            linestyle_obs="--",
        )
        # Remove legend from axWinter if it exists
        if axWinter.get_legend() is not None and hasAnnual:
            axWinter.get_legend().remove()

    if hasSummer:
        axSummer.set_title("Mean yearly summer point mass balance", fontsize=24)
        grouped_ids_summer = grouped_ids[grouped_ids.PERIOD == "summer"].sort_values(
            by="YEAR"
        )
        plotMeanPred(
            grouped_ids_summer,
            axSummer,
            linestyle_pred="-",
            linestyle_obs="--",
        )
        # Remove legend from axSummer if it exists
        if axSummer.get_legend() is not None and (hasAnnual or hasWinter):
            axSummer.get_legend().remove()

    plt.tight_layout()

    return fig  # To log figure during training
