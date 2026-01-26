import matplotlib.pyplot as plt
import seaborn as sns
import math

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
            roundedVal = f"{v:.{precLegend}f}"
            if modelName is None:
                legend.append(r"$\mathrm{%s}=%s$" % (leg, roundedVal))
            else:
                legend.append(r"$\mathrm{%s_{%s}}=%s$" % (leg, modelName, roundedVal))

        ax.text(
            0.03,
            0.98,
            ", ".join(legend),
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=20,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )
        ax.legend([], [], frameon=False)
    elif hue is not None:
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

    for i, test_gl in enumerate(custom_order):
        df_gl = grouped_ids[grouped_ids[order_key] == test_gl]

        ax = axs.flatten()[i]

        sns.scatterplot(
            df_gl,
            x="target",
            y="pred",
            ax=ax,
            hue_order=["annual", "winter"],
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
                            r"$\mathrm{%s_%s}=%s$" % (leg, k2, roundedVal)
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
