import matplotlib.pyplot as plt
import seaborn as sns


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
        overwrite_legend = {"pearson_corr": "\\rho"}
        legend = []
        for k, v in scores.items():
            leg = overwrite_legend.get(k, k.upper())
            if modelName is None:
                legend.append(r"$\mathrm{%s}=%.3f $" % (leg, v))
            else:
                legend.append(r"$\mathrm{%s_{%s}}=%.3f $" % (leg, modelName, v))

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
