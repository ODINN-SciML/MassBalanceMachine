import matplotlib.pyplot as plt
import numpy as np


def scatterplot_mb(
    grouped_ids,
    ax=None,
    title="",
    hue=None,
    nbin=100,
    xlabel="SMB (m w.e. / y)",
    ylabel="Surface elevation (m)",
    ax_xlim=None,
    ax_ylim=None,
    test_data=None,
    vvmin=None,
    vvmax=None,
):
    """
    Plots training point mass balance (PMB) as fonction of surface elevation with density of point within hexagons.

    Parameters
    ----------
    grouped_ids : pandas.DataFrame
        DataFrame containing at least 'target' (observed values) and 'pred' (predicted values) columns for the train data.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto; if None, a new figure and axes are created.
    title : str, optional
        Title for the plot.
    hue : str or None, optional
        Column name in grouped_ids used for coloring points by group.
    nbin : int, optional
        Number of bins in the histogram. Determine size of the hexagones in the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    ax_xlim : tuple or None, optional
        x-axis limits as (min, max).
    ax_ylim : tuple or None, optional
        y-axis limits as (min, max).
    test_data : pandas.DataFrame
        DataFrame containing at least 'target' (observed values) and 'pred' (predicted values) columns for the test data.
    vvmin, vvmax : float, optional
        Lower and upper limit of the colormap

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if a new one was created, otherwise None.
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    else:
        fig = None
    x = grouped_ids["POINT_BALANCE"]
    y = grouped_ids["POINT_ELEVATION"]

    counts, xedges, yedges = np.histogram2d(x, y, bins=100)

    # Assign each point its bin count
    ix = np.searchsorted(xedges, x) - 1
    iy = np.searchsorted(yedges, y) - 1
    z = counts[ix, iy]
    if vvmin is None:
        vvmax = max(z)
        vvmin = min(z)
    # sc = ax.scatter(x, y, c=z,vmin = vvmin, vmax = vvmax, cmap="viridis", s=10, marker = 'o',label = 'train_data')
    sc = ax.hexbin(x, y, cmap="viridis", gridsize=nbin, mincnt=1, label="train_data")
    plt.colorbar(sc, ax=ax, label="Local point count")
    if test_data is not None:

        x = test_data["POINT_BALANCE"]
        y = test_data["POINT_ELEVATION"]

        counts, xedges, yedges = np.histogram2d(x, y, bins=50)

        # Assign each point its bin count
        ix = np.searchsorted(xedges, x) - 1
        iy = np.searchsorted(yedges, y) - 1
        z = counts[ix, iy]
        ax.scatter(
            x,
            y,
            c=z,
            vmin=vvmin,
            vmax=vvmax,
            cmap="viridis",
            s=10,
            marker="x",
            label="test_data",
        )

    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)

    if test_data is not None:
        ax.legend()
    ax.grid()
    ax.set_title(title, fontsize=20)

    # Set axes limits
    if ax_xlim is not None:
        ax.set_xlim(ax_xlim)
    if ax_ylim is not None:
        ax.set_ylim(ax_ylim)

    plt.tight_layout()

    return fig  # To log figure during training


def histogram_mb(
    grouped_ids,
    axs=None,
    title="",
    xlabel="SMB (m/y)",
    ylabel="Count",
    nbins=50,
    ax_xlim=None,
    ax_ylim=None,
    test_data=None,
):
    """
    Plots training point mass balance (PMB) histogram,

    Parameters
    ----------
    grouped_ids : pandas.DataFrame
        DataFrame containing at least 'target' (observed values) and 'pred' (predicted values) columns.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto; if None, a new figure and axes are created.
    title : str, optional
        Title for the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    nbins : int, optional
        Number of bins in the histogram.
    ax_xlim : tuple or None, optional
        x-axis limits as (min, max).
    ax_ylim : tuple or None, optional
        y-axis limits as (min, max).
    test_data : pandas.DataFrame
        DataFrame containing at least 'target' (observed values) and 'pred' (predicted values) columns for the test data.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if a new one was created, otherwise None.
    """

    if axs is None:
        if test_data is None:
            fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        else:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    else:
        fig = None
    x = grouped_ids["POINT_BALANCE"]
    ax = axs[0] if test_data is not None else axs
    ax.hist(x, bins=nbins, label="training data")
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.legend()
    ax.grid()
    if test_data is not None:
        x = test_data["POINT_BALANCE"]
        ax = axs[1]
        ax.hist(x, bins=nbins, label="test data")
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.legend()
        ax.grid()
    fig.suptitle(title, fontsize=20)

    # Set axes limits
    if ax_xlim is not None:
        ax.set_xlim(ax_xlim)
    if ax_ylim is not None:
        ax.set_ylim(ax_ylim)

    plt.tight_layout()

    return fig  # To log figure during training
