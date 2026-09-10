import numpy as np
import matplotlib.pyplot as plt
from calendar import month_abbr

from plots.map_plots import mapGlacier

_month_to_number = {month_abbr[i].lower(): i for i in range(1, 13)}


def _month_number(months):
    """Calendar month number of MONTHS entries; the padding months ("oct_", ...)
    are merged with the month they refer to."""
    return months.str.rstrip("_").map(_month_to_number)


def _month_style(month):
    """Color and linestyle of a month in the monthly profiles. The color gets
    brighter from January to July and darker again until December, and the second
    half of the year is dashed so that months with the same color can be told
    apart."""
    distance_to_january = min(month - 1, 13 - month) / 6
    color = plt.get_cmap("plasma")(0.85 * distance_to_january)
    return color, "-" if month <= 7 else "--"


def monthlyProfile(
    df,
    column,
    ax=None,
    bin_width=50,
    xlabel=None,
    title=None,
    lapse_rate_unit=None,
):
    """
    Plots the elevation profile of a gridded variable, with one line per calendar
    month. Values are averaged over all the grid points of each elevation band and
    over all the years in df.

    Args:
        df (pd.DataFrame): Gridded values with columns MONTHS, POINT_ELEVATION and
            `column`.
        column (str): Variable to plot.
        ax (matplotlib.axes.Axes): Axis to plot on, a new figure is created if None.
        bin_width (float): Width of the elevation bands in meters.
        xlabel (str): Label of the x axis, defaults to `column`.
        title (str): Title of the plot.
        lapse_rate_unit (str): If provided, the slope of a linear fit of each
            profile is added to the legend with this unit, e.g. "°C/km".

    Returns the created figure, or None if ax is provided.
    """
    df = df.assign(
        MONTH=_month_number(df.MONTHS),
        ELEVATION_BAND=bin_width * (np.floor(df.POINT_ELEVATION / bin_width) + 0.5),
    )
    profiles = df.groupby(["MONTH", "ELEVATION_BAND"])[column].mean()

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = None

    for month in profiles.index.get_level_values("MONTH").unique().sort_values():
        profile = profiles.loc[month]
        label = month_abbr[month]
        if lapse_rate_unit is not None and len(profile) > 1:
            slope = np.polyfit(profile.index.values, profile.values, 1)[0]
            label += f" ({1000 * slope:.1f} {lapse_rate_unit})"
        color, linestyle = _month_style(month)
        ax.plot(
            profile.values,
            profile.index.values,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            label=label,
        )
    ax.set_xlabel(xlabel or column)
    ax.set_ylabel("Elevation (m)")
    if title is not None:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")

    if fig is not None:
        fig.tight_layout()
    return fig


def _point_means(df, column, by=[]):
    """Average `column` per grid point (and per `by` groups) into the "pred" column
    expected by mapGlacier."""
    return df.groupby(by + ["POINT_LAT", "POINT_LON"], as_index=False).agg(
        RGIId=("RGIId", "first"), pred=(column, "mean")
    )


def periodMap(df, column, rgi_id, cfg, gdir=None, title=None, label_cb=None):
    """
    Maps a gridded variable averaged over all the months and years in df.

    Args:
        df (pd.DataFrame): Gridded values with columns RGIId, POINT_LAT, POINT_LON
            and `column`.
        column (str): Variable to plot.
        rgi_id (str): Glacier to plot.
        cfg (config.Config): Configuration instance.
        gdir (oggm.GlacierDirectory): OGGM directory of the glacier, it is
            initialized if None.
        title (str): Title of the plot.
        label_cb (str): Label of the colorbar, defaults to `column`.

    Returns the created figure.
    """
    return mapGlacier(
        _point_means(df, column),
        rgi_id,
        cfg,
        gdir=gdir,
        mapOnly=True,
        reverse_cb=True,
        title=title,
        label_cb=label_cb or column,
    )


def monthlyMaps(df, column, rgi_id, cfg, gdir=None, title=None, label_cb=None):
    """
    Maps a gridded variable for each calendar month, averaged over all the years in
    df. All the maps share the same color scale.

    Args:
        df (pd.DataFrame): Gridded values with columns RGIId, MONTHS, POINT_LAT,
            POINT_LON and `column`.
        column (str): Variable to plot.
        rgi_id (str): Glacier to plot.
        cfg (config.Config): Configuration instance.
        gdir (oggm.GlacierDirectory): OGGM directory of the glacier, it is
            initialized if None.
        title (str): Title of the figure.
        label_cb (str): Label of the colorbars, defaults to `column`.

    Returns the created figure.
    """
    means = _point_means(df.assign(MONTH=_month_number(df.MONTHS)), column, ["MONTH"])
    max_abs = means.pred.abs().max()

    fig, axs = plt.subplots(3, 4, figsize=(22, 15))
    for month, ax in zip(range(1, 13), axs.flatten()):
        mapGlacier(
            means[means.MONTH == month].reset_index(drop=True),
            rgi_id,
            cfg,
            ax=ax,
            max_abs=max_abs,
            gdir=gdir,
            mapOnly=True,
            reverse_cb=True,
            title=month_abbr[month],
            label_cb=label_cb or column,
        )
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
