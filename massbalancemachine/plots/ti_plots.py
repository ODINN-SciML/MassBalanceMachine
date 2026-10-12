import numpy as np
import torch
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


# Markers of the geodetic glaciers of each side in `regionCorrectionMaps`
_flag_styles = {
    "train": dict(marker="o", edgecolor="black"),
    "val": dict(marker="s", edgecolor="magenta"),
    "test": dict(marker="^", edgecolor="darkorange"),
}


def regionCorrectionMaps(
    module, glaciers, elevations, variable, flagged=None, title=None
):
    """
    Maps a correction of a TIlike model at the centroids of the glaciers of a region,
    one panel per elevation. The temperature bias and the precipitation correction
    only depend on the location (and possibly the elevation), so they are evaluated
    directly at the centroids. A correction that does not depend on the elevation is
    drawn once. A correction that depends on ELEVATION_DIFFERENCE takes it as the
    elevation minus the altitude of the ERA5 cell of the glacier, so at a given
    elevation it also varies with the ERA5 topography.

    Args:
        module (TILikeModel): Model whose correction is mapped.
        glaciers (pd.DataFrame): Glaciers with columns CenLat and CenLon, and
            ALTITUDE_CLIMATE when the correction depends on ELEVATION_DIFFERENCE.
        elevations (list of float): Elevations in meters at which to evaluate the
            correction.
        variable (str): "temperature_bias" or "precipitation_correction".
        flagged (dict): Glaciers to flag, as {side: pd.DataFrame with columns
            POINT_LAT and POINT_LON}, side being one of "train", "val" and "test".
        title (str): Title of the figure.

    Returns the created figure.
    """
    if variable == "temperature_bias":
        label, inputs, predict, cmap = (
            "Temperature bias (°C)",
            module.inp_bias_T,
            module.predict_temp_bias_from_df,
            "RdBu_r",
        )
    elif variable == "precipitation_correction":
        label, inputs, predict, cmap = (
            "Precipitation correction $P_{cor}$ (-)",
            module.inp_P_cor,
            module.predict_bias_cor_from_df,
            "viridis",
        )
    else:
        raise ValueError(f"Unknown variable {variable}.")
    location = {"POINT_LAT", "POINT_LON", "POINT_ELEVATION", "ELEVATION_DIFFERENCE"}
    assert set(inputs) <= location, (
        f"{label} depends on {sorted(set(inputs) - location)}, which are not known "
        "at the glacier centroids."
    )
    assert "ELEVATION_DIFFERENCE" not in inputs or "ALTITUDE_CLIMATE" in glaciers, (
        f"{label} depends on ELEVATION_DIFFERENCE, which needs the column "
        "ALTITUDE_CLIMATE of the glaciers."
    )
    panelElevations = (
        elevations
        if {"POINT_ELEVATION", "ELEVATION_DIFFERENCE"} & set(inputs)
        else [None]
    )

    values = []
    df = glaciers.rename(columns={"CenLat": "POINT_LAT", "CenLon": "POINT_LON"})
    for z in panelElevations:
        with torch.no_grad():
            elevation = z if z is not None else 0.0
            values.append(
                predict(
                    df.assign(
                        POINT_ELEVATION=elevation,
                        ELEVATION_DIFFERENCE=elevation
                        - df.get("ALTITUDE_CLIMATE", 0.0),
                    )
                )
                .cpu()
                .numpy()
            )
    # Same color scale for all the elevations
    if cmap == "RdBu_r":
        vmax = max(np.abs(v).max() for v in values)
        vmin = -vmax
    else:
        vmin = min(v.min() for v in values)
        vmax = max(v.max() for v in values)

    fig, axs = plt.subplots(
        1, len(panelElevations), figsize=(5 * len(panelElevations), 4.5), squeeze=False
    )
    for ax, z, v in zip(axs[0], panelElevations, values):
        sc = ax.scatter(
            glaciers.CenLon, glaciers.CenLat, c=v, s=4, cmap=cmap, vmin=vmin, vmax=vmax
        )
        # Test first, so that the glaciers the model was trained on stay visible
        for side in ["test", "train", "val"]:
            flags = (flagged or {}).get(side, [])
            if len(flags) == 0:
                continue
            ax.scatter(
                flags.POINT_LON,
                flags.POINT_LAT,
                s=30,
                facecolor="none",
                linewidth=0.8,
                label=f"{side} geodetic ({len(flags)})",
                **_flag_styles[side],
            )
        ax.set_title(f"{z:.0f} m" if z is not None else "independent of the elevation")
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_aspect(1 / np.cos(np.deg2rad(glaciers.CenLat.mean())))
        fig.colorbar(sc, ax=ax, label=label, shrink=0.8)
    if flagged:
        axs[0, 0].legend(fontsize="small", loc="best")
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
