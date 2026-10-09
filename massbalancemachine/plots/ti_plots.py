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


# One hue per season (winter blue, spring green, summer red, autumn gold to ochre),
# the three months of a season going from light to dark
_season_colors = [
    ("Blues", [12, 1, 2], [0.5, 0.7, 0.9]),
    ("Greens", [3, 4, 5], [0.45, 0.65, 0.85]),
    ("Reds", [6, 7, 8], [0.5, 0.7, 0.9]),
    ("YlOrBr", [9, 10, 11], [0.35, 0.5, 0.65]),
]
_month_colors = {
    month: plt.get_cmap(cmap)(level)
    for cmap, months, levels in _season_colors
    for month, level in zip(months, levels)
}


def _month_color(month):
    """Color of a month in the monthly profiles, see `_season_colors`."""
    return _month_colors[month]


def _label_line_ends(ax, ends, min_gap=0.045):
    """Write each label at the top end of its line, the labels too close to the
    previous one being raised by one row so that they do not overlap. Lines ending
    at the same point share one label.

    Args:
        ends (list of (x, y, label)): Top end of each line, in data coordinates.
        min_gap (float): Smallest horizontal distance between two labels of the
            same row, in fraction of the axis width.
    """
    x0, x1 = ax.get_xlim()
    merged = []
    for x, y, label in ends:
        same = next(
            (m for m in merged if m[1] == y and abs(m[0] - x) < 0.01 * (x1 - x0)),
            None,
        )
        if same is None:
            merged.append([x, y, [label]])
        else:
            same[2].append(label)
    rows = []  # x position, in axis fraction, of the last label of each row
    placed = []
    for x, y, labels in sorted(merged, key=lambda m: m[0]):
        label = "all months" if len(labels) == 12 else ", ".join(labels)
        xf = (x - x0) / (x1 - x0)
        row = next((i for i, last in enumerate(rows) if xf - last >= min_gap), None)
        if row is None:
            row = len(rows)
            rows.append(xf)
        rows[row] = xf
        placed.append((x, y, label, row))
    for x, y, label, row in placed:
        ax.annotate(
            label,
            (x, y),
            xytext=(0, 4 + 11 * row),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.2",
        )
    # Room above the lines for the rows of labels
    ax.margins(y=0.04 + 0.035 * len(rows))


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

    ends = []
    for month in profiles.index.get_level_values("MONTH").unique().sort_values():
        profile = profiles.loc[month]
        label = month_abbr[month]
        if lapse_rate_unit is not None and len(profile) > 1:
            slope = np.polyfit(profile.index.values, profile.values, 1)[0]
            label += f" ({1000 * slope:.1f} {lapse_rate_unit})"
        ax.plot(
            profile.values,
            profile.index.values,
            color=_month_color(month),
            linewidth=2,
            label=label,
        )
        ends.append((profile.values[-1], profile.index.values[-1], month_abbr[month]))
    # Months named on the lines so that they can be read without the legend
    _label_line_ends(ax, ends)
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


def scalarParameters(module, temperatures=np.linspace(-10, 10, 401)):
    """
    Plots the two activation functions shaped by the learnable scalars of a TIlike
    model: the fraction of precipitation falling as snow, and the positive degree
    days, both against the corrected temperature T + cor_T. Their sharp versions
    (a step at the threshold and max(T, 0)) are drawn for reference.

    Args:
        module (TILikeModel): Model whose parameters are plotted.
        temperatures (np.ndarray): Corrected temperatures (°C) on the x axis.

    Returns the created figure.
    """
    values = module.scalar_parameters()
    slope, threshold = values["snow_slope"], values["snow_threshold"]
    curvature = values["pdd_curvature"]
    x = torch.as_tensor(temperatures)

    fig, (ax_snow, ax_pdd) = plt.subplots(1, 2, figsize=(11, 4.5))

    snow = torch.sigmoid(slope * (threshold - x))
    ax_snow.plot(temperatures, (temperatures < threshold), color="0.6", ls="--")
    ax_snow.plot(temperatures, snow, color="tab:blue", linewidth=2)
    ax_snow.axvline(threshold, color="0.6", linewidth=0.8)
    ax_snow.set_xlabel("Corrected temperature $T + cor_T$ (°C)")
    ax_snow.set_ylabel("Fraction of precipitation as snow (-)")
    ax_snow.set_title(
        f"Snow fraction: slope {slope:.2f} /°C, threshold {threshold:.2f} °C"
    )

    pdd = torch.nn.functional.softplus(x * curvature) / curvature
    ax_pdd.plot(temperatures, np.maximum(temperatures, 0), color="0.6", ls="--")
    ax_pdd.plot(temperatures, pdd, color="tab:red", linewidth=2)
    ax_pdd.set_xlabel("Corrected temperature $T + cor_T$ (°C)")
    ax_pdd.set_ylabel("Positive degree days (°C)")
    ax_pdd.set_title(f"PDD: curvature {curvature:.2f} /°C")

    for ax in (ax_snow, ax_pdd):
        ax.grid(alpha=0.3)
    ax_pdd.legend(["sharp version", "learned"], fontsize="small")
    fig.tight_layout()
    return fig
