"""Location maps of the study-region glaciers (Fig. 1 and Fig. A3b of the paper).

Used by notebook `2.2 Study_regions`. Not to be confused with
`pmb_maps.plot_glacier_measurements_map`, which plots train/test splits.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.lines import Line2D

from papers.Transformer_TF.scripts.plotting.style import (
    NATURE_COLORS,
    nature_figsize,
    NATURE_SPECS,
)
from papers.Transformer_TF.scripts.plotting.palettes import get_cmap_hex


def plot_glacier_locations_map(
    glacier_info,
    glacier_outline_rgi,
    *,
    lon_col="POINT_LON",
    lat_col="POINT_LAT",
    nmeas_col="Nb. measurements",
    hue_col=None,
    palette=None,
    cmap_for_hue=None,
    title="Glacier measurement locations",
    figsize=None,
    nature_cols=1,
    nature_height_mm=80,
    extent=(5.8, 15, 44, 48),
    use_size=True,  # : NEW — if False, plot fixed-size points (location only), skip size legend
    fixed_size=80,  # : NEW — marker size (points^2) used when use_size=False
    sizes=(20, 300),
    size_legend_values=(30, 100, 1000, 6000),
    alpha=0.6,
    point_color=NATURE_COLORS[5],
    land_facecolor="lightgray",
    land_alpha=0.5,
    add_features=True,
    add_gridlines=True,
    legend_loc="lower right",
    legend_ncol=2,
    legend_fontsize=NATURE_SPECS["font_max_pt"],
    legend_title_fontsize=NATURE_SPECS["font_max_pt"],
    hue_legend_title=None,
    gridline_label_fontsize=NATURE_SPECS["font_max_pt"] - 1,
    zorder_scatter=10,
    data_region=None,
    year_col="YEAR",
    summary_loc=(0.02, 0.02),
    summary_fontsize=NATURE_SPECS["font_max_pt"],
    legend_separate=False,
    legend_figsize=(3, 2),
    show=True,
):
    """
    Plot glacier point locations on a Cartopy map, styled to Nature figure specs
    (font sizes, dpi, colors via NATURE_COLORS).

    If use_size=True (default), marker size ~ sqrt(# measurements), with a size legend.
    If use_size=False, all points are drawn at a fixed size (location only) — no
    size legend is built, useful for a simpler "where are the glaciers" map.

    If hue_col is given, points are colored by that categorical column (e.g. O2Region)
    using NATURE_COLORS by default (or a custom palette / cmap_for_hue), and a combined
    hue(+size, if use_size) legend is built. Otherwise, single color for all points
    (legend then only shows hue, or nothing if hue_col is also None and use_size=False).

    If legend_separate=True, the legend is NOT drawn on the map axes; instead a second,
    standalone Figure containing only the legend is created and returned, so it can be
    saved independently (e.g. for separate placement in Illustrator).

    Requires (in your environment):
      - numpy as np
      - matplotlib.pyplot as plt
      - seaborn as sns
      - cartopy.crs as ccrs
      - cartopy.feature as cfeature
      - from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
      - matplotlib.lines import Line2D
      - utils.plotting.style: NATURE_COLORS, nature_figsize, NATURE_SPECS

    Returns
    -------
    If legend_separate=False (default):
        fig, ax, glacier_info_out, scaled_size_fn
    If legend_separate=True:
        fig, ax, glacier_info_out, scaled_size_fn, legend_fig

    Note: scaled_size_fn is a no-op-ish passthrough wrapper when use_size=False
    (returns fixed_size regardless of input), kept for return-signature consistency.
    """

    if figsize is None:
        figsize = nature_figsize(cols=nature_cols, height_mm=nature_height_mm)

    # --- 0) Copy input ---
    df = glacier_info.copy()

    # --- 1) Preprocessing: sqrt scaling (only if use_size) ---
    if use_size:
        if nmeas_col not in df.columns:
            raise KeyError(f"Missing '{nmeas_col}' in glacier_info.")
        df["sqrt_size"] = np.sqrt(df[nmeas_col].astype(float))

        sqrt_min = float(df["sqrt_size"].min())
        sqrt_max = float(df["sqrt_size"].max())

        def scaled_size(val, min_out=sizes[0], max_out=sizes[1]):
            sqrt_val = np.sqrt(float(val))
            if sqrt_max == sqrt_min:
                return (min_out + max_out) / 2
            return min_out + (max_out - min_out) * (
                (sqrt_val - sqrt_min) / (sqrt_max - sqrt_min)
            )

        df["scaled_size"] = df[nmeas_col].apply(scaled_size)
        size_arg = "scaled_size"
        size_kwarg = dict(size="scaled_size", sizes=sizes)
    else:

        def scaled_size(val, min_out=fixed_size, max_out=fixed_size):
            return fixed_size

        df["scaled_size"] = fixed_size
        size_kwarg = dict(s=fixed_size)  # fixed marker size, no size mapping

    # --- 1b) Hue setup ---
    use_hue = hue_col is not None
    if use_hue:
        if hue_col not in df.columns:
            raise KeyError(f"Missing '{hue_col}' in glacier_info.")
        df[hue_col] = df[hue_col].astype(str)
        categories = sorted(df[hue_col].dropna().unique())

        if palette is None:
            if cmap_for_hue is not None:
                try:
                    colors = get_cmap_hex(cmap_for_hue, len(categories))
                except Exception:
                    colors = NATURE_COLORS[: len(categories)]
            else:
                if len(categories) > len(NATURE_COLORS):
                    print(
                        f"Warning: {len(categories)} categories but only "
                        f"{len(NATURE_COLORS)} NATURE_COLORS available; colors will repeat."
                    )
                colors = [
                    NATURE_COLORS[i % len(NATURE_COLORS)]
                    for i in range(len(categories))
                ]
            palette = dict(zip(categories, colors))
        if hue_legend_title is None:
            hue_legend_title = hue_col

    # --- 2) Figure + map ---
    fig = plt.figure(figsize=figsize)

    lonW, lonE, latS, latN = extent

    max_lat, min_lat = float(df[lat_col].max()), float(df[lat_col].min())
    max_lon, min_lon = float(df[lon_col].max()), float(df[lon_col].min())

    if not (latS <= min_lat <= max_lat <= latN):
        print(
            f"Warning: latitude bounds may not fully contain glacier points: "
            f"glacier lat range [{min_lat:.2f}, {max_lat:.2f}] vs map bounds [{latS}, {latN}]"
        )
    if not (lonW <= min_lon <= max_lon <= lonE):
        print(
            f"Warning: longitude bounds may not fully contain glacier points: "
            f"glacier lon range [{min_lon:.2f}, {max_lon:.2f}] vs map bounds [{lonW}, {lonE}]"
        )

    projPC = ccrs.PlateCarree()
    ax = plt.axes(projection=projPC)
    ax.set_extent([lonW, lonE, latS, latN], crs=ccrs.Geodetic())

    if add_features:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
        ax.add_feature(cfeature.LAKES, linewidth=0.3)
        ax.add_feature(cfeature.RIVERS, linewidth=0.3)
        ax.add_feature(cfeature.BORDERS, linestyle="-", linewidth=0.4)
        ax.add_feature(cfeature.LAND, facecolor=land_facecolor, alpha=land_alpha)

    glacier_outline_rgi.plot(
        ax=ax, transform=projPC, color="black", alpha=0.7, linewidth=0.4
    )

    # --- 3) Scatter ---
    if use_hue:
        g = sns.scatterplot(
            data=df,
            x=lon_col,
            y=lat_col,
            hue=hue_col,
            alpha=alpha,
            palette=palette,
            transform=projPC,
            ax=ax,
            zorder=zorder_scatter,
            legend=True,
            **size_kwarg,
        )
    else:
        g = sns.scatterplot(
            data=df,
            x=lon_col,
            y=lat_col,
            alpha=alpha,
            color=point_color,
            transform=projPC,
            ax=ax,
            zorder=zorder_scatter,
            legend=False,
            **size_kwarg,
        )

    # --- 4) Gridlines ---
    if add_gridlines:
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.3,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {"size": gridline_label_fontsize, "color": "black"}
        gl.ylabel_style = {"size": gridline_label_fontsize, "color": "black"}
        gl.top_labels = gl.right_labels = False

    # --- 5) Legend handles ---
    if use_size:
        size_handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markersize=np.sqrt(scaled_size(val)),
                markerfacecolor=(point_color if not use_hue else "gray"),
                alpha=alpha,
                label=f"{val}",
            )
            for val in size_legend_values
        ]
    else:
        size_handles = []

    if use_hue:
        handles, labels = g.get_legend_handles_labels()
        expected_labels = list(palette.keys())
        label_to_handle = dict(zip(labels, handles))
        hue_entries = [
            (label_to_handle[l], l) for l in expected_labels if l in label_to_handle
        ]

        combined_handles = [h for h, _ in hue_entries] + size_handles
        combined_labels = [l for _, l in hue_entries] + [
            str(v) for v in size_legend_values
        ]
        legend_title = (
            f"{hue_legend_title} / Number of measurements"
            if use_size
            else hue_legend_title
        )
    else:
        combined_handles = size_handles
        combined_labels = [str(v) for v in size_legend_values]
        legend_title = "Number of measurements"

    # Remove seaborn's auto-attached hue legend (we build our own combined one, if any)
    if use_hue and g.legend_ is not None:
        g.legend_.remove()

    legend_fig = None
    has_legend_content = len(combined_handles) > 0

    if legend_separate:
        if has_legend_content:
            legend_fig = plt.figure(figsize=legend_figsize)
            legend_fig.legend(
                combined_handles,
                combined_labels,
                title=legend_title,
                loc="center",
                frameon=True,
                fontsize=legend_fontsize,
                title_fontsize=legend_title_fontsize,
                borderpad=0.8,
                labelspacing=0.8,
                ncol=legend_ncol,
            )
            legend_fig.gca().set_axis_off() if legend_fig.axes else None
    elif has_legend_content:
        leg = ax.legend(
            combined_handles,
            combined_labels,
            title=legend_title,
            loc=legend_loc,
            frameon=True,
            fontsize=legend_fontsize,
            title_fontsize=legend_title_fontsize,
            borderpad=0.8,
            labelspacing=0.8,
            ncol=legend_ncol,
        )
        leg.get_frame().set_linewidth(0.4)

    # --- 6) Summary text box ---
    if data_region is not None:
        n_glaciers = len(glacier_info.index.unique())

        if year_col in data_region.columns:
            years = data_region[year_col].dropna()
            if len(years) > 0:
                y_min, y_max = int(years.min()), int(years.max())
                year_str = f"{y_min}\u2013{y_max}" if y_min != y_max else f"{y_min}"
            else:
                year_str = "N/A"
        else:
            print(
                f"Note: '{year_col}' not found in data_region; skipping years in summary."
            )
            year_str = "N/A"

        summary_text = f"Glaciers: {n_glaciers}\nYears: {year_str}"
        print(summary_text)

    if title:
        ax.set_title(title, fontsize=NATURE_SPECS["font_max_pt"])

    plt.tight_layout()

    if show:
        plt.show()

    if legend_separate:
        return fig, ax, df, scaled_size, legend_fig
    return fig, ax, df, scaled_size
