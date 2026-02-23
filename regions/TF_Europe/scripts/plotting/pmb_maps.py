import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.lines import Line2D

from regions.TF_Europe.scripts.plotting.palettes import get_cmap_hex  # noqa: F401


def plot_glacier_measurements_map(
    glacier_info,
    glacier_outline_rgi,
    *,
    lon_col="POINT_LON",
    lat_col="POINT_LAT",
    nmeas_col="Nb. measurements",
    split_col="Train/Test glacier",
    title="Glacier measurement locations Central European Alps (5pct)",
    figsize=(18, 10),
    extent=(5.8, 15, 44, 48),  # (lonW, lonE, latS, latN)
    sizes=(100, 1500),  # scatter size range (points^2)
    size_legend_values=(30, 100, 1000, 6000),
    alpha=0.6,
    land_facecolor="lightgray",
    land_alpha=0.5,
    palette=None,  # dict like {"Train": "...", "Test": "..."} or None
    cmap_for_train=None,  # optional: e.g. cm.batlow to auto-pick a dark color if palette is None
    add_features=True,
    add_gridlines=True,
    legend_loc="lower right",
    legend_ncol=3,
    legend_fontsize=18,
    legend_title_fontsize=18,
    zorder_scatter=10,
    show=True,
):
    """
    Plot glacier point locations on a Cartopy map with marker size ~ sqrt(# measurements)
    and hue by a train/test column, including a custom combined legend.

    Requires (in your environment):
      - numpy as np
      - matplotlib.pyplot as plt
      - seaborn as sns
      - cartopy.crs as ccrs
      - cartopy.feature as cfeature
      - from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
      - matplotlib.lines import Line2D
      - matplotlib.patches import Patch
      - (optional) cm + get_cmap_hex if you want auto palette from a cmap

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    glacier_info_out : copy of glacier_info with added ['sqrt_size','scaled_size']
    scaled_size_fn : function(val)->float used for consistent scaling
    """

    # --- 0) Copy input ---
    df = glacier_info.copy()

    # --- 1) Preprocessing: sqrt scaling ---
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

    # --- 2) Figure + map ---
    fig = plt.figure(figsize=figsize)

    lonW, lonE, latS, latN = extent

    # Bounds check
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
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAKES)
        ax.add_feature(cfeature.RIVERS)
        ax.add_feature(cfeature.BORDERS, linestyle="-", linewidth=1)
        ax.add_feature(cfeature.LAND, facecolor=land_facecolor, alpha=land_alpha)

    # Glacier outlines
    glacier_outline_rgi.plot(ax=ax, transform=projPC, color="black", alpha=0.7)

    # --- 3) Palette ---
    if palette is None:
        # Try to auto-create something sane:
        # - If user gave cmap_for_train and they have get_cmap_hex, use it.
        # - Else: fallback to dark blue + red.
        train_color = "#1f4e79"
        if cmap_for_train is not None:
            try:
                # requires your helper
                colors = get_cmap_hex(cmap_for_train, 10)  # noqa: F821
                train_color = colors[0]
            except Exception:
                pass
        palette = {"Train": train_color, "Test": "#b2182b"}

    # --- 4) Scatter ---
    if split_col not in df.columns:
        raise KeyError(f"Missing '{split_col}' in glacier_info.")

    g = sns.scatterplot(
        data=df,
        x=lon_col,
        y=lat_col,
        size="scaled_size",
        hue=split_col,
        sizes=sizes,
        alpha=alpha,
        palette=palette,
        transform=projPC,
        ax=ax,
        zorder=zorder_scatter,
        legend=True,
    )

    # --- 5) Gridlines ---
    if add_gridlines:
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=1,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {"size": 16, "color": "black"}
        gl.ylabel_style = {"size": 16, "color": "black"}
        gl.top_labels = gl.right_labels = False

    # --- 6) Custom combined legend ---
    handles, labels = g.get_legend_handles_labels()
    expected_labels = list(palette.keys())
    hue_entries = [(h, l) for h, l in zip(handles, labels) if l in expected_labels]

    size_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=np.sqrt(
                scaled_size(val)
            ),  # matplotlib uses radius-like marker size
            markerfacecolor="gray",
            alpha=alpha,
            label=f"{val}",
        )
        for val in size_legend_values
    ]

    combined_handles = [h for h, _ in hue_entries] + size_handles
    combined_labels = [l for _, l in hue_entries] + [str(v) for v in size_legend_values]

    ax.legend(
        combined_handles,
        combined_labels,
        title="Number of measurements",
        loc=legend_loc,
        frameon=True,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
        borderpad=1.2,
        labelspacing=1.2,
        ncol=legend_ncol,
    )

    ax.set_title(title, fontsize=25)
    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax, df, scaled_size
