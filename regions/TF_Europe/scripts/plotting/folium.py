import folium
import pandas as pd


def plot_stakes_folium(
    df_pmb_clean,
    glacier_col="GLACIER",
    lat_col=None,
    lon_col=None,
    elev_col=None,
    id_col=None,
    tooltip_col=None,  # <-- new
    center=None,
    zoom_start=10,
    color_map=None,
):
    """
    Create an interactive Folium map of stake points grouped by glacier.

    Parameters
    ----------
    tooltip_col : str, optional
        Column name to show on hover (e.g. 'O2Region'). If None, no tooltip.
    """

    # Infer column names if not provided
    if lat_col is None:
        lat_col = "lat" if "lat" in df_pmb_clean.columns else "POINT_LAT"
    if lon_col is None:
        lon_col = "lon" if "lon" in df_pmb_clean.columns else "POINT_LON"
    if elev_col is None:
        elev_col = (
            "altitude" if "altitude" in df_pmb_clean.columns else "POINT_ELEVATION"
        )
    if id_col is None:
        id_col = (
            "stake_number" if "stake_number" in df_pmb_clean.columns else "POINT_ID"
        )

    # Compute center if not provided
    if center is None:
        center_lat = float(df_pmb_clean[lat_col].median())
        center_lon = float(df_pmb_clean[lon_col].median())
    else:
        center_lat, center_lon = center

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    default_colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "darkred",
        "cadetblue",
        "darkgreen",
        "darkpurple",
        "pink",
        "gray",
        "black",
    ]

    glaciers = sorted(df_pmb_clean[glacier_col].dropna().unique())

    if color_map is None:
        color_map = {
            g: default_colors[i % len(default_colors)] for i, g in enumerate(glaciers)
        }
    else:
        for i, g in enumerate(glaciers):
            color_map.setdefault(g, default_colors[i % len(default_colors)])

    for glacier_name, df_g in df_pmb_clean.groupby(glacier_col):
        if pd.isna(glacier_name):
            continue

        fg = folium.FeatureGroup(name=str(glacier_name))
        color = color_map[str(glacier_name)]

        for _, row in df_g.iterrows():
            stake_id = row.get(id_col, "NA")
            altitude = row.get(elev_col, "NA")

            tooltip = None
            if tooltip_col is not None and tooltip_col in row.index:
                tooltip = f"{tooltip_col}: {row[tooltip_col]}"

            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=f"{glacier_name} - Stake {stake_id}: {altitude} m",
                tooltip=tooltip,
            ).add_to(fg)

        fg.add_to(m)

    legend_rows = "\n".join(
        f'<p><span style="color: {color_map[g]};">●</span> {g}</p>' for g in glaciers
    )

    legend_html = f"""
    <div style="
        position: fixed; bottom: 50px; left: 50px; z-index: 1000;
        background-color: white; padding: 10px; border-radius: 5px;
        border: 1px solid #999;
    ">
        <p><strong>Glaciers</strong></p>
        {legend_rows}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m
