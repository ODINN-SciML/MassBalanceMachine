import os
from pandas.api.types import CategoricalDtype
import joypy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
import seaborn as sns
from cmcrameri import cm
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
)
import matplotlib as mpl
import massbalancemachine as mbm
import os
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import string
from regions.Switzerland.scripts.geodata import *
from regions.Switzerland.scripts.helpers import *

colors = get_cmap_hex(cm.batlow, 10)
color_annual = colors[0]
color_winter = "#c51b7d"


def plot_geodetic_MB(df, glacier_name, color_xgb="blue", color_tim="red"):
    df = df.dropna(subset=["geodetic_mb", "mbm_mb_mean", "glamos_mb_mean"])

    # Ensure data exists before plotting
    if len(df) == 0:
        print("No valid data points to plot.")
    else:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 5))

        # Scatter plot
        sns.scatterplot(
            df,
            x="geodetic_mb",
            y="mbm_mb_mean",
            color=color_xgb,
            #    hue = 'end_year',
            alpha=0.7,
            label="MBM MB",
            marker="o",
        )
        sns.scatterplot(
            df,
            x="geodetic_mb",
            y="glamos_mb_mean",
            color=color_tim,
            alpha=0.7,
            label="GLAMOS MB",
            marker="s",
        )

        # diagonal line
        pt = (0, 0)
        ax.axline(pt, slope=1, color="grey", linestyle="--", linewidth=1)
        ax.axvline(0, color="grey", linestyle="--", linewidth=1)
        ax.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax.grid(True, linestyle="--", linewidth=0.5)

        # Labels & Title
        ax.set_xlabel("Geodetic MB [m w.e.]", fontsize=12)
        ax.set_ylabel("Modeled MB [m w.e.]", fontsize=12)
        ax.set_title(f"{glacier_name.capitalize()} Glacier", fontsize=14)
        ax.legend(loc="upper left", fontsize=10)

        # return figure
        return fig


def scatter_geodetic_MB(df_all, hue="GLACIER", size=False):
    """
    Creates scatter plots comparing Geodetic MB to MBM MB and GLAMOS MB, with RMSE and correlation annotations.

    Parameters:
    -----------
    df_all : pd.DataFrame
        DataFrame containing 'Geodetic MB', 'MBM MB', 'GLAMOS MB', 'GLACIER', 'Area', and 'Test Glacier'.
    size : bool, optional
        If True, scales points based on glacier area.
    """
    # Drop rows where any required columns are NaN
    df_all = df_all.dropna(subset=["Geodetic MB", "MBM MB", "GLAMOS MB"])

    # Compute RMSE and Pearson correlation
    rmse_mbm = root_mean_squared_error(df_all["Geodetic MB"], df_all["MBM MB"])
    corr_mbm = np.corrcoef(df_all["Geodetic MB"], df_all["MBM MB"])[0, 1]
    rmse_glamos = root_mean_squared_error(df_all["Geodetic MB"], df_all["GLAMOS MB"])
    corr_glamos = np.corrcoef(df_all["Geodetic MB"], df_all["GLAMOS MB"])[0, 1]

    # Define figure and axes
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot MBM MB vs Geodetic MB
    plot_scatter(df_all, hue, size, axs[0], "MBM MB", rmse_mbm, corr_mbm)

    # Plot GLAMOS MB vs Geodetic MB
    plot_scatter(df_all, hue, size, axs[1], "GLAMOS MB", rmse_glamos, corr_glamos)

    axs[0].set_title("MBM MB vs Geodetic MB", fontsize=16)
    axs[1].set_title("GLAMOS MB vs Geodetic MB", fontsize=16)

    # Adjust legend outside of plot
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        ncol=2,
        fontsize=14,
    )

    plt.tight_layout()
    plt.show()


def plot_mass_balance(
    glacier_name, year, df_stakes, path_distributed_MB_glamos, PATH_PREDICTIONS
):
    """
    Plots the annual and winter mass balance for a given glacier and year,
    comparing GLAMOS distributed MB with model predictions.

    Parameters:
    -----------
    glacier_name : str
        Name of the glacier.
    year : int
        Year for which the mass balance should be plotted.
    df_stakes : pd.DataFrame
        DataFrame containing all stake measurements.
    path_distributed_MB_glamos : str
        Path to the GLAMOS distributed mass balance files.
    PATH_PREDICTIONS : str
        Path to the MBM model predictions.
    """
    # Extract stake data for this glacier and year
    stakes_data = df_stakes[
        (df_stakes.GLACIER == glacier_name) & (df_stakes.YEAR == year)
    ]
    stakes_data_ann = stakes_data[stakes_data.PERIOD == "annual"].copy()
    stakes_data_win = stakes_data[stakes_data.PERIOD == "winter"].copy()

    # Construct file paths
    file_ann = f"{year}_ann_fix_lv95.grid"
    file_win = f"{year}_win_fix_lv95.grid"
    grid_path_ann = os.path.join(path_distributed_MB_glamos, glacier_name, file_ann)
    grid_path_win = os.path.join(path_distributed_MB_glamos, glacier_name, file_win)

    # Load GLAMOS data (Annual)
    metadata_ann, grid_data_ann = load_grid_file(grid_path_ann)
    ds_glamos_ann = convert_to_xarray_geodata(grid_data_ann, metadata_ann)
    ds_glamos_wgs84_ann = transform_xarray_coords_lv95_to_wgs84(ds_glamos_ann)

    # Load GLAMOS data (Winter)
    metadata_win, grid_data_win = load_grid_file(grid_path_win)
    ds_glamos_win = convert_to_xarray_geodata(grid_data_win, metadata_win)
    ds_glamos_wgs84_win = transform_xarray_coords_lv95_to_wgs84(ds_glamos_win)

    # Load MBM predictions (Annual)
    mbm_file_ann = os.path.join(
        PATH_PREDICTIONS, glacier_name, f"{glacier_name}_{year}_annual.zarr"
    )
    ds_mbm_ann = xr.open_dataset(mbm_file_ann)
    ds_mbm_ann = apply_gaussian_filter(ds_mbm_ann)

    # Load MBM predictions (Winter)
    mbm_file_win = os.path.join(
        PATH_PREDICTIONS, glacier_name, f"{glacier_name}_{year}_winter.zarr"
    )
    ds_mbm_win = xr.open_dataset(mbm_file_win)
    ds_mbm_win = apply_gaussian_filter(ds_mbm_win)

    # Ensure correct coordinate names
    lon_name = "lon" if "lon" in ds_mbm_ann.coords else "longitude"
    lat_name = "lat" if "lat" in ds_mbm_ann.coords else "latitude"

    # Apply the function correctly using lambda
    stakes_data_ann["Predicted_MB"] = stakes_data_ann.apply(
        lambda row: get_predicted_mb(lon_name, lat_name, row, ds_mbm_ann), axis=1
    )
    stakes_data_ann.dropna(subset=["Predicted_MB"], inplace=True)
    stakes_data_ann["GLAMOS_MB"] = stakes_data_ann.apply(
        lambda row: get_predicted_mb_glamos(
            lon_name, lat_name, row, ds_glamos_wgs84_ann
        ),
        axis=1,
    )
    stakes_data_ann.dropna(subset=["GLAMOS_MB"], inplace=True)

    # Same for winter
    stakes_data_win["Predicted_MB"] = stakes_data_win.apply(
        lambda row: get_predicted_mb(lon_name, lat_name, row, ds_mbm_win), axis=1
    )
    stakes_data_win.dropna(subset=["Predicted_MB"], inplace=True)
    stakes_data_win["GLAMOS_MB"] = stakes_data_win.apply(
        lambda row: get_predicted_mb_glamos(
            lon_name, lat_name, row, ds_glamos_wgs84_win
        ),
        axis=1,
    )
    stakes_data_win.dropna(subset=["GLAMOS_MB"], inplace=True)

    # Compute color scale limits (Annual)
    vmin_ann = min(
        ds_glamos_wgs84_ann.min().item(), ds_mbm_ann.pred_masked.min().item()
    )
    vmax_ann = max(
        ds_glamos_wgs84_ann.max().item(), ds_mbm_ann.pred_masked.max().item()
    )

    # Compute color scale limits (Winter)
    vmin_win = min(
        ds_glamos_wgs84_win.min().item(), ds_mbm_win.pred_masked.min().item()
    )
    vmax_win = max(
        ds_glamos_wgs84_win.max().item(), ds_mbm_win.pred_masked.max().item()
    )

    cmap_ann, norm_ann, cmap_win, norm_win = get_color_maps(
        vmin_ann, vmax_ann, vmin_win, vmax_win
    )

    # Create figure with 2 rows (Annual & Winter)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    # Annual GLAMOS Plot
    ds_glamos_wgs84_ann.plot.imshow(
        ax=axes[0, 0],
        cmap=cmap_ann,
        norm=norm_ann,
        cbar_kwargs={"label": "Mass Balance [m w.e.]"},
    )
    axes[0, 0].set_title("GLAMOS (Annual)")
    sns.scatterplot(
        data=stakes_data_ann,
        x="POINT_LON",
        y="POINT_LAT",
        hue="POINT_BALANCE",
        palette=cmap_ann,
        hue_norm=norm_ann,
        ax=axes[0, 0],
        s=25,
        legend=False,
    )

    # add rmse if available
    if not stakes_data_ann.empty:
        rmse = root_mean_squared_error(
            stakes_data_ann.POINT_BALANCE, stakes_data_ann.GLAMOS_MB
        )
        axes[0, 0].text(
            0.05,
            0.1,
            f"RMSE: {rmse:.2f}",
            transform=axes[0, 0].transAxes,
            ha="left",
            va="top",
            fontsize=18,
        )

    # Annual MBM Predictions Plot
    ds_mbm_ann.pred_masked.plot.imshow(
        ax=axes[0, 1],
        cmap=cmap_ann,
        norm=norm_ann,
        cbar_kwargs={"label": "Mass Balance [m w.e.]"},
    )
    axes[0, 1].set_title("MBM (Annual)")

    # Add Annual Stake Coordinates
    sns.scatterplot(
        data=stakes_data_ann,
        x="POINT_LON",
        y="POINT_LAT",
        hue="POINT_BALANCE",
        palette=cmap_ann,
        hue_norm=norm_ann,
        ax=axes[0, 1],
        s=25,
        legend=False,
    )

    # add rmse
    rmse = root_mean_squared_error(
        stakes_data_ann.POINT_BALANCE, stakes_data_ann.Predicted_MB
    )
    axes[0, 1].text(
        0.05,
        0.1,
        f"RMSE: {rmse:.2f}",
        transform=axes[0, 1].transAxes,
        ha="left",
        va="top",
        fontsize=18,
    )

    # Winter GLAMOS & MBM Plots
    ds_glamos_wgs84_win.plot.imshow(
        ax=axes[1, 0],
        cmap=cmap_win,
        norm=norm_win,
        cbar_kwargs={"label": "Mass Balance [m w.e.]"},
    )
    axes[1, 0].set_title("GLAMOS (Winter)")
    sns.scatterplot(
        data=stakes_data_win,
        x="POINT_LON",
        y="POINT_LAT",
        hue="POINT_BALANCE",
        palette=cmap_win,
        hue_norm=norm_win,
        ax=axes[1, 0],
        s=25,
        legend=False,
    )

    # add rmse
    rmse = root_mean_squared_error(
        stakes_data_win.POINT_BALANCE, stakes_data_win.GLAMOS_MB
    )
    axes[1, 0].text(
        0.05,
        0.1,
        f"RMSE: {rmse:.2f}",
        transform=axes[1, 0].transAxes,
        ha="left",
        va="top",
        fontsize=18,
    )

    # Winter MBM Predictions Plot
    ds_mbm_win.pred_masked.plot.imshow(
        ax=axes[1, 1],
        cmap=cmap_win,
        norm=norm_win,
        cbar_kwargs={"label": "Mass Balance [m w.e.]"},
    )
    axes[1, 1].set_title("MBM (Winter)")
    sns.scatterplot(
        data=stakes_data_win,
        x="POINT_LON",
        y="POINT_LAT",
        hue="POINT_BALANCE",
        palette=cmap_win,
        hue_norm=norm_win,
        ax=axes[1, 1],
        s=25,
        legend=False,
    )

    # add rmse
    rmse = root_mean_squared_error(
        stakes_data_win.POINT_BALANCE, stakes_data_win.Predicted_MB
    )
    axes[1, 1].text(
        0.05,
        0.1,
        f"RMSE: {rmse:.2f}",
        transform=axes[1, 1].transAxes,
        ha="left",
        va="top",
        fontsize=18,
    )

    plt.suptitle(f"{glacier_name.capitalize()}: Mass Balance {year}")

    plt.tight_layout()

    return fig


def plot_snow_cover_scatter(df, add_corr=True):
    """
    Generate scatter plots of snow cover and corrected snow cover
    for each month in the dataset, including R^2 values in each plot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data. Must have columns:
      'monthNb', 'snow_cover_S2', 'snow_cover_glacier',
      'snow_cover_glacier_corr', and 'glacier_name'.

    Returns:
    - fig, axs: Matplotlib figure and axes objects for further customization.
    """
    # Number of unique months
    N_months = len(df["month"].unique())

    # Create a grid of subplots
    if add_corr:
        fig, axs = plt.subplots(2, N_months, figsize=(15, 8), squeeze=False)
    else:
        fig, axs = plt.subplots(1, N_months, figsize=(15, 4), squeeze=False)

    # Get sorted unique months
    months = np.sort(df["monthNb"].unique())

    # Loop over each month
    for i, monthNb in enumerate(months):
        # Subset data for the current month
        df_month = df[df["monthNb"] == monthNb]

        # Left column: scatter plot of snow cover
        ax = axs[0, i]
        sns.scatterplot(
            data=df_month,
            x="snow_cover_S2",
            y="snow_cover_glacier",
            marker="o",
            hue="glacier_name",
            ax=ax,
        )
        x = np.linspace(0, 1, 100)
        ax.plot(x, x, "k--")  # Identity line

        # Calculate and add R^2 value
        r2 = (
            np.corrcoef(df_month["snow_cover_S2"], df_month["snow_cover_glacier"])[0, 1]
            ** 2
        )
        mse = mean_squared_error(
            df_month["snow_cover_glacier"], df_month["snow_cover_S2"]
        )
        ax.text(
            0.05,
            0.85,
            f"R² = {r2:.2f}\nMSE = {mse:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            color="black",
        )

        ax.set_xlabel("Sentinel-2")
        ax.set_ylabel("Mass Balance Machine")
        ax.set_title(f'Snow Cover (Normal), {df_month["month"].values[0]}')
        ax.get_legend().remove()  # Remove legend

        if add_corr:
            # Right column: scatter plot of corrected snow cover
            ax = axs[1, i]
            sns.scatterplot(
                data=df_month,
                x="snow_cover_S2",
                y="snow_cover_glacier_corr",
                marker="o",
                hue="glacier_name",
                ax=ax,
            )
            ax.plot(x, x, "k--")  # Identity line

            # Calculate and add R^2 value
            r2_corr = (
                np.corrcoef(
                    df_month["snow_cover_S2"], df_month["snow_cover_glacier_corr"]
                )[0, 1]
                ** 2
            )
            mse_corr = mean_squared_error(
                df_month["snow_cover_glacier_corr"], df_month["snow_cover_S2"]
            )
            ax.text(
                0.05,
                0.85,
                f"R² = {r2_corr:.2f}\nMSE = {mse_corr:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                color="black",
            )

            ax.set_xlabel("Sentinel-2")
            ax.set_ylabel("Mass Balance Machine")
            ax.set_title(f'Snow Cover (Corrected), {df_month["month"].values[0]}')
            ax.get_legend().remove()  # Remove legend

    # Add a single legend underneath the last row of axes
    if add_corr:
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=5,
            bbox_to_anchor=(0.5, -0.05),
            title="Glacier Name",
        )

        # Adjust layout for better spacing
        plt.tight_layout(
            rect=[0, 0.08, 1, 1]
        )  # Leave space at the bottom for the legend

    return fig, axs


def plot_snow_cover_scatter_combined(df):
    """
    Generate two scatter plots:
    1. Snow cover for all months together.
    2. Corrected snow cover for all months together.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data. Must have columns:
      'monthNb', 'snow_cover_S2', 'snow_cover_glacier',
      'snow_cover_glacier_corr', and 'glacier_name'.

    Returns:
    - fig, axs: Matplotlib figure and axes objects for further customization.
    """

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # First subplot: Normal snow cover
    ax = axs[0]
    sns.scatterplot(
        data=df,
        x="snow_cover_S2",
        y="snow_cover_glacier",
        marker="o",
        style="month",
        ax=ax,
        s=200,
    )
    x = np.linspace(0, 1, 100)
    ax.plot(x, x, "k--")  # Identity line
    ax.set_xlabel("Sentinel-2", fontsize=14)
    ax.set_ylabel("Mass Balance Machine", fontsize=14)
    ax.set_title("Snow Cover (Normal)", fontsize=16)
    ax.get_legend().remove()  # Remove legend for now

    r2 = np.corrcoef(df["snow_cover_S2"], df["snow_cover_glacier"])[0, 1] ** 2
    ax.text(
        0.05, 0.9, f"R² = {r2:.2f}", transform=ax.transAxes, fontsize=16, color="black"
    )

    # Add a single legend to the right of the plots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(1.02, 0.8),  # Move legend to the side
        title="Glacier Month",
        fontsize=16,
        title_fontsize=14,
    )

    # Adjust layout for better spacing
    plt.suptitle(
        df.glacier_name.unique()[0].capitalize(), fontsize=20, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space on the right for the legend
    return fig, axs


def plot_snow_cover_geoplots(
    raster_res,
    path_S2,
    month_pos,
    add_snowline=False,
    band_size=50,
    percentage_threshold=50,
):
    """
    Plot geoplots of snow cover for a given raster file.

    Parameters:
    - raster_res (str): The name of the raster file to process.
    - path_S2 (str): Path to the directory containing the satellite rasters.
    - get_hydro_year_and_month (function): Function to determine the hydrological year and month from a date.
    - month_pos (dict): Mapping of hydrological months to their abbreviated names.
    - IceSnowCover (function): Function to calculate snow and ice cover from a GeoDataFrame.
    - snowCover (function): Function to load mass-balance predictions and calculate snow cover corrections.
    - plotClasses (function): Function to create the plots.
    """
    # Extract glacier name
    glacierName = raster_res.split("_")[0]

    # Extract date from satellite raster
    match = re.search(r"(\d{4})_(\d{2})_(\d{2})", raster_res)
    if not match:
        raise ValueError(f"Invalid raster filename format: {raster_res}")

    year, month, day = match.groups()
    date_str = f"{year}-{month}-{day}"
    raster_date = datetime.strptime(date_str, "%Y-%m-%d")

    # Find closest hydrological year and month
    closest_month, hydro_year = get_hydro_year_and_month(raster_date)
    monthNb = month_pos[closest_month]

    # Skip if the hydrological year is out of range
    if hydro_year > 2021:
        return

    # Read satellite raster over glacier
    raster_path = os.path.join(path_S2, "perglacier", raster_res)
    gdf_S2_res = gpd.read_file(raster_path)

    # Load MB predictions for that year and month
    path_nc_wgs84 = f"results/nc/sgi/{glacierName}/"
    filename_nc = f"{glacierName}_{hydro_year}_{monthNb}.nc"

    # Calculate snow and ice cover
    geoData_gl = mbm.geodata.GeoData(pd.DataFrame)
    geoData_gl.set_ds_latlon(filename_nc, path_nc_wgs84)
    geoData_gl.classify_snow_cover(tol=0.1)
    gdf_glacier = geoData_gl.gdf

    # Plot the results
    gl_date = f"{hydro_year}-{closest_month}"
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    plotClasses(
        gdf_glacier,
        gdf_S2_res,
        axs,
        gl_date,
        raster_date,
        add_snowline,
        band_size=band_size,
        percentage_threshold=percentage_threshold,
    )
    plt.show()


def plotClasses(
    gdf_glacier,
    gdf_S2_res,
    axs,
    gl_date,
    file_date,
    add_snowline=False,
    band_size=10,
    percentage_threshold=50,
):

    # Define the colors for categories (ensure that your categories match the color list)
    colors_cat = ["#a6cee3", "#1f78b4", "#8da0cb", "#b2df8a", "#fb9a99"]

    # Manually map categories to colors (assuming categories 0-5 for example)
    classes = {
        1.0: "snow",
        3.0: "clean ice",
        2.0: "firn / old snow / bright ice",
        4.0: "debris",
        5.0: "cloud",
    }
    map = dict(
        zip(classes.keys(), colors_cat[:6])
    )  # Adjust according to the number of categories

    # Set up the basemap provider
    API_KEY = "000378bd-b0f0-46e2-a46d-f2165b0c6c02"
    provider = cx.providers.Stadia.StamenTerrain(api_key=API_KEY)
    provider["url"] = provider["url"] + f"?api_key={API_KEY}"

    # Plot the first figure (Mass balance)
    vmin, vmax = gdf_glacier.pred_masked.min(), gdf_glacier.pred_masked.max()

    # Determine the colormap and normalization
    if vmin < 0 and vmax > 0:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        cmap = "RdBu"
    elif vmin < 0 and vmax <= 0:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = "Reds"
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = "Blues"

    gdf_clean = gdf_glacier.dropna(subset=["pred_masked"])
    gdf_clean.plot(
        column="pred_masked",  # Column to visualize
        cmap=cmap,  # Color map suitable for glacier data
        norm=norm,
        legend=True,  # Display a legend
        ax=axs[0],
        markersize=5,  # Adjust size if points are too small or large
        missing_kwds={"color": "lightgrey"},  # Define color for NaN datas
    )
    axs[0].set_title(f"Mass balance: {gl_date}")

    # Plot the second figure (MBM classes)
    gdf_clean = gdf_glacier.dropna(subset=["classes"])
    gdf_clean["color"] = gdf_clean["classes"].map(map)
    # Plot with manually defined colormap
    gdf_clean.plot(
        column="classes",  # Column to visualize
        legend=True,  # Display a legend
        markersize=5,  # Adjust size if points are too small or large
        missing_kwds={"color": "lightgrey"},  # Define color for NaN datas
        categorical=True,  # Ensure the plot uses categorical colors
        ax=axs[1],
        color=gdf_clean["color"],  # Use the custom colormap
    )

    # calculate snow and ice cover
    snow_cover_glacier = IceSnowCover(gdf_glacier, gdf_S2_res)
    AddSnowCover(snow_cover_glacier, axs[1])

    # cx.add_basemap(axs[1], crs=gdf_glacier.crs, source=provider)
    axs[1].set_title(f"MBM: {gl_date}")

    if add_snowline:
        # Overlay the selected band (where 'selected_band' is True)
        selected_band = gdf_clean[gdf_clean["selected_band"] == True]
        # Plot the selected elevation band with a distinct style (e.g., red border)
        selected_band.plot(ax=axs[1], color="red", linewidth=1, markersize=5, alpha=0.5)

    # Plot the fourth figure (Resampled Sentinel classes)
    gdf_clean = gdf_S2_res.dropna(subset=["classes"])
    gdf_clean["color"] = gdf_clean["classes"].map(map)
    # Plot with manually defined colormap
    gdf_clean.plot(
        column="classes",  # Column to visualize
        legend=True,  # Display a legend
        markersize=5,  # Adjust size if points are too small or large
        missing_kwds={"color": "lightgrey"},  # Define color for NaN datas
        categorical=True,  # Ensure the plot uses categorical colors
        ax=axs[2],
        color=gdf_clean["color"],  # Use the custom colormap
    )
    # calculate snow and ice cover
    snow_cover_glacier = IceSnowCover(gdf_S2_res, gdf_S2_res)
    AddSnowCover(snow_cover_glacier, axs[2])
    # cx.add_basemap(axs[2], crs=gdf_glacier.crs, source=provider)
    axs[2].set_title(f"Sentinel: {file_date.strftime('%Y-%m-%d')}")

    # Manually add custom legend for the third plot
    handles = [
        mpatches.Patch(color=color, label=classes[i]) for i, color in map.items()
    ]
    axs[2].legend(
        handles=handles, title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left"
    )

    # Show the plot with consistent colors
    # plt.tight_layout()
    plt.show()


def AddSnowCover(snow_cover_glacier, ax):
    # Custom legend for snow and ice cover
    legend_labels = "\n".join(((f"Snow cover: {snow_cover_glacier*100:.2f}%"),))
    #    (f"Ice cover: {ice_cover_glacier*100:.2f}%")))

    props = dict(boxstyle="round", facecolor="white", alpha=0.7)
    ax.text(
        0.03,
        0.08,
        legend_labels,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=12,
        bbox=props,
    )


def plot_mass_balance_comparison_annual_glamos_nn(
    glacier_name,
    year,
    cfg,
    df_stakes,
    path_distributed_mb,
    path_pred_nn,
):
    """Plot annual MB comparison (GLAMOS vs NN) for a glacier and year."""

    # Stake data
    stakes_data = df_stakes[
        (df_stakes.GLACIER == glacier_name) & (df_stakes.YEAR == year)
    ]
    stakes_data_ann = stakes_data[stakes_data.PERIOD == "annual"].copy()

    # GLAMOS grid path and loading
    file_ann = f"{year}_ann_fix_lv95.grid"
    grid_path_ann = os.path.join(
        cfg.dataPath, path_distributed_mb, "GLAMOS", glacier_name, file_ann
    )
    metadata_ann, grid_data_ann = load_grid_file(grid_path_ann)
    ds_glamos_ann = convert_to_xarray_geodata(grid_data_ann, metadata_ann)

    if glacier_name == "adler" or glacier_name == "findelen":
        ds_glamos_wgs84_ann = transform_xarray_coords_lv03_to_wgs84(ds_glamos_ann)
    else:
        ds_glamos_wgs84_ann = transform_xarray_coords_lv95_to_wgs84(ds_glamos_ann)

    # Load NN predictions
    mbm_file_nn = os.path.join(
        path_pred_nn, glacier_name, f"{glacier_name}_{year}_annual.zarr"
    )
    ds_mbm_nn = apply_gaussian_filter(xr.open_dataset(mbm_file_nn))

    # Coordinate names
    lon_name = "lon" if "lon" in ds_mbm_nn.coords else "longitude"
    lat_name = "lat" if "lat" in ds_mbm_nn.coords else "latitude"

    # Add predictions to stake data
    if not stakes_data_ann.empty:
        stakes_data_ann["Predicted_MB_NN"] = stakes_data_ann.apply(
            lambda row: get_predicted_mb(lon_name, lat_name, row, ds_mbm_nn), axis=1
        )
        stakes_data_ann["GLAMOS_MB"] = stakes_data_ann.apply(
            lambda row: get_predicted_mb_glamos(
                lon_name, lat_name, row, ds_glamos_wgs84_ann
            ),
            axis=1,
        )
        stakes_data_ann.dropna(subset=["Predicted_MB_NN", "GLAMOS_MB"], inplace=True)

    # Color scale
    vmin = min(ds_glamos_wgs84_ann.min().item(), ds_mbm_nn.pred_masked.min().item())
    vmax = max(ds_glamos_wgs84_ann.max().item(), ds_mbm_nn.pred_masked.max().item())
    (
        cmap_ann,
        norm_ann,
    ) = get_color_maps(vmin, vmax)

    # Plot setup
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharex=False, sharey=False)

    # GLAMOS plot
    ds_glamos_wgs84_ann.plot.imshow(
        ax=axes[0],
        cmap=cmap_ann,
        norm=norm_ann,
        cbar_kwargs={"label": "Mass Balance [m w.e.]"},
    )
    axes[0].set_title("GLAMOS (Annual)")

    var_glamos = ds_glamos_wgs84_ann.var().item()
    if not stakes_data_ann.empty:
        sns.scatterplot(
            data=stakes_data_ann,
            x="POINT_LON",
            y="POINT_LAT",
            hue="POINT_BALANCE",
            palette=cmap_ann,
            hue_norm=norm_ann,
            ax=axes[0],
            s=25,
            legend=False,
        )
        rmse_glamos = root_mean_squared_error(
            stakes_data_ann.POINT_BALANCE, stakes_data_ann.GLAMOS_MB
        )
        text_glamos = f"RMSE: {rmse_glamos:.2f},\nmean MB: {ds_glamos_wgs84_ann.mean().item():.2f},\nvar: {var_glamos:.2f}"
    else:
        text_glamos = (
            f"mean MB: {ds_glamos_wgs84_ann.mean().item():.2f},\nvar: {var_glamos:.2f}"
        )
    axes[0].text(
        0.05,
        0.15,
        text_glamos,
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=18,
    )

    # NN plot
    ds_mbm_nn.pred_masked.plot.imshow(
        ax=axes[1],
        cmap=cmap_ann,
        norm=norm_ann,
        cbar_kwargs={"label": "Mass Balance [m w.e.]"},
    )
    axes[1].set_title("MBM NN (Annual)")

    var_nn = ds_mbm_nn.pred_masked.var().item()
    if not stakes_data_ann.empty:
        sns.scatterplot(
            data=stakes_data_ann,
            x="POINT_LON",
            y="POINT_LAT",
            hue="POINT_BALANCE",
            palette=cmap_ann,
            hue_norm=norm_ann,
            ax=axes[1],
            s=25,
            legend=False,
        )
        rmse_nn = root_mean_squared_error(
            stakes_data_ann.POINT_BALANCE,
            stakes_data_ann.Predicted_MB_NN,
        )
        text_nn = (
            f"RMSE: {rmse_nn:.2f},\n"
            f"mean MB: {ds_mbm_nn.pred_masked.mean().item():.2f},\n"
            f"var: {var_nn:.2f}"
        )
    else:
        text_nn = (
            f"mean MB: {ds_mbm_nn.pred_masked.mean().item():.2f},\n"
            f"var: {var_nn:.2f}"
        )
    axes[1].text(
        0.05,
        0.15,
        text_nn,
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=18,
    )

    plt.suptitle(
        f"{glacier_name.capitalize()} Glacier – Annual MB (GLAMOS vs NN, {year})",
        fontsize=20,
    )
    plt.tight_layout()
    plt.show()


# --- optional: wrapper to reproduce original plot ----------------------------


# -----------------------------------------------------------
# Shared helper
# -----------------------------------------------------------
def lonlat_names(obj):
    coords = getattr(obj, "coords", {})
    if "lon" in coords and "lat" in coords:
        return "lon", "lat"
    if "longitude" in coords and "latitude" in coords:
        return "longitude", "latitude"
    return "lon", "lat"


# -----------------------------------------------------------
# Stake extraction for scatter plots
# -----------------------------------------------------------
def extract_stake_scatter_points(
    df_stakes,
    glacier,
    year,
    period="annual",
):
    """
    Returns arrays:
    obs_stake (x), obs_stake (y)
    """

    if df_stakes is None:
        return None, None

    sub = df_stakes[(df_stakes.GLACIER == glacier) & (df_stakes.YEAR == year)].copy()

    if period == "annual" and "PERIOD" in sub.columns:
        sub = sub[sub.PERIOD == "annual"]

    if sub.empty or "POINT_BALANCE" not in sub.columns:
        return None, None

    obs = sub["POINT_BALANCE"].values
    obs = obs[np.isfinite(obs)]

    if obs.size == 0:
        return None, None

    return obs, obs


# -----------------------------------------------------------
# Single glacier–year scatter plot
# -----------------------------------------------------------
def plot_glamos_vs_mbm_pixel_scatter(
    glacier,
    year,
    cfg,
    path_distributed_mb,
    path_pred_lstm,
    period="annual",
    apply_smoothing_fn=None,
    ax=None,
    df_stakes=None,
):
    """
    Scatter plot of GLAMOS vs MBM (LSTM) mass balance
    for overlapping pixels only, with optional stake overlay.
    """

    # ---------- helpers ----------
    def pick_file_glamos(glacier, year, period="annual"):
        suffix = "ann" if period == "annual" else "win"
        base = os.path.join(cfg.dataPath, path_distributed_mb, "GLAMOS", glacier)
        cand_lv95 = os.path.join(base, f"{year}_{suffix}_fix_lv95.grid")
        cand_lv03 = os.path.join(base, f"{year}_{suffix}_fix_lv03.grid")
        if os.path.exists(cand_lv95):
            return cand_lv95, "lv95"
        if os.path.exists(cand_lv03):
            return cand_lv03, "lv03"
        return None, None

    def load_glamos_wgs84(glacier, year):
        path, cs = pick_file_glamos(glacier, year, period)
        if path is None:
            return None
        meta, arr = load_grid_file(path)
        da = convert_to_xarray_geodata(arr, meta)
        if cs == "lv03":
            return transform_xarray_coords_lv03_to_wgs84(da)
        if cs == "lv95":
            return transform_xarray_coords_lv95_to_wgs84(da)
        return None

    def load_lstm_ds(glacier, year):
        zpath = os.path.join(path_pred_lstm, glacier, f"{glacier}_{year}_{period}.zarr")
        if not os.path.exists(zpath):
            return None
        ds = xr.open_zarr(zpath)
        if apply_smoothing_fn is not None:
            ds = apply_smoothing_fn(ds)
        return ds

    # ---------- load data ----------
    da_g = load_glamos_wgs84(glacier, year)
    ds_m = load_lstm_ds(glacier, year)

    if da_g is None or ds_m is None or "pred_masked" not in ds_m:
        raise ValueError(f"No overlapping data for {glacier} {year}")

    # ---------- interpolate MBM → GLAMOS grid ----------
    lon_g, lat_g = lonlat_names(da_g)
    lon_m, lat_m = lonlat_names(ds_m)

    mbm_interp = ds_m["pred_masked"].interp(
        {lon_m: da_g[lon_g], lat_m: da_g[lat_g]},
        method="nearest",
    )

    # ---------- overlap mask ----------
    mask = np.isfinite(da_g.values) & np.isfinite(mbm_interp.values)

    glamos_vals = da_g.values[mask]
    mbm_vals = mbm_interp.values[mask]

    if glamos_vals.size == 0:
        raise ValueError(f"No overlapping valid pixels for {glacier} {year}")

    # ---------- plot ----------
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    sns.scatterplot(
        x=glamos_vals,
        y=mbm_vals,
        s=6,
        alpha=0.35,
        edgecolor=None,
        ax=ax,
    )

    # ---------- stake overlay ----------
    g_s = m_s = None
    if df_stakes is not None:
        obs_x, obs_y = extract_stake_scatter_points(
            df_stakes,
            glacier,
            year,
            period=period,
        )

        if obs_x is not None and len(obs_x) > 0:
            ax.scatter(
                obs_x,
                obs_y,
                c="red",
                s=70,
                marker="x",
                linewidths=1.8,
                label="Stakes (observations)",
                zorder=6,
            )
            ax.legend(frameon=True, loc="lower right")

    # ---------- 1:1 line ----------
    lims = [
        min(glamos_vals.min(), mbm_vals.min()),
        max(glamos_vals.max(), mbm_vals.max()),
    ]
    ax.plot(lims, lims, "k--", lw=1)

    # ---------- statistics ----------
    r = np.corrcoef(glamos_vals, mbm_vals)[0, 1]
    r2 = r**2

    # ---------- formatting ----------
    ax.set_title(f"{glacier.capitalize()} – {year}")
    ax.set_xlabel("GLAMOS MB [m w.e.]")
    ax.set_ylabel("MBM MB [m w.e.]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    ax.text(
        0.05,
        0.95,
        f"N = {glamos_vals.size}\n" f"r = {r:.2f}\n" f"R² = {r2:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    return ax, glamos_vals, mbm_vals


# -----------------------------------------------------------
# 2 glaciers × 2 years figure
# -----------------------------------------------------------
def plot_2glaciers_2years_glamos_vs_mbm_scatter(
    glacier_names,
    years_by_glacier,
    cfg,
    path_distributed_mb,
    path_pred_lstm,
    period="annual",
    apply_smoothing_fn=None,
    df_stakes=None,
):
    """
    Scatter plots of GLAMOS vs MBM (pixel-wise overlap)
    Layout: 2 rows × 2 columns (glacier × year)
    """

    assert len(glacier_names) == 2
    assert len(years_by_glacier) == 2
    assert all(len(y) == 2 for y in years_by_glacier)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12, 12),
        sharex=True,
        sharey=True,
    )

    axes = np.atleast_2d(axes)
    all_axes = []

    for r, glacier in enumerate(glacier_names):
        for c, year in enumerate(years_by_glacier[r]):
            ax = axes[r, c]

            try:
                plot_glamos_vs_mbm_pixel_scatter(
                    glacier=glacier,
                    year=year,
                    cfg=cfg,
                    path_distributed_mb=path_distributed_mb,
                    path_pred_lstm=path_pred_lstm,
                    period=period,
                    apply_smoothing_fn=apply_smoothing_fn,
                    ax=ax,
                    df_stakes=df_stakes,
                )
            except Exception:
                ax.text(
                    0.5,
                    0.5,
                    f"No data\n{glacier} {year}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()

            if r == 0:
                ax.set_title(f"{glacier.capitalize()} – {year}")
            if c == 0:
                ax.set_ylabel("MBM MB [m w.e.]")
            if r == 1:
                ax.set_xlabel("GLAMOS MB [m w.e.]")

            all_axes.append(ax)

    # ---------- synchronize limits ----------
    lims = all_axes[0].get_xlim()
    for ax in all_axes:
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    plt.tight_layout()
    return fig, all_axes


def plot_glacier_monthly_series_lstm_sharedcmap_center0(
    glacier_name: str,
    year: int,
    path_pred_lstm: str,
    *,
    var: str = "pred_masked",
    months_order=None,
    apply_smoothing_fn=None,
    cmap_name: str = "coolwarm_r",
):
    """
    Plot all monthly LSTM predictions (e.g. cumulative MB) for a glacier–year.
    All panels share the same color normalization centered at 0, like in
    plot_2glaciers_2years_glamos_vs_lstm.

    Parameters
    ----------
    glacier_name : str
        Glacier folder name under path_pred_lstm.
    year : int
        Year to plot.
    path_pred_lstm : str
        Root path where monthly zarrs are stored.
    var : str
        Variable to plot (default "cum_pred").
    months_order : list[str], optional
        Order of months to display (hydrological order). Defaults to Sep–Aug.
    apply_smoothing_fn : callable, optional
        Function to apply to the dataset before plotting.
    cmap_name : str
        Matplotlib colormap name (default "coolwarm_r").
    """

    # Default hydrological year order (Sep–Aug)
    if months_order is None:
        months_order = [
            "oct",
            "nov",
            "dec",
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
        ]

    n_months = len(months_order)
    ncols = 4
    nrows = int(np.ceil(n_months / ncols))

    # --- First pass: determine global min/max ---
    vmin, vmax = np.inf, -np.inf
    month_datasets = {}

    for month in months_order:
        zpath = os.path.join(
            path_pred_lstm, glacier_name, f"{glacier_name}_{year}_{month}.zarr"
        )
        if not os.path.exists(zpath):
            continue
        ds = xr.open_zarr(zpath)
        if var not in ds:
            continue
        if apply_smoothing_fn is not None:
            ds = apply_smoothing_fn(ds)
        da = ds[var]
        vmin = min(vmin, float(da.min().item()))
        vmax = max(vmax, float(da.max().item()))
        month_datasets[month] = da

    if not month_datasets:
        raise FileNotFoundError(
            f"No {var} data found for {glacier_name} {year} in {path_pred_lstm}"
        )

    # --- Shared colormap centered at 0 ---
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # --- Plot grid ---
    fig = plt.figure(figsize=(4.8 * ncols, 5.0 * nrows))
    gs = GridSpec(nrows=nrows, ncols=ncols, figure=fig, wspace=0.3, hspace=0.25)

    for i, month in enumerate(months_order):
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[row, col])

        if month not in month_datasets:
            ax.text(
                0.5, 0.5, f"No data\n{month}", ha="center", va="center", fontsize=11
            )
            ax.set_axis_off()
            continue

        da = month_datasets[month]

        im = da.plot.imshow(ax=ax, cmap=cmap, norm=norm, add_colorbar=False)

        cb = fig.colorbar(im, ax=ax, shrink=0.75)
        cb.set_label("Cumulative MB [m w.e.]", fontsize=10)

        mean_val = float(da.mean().item())
        ax.set_title(f"{month.capitalize()} ({mean_val:+.2f} m w.e.)", fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    fig.suptitle(
        f"{glacier_name.capitalize()} – {year} Monthly Cumulative Predictions",
        fontsize=18,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()

    return fig


def plot_monthly_joyplot(
    df_long,
    month_order=None,
    color_lstm="tab:blue",
    color_nn="tab:orange",
    color_xgb="tab:green",
    color_glamos="gray",
    figsize_cm=(12, 14),
    x_range=(-2.2, 2.2),
    alpha=1,
):
    """
    Plot a JoyPy ridge plot comparing monthly MB distributions for
    LSTM, NN, XGB, and GLAMOS.

    Parameters
    ----------
    df_long : pd.DataFrame
        Long-format dataframe from `prepare_monthly_long_df`.
    month_order : list
        Ordered months (default: Oct–Sep hydrological order).
    """

    if month_order is None:
        month_order = [
            "oct",
            "nov",
            "dec",
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
        ]

    cm = 1 / 2.54

    model_cols = ["mb_lstm", "mb_nn", "mb_xgb", "mb_glamos"]
    model_labels = ["LSTM", "NN", "XGB", "GLAMOS"]
    model_colors = [color_lstm, color_nn, color_xgb, color_glamos]

    fig, ax = joypy.joyplot(
        df_long,
        by="Month",
        column=model_cols,
        alpha=0.8,
        overlap=0,
        fill=False,
        linewidth=1.5,
        xlabelsize=8.5,
        ylabelsize=8.5,
        x_range=x_range,
        grid=False,
        color=model_colors,
        figsize=(figsize_cm[0] * cm, figsize_cm[1] * cm),
        ylim="own",
    )

    # Zero-line
    plt.axvline(x=0, color="grey", alpha=0.5, linewidth=1)

    # Axis labels & ticks
    plt.xlabel("Mass balance (m w.e.)", fontsize=8.5)
    plt.yticks(ticks=range(1, 13), labels=month_order, fontsize=8.5)
    plt.gca().set_yticklabels(month_order)

    # Legend
    legend_patches = [
        Patch(facecolor=color, label=label, alpha=alpha, edgecolor="k")
        for label, color in zip(model_labels, model_colors)
    ]

    plt.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.48, -0.1),
        ncol=4,
        fontsize=8.5,
        handletextpad=0.5,
        columnspacing=1,
    )

    plt.show()
    return fig
