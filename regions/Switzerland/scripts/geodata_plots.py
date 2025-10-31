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
from scipy.stats import pearsonr
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
)
import matplotlib as mpl
import massbalancemachine as mbm

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


def get_color_maps(vmin_ann, vmax_ann, vmin_win, vmax_win):
    # print(
    #     f"Color scale range (Annual): vmin={vmin_ann:.3f}, vmax={vmax_ann:.3f}"
    # )
    # print(
    #     f"Color scale range (Winter): vmin={vmin_win:.3f}, vmax={vmax_win:.3f}"
    # )
    if vmin_ann < 0 and vmax_ann > 0:
        norm_ann = mcolors.TwoSlopeNorm(vmin=vmin_ann, vcenter=0, vmax=-vmin_ann)
        cmap_ann = "coolwarm_r"
    elif vmin_ann < 0 and vmax_ann < 0:
        norm_ann = mcolors.Normalize(vmin=vmin_ann, vmax=vmax_ann)
        cmap_ann = "Reds_r"
    else:
        norm_ann = mcolors.Normalize(vmin=vmin_ann, vmax=vmax_ann)
        cmap_ann = "Blues"

    if vmin_win < 0:
        norm_win = mcolors.TwoSlopeNorm(vmin=-vmax_win, vcenter=0, vmax=vmax_win)
        cmap_win = "coolwarm_r"
    else:
        norm_win = mcolors.Normalize(vmin=vmin_win, vmax=vmax_win)
        cmap_win = "Blues"

    return cmap_ann, norm_ann, cmap_win, norm_win


# Function to extract mass balance for each stake
def get_predicted_mb(lon_name, lat_name, row, ds):
    try:
        return ds.sel(
            {lon_name: row.POINT_LON, lat_name: row.POINT_LAT}, method="nearest"
        ).pred_masked.item()  # Convert to scalar
    except Exception:
        print(f"Warning: Stake at ({row.POINT_LON}, {row.POINT_LAT}) is out of bounds.")
        return np.nan


def get_predicted_mb_glamos(lon_name, lat_name, row, ds):
    try:
        return ds.sel(
            {lon_name: row.POINT_LON, lat_name: row.POINT_LAT}, method="nearest"
        ).item()  # Convert to scalar
    except Exception:
        print(f"Warning: Stake at ({row.POINT_LON}, {row.POINT_LAT}) is out of bounds.")
        return np.nan


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


def plot_mass_balance_comparison(
    glacier_name,
    geoMB_periods,
    MBM_glwmb,
    df,
    color_mbm="#1f77b4",  # Default matplotlib blue
    color_model2="#ff7f0e",  # Default matplotlib orange
):
    """
    Plots time series and comparisons of modeled and geodetic mass balances.

    Parameters:
    - glacier_name (str): Name of the glacier (used in title).
    - geoMB_periods (list of tuples): Periods for geodetic MB, e.g., [(2000, 2005), (2005, 2010)].
    - MBM_glwmb (pd.DataFrame): DataFrame with modeled MB time series, indexed by year.
    - df (pd.DataFrame): DataFrame with columns ['geoMB_periods', 'mbm_geod', 'glamos_geod', 'target_geod'].
    - color_xgb (str): Color for MBM model.
    - color_tim (str): Color for GLAMOS model.
    """
    min_year, max_year = min(geoMB_periods)[0], max(geoMB_periods)[1]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # --- Plot 1: Time series ---
    MBM_glwmb[MBM_glwmb.index >= min_year].plot(
        ax=axs[0], marker="o", color=[color_mbm, color_model2]
    )
    axs[0].set_title(f"{glacier_name.capitalize()} Glacier")
    axs[0].set_ylabel("Mass Balance [m w.e.]")
    axs[0].set_xlabel("Year")
    axs[0].grid(True, linestyle="--", linewidth=0.5)

    # --- Plot 2: Geodetic MB lines ---
    for _, row in df.iterrows():
        x_start, x_end = row["geoMB_periods"]
        axs[1].hlines(
            y=row["mbm_geod"], xmin=x_start, xmax=x_end, color=color_mbm, linewidth=2
        )
        axs[1].hlines(
            y=row["glamos_geod"],
            xmin=x_start,
            xmax=x_end,
            color=color_model2,
            linewidth=2,
        )

    axs[1].set_xlabel("Year")
    axs[1].set_ylabel("Geodetic MB [m w.e.]")
    axs[1].set_title("Geodetic MB")
    axs[1].grid(True)

    # --- Plot 3: Scatter comparison ---
    df.plot.scatter(
        x="target_geod",
        y="mbm_geod",
        ax=axs[2],
        color=color_mbm,
        alpha=0.7,
        label="MBM MB",
        marker="o",
    )
    df.plot.scatter(
        x="target_geod",
        y="glamos_geod",
        ax=axs[2],
        color=color_model2,
        alpha=0.7,
        label="GLAMOS MB",
        marker="s",
    )
    axs[2].set_xlabel("Geodetic MB [m w.e.]", fontsize=12)
    axs[2].set_ylabel("Modeled MB [m w.e.]", fontsize=12)
    axs[2].axline((0, 0), slope=1, color="grey", linestyle="--", linewidth=1)
    axs[2].axvline(0, color="grey", linestyle="--", linewidth=1)
    axs[2].axhline(0, color="grey", linestyle="--", linewidth=1)
    axs[2].grid(True, linestyle="--", linewidth=0.5)
    axs[2].set_title("Geodetic MB vs. Modeled MB")

    plt.tight_layout()
    plt.show()


def plot_mass_balance_comparison_annual(
    glacier_name,
    year,
    cfg,
    df_stakes,
    path_distributed_mb,  # base for GLAMOS grids
    path_pred_lstm,  # base for LSTM/XGB zarrs
    period="annual",
):
    """Plot annual MB comparison (GLAMOS vs LSTM/XGB) for a glacier and year (no bias correction)."""

    import os
    import xarray as xr
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Stake data (annual only)
    stakes_data = df_stakes[
        (df_stakes.GLACIER == glacier_name) & (df_stakes.YEAR == year)
    ]
    stakes_data_ann = stakes_data[stakes_data.PERIOD == "annual"].copy()

    # ---- Locate GLAMOS grid (ann/win in lv95/lv03) ----
    def pick_ann_file(cfg, glacier_name, year, period="annual"):
        suffix = "ann" if period == "annual" else "win"
        base = os.path.join(cfg.dataPath, path_distributed_mb, "GLAMOS", glacier_name)
        cand_lv95 = os.path.join(base, f"{year}_{suffix}_fix_lv95.grid")
        cand_lv03 = os.path.join(base, f"{year}_{suffix}_fix_lv03.grid")
        if os.path.exists(cand_lv95):
            return cand_lv95, "lv95"
        if os.path.exists(cand_lv03):
            return cand_lv03, "lv03"
        return None, None

    grid_path_ann, coord_system = pick_ann_file(cfg, glacier_name, year, period)
    if grid_path_ann is None:
        raise FileNotFoundError(
            f"No GLAMOS {period} grid found for {glacier_name} {year}"
        )

    # ---- Load GLAMOS and transform to WGS84 ----
    metadata_ann, grid_data_ann = load_grid_file(grid_path_ann)
    da_glamos_ann = convert_to_xarray_geodata(grid_data_ann, metadata_ann)  # DataArray

    if coord_system == "lv03":
        da_glamos_wgs84_ann = transform_xarray_coords_lv03_to_wgs84(da_glamos_ann)
    elif coord_system == "lv95":
        da_glamos_wgs84_ann = transform_xarray_coords_lv95_to_wgs84(da_glamos_ann)
    else:
        raise ValueError(f"Unknown coord system for GLAMOS grid: {coord_system}")

    # ---- Load LSTM predictions (Zarr) and (optionally) smooth ----
    mbm_file_lstm = os.path.join(
        path_pred_lstm, glacier_name, f"{glacier_name}_{year}_{period}.zarr"
    )
    if not os.path.exists(mbm_file_lstm):
        raise FileNotFoundError(f"Missing LSTM/XGB zarr: {mbm_file_lstm}")

    ds_mbm_lstm = apply_gaussian_filter(xr.open_zarr(mbm_file_lstm))

    # ---- Coordinate names for stake sampling ----
    lon_name = "lon" if "lon" in ds_mbm_lstm.coords else "longitude"
    lat_name = "lat" if "lat" in ds_mbm_lstm.coords else "latitude"

    # ---- Sample model & GLAMOS at stake points ----
    if not stakes_data_ann.empty:
        stakes_data_ann["Predicted_MB_LSTM"] = stakes_data_ann.apply(
            lambda row: get_predicted_mb(lon_name, lat_name, row, ds_mbm_lstm), axis=1
        )
        stakes_data_ann["GLAMOS_MB"] = stakes_data_ann.apply(
            lambda row: get_predicted_mb_glamos(
                lon_name, lat_name, row, da_glamos_wgs84_ann
            ),
            axis=1,
        )
        stakes_data_ann.dropna(subset=["Predicted_MB_LSTM", "GLAMOS_MB"], inplace=True)

    # ---- Color limits from raw (unbiased) fields ----
    vmin = min(
        float(da_glamos_wgs84_ann.min().item()),
        float(ds_mbm_lstm["pred_masked"].min().item()),
    )
    vmax = max(
        float(da_glamos_wgs84_ann.max().item()),
        float(ds_mbm_lstm["pred_masked"].max().item()),
    )
    cmap_ann, norm_ann, _, _ = get_color_maps(vmin, vmax, 0, 0)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=False, sharey=False)

    # GLAMOS
    da_glamos_wgs84_ann.plot.imshow(
        ax=axes[0],
        cmap=cmap_ann,
        norm=norm_ann,
        cbar_kwargs={"label": "Mass Balance [m w.e.]"},
    )
    axes[0].set_title("GLAMOS (Annual)")
    var_glamos = float(da_glamos_wgs84_ann.var().item())

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
        text_glamos = (
            f"RMSE: {rmse_glamos:.2f},\n"
            f"mean MB: {float(da_glamos_wgs84_ann.mean().item()):.2f},\n"
            f"var: {var_glamos:.2f}"
        )
    else:
        text_glamos = (
            f"mean MB: {float(da_glamos_wgs84_ann.mean().item()):.2f},\n"
            f"var: {var_glamos:.2f}"
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

    # LSTM
    ds_mbm_lstm["pred_masked"].plot.imshow(
        ax=axes[1],
        cmap=cmap_ann,
        norm=norm_ann,
        cbar_kwargs={"label": "Mass Balance [m w.e.]"},
    )
    axes[1].set_title("MBM LSTM (Annual)")
    var_lstm = float(ds_mbm_lstm["pred_masked"].var().item())

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
        rmse_lstm = root_mean_squared_error(
            stakes_data_ann.POINT_BALANCE, stakes_data_ann.Predicted_MB_LSTM
        )
        text_lstm = (
            f"RMSE: {rmse_lstm:.2f},\n"
            f"mean MB: {float(ds_mbm_lstm['pred_masked'].mean().item()):.2f},\n"
            f"var: {var_lstm:.2f}"
        )
    else:
        text_lstm = (
            f"mean MB: {float(ds_mbm_lstm['pred_masked'].mean().item()):.2f},\n"
            f"var: {var_lstm:.2f}"
        )

    axes[1].text(
        0.05,
        0.15,
        text_lstm,
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=18,
    )

    plt.suptitle(
        f"{glacier_name.capitalize()} Glacier – Annual MB Comparison ({year})",
        fontsize=20,
    )
    plt.tight_layout()
    plt.show()


def plot_scatter_comparison(
    ax, df, glacier_name, color_mbm, color_glamos, title_suffix=""
):
    sns.scatterplot(
        df,
        x="Geodetic MB",
        y="MBM MB",
        color=color_mbm,
        alpha=0.7,
        label="MBM MB",
        marker="o",
        ax=ax,
    )
    sns.scatterplot(
        df,
        x="Geodetic MB",
        y="GLAMOS MB",
        color=color_glamos,
        alpha=0.7,
        label="GLAMOS MB",
        marker="s",
        ax=ax,
    )

    ax.axline((0, 0), slope=1, color="grey", linestyle="--", linewidth=1)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.grid(True, linestyle="--", linewidth=0.5)

    ax.set_xlabel("Geodetic MB [m w.e.]", fontsize=12)
    ax.set_ylabel("Modeled MB [m w.e.]", fontsize=12)
    ax.set_title(f"{glacier_name.capitalize()} Glacier {title_suffix}", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)


def mbm_glwd_pred(PATH_PREDICTIONS, GLACIER_NAME):
    # Define the path to model predictions
    path_results = os.path.join(PATH_PREDICTIONS, GLACIER_NAME)

    # Extract available years from NetCDF filenames
    years = sorted(
        [
            int(f.split("_")[1])
            for f in os.listdir(path_results)
            if f.endswith("_annual.zarr")
        ]
    )

    # Extract model-predicted mass balance
    pred_gl = []
    for year in years:
        file_path = os.path.join(path_results, f"{GLACIER_NAME}_{year}_annual.zarr")
        if not os.path.exists(file_path):
            print(f"Warning: Missing MBM file for {GLACIER_NAME} ({year}). Skipping...")
            pred_gl.append(np.nan)
            continue

        ds = xr.open_dataset(file_path)
        pred_gl.append(ds.pred_masked.mean().item())

    # Create DataFrame
    MBM_glwmb = pd.DataFrame(pred_gl, index=years, columns=["MBM Balance"])
    return MBM_glwmb


def plot_mbm_vs_geodetic_by_area_bin(
    df,
    bins=[0, 1, 5, 10, 100, np.inf],
    labels=["<1", "1–5", "5–10", "10–100", ">100"],
    max_bins=4,
    figsize=(25, 10),
    annotate_rmse=True,
    rmse_box=True,
    geodetic_sigma_col="Geodetic MB sigma",  # <-- NEW
    errorbar_alpha=0.5,  # <-- NEW
    errorbar_elinewidth=1.0,  # <-- NEW
    errorbar_capsize=2.0,  # <-- NEW
):
    """
    Scatter MBM MB vs observed geodetic MB by area bin, with x-error bars showing
    GLAMOS geodetic MB uncertainties (sigma). Expects a column named
    `geodetic_sigma_col` (default: "Geodetic MB sigma").
    """
    subplot_labels = ["(a)", "(b)", "(c)", "(d)"]
    df = df.copy().replace([np.inf, -np.inf], np.nan)

    # Ensure sigma column exists; if missing, use zeros (no errorbars)
    if geodetic_sigma_col not in df.columns:
        print(
            f"Warning: '{geodetic_sigma_col}' not found. Plotting without uncertainties."
        )
        df[geodetic_sigma_col] = 0.0

    df["Area_bin"] = pd.cut(
        df["Area"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
        ordered=True,
    )
    categories = list(df["Area_bin"].cat.categories)
    bins_in_use = [b for b in categories if (df["Area_bin"] == b).any()]
    if not bins_in_use:
        raise ValueError("No data fall into the specified area bins.")

    n_plots = min(max_bins, len(bins_in_use))
    fig, axs = plt.subplots(1, n_plots, figsize=figsize, sharex=True, sharey=True)
    if n_plots == 1:
        axs = np.array([axs])

    # Global limits – include x-uncertainty so error bars aren’t clipped
    mask_bins = df["Area_bin"].isin(bins_in_use[:n_plots])
    all_x = df.loc[mask_bins, "Geodetic MB"]
    all_y = df.loc[mask_bins, "MBM MB"]
    all_sig = df.loc[mask_bins, geodetic_sigma_col].fillna(0.0)

    valid = ~(all_x.isna() | all_y.isna())
    if valid.any():
        xmin = float((all_x[valid] - all_sig[valid]).min())
        xmax = float((all_x[valid] + all_sig[valid]).max())
        ymin = float(all_y[valid].min())
        ymax = float(all_y[valid].max())
        pad = 0.25
        vmin, vmax = min(xmin, ymin) - pad, max(xmax, ymax) + pad
    else:
        vmin, vmax = -1.0, 1.0

    for i, area_bin in enumerate(bins_in_use[:n_plots]):
        ax = axs[i]
        df_bin = (
            df[df["Area_bin"] == area_bin]
            .dropna(subset=["Geodetic MB", "MBM MB"])
            .copy()
        )
        df_bin["GLACIER"] = df_bin.get("GLACIER", pd.Series(index=df_bin.index)).apply(
            lambda x: x.capitalize() if isinstance(x, str) else x
        )

        # Scatter
        hue_kw = {}
        if "GLACIER" in df_bin.columns:
            hue_kw = dict(hue="GLACIER", style="GLACIER")

        sns.scatterplot(
            data=df_bin,
            x="Geodetic MB",
            y="MBM MB",
            alpha=0.85,
            ax=ax,
            s=250,
            palette=sns.color_palette(
                get_cmap_hex(
                    cm.batlow, 1 + df_bin.get("GLACIER", pd.Series()).nunique()
                )
            ),
            **hue_kw,
        )

        # X-error bars for geodetic uncertainties
        if geodetic_sigma_col in df_bin.columns:
            xvals = df_bin["Geodetic MB"].to_numpy(dtype=float)
            yvals = df_bin["MBM MB"].to_numpy(dtype=float)
            xerr = df_bin[geodetic_sigma_col].fillna(0.0).to_numpy(dtype=float)

            # Draw errorbars without adding legend entries
            ax.errorbar(
                xvals,
                yvals,
                xerr=xerr,
                yerr=None,
                fmt="none",
                ecolor="k",
                alpha=errorbar_alpha,
                elinewidth=errorbar_elinewidth,
                capsize=errorbar_capsize,
                capthick=errorbar_elinewidth,
                zorder=0,  # behind points
            )

        # Axes decorations
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.axvline(0, color="grey", linestyle="--", linewidth=1)
        ax.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax.axline((0, 0), slope=1, color="grey", linestyle="--", linewidth=1)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)

        ax.text(
            0.02,
            1,
            subplot_labels[i],
            transform=ax.transAxes,
            fontsize=24,
            va="top",
            ha="left",
        )

        ax.set_xlabel("")  # global x-label handled below
        ax.set_ylabel("Modelled MB [m w.e.]", fontsize=20)
        ax.set_title(f"Area: {area_bin} km²", fontsize=24)

        # RMSE + r
        n = len(df_bin)
        if annotate_rmse and n > 1:
            resid = df_bin["MBM MB"].to_numpy() - df_bin["Geodetic MB"].to_numpy()
            rmse = float(np.sqrt(np.nanmean(resid**2)))
            r, _ = pearsonr(df_bin["Geodetic MB"], df_bin["MBM MB"])
            box = (
                dict(facecolor="white", alpha=0.7, edgecolor="none")
                if rmse_box
                else None
            )
            ax.text(
                0.93,
                0.02,
                f"RMSE = {rmse:.2f}, r = {r:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=20,
                bbox=box,
            )
        elif annotate_rmse:
            box = (
                dict(facecolor="white", alpha=0.7, edgecolor="none")
                if rmse_box
                else None
            )
            ax.text(
                0.93,
                0.02,
                "No data",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=20,
                bbox=box,
            )

        # Legend handling
        if "GLACIER" in df_bin.columns:
            handles, labels_ = ax.get_legend_handles_labels()
            ax.legend(
                handles,
                labels_,
                loc="lower center",
                borderaxespad=0.5,
                fontsize=16,
                bbox_to_anchor=(0.5, -0.3),
                ncol=2,
            )
        else:
            if ax.legend_:
                ax.legend_.remove()

    fig.canvas.draw()  # need a renderer for legend extents

    # Global x-label placement respecting legends
    legend_tops = []
    for ax in np.atleast_1d(axs).ravel():
        leg = ax.get_legend()
        if leg is None:
            continue
        bb_fig = leg.get_window_extent(fig.canvas.get_renderer()).transformed(
            fig.transFigure.inverted()
        )
        legend_tops.append(bb_fig.y1)

    if legend_tops:
        top_of_legends = max(legend_tops)
        gap = 0.015
        y_xlabel = min(0.95, top_of_legends + gap)
        fig.supxlabel("Observed geodetic MB [m w.e.]", fontsize=20, y=y_xlabel)
        plt.subplots_adjust(bottom=0.12)
    else:
        fig.supxlabel("Observed geodetic MB [m w.e.]", fontsize=20, y=0.05)
        plt.subplots_adjust(bottom=0.18)

    plt.show()
    return fig


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
    cmap_ann, norm_ann, _, _ = get_color_maps(vmin, vmax, 0, 0)

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


# --- shared helpers ----------------------------------------------------------


def _aggregate_source(df, source_name):
    """
    Aggregate a single SOURCE (e.g., 'LSTM' or 'GLAMOS') per period:
      1) mean per (SOURCE, altitude_interval, YEAR)
      2) min/mean/max across YEARS

    Returns a DataFrame with columns:
      ['altitude_interval','mean_Ba','min_Ba','max_Ba']
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["altitude_interval", "mean_Ba", "min_Ba", "max_Ba"]
        )

    df = df[df["SOURCE"].str.upper() == source_name.upper()]
    if df.empty:
        return pd.DataFrame(
            columns=["altitude_interval", "mean_Ba", "min_Ba", "max_Ba"]
        )

    per_year = (
        df.groupby(["SOURCE", "altitude_interval", "YEAR"])["pred"].mean().reset_index()
    )
    agg = (
        per_year.groupby(["SOURCE", "altitude_interval"])["pred"]
        .agg(mean_Ba="mean", min_Ba="min", max_Ba="max")
        .reset_index()
    )
    return agg.drop(columns=["SOURCE"])


def _default_style(color_annual, color_winter):
    return {
        "annual": {"color": color_annual, "ls": "-", "label": "Annual"},
        "winter": {"color": color_winter, "ls": "-", "label": "Winter"},
    }


# --- function 1: LSTM bands + means -----------------------------------------


def plot_lstm_by_elevation_periods(
    df_all_a,
    df_all_w,
    ax=None,
    color_annual=color_annual,
    color_winter=color_winter,
    band_alpha=0.25,
    lw=1.2,
    mean_linestyle="-",
    label_prefix="LSTM",
    show_band=True,
):
    """
    Plot LSTM results for annual & winter.
      - If show_band=True: min/max band + mean
      - If show_band=False: mean only (no band), useful for overlays

    Expects df_all_a / df_all_w with columns:
      ['SOURCE','altitude_interval','YEAR','PERIOD','pred'] (including LSTM rows).
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 6))

    def _aggregate_source(df, source_name):
        if df is None or df.empty:
            return pd.DataFrame(
                columns=["altitude_interval", "mean_Ba", "min_Ba", "max_Ba"]
            )
        df = df[df["SOURCE"].str.upper() == source_name.upper()]
        if df.empty:
            return pd.DataFrame(
                columns=["altitude_interval", "mean_Ba", "min_Ba", "max_Ba"]
            )
        per_year = (
            df.groupby(["SOURCE", "altitude_interval", "YEAR"])["pred"]
            .mean()
            .reset_index()
        )
        agg = (
            per_year.groupby(["SOURCE", "altitude_interval"])["pred"]
            .agg(mean_Ba="mean", min_Ba="min", max_Ba="max")
            .reset_index()
            .drop(columns=["SOURCE"])
        )
        return agg

    agg_lstm_a = _aggregate_source(df_all_a, "LSTM")
    agg_lstm_w = _aggregate_source(df_all_w, "LSTM")

    # --- annual ---
    if not agg_lstm_a.empty:
        if show_band:
            ax.fill_betweenx(
                agg_lstm_a["altitude_interval"],
                agg_lstm_a["min_Ba"],
                agg_lstm_a["max_Ba"],
                color=color_annual,
                alpha=band_alpha,
                label=f"{label_prefix} band (annual)",
            )
        ax.plot(
            agg_lstm_a["mean_Ba"],
            agg_lstm_a["altitude_interval"],
            color=color_annual,
            linestyle=mean_linestyle,
            linewidth=lw,
            label=f"{label_prefix} mean (annual)",
        )

    # --- winter ---
    if not agg_lstm_w.empty:
        if show_band:
            ax.fill_betweenx(
                agg_lstm_w["altitude_interval"],
                agg_lstm_w["min_Ba"],
                agg_lstm_w["max_Ba"],
                color=color_winter,
                alpha=band_alpha,
                label=f"{label_prefix} band (winter)",
            )
        ax.plot(
            agg_lstm_w["mean_Ba"],
            agg_lstm_w["altitude_interval"],
            color=color_winter,
            linestyle=mean_linestyle,
            linewidth=lw,
            label=f"{label_prefix} mean (winter)",
        )

    return ax


# --- function 2: GLAMOS  -------------------------------------------


def plot_glamos_by_elevation_periods(
    df_all_a,
    df_all_w,
    ax=None,
    color_annual=color_annual,
    color_winter=color_winter,
    lw=1.3,
    mean_linestyle=":",
    label_prefix="GLAMOS",
    show_band=False,
    band_alpha=0.25,
):
    """
    Plot GLAMOS results for annual & winter.
      - If show_band=True: min/max band + mean
      - If show_band=False: mean only (no band, legacy behavior)

    Expects df_all_a / df_all_w with columns:
      ['SOURCE','altitude_interval','YEAR','PERIOD','pred'] (including GLAMOS rows).
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 6))

    def _aggregate_source(df, source_name):
        if df is None or df.empty:
            return pd.DataFrame(
                columns=["altitude_interval", "mean_Ba", "min_Ba", "max_Ba"]
            )
        df = df[df["SOURCE"].str.upper() == source_name.upper()]
        if df.empty:
            return pd.DataFrame(
                columns=["altitude_interval", "mean_Ba", "min_Ba", "max_Ba"]
            )
        per_year = (
            df.groupby(["SOURCE", "altitude_interval", "YEAR"])["pred"]
            .mean()
            .reset_index()
        )
        agg = (
            per_year.groupby(["SOURCE", "altitude_interval"])["pred"]
            .agg(mean_Ba="mean", min_Ba="min", max_Ba="max")
            .reset_index()
            .drop(columns=["SOURCE"])
        )
        return agg

    agg_glam_a = _aggregate_source(df_all_a, "GLAMOS")
    agg_glam_w = _aggregate_source(df_all_w, "GLAMOS")

    # --- annual ---
    if not agg_glam_a.empty:
        if show_band:
            ax.fill_betweenx(
                agg_glam_a["altitude_interval"],
                agg_glam_a["min_Ba"],
                agg_glam_a["max_Ba"],
                color=color_annual,
                alpha=band_alpha,
                label=f"{label_prefix} band (annual)",
            )
        ax.plot(
            agg_glam_a["mean_Ba"],
            agg_glam_a["altitude_interval"],
            color=color_annual,
            linestyle=mean_linestyle,
            linewidth=lw,
            label=f"{label_prefix} mean (annual)",
        )

    # --- winter ---
    if not agg_glam_w.empty:
        if show_band:
            ax.fill_betweenx(
                agg_glam_w["altitude_interval"],
                agg_glam_w["min_Ba"],
                agg_glam_w["max_Ba"],
                color=color_winter,
                alpha=band_alpha,
                label=f"{label_prefix} band (winter)",
            )
        ax.plot(
            agg_glam_w["mean_Ba"],
            agg_glam_w["altitude_interval"],
            color=color_winter,
            linestyle=mean_linestyle,
            linewidth=lw,
            label=f"{label_prefix} mean (winter)",
        )

    return ax


# --- optional: stakes helper -------------------------------------------------


def plot_stakes_by_elevation_periods(
    df_stakes,
    glacier_name,
    valid_bins=None,
    ax=None,
    color_annual="#1f77b4",
    color_winter="#ff7f0e",
    marker_size=14,
):
    """
    Plot observed stake means per elevation bin for annual & winter.

    Parameters
    ----------
    df_stakes : DataFrame with columns
        ['GLACIER','PERIOD','YEAR','POINT_ELEVATION','POINT_BALANCE'].
    glacier_name : str
    valid_bins : set or None
        If provided, only plot bins present in model outputs.
    ax : matplotlib.axes.Axes or None
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 6))

    style = _default_style(color_annual, color_winter)

    def bin_center_100m(z):
        left = np.floor(z / 100.0) * 100.0
        return left + 50.0

    stakes = df_stakes[df_stakes["GLACIER"] == glacier_name].copy()
    if stakes.empty:
        return ax

    stakes["altitude_interval"] = bin_center_100m(stakes["POINT_ELEVATION"])

    if valid_bins is not None:
        stakes = stakes[stakes["altitude_interval"].isin(valid_bins)]
        if stakes.empty:
            return ax

    for period_key, tag in (("annual", "annual"), ("winter", "winter")):
        ssub = stakes[stakes["PERIOD"].str.lower() == tag]
        if ssub.empty:
            continue
        sagg = (
            ssub.groupby("altitude_interval", as_index=False)["POINT_BALANCE"]
            .mean()
            .rename(columns={"POINT_BALANCE": "mean_obs_Ba"})
            .sort_values("altitude_interval")
        )
        ax.scatter(
            sagg["mean_obs_Ba"],
            sagg["altitude_interval"],
            s=marker_size,
            marker="o" if tag == "annual" else "s",
            facecolors="none",
            edgecolors=style[period_key]["color"],
            linewidths=0.9,
            label=f"Stakes mean ({tag})",
        )

    return ax


# --- optional: wrapper to reproduce original plot ----------------------------


def plot_mb_by_elevation_periods_combined(
    df_all_a,
    df_all_w,
    df_stakes,
    glacier_name,
    ax=None,
    color_annual=color_annual,
    color_winter=color_winter,
):
    """
    Reproduces the original plot: LSTM (band+mean), GLAMOS (mean only),
    and stakes, for annual & winter.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 6))

    # Draw LSTM + GLAMOS
    plot_lstm_by_elevation_periods(
        df_all_a, df_all_w, ax=ax, color_annual=color_annual, color_winter=color_winter
    )
    plot_glamos_by_elevation_periods(
        df_all_a, df_all_w, ax=ax, color_annual=color_annual, color_winter=color_winter
    )

    # Compute valid bins from whatever was plotted (union of bins present)
    bins = set()
    for df in (df_all_a, df_all_w):
        if df is None or df.empty:
            continue
        if "altitude_interval" in df.columns:
            bins.update(pd.unique(df["altitude_interval"].dropna()))

    plot_stakes_by_elevation_periods(
        df_stakes,
        glacier_name,
        valid_bins=bins,
        ax=ax,
        color_annual=color_annual,
        color_winter=color_winter,
    )

    # cosmetics (same as before)
    ax.set_ylabel("Elevation bin (m a.s.l.)")
    ax.set_xlabel("Mass balance (m w.e.)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="best", fontsize=8)

    return ax


def plot_2glaciers_2years_glamos_vs_lstm(
    glacier_names,  # e.g. ("aletsch", "silvretta")
    years_by_glacier,  # e.g. ((2016, 2022), (2014, 2021))  <-- NEW
    cfg,
    df_stakes=None,
    path_distributed_mb=None,  # GLAMOS grids base
    path_pred_lstm=None,  # LSTM zarrs base
    period="annual",
    apply_smoothing_fn=None,  # optional callable(ds)->ds
):
    """
    Layout (2 rows × 4 panels) with one colorbar per glacier–year (outside maps):

      Row 1 (glacier_names[0]): G1-GLAMOS(y1a), G1-LSTM(y1a), |cbar|, G1-GLAMOS(y1b), G1-LSTM(y1b), |cbar|
      Row 2 (glacier_names[1]): G2-GLAMOS(y2a), G2-LSTM(y2a), |cbar|, G2-GLAMOS(y2b), G2-LSTM(y2b), |cbar|

    Each row has its own pair of years from `years_by_glacier[r]`, allowing different years per glacier.
    Annotations (lower-left of each panel): RMSE (vs stakes), mean MB, variance.
    """

    assert len(glacier_names) == 2, "glacier_names must have length 2"
    assert len(years_by_glacier) == 2 and all(
        len(p) == 2 for p in years_by_glacier
    ), "years_by_glacier must be ((y1a,y1b),(y2a,y2b))"
    assert path_distributed_mb and path_pred_lstm

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

    def lonlat_names(obj):
        coords = getattr(obj, "coords", {})
        if "lon" in coords and "lat" in coords:
            return "lon", "lat"
        if "longitude" in coords and "latitude" in coords:
            return "longitude", "latitude"
        return "lon", "lat"

    def stake_overlay_rmse(ax, glacier, year, cmap, norm, da_glamos, ds_lstm, which):
        """Overlay stakes & return RMSE for 'GLAMOS' or 'LSTM' vs POINT_BALANCE."""
        if df_stakes is None:
            return None
        sub = df_stakes[
            (df_stakes.GLACIER == glacier) & (df_stakes.YEAR == year)
        ].copy()
        if period == "annual" and "PERIOD" in sub.columns:
            sub = sub[sub.PERIOD == "annual"].copy()
        if sub.empty:
            return None

        lx, ly = lonlat_names(
            ds_lstm if which == "LSTM" and ds_lstm is not None else da_glamos
        )

        def _safe_pred(ds, row):
            try:
                return get_predicted_mb(lx, ly, row, ds)
            except Exception:
                return np.nan

        def _safe_glamos(row):
            try:
                return get_predicted_mb_glamos(lx, ly, row, da_glamos)
            except Exception:
                return np.nan

        if which == "GLAMOS":
            sub["FIELD"] = sub.apply(_safe_glamos, axis=1)
        else:
            sub["FIELD"] = (
                sub.apply(lambda r: _safe_pred(ds_lstm, r), axis=1)
                if ds_lstm is not None
                else np.nan
            )

        hue_col = "POINT_BALANCE" if "POINT_BALANCE" in sub.columns else "FIELD"
        sns.scatterplot(
            data=sub,
            x="POINT_LON",
            y="POINT_LAT",
            hue=hue_col,
            palette=cmap,
            hue_norm=norm,
            ax=ax,
            s=18,
            legend=False,
        )

        if "POINT_BALANCE" in sub.columns and not np.all(np.isnan(sub["FIELD"])):
            return root_mean_squared_error(sub["POINT_BALANCE"], sub["FIELD"])
        return None

    # ---------- figure & gridspec (2 rows × 6 columns with CB slots) ----------
    fig = plt.figure(figsize=(28, 15))
    gs = GridSpec(
        nrows=2,
        ncols=6,
        figure=fig,
        width_ratios=[1, 1, 0.045, 1, 1, 0.045],  # slim cbar columns
        wspace=0.30,
        hspace=0.12,
    )

    # Keep the first axis per row to share y across the row
    first_ax_in_row = [None, None]

    for r, glacier in enumerate(glacier_names):
        row_years = years_by_glacier[r]  # (y_a, y_b) for this glacier/row
        for j, year in enumerate(row_years):
            col_base = 3 * j  # (0 or 3)
            # sharey with the first axis in this row
            ax_g = fig.add_subplot(gs[r, col_base + 0], sharey=first_ax_in_row[r])
            if first_ax_in_row[r] is None:
                first_ax_in_row[r] = (
                    ax_g  # set after creation of the first GLAMOS axis in row
                )
            ax_m = fig.add_subplot(gs[r, col_base + 1], sharey=first_ax_in_row[r])
            ax_cb = fig.add_subplot(gs[r, col_base + 2])

            # load data
            da_g = load_glamos_wgs84(glacier, year)
            ds_m = load_lstm_ds(glacier, year)

            # compute pair vmin/vmax
            vals = []
            if da_g is not None:
                vals += [float(da_g.min().item()), float(da_g.max().item())]
            if ds_m is not None and "pred_masked" in ds_m:
                vals += [
                    float(ds_m["pred_masked"].min().item()),
                    float(ds_m["pred_masked"].max().item()),
                ]
            if not vals:
                for ax in (ax_g, ax_m):
                    ax.text(
                        0.5, 0.5, f"No data\n{glacier} {year}", ha="center", va="center"
                    )
                    ax.set_axis_off()
                ax_cb.set_axis_off()
                continue

            vmin, vmax = min(vals), max(vals)
            cmap, norm, _, _ = get_color_maps(vmin, vmax, 0, 0)

            # --- GLAMOS panel ---
            mappable_g = None
            if da_g is None:
                ax_g.text(
                    0.5, 0.5, f"No GLAMOS\n{glacier} {year}", ha="center", va="center"
                )
                ax_g.set_axis_off()
            else:
                mappable_g = da_g.plot.imshow(
                    ax=ax_g, cmap=cmap, norm=norm, add_colorbar=False
                )
                ax_g.set_title(f"{glacier.capitalize()} – GLAMOS ({year})", fontsize=16)

                mean_g = float(da_g.mean().item())
                var_g = float(da_g.var().item())
                rmse_g = stake_overlay_rmse(
                    ax_g, glacier, year, cmap, norm, da_g, ds_m, which="GLAMOS"
                )
                text_g = (
                    f"RMSE: {rmse_g:.2f}\n" if rmse_g is not None else ""
                ) + f"mean MB: {mean_g:.2f}\nvar: {var_g:.2f}"
                ax_g.text(
                    0.03,
                    0.03,
                    text_g,
                    transform=ax_g.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )

            # --- LSTM panel ---
            mappable_m = None
            if ds_m is None or "pred_masked" not in ds_m:
                ax_m.text(
                    0.5, 0.5, f"No LSTM\n{glacier} {year}", ha="center", va="center"
                )
                ax_m.set_axis_off()
            else:
                mappable_m = ds_m["pred_masked"].plot.imshow(
                    ax=ax_m, cmap=cmap, norm=norm, add_colorbar=False
                )
                ax_m.set_title(f"{glacier.capitalize()} – LSTM ({year})", fontsize=16)

                mean_m = float(ds_m["pred_masked"].mean().item())
                var_m = float(ds_m["pred_masked"].var().item())
                rmse_m = stake_overlay_rmse(
                    ax_m, glacier, year, cmap, norm, da_g, ds_m, which="LSTM"
                )
                text_m = (
                    f"RMSE: {rmse_m:.2f}\n" if rmse_m is not None else ""
                ) + f"mean MB: {mean_m:.2f}\nvar: {var_m:.2f}"
                ax_m.text(
                    0.03,
                    0.03,
                    text_m,
                    transform=ax_m.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )

            # --- shared colorbar outside (for this glacier–year pair) ---
            pair_mappable = mappable_m or mappable_g
            if pair_mappable is not None:
                cb = fig.colorbar(pair_mappable, cax=ax_cb)
                cb.set_label("Mass Balance [m w.e.]", fontsize=16)

            # ---- tidy y labels: only leftmost panel shows them ----
            if j == 0:
                ax_g.set_ylabel("Latitude")
                ax_m.tick_params(labelleft=False)
                ax_m.set_ylabel("")
            else:
                ax_g.tick_params(labelleft=False)
                ax_m.tick_params(labelleft=False)
                ax_g.set_ylabel("")
                ax_m.set_ylabel("")
            # Only bottom row shows x labels
            if r == 0:
                ax_g.tick_params(labelbottom=False)
                ax_m.tick_params(labelbottom=False)
            else:
                ax_g.set_xlabel("Longitude")
                ax_m.set_xlabel("Longitude")

    plt.tight_layout()
    plt.show()

    return fig


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


def prepare_monthly_long_df(df_lstm, df_nn, df_glamos_w, df_glamos_a, month_order=None):
    """
    Convert LSTM, NN, and GLAMOS glacier-month DataFrames into a long-format DataFrame
    for plotting.
    GLAMOS winter contains April data, GLAMOS annual contains September data.

    Parameters
    ----------
    df_lstm : pd.DataFrame
        Monthly LSTM results with columns ['glacier', 'year', ...months...].
    df_nn : pd.DataFrame
        Monthly NN results with columns ['glacier', 'year', ...months...].
    df_glamos_w : pd.DataFrame
        GLAMOS winter data with ['glacier', 'year', 'apr'].
    df_glamos_a : pd.DataFrame
        GLAMOS annual data with ['glacier', 'year', 'sep'].
    month_order : list, optional
        Ordered list of months in hydrological order (default: Oct–Sep).

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        ['glacier', 'year', 'Month', 'mb_nn', 'mb_lstm', 'mb_glamos'].
        'mb_glamos' is NaN for all months except April (winter) and September (annual).
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

    common_cols = ["glacier", "year"]

    # Keep only glaciers/years that exist in at least one GLAMOS dataset
    valid_pairs = pd.concat(
        [df_glamos_w[common_cols], df_glamos_a[common_cols]], ignore_index=True
    ).drop_duplicates()

    df_lstm = df_lstm.merge(valid_pairs, on=common_cols, how="inner")
    df_nn = df_nn.merge(valid_pairs, on=common_cols, how="inner")

    # --- Prepare long-format arrays ---
    array_nn, array_lstm, months, glaciers, years = [], [], [], [], []

    for col in month_order:
        array_nn.append(df_nn[col].values)
        array_lstm.append(df_lstm[col].values)
        months.append(np.tile(col, len(df_nn)))
        glaciers.append(df_nn["glacier"].values)
        years.append(df_nn["year"].values)

    df_long = pd.DataFrame(
        {
            "glacier": np.concatenate(glaciers),
            "year": np.concatenate(years),
            "mb_nn": np.concatenate(array_nn),
            "mb_lstm": np.concatenate(array_lstm),
            "Month": np.concatenate(months),
        }
    )

    # ---- Add a single mb_glamos column ----
    df_long["mb_glamos"] = np.nan

    # Merge winter (April)
    if "apr" in df_glamos_w.columns:
        apr_mask = df_long["Month"] == "apr"
        df_apr = df_long.loc[apr_mask, ["glacier", "year"]].merge(
            df_glamos_w[["glacier", "year", "apr"]], on=["glacier", "year"], how="left"
        )
        df_long.loc[apr_mask, "mb_glamos"] = df_apr["apr"].values

    # Merge annual (September)
    if "sep" in df_glamos_a.columns:
        sep_mask = df_long["Month"] == "sep"
        df_sep = df_long.loc[sep_mask, ["glacier", "year"]].merge(
            df_glamos_a[["glacier", "year", "sep"]], on=["glacier", "year"], how="left"
        )
        df_long.loc[sep_mask, "mb_glamos"] = df_sep["sep"].values

    # ---- Order months ----
    cat_month = CategoricalDtype(month_order, ordered=True)
    df_long["Month"] = df_long["Month"].astype(cat_month)

    return df_long


def plot_monthly_joyplot(
    df_long,
    month_order=None,
    color_annual="tab:blue",
    color_winter="tab:orange",
    figsize_cm=(12, 14),
    x_range=(-2.2, 2.2),
    alpha=1,
):
    """
    Plot a JoyPy ridge plot comparing LSTM and NN monthly MB distributions.

    Parameters
    ----------
    df_long : pd.DataFrame
        Long-format dataframe from `prepare_monthly_long_df`.
    month_order : list, optional
        Ordered list of months (default: hydrological year Oct–Sep).
    color_annual, color_winter : str
        Colors for LSTM and NN models respectively.
    figsize_cm : tuple
        Figure size in centimeters.
    x_range : tuple
        x-axis range for MB (m w.e.).
    alpha : float
        Transparency for lines and legend patches.
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
    model_colors = [color_annual, color_winter, "gray"]

    fig, ax = joypy.joyplot(
        df_long,
        by="Month",
        column=["mb_lstm", "mb_nn", "mb_glamos"],
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

    plt.axvline(x=0, color="grey", alpha=0.5, linewidth=1)
    plt.xlabel("Mass balance (m w.e.)", fontsize=8.5)
    plt.yticks(ticks=range(1, 13), labels=month_order, fontsize=8.5)
    plt.gca().set_yticklabels(month_order)

    legend_patches = [
        Patch(facecolor=color, label=model, alpha=alpha, edgecolor="k")
        for model, color in zip(["LSTM", "NN", "GLAMOS"], model_colors)
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

    # plt.tight_layout()
    plt.show()
    return fig
