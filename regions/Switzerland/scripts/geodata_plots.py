import matplotlib.pyplot as plt
import numpy as np
from cartopy import crs as ccrs, feature as cfeature
import os
from os import listdir
from os.path import isfile, join
import re
from matplotlib.colors import to_hex
import seaborn as sns
import geopandas as gpd
import contextily as cx
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error
from scripts.geodata import *
import massbalancemachine as mbm

def plotClasses(gdf_glacier,
                gdf_S2_res,
                axs,
                gl_date,
                file_date,
                add_snowline=False,
                band_size=10,
                percentage_threshold=50):

    # Define the colors for categories (ensure that your categories match the color list)
    colors_cat = ['#a6cee3', '#1f78b4', '#8da0cb', '#b2df8a', '#fb9a99']

    # Manually map categories to colors (assuming categories 0-5 for example)
    classes = {
        1.0: 'snow',
        3.0: 'clean ice',
        2.0: 'firn / old snow / bright ice',
        4.0: 'debris',
        5.0: 'cloud'
    }
    map = dict(
        zip(classes.keys(),
            colors_cat[:6]))  # Adjust according to the number of categories

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
        missing_kwds={"color": "lightgrey"}  # Define color for NaN datas
    )
    # make colorbar small
    # cbar = axs[0].get_figure().get_axes()[1]
    # cbar.set_ylabel("Mass balance [m w.e.]", fontsize=12)   
    #cx.add_basemap(axs[0], crs=gdf_glacier.crs, source=provider)
    axs[0].set_title(f"Mass balance: {gl_date}")

    # Plot the second figure (MBM classes)
    gdf_clean = gdf_glacier.dropna(subset=["classes"])
    gdf_clean['color'] = gdf_clean['classes'].map(map)
    # Plot with manually defined colormap
    gdf_clean.plot(
        column="classes",  # Column to visualize
        legend=True,  # Display a legend
        markersize=5,  # Adjust size if points are too small or large
        missing_kwds={"color": "lightgrey"},  # Define color for NaN datas
        categorical=True,  # Ensure the plot uses categorical colors
        ax=axs[1],
        color=gdf_clean['color']  # Use the custom colormap
    )

    # calculate snow and ice cover
    snow_cover_glacier = IceSnowCover(gdf_glacier, gdf_S2_res)
    AddSnowCover(snow_cover_glacier, axs[1])

    #cx.add_basemap(axs[1], crs=gdf_glacier.crs, source=provider)
    axs[1].set_title(f"MBM: {gl_date}")

    if add_snowline:
        # Overlay the selected band (where 'selected_band' is True)
        selected_band = gdf_clean[gdf_clean['selected_band'] == True]
        # Plot the selected elevation band with a distinct style (e.g., red border)
        selected_band.plot(ax=axs[1], color='red', linewidth=1, markersize=5, alpha = 0.5)

    # Plot the fourth figure (Resampled Sentinel classes)
    gdf_clean = gdf_S2_res.dropna(subset=["classes"])
    gdf_clean['color'] = gdf_clean['classes'].map(map)
    # Plot with manually defined colormap
    gdf_clean.plot(
        column="classes",  # Column to visualize
        legend=True,  # Display a legend
        markersize=5,  # Adjust size if points are too small or large
        missing_kwds={"color": "lightgrey"},  # Define color for NaN datas
        categorical=True,  # Ensure the plot uses categorical colors
        ax=axs[2],
        color=gdf_clean['color']  # Use the custom colormap
    )
    # calculate snow and ice cover
    snow_cover_glacier = IceSnowCover(gdf_S2_res, gdf_S2_res)
    AddSnowCover(snow_cover_glacier, axs[2])
    #cx.add_basemap(axs[2], crs=gdf_glacier.crs, source=provider)
    axs[2].set_title(f"Sentinel: {file_date.strftime('%Y-%m-%d')}")
    
    # Manually add custom legend for the third plot
    handles = [
        mpatches.Patch(color=color, label=classes[i])
        for i, color in map.items()
    ]
    axs[2].legend(handles=handles,
                  title="Classes",
                  bbox_to_anchor=(1.05, 1),
                  loc='upper left')
    
    # Show the plot with consistent colors
    # plt.tight_layout()
    plt.show()


def AddSnowCover(snow_cover_glacier, ax):
    # Custom legend for snow and ice cover
    legend_labels = "\n".join(
        ((f"Snow cover: {snow_cover_glacier*100:.2f}%"), ))
    #    (f"Ice cover: {ice_cover_glacier*100:.2f}%")))

    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.03,
            0.08,
            legend_labels,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=12,
            bbox=props)


def plot_snow_cover_scatter(df, add_corr = True):
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
    N_months = len(df['month'].unique())

    # Create a grid of subplots
    if add_corr:
        fig, axs = plt.subplots(2, N_months, figsize=(15, 8), squeeze=False)
    else:
        fig, axs = plt.subplots(1, N_months, figsize=(15, 4), squeeze=False)
    
    # Get sorted unique months
    months = np.sort(df['monthNb'].unique())

    # Loop over each month
    for i, monthNb in enumerate(months):
        # Subset data for the current month
        df_month = df[df['monthNb'] == monthNb]

        # Left column: scatter plot of snow cover
        ax = axs[0, i]
        sns.scatterplot(data=df_month,
                        x='snow_cover_S2',
                        y='snow_cover_glacier',
                        marker='o',
                        hue='glacier_name',
                        ax=ax)
        x = np.linspace(0, 1, 100)
        ax.plot(x, x, 'k--')  # Identity line
        
        # Calculate and add R^2 value
        r2 = np.corrcoef(df_month['snow_cover_S2'], df_month['snow_cover_glacier'])[0, 1]**2
        mse = mean_squared_error(df_month['snow_cover_glacier'], df_month['snow_cover_S2'])
        ax.text(0.05, 0.85, f"R² = {r2:.2f}\nMSE = {mse:.2f}", transform=ax.transAxes, fontsize=10, color="black")
        
        ax.set_xlabel('Sentinel-2')
        ax.set_ylabel('Mass Balance Machine')
        ax.set_title(f'Snow Cover (Normal), {df_month["month"].values[0]}')
        ax.get_legend().remove()  # Remove legend

        
        if add_corr:
            # Right column: scatter plot of corrected snow cover
            ax = axs[1, i]
            sns.scatterplot(data=df_month,
                            x='snow_cover_S2',
                            y='snow_cover_glacier_corr',
                            marker='o',
                            hue='glacier_name',
                            ax=ax)
            ax.plot(x, x, 'k--')  # Identity line
            
            # Calculate and add R^2 value
            r2_corr = np.corrcoef(df_month['snow_cover_S2'], df_month['snow_cover_glacier_corr'])[0, 1]**2
            mse_corr = mean_squared_error(df_month['snow_cover_glacier_corr'], df_month['snow_cover_S2'])
            ax.text(0.05, 0.85, f"R² = {r2_corr:.2f}\nMSE = {mse_corr:.2f}", transform=ax.transAxes, fontsize=10, color="black")

            ax.set_xlabel('Sentinel-2')
            ax.set_ylabel('Mass Balance Machine')
            ax.set_title(f'Snow Cover (Corrected), {df_month["month"].values[0]}')
            ax.get_legend().remove()  # Remove legend

    # Add a single legend underneath the last row of axes
    if add_corr:
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles,
               labels,
               loc='lower center',
               ncol=5,
               bbox_to_anchor=(0.5, -0.05),
               title="Glacier Name")

        # Adjust layout for better spacing
        plt.tight_layout(rect=[0, 0.08, 1,
                           1])  # Leave space at the bottom for the legend

    return fig, axs



def plot_snow_cover_geoplots(raster_res,
                             path_S2,
                             month_abbr_hydr,
                             add_snowline=False,
                             band_size=50,
                             percentage_threshold=50):
    """
    Plot geoplots of snow cover for a given raster file.

    Parameters:
    - raster_res (str): The name of the raster file to process.
    - path_S2 (str): Path to the directory containing the satellite rasters.
    - get_hydro_year_and_month (function): Function to determine the hydrological year and month from a date.
    - month_abbr_hydr (dict): Mapping of hydrological months to their abbreviated names.
    - IceSnowCover (function): Function to calculate snow and ice cover from a GeoDataFrame.
    - snowCover (function): Function to load mass-balance predictions and calculate snow cover corrections.
    - plotClasses (function): Function to create the plots.
    """
    # Extract glacier name
    glacierName = raster_res.split('_')[0]

    # Extract date from satellite raster
    match = re.search(r"(\d{4})_(\d{2})_(\d{2})", raster_res)
    if not match:
        raise ValueError(f"Invalid raster filename format: {raster_res}")

    year, month, day = match.groups()
    date_str = f"{year}-{month}-{day}"
    raster_date = datetime.strptime(date_str, "%Y-%m-%d")

    # Find closest hydrological year and month
    closest_month, hydro_year = get_hydro_year_and_month(raster_date)
    monthNb = month_abbr_hydr[closest_month]

    # Skip if the hydrological year is out of range
    if hydro_year > 2021:
        return

    # Read satellite raster over glacier
    raster_path = os.path.join(path_S2, 'perglacier', raster_res)
    gdf_S2_res = gpd.read_file(raster_path)

    # Load MB predictions for that year and month
    path_nc_wgs84 = f"results/nc/sgi/{glacierName}/"
    filename_nc = f"{glacierName}_{hydro_year}_{monthNb}.nc"

    # Calculate snow and ice cover
    geoData_gl = mbm.GeoData(pd.DataFrame)
    geoData_gl.set_ds_latlon(filename_nc, path_nc_wgs84)
    geoData_gl.classify_snow_cover(tol=0.1)
    gdf_glacier = geoData_gl.gdf

    # Plot the results
    gl_date = f"{hydro_year}-{closest_month}"
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    plotClasses(gdf_glacier,
                gdf_S2_res,
                axs,
                gl_date,
                raster_date,
                add_snowline,
                band_size=band_size,
                percentage_threshold=percentage_threshold)
    plt.show()


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
    sns.scatterplot(data=df,
                    x='snow_cover_S2',
                    y='snow_cover_glacier',
                    marker='o',
                    style='month',
                    ax=ax,
                    s=200)
    x = np.linspace(0, 1, 100)
    ax.plot(x, x, 'k--')  # Identity line
    ax.set_xlabel('Sentinel-2', fontsize=14)
    ax.set_ylabel('Mass Balance Machine', fontsize=14)
    ax.set_title('Snow Cover (Normal)', fontsize=16)
    ax.get_legend().remove()  # Remove legend for now

    r2 = np.corrcoef(df['snow_cover_S2'], df['snow_cover_glacier'])[0, 1]**2
    ax.text(0.05, 0.9, f"R² = {r2:.2f}", transform=ax.transAxes, fontsize=16, color="black")
 
    # # Second subplot: Corrected snow cover
    # ax = axs[1]
    # sns.scatterplot(data=df,
    #                 x='snow_cover_S2',
    #                 y='snow_cover_glacier_corr',
    #                 marker='o',
    #                 style='month',
    #                 ax=ax,
    #                 s=200)
    # ax.plot(x, x, 'k--')  # Identity line
    # ax.set_xlabel('Sentinel-2', fontsize=14)
    # ax.set_ylabel('Mass Balance Machine', fontsize=14)
    # ax.set_title('Snow Cover (Corrected)', fontsize=16)
    # ax.get_legend().remove()  # Remove legend for now
    # r2 = np.corrcoef(df['snow_cover_S2'], df['snow_cover_glacier_corr'])[0, 1]**2
    # ax.text(0.05, 0.9, f"R² = {r2:.2f}", transform=ax.transAxes, fontsize=16, color="black")

    # Add a single legend to the right of the plots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='center right',
        bbox_to_anchor=(1.02, 0.8),  # Move legend to the side
        title="Glacier Month",
        fontsize=16,
        title_fontsize=14)

    # Adjust layout for better spacing
    plt.suptitle(df.glacier_name.unique()[0].capitalize(),
                 fontsize=20,
                 fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9,
                           1])  # Leave space on the right for the legend
    return fig, axs
