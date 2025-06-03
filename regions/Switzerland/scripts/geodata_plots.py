import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.colors as mcolors
import massbalancemachine as mbm
import matplotlib.patches as mpatches

from scripts.geodata import *


def plot_geodetic_MB(df, glacier_name, color_xgb='blue', color_tim='red'):
    # # Convert lists to NumPy arrays for filtering NaNs
    # geodetic_mb = np.array(geodetic_mb)
    # mbm_mb_mean = np.array(mbm_mb_mean)
    # glamos_mb_mean = np.array(glamos_mb_mean)

    # # Remove NaN values to avoid plotting errors
    # valid_idx = ~np.isnan(geodetic_mb) & ~np.isnan(mbm_mb_mean) & ~np.isnan(
    #     glamos_mb_mean)
    # geodetic_mb, mbm_mb_mean, glamos_mb_mean = geodetic_mb[
    #     valid_idx], mbm_mb_mean[valid_idx], glamos_mb_mean[valid_idx]

    df = df.dropna(subset=['geodetic_mb', 'mbm_mb_mean', 'glamos_mb_mean'])
    
    # Ensure data exists before plotting
    if len(df) == 0:
        print("No valid data points to plot.")
    else:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 5))

        # Scatter plot
        sns.scatterplot(df, x = 'geodetic_mb',
                   y = 'mbm_mb_mean',
                   color=color_xgb,
                #    hue = 'end_year',
                  alpha=0.7,
                   label="MBM MB",
                   marker="o")
        sns.scatterplot(df, x = 'geodetic_mb',
                   y = 'glamos_mb_mean',
                   color=color_tim,
                   alpha=0.7,
                   label="GLAMOS MB",
                   marker="s")

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


def scatter_geodetic_MB(df_all, hue='GLACIER', size=False):
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
    df_all = df_all.dropna(subset=['Geodetic MB', 'MBM MB', 'GLAMOS MB'])

    # Compute RMSE and Pearson correlation
    rmse_mbm = mean_squared_error(df_all["Geodetic MB"],
                                  df_all["MBM MB"],
                                  squared=False)
    corr_mbm = np.corrcoef(df_all["Geodetic MB"], df_all["MBM MB"])[0, 1]
    rmse_glamos = mean_squared_error(df_all["Geodetic MB"],
                                     df_all["GLAMOS MB"],
                                     squared=False)
    corr_glamos = np.corrcoef(df_all["Geodetic MB"], df_all["GLAMOS MB"])[0, 1]

    # Define figure and axes
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)


    # Plot MBM MB vs Geodetic MB
    plot_scatter(df_all, hue, size, axs[0], "MBM MB", "Geodetic vs MBM MB", rmse_mbm, corr_mbm)

    # Plot GLAMOS MB vs Geodetic MB
    plot_scatter(df_all, hue, size, axs[1], "GLAMOS MB", "Geodetic vs GLAMOS MB", rmse_glamos,
                 corr_glamos)

    # Adjust legend outside of plot
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles,
               labels,
               bbox_to_anchor=(1.05, 1),
               loc="upper left",
               borderaxespad=0.,
               ncol=2,
               fontsize=14)

    plt.tight_layout()
    plt.show()
    
def plot_scatter(df_all, hue, size, ax, y_col, rmse, corr):
    """ Helper function to plot a scatter plot with annotations """
    sns.scatterplot(data=df_all,
                    x="Geodetic MB",
                    y=y_col,
                    hue=hue,
                    size="Area" if size else None,
                    sizes=(10, 1000),
                    alpha=0.7,
                    ax=ax)

    # Identity line (diagonal y=x)
    # diagonal line
    pt = (0, 0)
    ax.axline(pt, slope=1, color="grey", linestyle="--", linewidth=1)

    # Grid and axis labels
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.set_xlabel("Geodetic MB [m w.e.]")
    ax.set_ylabel(f"{y_col} [m w.e.]")

    # RMSE and correlation annotation
    legend_text = "\n".join(
        (r"$\mathrm{RMSE}=%.3f$" % rmse, r"$\mathrm{\rho}=%.3f$" % corr))
    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    ax.text(0.03,
            0.94,
            legend_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=18,
            bbox=props)
    ax.legend([], [], frameon=False)


def plot_mass_balance(glacier_name, year, df_stakes,
                      path_distributed_MB_glamos, PATH_PREDICTIONS):
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

    print(f"Processing: {glacier_name}, Year: {year}")

    # Extract stake data for this glacier and year
    stakes_data = df_stakes[(df_stakes.GLACIER == glacier_name)
                            & (df_stakes.YEAR == year)]
    stakes_data_ann = stakes_data[stakes_data.PERIOD == "annual"].copy()
    stakes_data_win = stakes_data[stakes_data.PERIOD == "winter"].copy()

    # Construct file paths
    file_ann = f"{year}_ann_fix_lv95.grid"
    file_win = f"{year}_win_fix_lv95.grid"
    grid_path_ann = os.path.join(path_distributed_MB_glamos, glacier_name,
                                 file_ann)
    grid_path_win = os.path.join(path_distributed_MB_glamos, glacier_name,
                                 file_win)

    # Load GLAMOS data (Annual)
    # if exists
    if not os.path.exists(grid_path_ann):
        print(f"Skipping {year}: GLAMOS file missing for {glacier_name}.")
        return None
    metadata_ann, grid_data_ann = load_grid_file(grid_path_ann)
    ds_glamos_ann = convert_to_xarray_geodata(grid_data_ann, metadata_ann)
    ds_glamos_wgs84_ann = transform_xarray_coords_lv95_to_wgs84(ds_glamos_ann)
    print(ds_glamos_wgs84_ann.mean().values)
    
    # Load GLAMOS data (Winter)
    try:
        metadata_win, grid_data_win = load_grid_file(grid_path_win)
        ds_glamos_win = convert_to_xarray_geodata(grid_data_win, metadata_win)
        ds_glamos_wgs84_win = transform_xarray_coords_lv95_to_wgs84(
            ds_glamos_win)
    except FileNotFoundError:
        print(
            f"Warning: Winter grid data missing for {glacier_name} ({year}).")
        ds_glamos_wgs84_win = None

    # Load MBM predictions (Annual)
    mbm_file_ann = os.path.join(PATH_PREDICTIONS, glacier_name,
                                f"{glacier_name}_{year}_annual.zarr")
    ds_mbm_ann = xr.open_dataset(mbm_file_ann)
    ds_mbm_ann = apply_gaussian_filter(ds_mbm_ann)
    print(ds_mbm_ann.pred_masked.mean().item())

    # Load MBM predictions (Winter)
    mbm_file_win = os.path.join(PATH_PREDICTIONS, glacier_name,
                                f"{glacier_name}_{year}_winter.zarr")
    ds_mbm_win = xr.open_dataset(mbm_file_win)
    ds_mbm_win = apply_gaussian_filter(ds_mbm_win)

    # Ensure correct coordinate names
    lon_name = "lon" if "lon" in ds_mbm_ann.coords else "longitude"
    lat_name = "lat" if "lat" in ds_mbm_ann.coords else "latitude"

    # Function to extract mass balance for each stake
    def get_predicted_mb(row, ds):
        try:
            return ds.sel(
                {
                    lon_name: row.POINT_LON,
                    lat_name: row.POINT_LAT
                },
                method="nearest").pred_masked.item()  # Convert to scalar
        except Exception:
            print(
                f"Warning: Stake at ({row.POINT_LON}, {row.POINT_LAT}) is out of bounds."
            )
            return np.nan

    def get_predicted_mb_glamos(row, ds):
        try:
            return ds.sel({
                lon_name: row.POINT_LON,
                lat_name: row.POINT_LAT
            },
                          method="nearest").item()  # Convert to scalar
        except Exception:
            print(
                f"Warning: Stake at ({row.POINT_LON}, {row.POINT_LAT}) is out of bounds."
            )
            return np.nan

    # Apply the function correctly using lambda
    stakes_data_ann["Predicted_MB"] = stakes_data_ann.apply(
        lambda row: get_predicted_mb(row, ds_mbm_ann), axis=1)
    stakes_data_ann.dropna(subset=["Predicted_MB"], inplace=True)
    stakes_data_ann["GLAMOS_MB"] = stakes_data_ann.apply(
        lambda row: get_predicted_mb_glamos(row, ds_glamos_wgs84_ann), axis=1)
    stakes_data_ann.dropna(subset=["GLAMOS_MB"], inplace=True)

    # Same for winter
    stakes_data_win["Predicted_MB"] = stakes_data_win.apply(
        lambda row: get_predicted_mb(row, ds_mbm_win), axis=1)
    stakes_data_win.dropna(subset=["Predicted_MB"], inplace=True)
    stakes_data_win["GLAMOS_MB"] = stakes_data_win.apply(
        lambda row: get_predicted_mb_glamos(row, ds_glamos_wgs84_win), axis=1)
    stakes_data_win.dropna(subset=["GLAMOS_MB"], inplace=True)

    # Compute color scale limits (Annual)
    vmin_ann = min(ds_glamos_wgs84_ann.min().item(),
                   ds_mbm_ann.pred_masked.min().item())
    vmax_ann = max(ds_glamos_wgs84_ann.max().item(),
                   ds_mbm_ann.pred_masked.max().item())

    # Compute color scale limits (Winter)
    vmin_win = min(ds_glamos_wgs84_win.min().item(),
                   ds_mbm_win.pred_masked.min().item())
    vmax_win = max(ds_glamos_wgs84_win.max().item(),
                   ds_mbm_win.pred_masked.max().item())

    print(
        f"Color scale range (Annual): vmin={vmin_ann:.3f}, vmax={vmax_ann:.3f}"
    )
    print(
        f"Color scale range (Winter): vmin={vmin_win:.3f}, vmax={vmax_win:.3f}"
    )

    if vmin_ann < 0 and vmax_ann > 0:
        norm_ann = mcolors.TwoSlopeNorm(vmin=vmin_ann,
                                        vcenter=0,
                                        vmax=-vmin_ann)
        cmap_ann = "coolwarm_r"
    elif vmin_ann < 0 and vmax_ann < 0:
        norm_ann = mcolors.Normalize(vmin=vmin_ann, vmax=vmax_ann)
        cmap_ann = "Reds"
    else:
        norm_ann = mcolors.Normalize(vmin=vmin_ann, vmax=vmax_ann)
        cmap_ann = "Blues"

    if vmin_win < 0:
        norm_win = mcolors.TwoSlopeNorm(vmin=-vmax_win,
                                        vcenter=0,
                                        vmax=vmax_win)
        cmap_win = "coolwarm_r"
    else:
        norm_win = mcolors.Normalize(vmin=vmin_win, vmax=vmax_win)
        cmap_win = "Blues"

    # Create figure with 2 rows (Annual & Winter)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    # Annual GLAMOS Plot
    ds_glamos_wgs84_ann.plot.imshow(
        ax=axes[0, 0],
        cmap=cmap_ann,
        norm=norm_ann,
        cbar_kwargs={"label": "Mass Balance [m w.e.]"})
    axes[0, 0].set_title("GLAMOS (Annual)")
    sns.scatterplot(data=stakes_data_ann,
                    x="POINT_LON",
                    y="POINT_LAT",
                    hue="POINT_BALANCE",
                    palette=cmap_ann,
                    hue_norm=norm_ann,
                    ax=axes[0, 0],
                    s=25,
                    legend=False)

    # add rmse if available
    if not stakes_data_ann.empty:
        rmse = mean_squared_error(stakes_data_ann.POINT_BALANCE,
                                  stakes_data_ann.GLAMOS_MB,
                                  squared=False)
        axes[0, 0].text(0.05,
                        0.1,
                        f"RMSE: {rmse:.2f}",
                        transform=axes[0, 0].transAxes,
                        ha='left',
                        va='top',
                        fontsize=18)

    # Annual MBM Predictions Plot
    ds_mbm_ann.pred_masked.plot.imshow(
        ax=axes[0, 1],
        cmap=cmap_ann,
        norm=norm_ann,
        cbar_kwargs={"label": "Mass Balance [m w.e.]"})
    axes[0, 1].set_title("MBM (Annual)")

    # Add Annual Stake Coordinates
    sns.scatterplot(data=stakes_data_ann,
                    x="POINT_LON",
                    y="POINT_LAT",
                    hue="POINT_BALANCE",
                    palette=cmap_ann,
                    hue_norm=norm_ann,
                    ax=axes[0, 1],
                    s=25,
                    legend=False)

    # add rmse
    if not stakes_data_ann.empty:
        rmse = mean_squared_error(stakes_data_ann.POINT_BALANCE,
                                  stakes_data_ann.Predicted_MB,
                                  squared=False)
        axes[0, 1].text(0.05,
                        0.1,
                        f"RMSE: {rmse:.2f}",
                        transform=axes[0, 1].transAxes,
                        ha='left',
                        va='top',
                        fontsize=18)

    # Winter GLAMOS & MBM Plots
    ds_glamos_wgs84_win.plot.imshow(
        ax=axes[1, 0],
        cmap=cmap_win,
        norm=norm_win,
        cbar_kwargs={"label": "Mass Balance [m w.e.]"})
    axes[1, 0].set_title("GLAMOS (Winter)")
    sns.scatterplot(data=stakes_data_win,
                    x="POINT_LON",
                    y="POINT_LAT",
                    hue="POINT_BALANCE",
                    palette=cmap_win,
                    hue_norm=norm_win,
                    ax=axes[1, 0],
                    s=25,
                    legend=False)

    # add rmse
    if not stakes_data_win.empty:
        rmse = mean_squared_error(stakes_data_win.POINT_BALANCE,
                                  stakes_data_win.GLAMOS_MB,
                                  squared=False)
        axes[1, 0].text(0.05,
                        0.1,
                        f"RMSE: {rmse:.2f}",
                        transform=axes[1, 0].transAxes,
                        ha='left',
                        va='top',
                        fontsize=18)

    # Winter MBM Predictions Plot
    ds_mbm_win.pred_masked.plot.imshow(
        ax=axes[1, 1],
        cmap=cmap_win,
        norm=norm_win,
        cbar_kwargs={"label": "Mass Balance [m w.e.]"})
    axes[1, 1].set_title("MBM (Winter)")
    sns.scatterplot(data=stakes_data_win,
                    x="POINT_LON",
                    y="POINT_LAT",
                    hue="POINT_BALANCE",
                    palette=cmap_win,
                    hue_norm=norm_win,
                    ax=axes[1, 1],
                    s=25,
                    legend=False)

    # add rmse
    if not stakes_data_win.empty:
        rmse = mean_squared_error(stakes_data_win.POINT_BALANCE,
                                  stakes_data_win.Predicted_MB,
                                  squared=False)
        axes[1, 1].text(0.05,
                        0.1,
                        f"RMSE: {rmse:.2f}",
                        transform=axes[1, 1].transAxes,
                        ha='left',
                        va='top',
                        fontsize=18)

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
        r2 = np.corrcoef(df_month['snow_cover_S2'],
                         df_month['snow_cover_glacier'])[0, 1]**2
        mse = mean_squared_error(df_month['snow_cover_glacier'],
                                 df_month['snow_cover_S2'])
        ax.text(0.05,
                0.85,
                f"R² = {r2:.2f}\nMSE = {mse:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                color="black")

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
            r2_corr = np.corrcoef(df_month['snow_cover_S2'],
                                  df_month['snow_cover_glacier_corr'])[0, 1]**2
            mse_corr = mean_squared_error(df_month['snow_cover_glacier_corr'],
                                          df_month['snow_cover_S2'])
            ax.text(0.05,
                    0.85,
                    f"R² = {r2_corr:.2f}\nMSE = {mse_corr:.2f}",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="black")

            ax.set_xlabel('Sentinel-2')
            ax.set_ylabel('Mass Balance Machine')
            ax.set_title(
                f'Snow Cover (Corrected), {df_month["month"].values[0]}')
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
    ax.text(0.05,
            0.9,
            f"R² = {r2:.2f}",
            transform=ax.transAxes,
            fontsize=16,
            color="black")

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
        selected_band.plot(ax=axs[1],
                           color='red',
                           linewidth=1,
                           markersize=5,
                           alpha=0.5)

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


def plot_mass_balance_comparison(
    glacier_name,
    geoMB_periods,
    MBM_glwmb,
    df,
    color_mbm="#1f77b4",  # Default matplotlib blue
    color_model2="#ff7f0e"   # Default matplotlib orange
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
        ax=axs[0],
        marker="o",
        color=[color_mbm, color_model2]
    )
    axs[0].set_title(f"{glacier_name.capitalize()} Glacier")
    axs[0].set_ylabel("Mass Balance [m w.e.]")
    axs[0].set_xlabel("Year")
    axs[0].grid(True, linestyle="--", linewidth=0.5)

    # --- Plot 2: Geodetic MB lines ---
    for _, row in df.iterrows():
        x_start, x_end = row['geoMB_periods']
        axs[1].hlines(y=row['mbm_geod'], xmin=x_start, xmax=x_end, color=color_mbm, linewidth=2)
        axs[1].hlines(y=row['glamos_geod'], xmin=x_start, xmax=x_end, color=color_model2, linewidth=2)

    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Geodetic MB [m w.e.]')
    axs[1].set_title('Geodetic MB')
    axs[1].grid(True)

    # --- Plot 3: Scatter comparison ---
    df.plot.scatter(
        x='target_geod', y='mbm_geod',
        ax=axs[2], color=color_mbm, alpha=0.7, label="MBM MB", marker="o"
    )
    df.plot.scatter(
        x='target_geod', y='glamos_geod',
        ax=axs[2], color=color_model2, alpha=0.7, label="GLAMOS MB", marker="s"
    )
    axs[2].set_xlabel("Geodetic MB [m w.e.]", fontsize=12)
    axs[2].set_ylabel("Modeled MB [m w.e.]", fontsize=12)
    axs[2].axline((0, 0), slope=1, color="grey", linestyle="--", linewidth=1)
    axs[2].axvline(0, color="grey", linestyle="--", linewidth=1)
    axs[2].axhline(0, color="grey", linestyle="--", linewidth=1)
    axs[2].grid(True, linestyle="--", linewidth=0.5)
    axs[2].set_title('Geodetic MB vs. Modeled MB')

    plt.tight_layout()
    plt.show()