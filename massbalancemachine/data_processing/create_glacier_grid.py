import numpy as np
import salem
import pyproj
import pandas as pd
import xarray as xr
import oggm


def create_glacier_grids(ds: xr.Dataset, years: list, glacier_indices: "tuple[np.array, np.array]",
                        gdir: oggm.GlacierDirectory,
                        rgi_gl: str) -> pd.DataFrame:
    """Creates a DataFrame of glacier grid data for each year

    Args:
        ds (xarray.Dataset): glacier attributes masked over its outline
        years (list): years of the data
        glacier_indices (np.array): indices of glacier mask in the OGGM grid
        gdir (oggm directory): oggm glacier directory
        rgi_gl (str): RGI Id of the glacier
    Returns:
        df_grid (pd.DataFrame): dataframe of glacier grid data, for each year
    """
    # Assuming the coordinate variables are named 'x' and 'y' in your dataset
    x_coords = ds['x'].values
    y_coords = ds['y'].values

    # Retrieve the x and y values using the glacier indices
    glacier_x_vals = x_coords[glacier_indices[1]]
    glacier_y_vals = y_coords[glacier_indices[0]]
    
    # Convert glacier coordinates to latitude and longitude
    # Transform stake coord to glacier system:
    transf = pyproj.Transformer.from_proj(gdir.grid.proj,
                                          salem.wgs84,
                                          always_xy=True)
    lon, lat = transf.transform(glacier_x_vals, glacier_y_vals)

    # Glacier mask as boolean array:
    gl_mask_bool = ds['glacier_mask'].values.astype(bool)
    
    # Create a DataFrame
    data_grid = {
        'RGIId': [rgi_gl] * len(ds.masked_elev.values[gl_mask_bool]),
        'POINT_LAT': lat,
        'POINT_LON': lon,
        'aspect': ds.masked_aspect.values[gl_mask_bool],
        'slope': ds.masked_slope.values[gl_mask_bool],
        'topo': ds.masked_elev.values[gl_mask_bool],
        'dis_from_border': ds.masked_dis.values[gl_mask_bool],
        'hugonnet_dhdt': ds.masked_hug.values[gl_mask_bool],
        'consensus_ice_thickness': ds.masked_cit.values[gl_mask_bool],
        'millan_ice_thickness': ds.masked_mit.values[gl_mask_bool],
        'millan_v': ds.masked_miv.values[gl_mask_bool],
        'millan_vx': ds.masked_mivx.values[gl_mask_bool],
        'millan_vy': ds.masked_mivy.values[gl_mask_bool]
    }

    df_grid = pd.DataFrame(data_grid)

    # Match to WGMS format:
    df_grid['POINT_ID'] = np.arange(1, len(df_grid) + 1)
    df_grid['N_MONTHS'] = 12
    df_grid['POINT_ELEVATION'] = df_grid[
        'topo']  # no other elevation available
    df_grid['POINT_BALANCE'] = 0  # fake PMB for simplicity (not used)
    num_rows_per_year = len(df_grid)
    # Repeat the DataFrame num_years times
    df_grid = pd.concat([df_grid] * len(years), ignore_index=True)
    # Add the 'year' and date columns to the DataFrame
    df_grid['YEAR'] = np.repeat(
        years, num_rows_per_year
    )  # 'year' column that has len(df_grid) instances of year
    df_grid['FROM_DATE'] = df_grid['YEAR'].apply(lambda x: str(x) + '1001')
    df_grid['TO_DATE'] = df_grid['YEAR'].apply(lambda x: str(x + 1) + '0930')

    return df_grid
