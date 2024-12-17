import pandas as pd

import config
import xarray as xr
import numpy as np
import salem
import pyproj
from scipy.ndimage import gaussian_filter
import os


class GeoData:
    """Class for handling geodata objects such as raster files (nc, tif),
       xarray datasets, shapefiles and geopandas dataframes.
       Attributes:
       - data (pd.DataFrame): Dataframe with the monthly point MB predictions.
       - ds_latlon (xr.Dataset): Dataset with the predictions in WGS84 coordinates.
       - ds_xy (xr.Dataset): Dataset with the predictions in the projection of OGGM. For CH, this is LV95.
    """

    def __init__(
        self,
        data: pd.DataFrame,
    ):
        self.data = data
        self.ds_latlon = None
        self.ds_xy = None

    def pred_to_xr(self, ds, gdir, pred_var='pred'):
        """Transforms MB predictions to xarray dataset. 
           Makes it easier for plotting and saving to netcdf.
           Keeps on netcdf in OGGM projection and transforms one to WGS84.

        Args:
            ds (xarray.Dataset): OGGM glacier grid with attributes.
            gdir gdir (oggm.GlacierDirectory): the OGGM glacier directory
            pred_var (str, optional): Name of prediction column in self.data. Defaults to 'pred'.
        """
        glacier_indices = np.where(ds['glacier_mask'].values == 1)
        pred_masked = ds.glacier_mask.values

        # Set pred_masked to nan where 0
        pred_masked = np.where(pred_masked == 0, np.nan, pred_masked)
        for i, (x_index, y_index) in enumerate(
                zip(glacier_indices[0], glacier_indices[1])):
            pred_masked[x_index, y_index] = self.data.iloc[i][pred_var]

        pred_masked = np.where(pred_masked == 1, np.nan, pred_masked)
        self.ds_xy = ds.assign(pred_masked=(('y', 'x'), pred_masked))

        # Change from OGGM proj. to wgs84
        self.ds_latlon = self.oggmToWgs84(self.ds_xy, gdir)

    def save_arrays(self, path_wgs84: str, path_lv95: str, filename: str):
        """Saves the xarray datasets in OGGM projection and WGMS84 to netcdf files.

        Args:
            path_wgs84 (str): path to save the dataset in WGS84 projection.
            path_lv95 (str): path to save the dataset in LV95 projection.
            filename (str): filename for the netcdf file.
        """
        self.__class__.save_to_netcdf(self.ds_latlon, path_wgs84, filename)
        self.__class__.save_to_netcdf(self.ds_xy, path_lv95, filename)

    @staticmethod
    def save_to_netcdf(ds: xr.Dataset, path: str, filename: str):
        """Saves the xarray dataset to a netcdf file.
        """
        # Create path if not exists
        if not os.path.exists(path):
            os.makedirs(path)

        # delete file if already exists
        if os.path.exists(path + filename):
            os.remove(path + filename)

        # save prediction to netcdf
        ds.to_netcdf(path + filename)

    @staticmethod
    def oggmToWgs84(ds, gdir):
        """Transforms a xarray dataset from OGGM projection to WGS84.

        Args:
            ds (xr.Dataset): xr.Dataset with the predictions in OGGM projection.
            gdir (oggm.GlacierDirectory): oggm glacier directory

        Returns:
            xr.Dataset: xr.Dataset with the predictions in WGS84 projection.
        """
        # Define the Swiss coordinate system (EPSG:2056) and WGS84 (EPSG:4326)
        transformer = pyproj.Transformer.from_proj(gdir.grid.proj,
                                                   salem.wgs84,
                                                   always_xy=True)

        # Get the Swiss x and y coordinates from the dataset
        x_coords = ds['x'].values
        y_coords = ds['y'].values

        # Create a meshgrid for all x, y coordinate pairs
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)

        # Flatten the meshgrid arrays for transformation
        x_flat = x_mesh.ravel()
        y_flat = y_mesh.ravel()

        # Transform the flattened coordinates
        lon_flat, lat_flat = transformer.transform(x_flat, y_flat)

        # Reshape transformed coordinates back to the original grid shape
        lon = lon_flat.reshape(x_mesh.shape)
        lat = lat_flat.reshape(y_mesh.shape)

        # Extract unique 1D coordinates for lat and lon
        lon_1d = lon[
            0, :]  # Take the first row for unique longitudes along x-axis
        lat_1d = lat[:,
                     0]  # Take the first column for unique latitudes along y-axis

        # Assign the 1D coordinates to x and y dimensions
        ds = ds.assign_coords(longitude=("x", lon_1d), latitude=("y", lat_1d))

        # Swap x and y dimensions with lon and lat
        ds = ds.swap_dims({"x": "longitude", "y": "latitude"})

        # Optionally, drop the old x and y coordinates if no longer needed
        ds = ds.drop_vars(["x", "y"])

        return ds

    @staticmethod
    def GaussianFilter(ds: xr.Dataset,
                       variable_name: str = 'pred_masked',
                       sigma: float = 1):
        """
        Apply Gaussian filter only to the specified variable in the xarray.Dataset.
        
        Parameters:
        - ds (xarray.Dataset): Input dataset
        - variable_name (str): The name of the variable to apply the filter to (default 'pred_masked')
        - sigma (float): The standard deviation for the Gaussian filter. Default is 1.
        
        Returns:
        - xarray.Dataset: New dataset with smoothed variable
        """
        # Check if the variable exists in the dataset
        if variable_name not in ds:
            raise ValueError(
                f"Variable '{variable_name}' not found in the dataset.")

        # Get the DataArray for the specified variable
        data_array = ds[variable_name]

        # Step 1: Create a mask of valid data (non-NaN values)
        mask = ~np.isnan(data_array)

        # Step 2: Replace NaNs with zero (or a suitable neutral value)
        filled_data = data_array.fillna(0)

        # Step 3: Apply Gaussian filter to the filled data
        smoothed_data = gaussian_filter(filled_data, sigma=sigma)

        # Step 4: Restore NaNs to their original locations
        smoothed_data = xr.DataArray(
            smoothed_data,
            dims=data_array.dims,
            coords=data_array.coords,
            attrs=data_array.attrs).where(
                mask)  # Apply the mask to restore NaNs

        # Create a new dataset with the smoothed data
        smoothed_dataset = ds.copy()  # Make a copy of the original dataset
        smoothed_dataset[
            variable_name] = smoothed_data  # Replace the original variable with the smoothed one

        return smoothed_dataset
