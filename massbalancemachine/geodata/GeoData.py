import pandas as pd
import xarray as xr
import numpy as np
import salem
import pyproj
from scipy.ndimage import gaussian_filter
import os
import rasterio
from shapely.geometry import Point, box
import geopandas as gpd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

import config
from data_processing.utils import (
    _rebuild_month_index,
    build_head_tail_pads_from_monthly_df,
    _compute_head_tail_pads_from_df,
)


class GeoData:
    """Class for handling geodata objects such as raster files (nc, tif),
    xarray datasets, shapefiles and geopandas dataframes.
    Attributes:
    - data (pd.DataFrame): Dataframe with the monthly point MB predictions.
    - ds_latlon (xr.Dataset): Dataset with the predictions in WGS84 coordinates.
    - ds_xy (xr.Dataset): Dataset with the predictions in the projection of OGGM. For CH, this is LV95.
    - gdf (gpd.GeoDataFrame): Geopandas dataframe with the predictions in WGS84 coordinates.
    """

    def __init__(self, data: pd.DataFrame, months_head_pad=None, months_tail_pad=None):
        self.data = data
        self.ds_latlon = None
        self.ds_xy = None
        self.gdf = None

        self.months_tail_pad = months_tail_pad
        self.months_head_pad = months_head_pad

        assert (months_head_pad is None) == (
            months_tail_pad is None
        ), "If any of months_head_pad or months_tail_pad is provided, the other variable must also be provided."

        try:
            if months_head_pad is None and months_tail_pad is None:
                months_head_pad, months_tail_pad = _compute_head_tail_pads_from_df(
                    self.data
                )
                self.months_tail_pad = months_tail_pad
                self.months_head_pad = months_head_pad
        except AttributeError as e:
            raise ValueError(
                "Could not compute months_head_pad / months_tail_pad from dataframe. Please provide them explicitly as arguments in constructor."
            ) from e

        _, self.month_pos = _rebuild_month_index(months_head_pad, months_tail_pad)

    def set_gdf(self, gdf):
        """Set the gdf attribute to a geopandas dataframe."""
        self.gdf = gdf

    def set_ds_latlon(self, ds_or_path, path_nc_wgs84=""):
        """Set the ds_latlon attribute to an xarray Dataset or load it from a NetCDF file."""
        if isinstance(ds_or_path, xr.Dataset):
            # If the input is an xarray.Dataset, use it directly
            self.ds_latlon = ds_or_path
        elif isinstance(ds_or_path, str):
            # If the input is a string and corresponds to a valid file, open the xarray from the file
            full_path = os.path.join(path_nc_wgs84, ds_or_path)
            self.ds_latlon = xr.open_dataset(full_path)
        else:
            raise TypeError(
                "ds_latlon must be either an xarray.Dataset or a valid file path to a NetCDF file"
            )
        # Apply Gaussian filter
        self.apply_gaussian_filter()

        # Convert to geopandas
        self.xr_to_gpd()

    def set_ds_xy(self, ds_or_path, path_nc_xy=""):
        """Set the ds_xy attribute to an xarray Dataset or load it from a NetCDF file."""
        if isinstance(ds_or_path, xr.Dataset):
            # If the input is an xarray.Dataset, use it directly
            self.ds_xy = ds_or_path
        elif isinstance(ds_or_path, str) and os.path.isfile(
            os.path.join(path_nc_xy, ds_or_path)
        ):
            # If the input is a string and corresponds to a valid file, open the xarray from the file
            full_path = os.path.join(path_nc_xy, ds_or_path)
            self.ds_xy = xr.open_dataset(full_path)
        else:
            raise TypeError(
                "ds_xy must be either an xarray.Dataset or a valid file path to a NetCDF file"
            )

    def pred_to_xr(self, ds, gdir=None, pred_var="pred", source_type="oggm"):
        """Transforms MB predictions to xarray dataset.
           Makes it easier for plotting and saving to netcdf.
           Keeps on netcdf in OGGM projection and transforms one to WGS84.

        Args:
            ds (xarray.Dataset): OGGM glacier grid with attributes.
            gdir gdir (oggm.GlacierDirectory): the OGGM glacier directory
            pred_var (str, optional): Name of prediction column in self.data. Defaults to 'pred'.
        """
        if source_type == "oggm":
            glacier_indices = np.where(ds["glacier_mask"].values == 1)
            pred_masked = ds.glacier_mask.values

            # Set pred_masked to nan where 0
            pred_masked = np.where(pred_masked == 0, np.nan, pred_masked)

            for i, (x_index, y_index) in enumerate(
                zip(glacier_indices[0], glacier_indices[1])
            ):
                print(x_index, y_index)
                pred_masked[x_index, y_index] = self.data.iloc[i][pred_var]

            pred_masked = np.where(pred_masked == 1, np.nan, pred_masked)

            self.ds_xy = ds.assign(pred_masked=(("y", "x"), pred_masked))

            # Change from OGGM proj. to wgs84
            self.ds_latlon = self.oggmToWgs84(self.ds_xy, gdir)

        if source_type == "sgi":
            # Faster way:
            # Create a new variable pred_masked initialized with NaN
            ds["pred_masked"] = xr.DataArray(
                np.full((ds.dims["lat"], ds.dims["lon"]), np.nan),
                dims=("lat", "lon"),
                coords={"lat": ds["lat"], "lon": ds["lon"]},
            )

            # Extract the DataFrame columns as numpy arrays
            point_lon = self.data["POINT_LON"].values
            point_lat = self.data["POINT_LAT"].values
            pred_values = self.data[pred_var].values

            # Use vectorized nearest neighbor selection
            nearest_points = ds.sel(
                lon=xr.DataArray(point_lon, dims="points"),
                lat=xr.DataArray(point_lat, dims="points"),
                method="nearest",
            )

            # Extract the nearest grid indices
            lon_indices = nearest_points["lon"].values
            lat_indices = nearest_points["lat"].values

            # Create a mask with NaN, then fill with the predictions at the corresponding indices
            pred_masked = np.full((ds.dims["lat"], ds.dims["lon"]), np.nan)

            # Loop over the matched points to assign values
            for lon_idx, lat_idx, pred in zip(lon_indices, lat_indices, pred_values):
                pred_masked[
                    np.where(ds["lat"] == lat_idx)[0][0],
                    np.where(ds["lon"] == lon_idx)[0][0],
                ] = pred

            # Assign pred_masked back to the dataset
            ds["pred_masked"] = (("lat", "lon"), pred_masked)
            self.ds_latlon = ds

    def save_arrays(self, filename: str, path: str = None, proj_type: str = "wgs840"):
        """Saves the xarray datasets in OGGM projection and WGMS84 to netcdf files.

        Args:
            path_wgs84 (str): path to save the dataset in WGS84 projection.
            path_lv95 (str): path to save the dataset in LV95 projection.
            filename (str): filename for the netcdf file.
        """
        if proj_type == "wgs84":
            self.__class__.save_to_zarr(self.ds_latlon, path, filename)
        elif proj_type == "lv95":
            self.__class__.save_to_zarr(self.ds_xy, path, filename)
        else:
            raise ValueError("proj_type must be either 'wgs84' or 'lv95'.")

    def xr_to_gpd(self):
        """Converts an xarray dataset to a geopandas dataframe."""
        # Get lat and lon, and variables data
        lat = self.ds_latlon["lat"].values
        lon = self.ds_latlon["lon"].values
        pred_masked_data = self.ds_latlon["pred_masked"].values
        masked_elev_data = self.ds_latlon["masked_elev"].values
        # masked_dis_data = self.ds_latlon['masked_dis'].values

        # Create meshgrid of coordinates
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # Flatten all arrays to match shapes
        lon_flat = lon_grid.flatten()
        lat_flat = lat_grid.flatten()
        pred_masked_data_flat = pred_masked_data.flatten()
        masked_elev_data_flat = masked_elev_data.flatten()
        # masked_dis_data_flat = masked_dis_data.flatten()

        # Verify shapes
        assert (
            len(lon_flat)
            == len(lat_flat)
            == len(pred_masked_data_flat)
            == len(masked_elev_data_flat)
        ), "Shapes don't match!"

        # Create GeoDataFrame
        points = [Point(xy) for xy in zip(lon_flat, lat_flat)]
        gdf = gpd.GeoDataFrame(
            {
                "pred_masked": pred_masked_data_flat,
                "elev_masked": masked_elev_data_flat,
                # "dis_masked": masked_dis_data_flat
            },
            geometry=points,
            crs="EPSG:4326",
        )

        # return gdf, lon, lat
        self.gdf = gdf

    def apply_gaussian_filter(
        self, variable_name: str = "pred_masked", sigma: float = 1
    ):
        """
        Apply Gaussian filter only to the specified variable in the xarray.Dataset.

        Parameters:
        - variable_name (str): The name of the variable to apply the filter to (default 'pred_masked')
        - sigma (float): The standard deviation for the Gaussian filter. Default is 1.

        Returns:
        - self: Returns the instance for method chaining.
        """
        if self.ds_latlon is None:
            raise ValueError("ds_latlon attribute is not set. Please set it first.")

        # Check if the variable exists in the dataset
        if variable_name not in self.ds_latlon:
            raise ValueError(f"Variable '{variable_name}' not found in the dataset.")

        # Get the DataArray for the specified variable
        data_array = self.ds_latlon[variable_name]

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
            attrs=data_array.attrs,
        ).where(
            mask
        )  # Apply the mask to restore NaNs

        # Replace the original variable with the smoothed one in the dataset
        self.ds_latlon[variable_name] = smoothed_data

        # Return self to allow method chaining
        return self

    def gridded_MB_pred(
        self,
        df_grid_monthly,
        custom_model,
        glacier_name,
        year,
        all_columns,
        path_glacier_dem,
        path_save_glw,
        save_monthly_pred=True,
        save_seasonal_pred=True,
        type_model="XGBoost",
    ):
        """
        Computes and saves gridded mass balance (MB) predictions for a given glacier and year.

        This function predicts seasonal and annual surface mass balance (SMB) using
        the ML model, saves the results as Zarr files, and optionally
        saves monthly predictions.

        Args:
            custom_model (object): The trained MassBalanceMachine model used for prediction.
            glacier_name (str): Name of the glacier being processed.
            year (int): Year for which predictions are made.
            all_columns (list of str): List of feature column names used for predictions.
            path_glacier_dem (str): Path to the directory containing glacier DEM files.
            path_save_glw (str): Path to the directory where output files will be saved.
            cfg (dict): Configuration dictionary containing additional processing parameters.
            save_monthly_pred (bool, optional): If True, saves monthly mass balance predictions. Defaults to True.

        Returns:
            None

        Notes:
            - The function first computes cumulative SMB predictions using the MassBalanceMachine model.
            - Annual and winter SMB predictions are extracted and saved.
            - The function loads the glacier's DEM dataset from a Zarr file.
            - If the DEM file is missing, the function prints a warning and exits.
            - Predictions are saved using `_save_prediction` for annual and winter MB.
            - If `save_monthly_pred` is True, `_save_monthly_predictions` is called.

        Raises:
            FileNotFoundError: If the DEM file for the glacier and year is not found.
        """

        if type_model == "XGBoost":
            # Compute cumulative SMB predictions
            df_grid_monthly = custom_model.cumulative_pred(self.data, self.month_pos)
            self.data = df_grid_monthly

        # Generate annual and winter predictions
        pred_winter, df_pred_months_winter = custom_model.glacier_wide_pred(
            self.data[all_columns],
            self.months_head_pad,
            self.months_tail_pad,
            type_pred="winter",
        )
        pred_annual, df_pred_months_annual = custom_model.glacier_wide_pred(
            self.data[all_columns],
            self.months_head_pad,
            self.months_tail_pad,
            type_pred="annual",
        )

        # Filter results for the current year
        pred_y_annual = pred_annual.drop(columns=["YEAR"], errors="ignore")
        pred_y_winter = pred_winter.drop(columns=["YEAR"], errors="ignore")

        # Save seasonal predictions
        if not os.path.exists(path_glacier_dem):
            print(f"DEM file not found for {path_glacier_dem}, skipping...")
            return
        ds = xr.open_dataset(path_glacier_dem)

        # Save both annual and winter predictions using the helper function
        if save_seasonal_pred:
            self._save_prediction(
                ds, pred_y_winter, glacier_name, year, path_save_glw, "winter"
            )
            self._save_prediction(
                ds, pred_y_annual, glacier_name, year, path_save_glw, "annual"
            )

        # Save monthly grids
        if save_monthly_pred and type_model == "NN":
            coordinates = (
                df_grid_monthly.groupby("ID")[["POINT_LAT", "POINT_LON"]]
                .mean()
                .reset_index()
            )
            df_pred_months_annual = df_pred_months_annual.merge(
                coordinates, on="ID", how="left"
            )

            self._save_monthly_predictions_NN(
                df_pred_months_annual, ds, glacier_name, year, path_save_glw
            )
        elif save_monthly_pred and type_model == "XGBoost":
            self._save_monthly_predictions_XGB(
                df_grid_monthly, ds, glacier_name, year, path_save_glw
            )

        return df_pred_months_annual

    def get_mean_SMB(
        self, custom_model, all_columns, months_head_pad=None, months_tail_pad=None
    ):
        """Computes the mean surface mass balance (SMB) for a glacier using the MassBalanceMachine model."""
        # Compute cumulative SMB predictions
        df_grid_monthly = custom_model.cumulative_pred(
            self.data[all_columns], self.month_pos
        )

        # Generate annual and winter predictions
        pred_annual, df_pred_months = custom_model.glacier_wide_pred(
            custom_model,
            df_grid_monthly[all_columns],
            months_head_pad,
            months_tail_pad,
            type_pred="annual",
        )

        # Drop year column
        pred_y_annual = pred_annual.drop(columns=["YEAR"], errors="ignore")

        # Take mean over all points:
        mean_SMB = pred_y_annual.pred.mean()

        return mean_SMB

    def _save_prediction(
        self, ds, pred_data, glacier_name, year, path_save_glw, season
    ):
        """Helper function to save seasonal glacier-wide predictions."""
        self.data = pred_data
        self.pred_to_xr(ds, pred_var="pred", source_type="sgi")

        save_path = os.path.join(path_save_glw, glacier_name)
        os.makedirs(save_path, exist_ok=True)

        self.save_arrays(
            f"{glacier_name}_{year}_{season}.zarr",
            path=save_path + "/",
            proj_type="wgs84",
        )

    def _save_monthly_predictions_NN(self, df, ds, glacier_name, year, path_save_glw):
        """Helper function to save monthly predictions."""
        hydro_months = [
            "sep",
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
        ]
        df_cumulative = df[hydro_months].cumsum(axis=1)

        for month in hydro_months:
            df_month = df[[month, "ID", "POINT_LON", "POINT_LAT"]]
            df_month["pred"] = df_month[month]
            df_month["cum_pred"] = df_cumulative[month]
            self.data = df_month
            self.pred_to_xr(ds, pred_var="cum_pred", source_type="sgi")
            save_path = os.path.join(path_save_glw, glacier_name)
            os.makedirs(save_path, exist_ok=True)
            self.save_arrays(
                f"{glacier_name}_{year}_{month}.zarr",
                path=save_path + "/",
                proj_type="wgs84",
            )

    def _save_monthly_predictions_XGB(self, df, ds, glacier_name, year, path_save_glw):
        """Helper function to save monthly predictions."""
        hydro_months = [
            "sep",
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
        ]
        for month in hydro_months:
            df_month = (
                df[df["MONTHS"] == month]
                .groupby("ID")
                .agg(
                    {
                        "YEAR": "mean",
                        "POINT_LAT": "mean",
                        "POINT_LON": "mean",
                        "pred": "mean",
                        "cum_pred": "mean",
                    }
                )
                .drop(columns=["YEAR"], errors="ignore")
            )

            self.data = df_month
            self.pred_to_xr(ds, pred_var="cum_pred", source_type="sgi")
            save_path = os.path.join(path_save_glw, glacier_name)
            os.makedirs(save_path, exist_ok=True)
            self.save_arrays(
                f"{glacier_name}_{year}_{month}.zarr",
                path=save_path + "/",
                proj_type="wgs84",
            )

    @staticmethod
    def save_to_zarr(ds: xr.Dataset, path: str, filename: str):
        """Saves the xarray dataset to a netcdf file."""
        # Create path if not exists
        if not os.path.exists(path):
            os.makedirs(path)

        # delete file if already exists
        if os.path.exists(path + filename):
            os.remove(path + filename)

        # save prediction to netcdf
        ds.to_zarr(path + filename)

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
        transformer = pyproj.Transformer.from_proj(
            gdir.grid.proj, salem.wgs84, always_xy=True
        )

        # Get the Swiss x and y coordinates from the dataset
        x_coords = ds["x"].values
        y_coords = ds["y"].values

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
        lon_1d = lon[0, :]  # Take the first row for unique longitudes along x-axis
        lat_1d = lat[:, 0]  # Take the first column for unique latitudes along y-axis

        # Assign the 1D coordinates to x and y dimensions
        ds = ds.assign_coords(longitude=("x", lon_1d), latitude=("y", lat_1d))

        # Swap x and y dimensions with lon and lat
        ds = ds.swap_dims({"x": "longitude", "y": "latitude"})

        # Optionally, drop the old x and y coordinates if no longer needed
        ds = ds.drop_vars(["x", "y"])

        return ds

    @staticmethod
    def raster_to_gpd(input_raster):
        # Open the raster
        with rasterio.open(input_raster) as src:
            data = src.read(1)  # Read the first band
            transform = src.transform
            crs = src.crs
            nodata = src.nodata

        # Get indices of non-NaN values and their corresponding values
        mask = data != nodata
        rows, cols = np.nonzero(mask)
        values = data[mask]

        # Vectorize coordinate transformation
        x_coords, y_coords = rasterio.transform.xy(
            transform, rows, cols, offset="center"
        )

        # Create GeoDataFrame directly using vectorized data
        gdf_raster = gpd.GeoDataFrame(
            {"classes": values},
            geometry=gpd.points_from_xy(x_coords, y_coords),
            crs=crs,
        )

        return gdf_raster

    @staticmethod
    def resample_satellite_to_glacier(gdf_glacier, gdf_raster):
        # Clip raster to glacier extent
        # Step 1: Get the bounding box of the points GeoDataFrame
        bounding_box = gdf_glacier.total_bounds  # [minx, miny, maxx, maxy]
        raster_bounds = gdf_raster.total_bounds  # [minx, miny, maxx, maxy]

        # Problem 1: check if glacier bounds are within raster bounds
        if not (
            bounding_box[0] >= raster_bounds[0]  # minx of glacier >= minx of raster
            and bounding_box[1] >= raster_bounds[1]  # miny of glacier >= miny of raster
            and bounding_box[2] <= raster_bounds[2]  # maxx of glacier <= maxx of raster
            and bounding_box[3] <= raster_bounds[3]  # maxy of glacier <= maxy of raster
        ):
            return 0

        # Step 2: Create a rectangular geometry from the bounding box
        bbox_polygon = box(*bounding_box)

        # Problem 2: Glacier is in regions where raster is NaN
        gdf_clipped = gpd.clip(gdf_raster, bbox_polygon)
        if gdf_clipped.empty:
            return 1

        # Step 3: Clip the raster-based GeoDataFrame to this bounding box
        gdf_clipped = gdf_raster[gdf_raster.intersects(bbox_polygon)]

        # Optionally, further refine the clipping if exact match is needed
        gdf_clipped = gpd.clip(gdf_raster, bbox_polygon)

        # Resample clipped raster to glacier points
        # Extract coordinates and values from gdf_clipped
        clipped_coords = np.array([(geom.x, geom.y) for geom in gdf_clipped.geometry])
        clipped_values = gdf_clipped["classes"].values

        # Extract coordinates from gdf_glacier
        points_coords = np.array([(geom.x, geom.y) for geom in gdf_glacier.geometry])

        # Build a KDTree for efficient nearest-neighbor search
        tree = cKDTree(clipped_coords)

        # Query the tree for the nearest neighbor to each point in gdf_glacier
        distances, indices = tree.query(points_coords)

        # Assign the values from the nearest neighbors
        gdf_clipped_res = gdf_glacier.copy()
        gdf_clipped_res = gdf_clipped_res[["geometry"]]
        gdf_clipped_res["classes"] = clipped_values[indices]

        # Assuming 'value' is the column storing the resampled values
        gdf_clipped_res["classes"] = np.where(
            gdf_glacier["pred_masked"].isna(),  # Check where original values are NaN
            np.nan,  # Assign NaN to those locations
            gdf_clipped_res["classes"],  # Keep the resampled values elsewhere
        )

        return gdf_clipped_res
