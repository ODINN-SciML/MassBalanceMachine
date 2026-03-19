import pandas as pd
import xarray as xr
from tqdm import tqdm
import re
from calendar import monthrange

from regions.Switzerland.scripts.config_CH import *
from regions.Switzerland.scripts.utils.helpers import *
from regions.Switzerland.scripts.geo_data.geodata import LV03_to_WGS84, load_grid_file


def process_pcsr(cfg):
    """
    Process daily PCSR raster grids into monthly mean xarray datasets (WGS84) and save to Zarr.

    For each glacier directory under `<dataPath>/<path_pcsr>/raw/`, this function:
    - loads daily grids from `.grid` files
    - aggregates daily grids into 12 monthly means
    - converts to an xarray DataArray with a time dimension
    - transforms LV03 coordinates to WGS84 (lat/lon)
    - writes glacier-specific Zarr outputs

    Parameters
    ----------
    cfg : object
        Configuration object with attribute `dataPath`. Uses `path_pcsr` constant from config.

    Returns
    -------
    None

    Side Effects
    ------------
    Creates/empties the output Zarr folder and writes Zarr datasets.

    Raises
    ------
    FileNotFoundError
        If expected raw directories/files are missing.
    ValueError
        If grid shapes are inconsistent or date aggregation fails.
    """

    glDirect = np.sort(
        os.listdir(cfg.dataPath + path_pcsr + "raw/")
    )  # Glaciers with data
    path_pcsr_save = cfg.dataPath + path_pcsr + "zarr/"

    # check folder exists otherwise create it
    if not os.path.exists(path_pcsr_save):
        os.makedirs(path_pcsr_save)
    # Clean output folder
    emptyfolder(path_pcsr_save)

    for glacierName in tqdm(glDirect, desc="glaciers", position=0):
        grid = os.listdir(cfg.dataPath + path_pcsr + "raw/" + glacierName)
        grid_year = int(re.findall(r"\d+", grid[0])[0])
        daily_grids = os.listdir(
            cfg.dataPath + path_pcsr + "raw/" + glacierName + "/" + grid[0]
        )
        # Sort by day number from 001 to 365
        daily_grids.sort()
        grids = []
        for fileName in daily_grids:
            if "grid" not in fileName:
                continue

            # Load daily grid file
            file_path = (
                cfg.dataPath
                + path_pcsr
                + "raw/"
                + glacierName
                + "/"
                + grid[0]
                + "/"
                + fileName
            )
            metadata, grid_data = load_grid_file(file_path)
            grids.append(grid_data)

        # Take monthly means:
        monthly_grids = []
        for i in range(12):
            num_days_month = monthrange(grid_year, i + 1)[1]
            monthly_grids.append(
                np.mean(
                    np.stack(
                        grids[i * num_days_month : (i + 1) * num_days_month], axis=0
                    ),
                    axis=0,
                )
            )

        monthly_grids = np.array(monthly_grids)
        num_months = monthly_grids.shape[0]

        # Convert to xarray (CH coordinates)
        data_array = convert_to_xarray(monthly_grids, metadata, num_months)

        # Convert to WGS84 (lat/lon) coordinates
        data_array_transf = transform_xarray_coords_lv03_to_wgs84_time(data_array)

        # Save xarray
        if glacierName == "findelen":
            data_array_transf.to_zarr(path_pcsr_save + f"xr_direct_{glacierName}.zarr")
            data_array_transf.to_zarr(path_pcsr_save + f"xr_direct_adler.zarr")
        elif glacierName == "stanna":
            data_array_transf.to_zarr(path_pcsr_save + f"xr_direct_sanktanna.zarr")
        else:
            data_array_transf.to_zarr(path_pcsr_save + f"xr_direct_{glacierName}.zarr")


def convert_to_xarray(grid_data, metadata, num_months):
    """
    Convert a time stack of gridded rasters into an xarray DataArray with (time, y, x) dims.

    Parameters
    ----------
    grid_data : numpy.ndarray
        3D array of shape (num_months, nrows, ncols).
    metadata : dict
        Grid metadata containing `ncols`, `nrows`, `xllcorner`, `yllcorner`, `cellsize`.
    num_months : int
        Number of time steps expected along the first axis.

    Returns
    -------
    xarray.DataArray
        DataArray with dims ("time", "y", "x") and coordinates built from metadata.
        The data are flipped along the y axis (axis=1) so that the y-coordinate
        increases in the expected direction.

    Raises
    ------
    ValueError
        If `grid_data` does not match the expected shape.
    KeyError
        If required metadata keys are missing.
    """

    # Extract metadata values
    ncols = int(metadata["ncols"])
    nrows = int(metadata["nrows"])
    xllcorner = metadata["xllcorner"]
    yllcorner = metadata["yllcorner"]
    cellsize = metadata["cellsize"]

    # Create x and y coordinates based on the metadata
    x_coords = xllcorner + np.arange(ncols) * cellsize
    y_coords = yllcorner + np.arange(nrows) * cellsize

    time_coords = np.arange(num_months)

    if grid_data.shape != (num_months, nrows, ncols):
        raise ValueError(
            f"Expected grid_data shape ({num_months}, {nrows}, {ncols}), got {grid_data.shape}"
        )

    # Create the xarray DataArray
    data_array = xr.DataArray(
        np.flip(grid_data, axis=1),
        # grid_data,
        dims=("time", "y", "x"),
        coords={"time": time_coords, "y": y_coords, "x": x_coords},
        name="grid_data",
    )
    return data_array


def transform_xarray_coords_lv03_to_wgs84_time(data_array):
    """
    Reassign LV03 (CH1903) x/y coordinates of a (time, y, x) DataArray to WGS84 lon/lat.

    This function transforms every grid point from LV03 to WGS84, assigns 1D lon/lat
    coordinates, swaps dimensions to ("time", "lon", "lat"), and returns the result.

    Parameters
    ----------
    data_array : xarray.DataArray
        3D DataArray with dims ("time", "y", "x") and LV03 coordinates in meters.

    Returns
    -------
    xarray.DataArray
        DataArray with WGS84 coordinates and dims reordered to ("time", "lon", "lat").

    Notes
    -----
    Assumes the transformed grid can be represented by 1D lon and 1D lat vectors
    (uses the first time slice to extract these).
    """
    # Extract time, y, and x dimensions
    time_dim = data_array.coords["time"]

    # Flatten the DataArray (values) and extract x and y coordinates for each time step
    flattened_values = data_array.values.reshape(
        -1
    )  # Flatten entire 3D array (time, y, x)

    # flattened_values = data_array.values.flatten()
    y_coords, x_coords = np.meshgrid(
        data_array.y.values, data_array.x.values, indexing="ij"
    )

    # Flatten the coordinate arrays
    flattened_x = np.tile(
        x_coords.flatten(), len(time_dim)
    )  # Repeat for each time step
    flattened_y = np.tile(
        y_coords.flatten(), len(time_dim)
    )  # Repeat for each time step

    # Create a DataFrame with columns for x, y, and value
    df = pd.DataFrame(
        {"x_pos": flattened_x, "y_pos": flattened_y, "value": flattened_values}
    )
    df["z_pos"] = 0

    # Convert to lat/lon
    df = LV03_to_WGS84(df)

    # Transform LV03 to WGS84 (lat, lon)
    lon, lat = df.lon.values, df.lat.values

    # Reshape the flattened WGS84 coordinates back to the original grid shape (time, y, x)
    lon = lon.reshape((len(time_dim), *x_coords.shape))  # Shape: (time, y, x)
    lat = lat.reshape((len(time_dim), *y_coords.shape))  # Shape: (time, y, x)

    # Assign the 1D WGS84 coordinates for swapping
    lon_1d = lon[0, 0, :]  # Use the first time slice, and take x (lon) values
    lat_1d = lat[0, :, 0]  # Use the first time slice, and take y (lat) values

    # Assign the WGS84 coordinates back to the xarray
    data_array = data_array.assign_coords(lon=("x", lon_1d))  # Assign longitudes
    data_array = data_array.assign_coords(lat=("y", lat_1d))  # Assign latitudes

    # First, swap 'x' with 'lon' and 'y' with 'lat', keeping the time dimension intact
    data_array = data_array.swap_dims({"x": "lon", "y": "lat"})

    # Reorder the dimensions to be (time, lon, lat)
    data_array = data_array.transpose("time", "lon", "lat")

    return data_array
