# MassBalanceMachine Core

The MassBalanceMachine core consists of the following sub-packages:

### 1. Data Processing

The `data_processing` package includes the `Dataset` class and methods to retrieve topographical and meteorological data, and transform the dataset to a monthly resolution.

#### Methods

* `__init__(self, *, data: pd.DataFrame, region_name: str, data_path: str)`

  * What it does: Initializes the Dataset object with input data, region name, and data path.
  * Input:
    * `data`: Pandas DataFrame containing raw data
    * `region_name`: String representing the region name
    * `data_path`: String path to the data directory
  * Output: None (initializes object attributes)
* `get_topo_features(self, *, vois: list[str]) -> None`

  * What it does: Fetches topographical data for specified variables of interest using OGGM.
  * Input:
    * `vois`: List of strings representing topographical variables of interest
  * Output: None (updates `self.data` with topographical features)
* `get_climate_features(self, *, climate_data: str, geopotential_data: str) -> None`

  * What it does: Fetches climate data for the specified RGI IDs.
  * Input:
    * `climate_data`: String path to netCDF-3 file containing climate data
    * `geopotential_data`: String path to netCDF-3 file containing geopotential data
  * Output: None (updates `self.data` with climate features)
* `convert_to_monthly(self, *, vois_climate: list[str], vois_topographical: list[str]) -> None`

  * What it does: Converts variable period SMB target data to monthly time resolution.
  * Input:
    * `vois_climate`: List of strings representing climate variables of interest
    * `vois_topographical`: List of strings representing topographical variables of interest
  * Output: None (updates `self.data` with monthly resolution data)

### 2. Data Loader

[WIP]

### 3. Models

[WIP]

### 4. Utils

The `utils` package includes methods for data exploration and data pre-processing. These methods do not have to be called with a `Dataset` object, but can be directly called from the `massbalancemachine` package, for example: `massbalancemachine.plot_stake_timeseries()`.

#### Methods

* `plot_stake_timeseries(df: pd.DataFrame, save_img=None, stakeIDs=None) -> None`

  * What it does: Plots the timeseries of individual stakes in the dataset, including mean and standard deviation. Allows highlighting specific stakes.
  * Input:
    * `df`: Pandas DataFrame containing all available stakes with monthly time resolution
    * `save_img` (optional): List containing image format and directory to save the figure
    * `stakeIDs` (optional): List of stake IDs to highlight in the figure
  * Output: None (displays and optionally saves a plot)
* `plot_cumulative_smb(df: pd.DataFrame, save_img=None) -> None`

  * What it does: Plots the cumulative annual Surface Mass Balance (SMB) of all stakes in the region of interest.
  * Input:
    * `df`: Pandas DataFrame containing the SMB data
    * `save_img` (optional): List containing image format and directory to save the figure
  * Output: None (displays and optionally saves a plot)
* `convert_to_wgs84(*, data: pd.DataFrame, from_crs: str | int) -> pd.DataFrame`

  * What it does: Transforms coordinates from a given Coordinate Reference System (CRS) to WGS84.
  * Input:
    * `data`: Pandas DataFrame containing 'POINT\_LAT' and 'POINT\_LON' columns
    * `from_crs`: EPSG code (as string or int) of the source coordinate reference system
  * Output: Pandas DataFrame with transformed coordinates in WGS84
* `convert_to_wgms(*, wgms_data_columns: dict, data: pd.DataFrame, date_columns: list[str], smb_columns: list[str]) -> pd.DataFrame`

  * What it does: Converts dataset to WGMS-like format with individual records for each measurement period.
  * Input:
    * `wgms_data_columns`: Dictionary mapping WGMS column names to corresponding data column names
    * `data`: Input Pandas DataFrame containing the raw data
    * `date_columns`: List of column names containing measurement dates
    * `smb_columns`: List of column names containing surface mass balance values
  * Output: Pandas DataFrame in WGMS-like format with individual records for each measurement period
* `get_rgi(*, data: pd.DataFrame, glacier_outlines: gpd.GeoDataFrame) -> pd.DataFrame`

  * What it does: Assigns RGI IDs to stake measurements based on their spatial location within glacier outlines.
  * Input:
    * `data`: Pandas DataFrame containing stake measurements with 'POINT\_LON' and 'POINT\_LAT' columns
    * `glacier_outlines`: GeoDataFrame containing glacier outlines with 'RGIId' column
  * Output: GeoDataFrame with original data and added 'RGIId' column for each stake measurement
