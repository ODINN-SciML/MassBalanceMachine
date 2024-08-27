# MassBalanceMachine Core

In this README, for all packages included in the `massbalancemachine`, all methods that are relevant to the user are listed together with their input and output. The MassBalanceMachine core consists of the following sub-packages:

### 1. Data Processing

The `data_processing` package includes the `Dataset` class and methods to retrieve topographical and meteorological data, and transform the dataset to a monthly resolution.

#### Dataset

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

#### Utils

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

### 2. Data Loader

The `dataloader` package includes the `DataLoader` class that includes methods to split the data into a train and test dataset and create data partitions (splits) for cross validation.

#### Methods

`set_train_test_split(self, *, test_size: float = 0.3, random_seed: int = None, shuffle: bool = True) -> Tuple[Iterator[Any], Iterator[Any]]`

* What it does: Splits the dataset into training and testing sets based on indices.
* Input:
  * `test_size` (float): Proportion of the dataset to include in the test split.
  * `random_seed` (int, optional): Seed for the random number generator. If None, a random seed is used.
  * `shuffle` (bool): Whether to shuffle the data before splitting.
* Output: A tuple of two iterators:
  * `train_iterator`: Iterator for the indices of the training data.
  * `test_iterator`: Iterator for the indices of the testing data.

`get_cv_split(self, *, n_splits: int = 5) -> Tuple[List[Tuple[ndarray, ndarray]]]`

* What it does: Creates a cross-validation split of the training data using GroupKFold.
* Input:
  * `n_splits` (int): Number of splits for cross-validation.
* Output: A tuple containing:
  * A list of tuples, where each tuple represents a fold and contains:
    * `train_indices` (ndarray): Indices for the training data in the fold.
    * `val_indices` (ndarray): Indices for the validation data in the fold.
* Raises:
  * `ValueError`: If `train_indices` is None (i.e., if `set_train_test_split` hasn't been called).

### 3. Models

The `models` package includes different machine learning models. Users can create a new instance of one of these models and then train it with their data.

#### 3.1 CustomXGBoostRegressor

#### Methods

`gridsearch(self, parameters: Dict[str, Union[list, np.ndarray]], splits: Dict[str, Union[list, np.ndarray]], features: pd.DataFrame, targets: np.ndarray, num_jobs: int = -1) -> None`

* What it does: Performs a grid search for hyperparameter tuning using GridSearchCV.
* Input:
  * `parameters` (dict): A dictionary of parameters to search over.
  * `splits` (dict): A dictionary containing cross-validation split information.
  * `features` (pd.DataFrame): The input features for training.
  * `targets` (np.ndarray): The target values for training.
  * `num_jobs` (int, optional): The number of parallel jobs to run. Defaults to -1 (uses all processors).
* Output: None
* Sets:
  * `self.param_search` (GridSearchCV): The fitted GridSearchCV object.

`randomsearch(self, parameters: Dict[str, Union[list, np.ndarray]], n_iter: int, splits: Dict[str, Union[list, np.ndarray]], features: pd.DataFrame, targets: np.ndarray, num_jobs: int = -1) -> None`

* What it does: Performs a randomized search for hyperparameter tuning using RandomizedSearchCV.
* Input:
  * `parameters` (dict): A dictionary of parameters and their distributions to sample from.
  * `n_iter` (int): Number of parameter settings that are sampled.
  * `splits` (dict): A dictionary containing cross-validation split information.
  * `features` (pd.DataFrame): The input features for training.
  * `targets` (np.ndarray): The target values for training.
  * `num_jobs` (int, optional): The number of parallel jobs to run. Defaults to -1 (uses all processors).
* Output: None
* Sets:
  * `self.param_search` (RandomizedSearchCV): The fitted RandomizedSearchCV object.

`save_model(self, fname: str) -> None`

* What it does: Saves a grid search or randomized search CV instance to a file.
* Input:
  * `fname` (str): The filename to save the model.
* Output: None

`load_model(cls, fname: str) -> Union[GridSearchCV, RandomizedSearchCV]`

* What it does: Loads a grid search or randomized search CV instance from a file.
* Input:
  * `fname` (str): The filename from which to load the model.
* Output: The loaded GridSearchCV or RandomizedSearchCV object.
