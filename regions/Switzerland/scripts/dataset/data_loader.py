import re
import logging

from sklearn.model_selection import (
    train_test_split,
)

import massbalancemachine as mbm
from regions.Switzerland.scripts.config_CH import *
from regions.Switzerland.scripts.utils import *


def get_stakes_data(cfg):
    """
    Load and filter GLAMOS stake mass-balance observations.

    This function reads the processed GLAMOS/WGMS stake dataset from disk and
    restricts it to glaciers for which potential clear-sky radiation (PCSR)
    data are available. Availability of PCSR is determined by the presence of
    corresponding Zarr files in the configured PCSR directory.

    Parameters
    ----------
    cfg : object
        Configuration object containing at least a `dataPath` attribute that
        points to the base directory where GLAMOS and PCSR data are stored.

    Returns
    -------
    pandas.DataFrame
        Filtered GLAMOS stake dataset containing only records for glaciers
        that have corresponding PCSR data available.

    Notes
    -----
    - The function expects the file
      ``CH_wgms_dataset_all.csv`` to be located under:
        ``cfg.dataPath + path_PMB_GLAMOS_csv``.
    - Glacier names are matched against available PCSR files named in the form
      ``xr_direct_<glacier>.zarr``.
    - Glaciers without PCSR data are removed from the returned dataset.
    - The function does not modify or validate the content of the CSV file
      beyond this filtering step.
    """
    data_glamos = pd.read_csv(
        cfg.dataPath + path_PMB_GLAMOS_csv + "CH_wgms_dataset_all.csv"
    )

    # Glaciers with data of potential clear sky radiation
    # Format to same names as stakes:
    glDirect = np.sort(
        [
            re.search(r"xr_direct_(.*?)\.zarr", f).group(1)
            for f in os.listdir(cfg.dataPath + path_pcsr + "zarr/")
        ]
    )

    # Filter out glaciers without data:
    data_glamos = data_glamos[data_glamos.GLACIER.isin(glDirect)]
    return data_glamos


def process_or_load_data(
    run_flag,
    df,
    paths,
    cfg,
    vois_climate,
    vois_topographical,
    add_pcsr=True,
    region_name="CH",
    region_id=11,
    output_file="CH_wgms_dataset_monthly_full.csv",
):
    """
    Process GLAMOS stake data into monthly resolution or load an existing processed file.

    Depending on the value of `run_flag`, this function either:
      - performs full preprocessing of the input GLAMOS dataset (adding climate
        features, optional radiation, and converting to monthly resolution), or
      - loads a previously processed CSV file from disk.

    Processing mode (run_flag=True)
    --------------------------------
    When `run_flag` is True, the function performs the following steps:

    1) Adds a unique glacier-wide identifier (GLWD_ID) for each (GLACIER, YEAR).
    2) Wraps the input DataFrame in an mbm.data_processing.Dataset object.
    3) Adds ERA5 climate features (temperature, precipitation, etc.).
    4) Optionally adds potential clear-sky radiation (PCSR) data.
    5) Converts the dataset to monthly temporal resolution.
    6) Recomputes GLWD_ID for the monthly dataset.
    7) Saves the final monthly dataset to CSV.
    8) Returns the processed DataFrame.

    Load mode (run_flag=False)
    ---------------------------
    When `run_flag` is False, the function attempts to read an already
    processed CSV file from disk. If successful, it returns the loaded
    DataFrame; otherwise, it logs an error and returns None.

    Parameters
    ----------
    run_flag : bool
        If True, perform full data processing.
        If False, load an existing processed dataset from CSV.

    data_glamos : pandas.DataFrame
        Raw GLAMOS stake dataset to be processed. Only used when `run_flag=True`.

    paths : dict
        Dictionary containing required file paths. Must include at least:
        - "csv_path": base directory for CSV input/output files
        - "era5_climate_data": path to ERA5 monthly climate NetCDF
        - "geopotential_data": path to ERA5 geopotential data
        - "radiation_save_path": path to PCSR radiation data (if add_pcsr=True)

    cfg : object
        Configuration object used to initialize the Dataset wrapper.
        Must contain attributes such as `metaData` required by
        `convert_to_monthly`.

    vois_climate : list of str
        Names of climate variables to include in the monthly conversion.

    vois_topographical : list of str
        Names of topographic variables to include in the monthly conversion.

    add_pcsr : bool, optional (default=True)
        If True, potential clear-sky radiation features are added before
        monthly conversion.

    output_file : str, optional
        Name of the CSV file used to save or load the processed monthly data.
        The file is written to / read from `paths["csv_path"]`.

    Returns
    -------
    pandas.DataFrame or None
        - Processed monthly dataset if `run_flag=True`
        - Loaded dataset if `run_flag=False`
        - None if processing or loading fails

    Raises
    ------
    FileNotFoundError
        If `run_flag=False` and the expected processed CSV file does not exist.

    Notes
    -----
    - The function logs detailed information about dataset sizes at each step.
    - A new column `GLWD_ID` is added to uniquely identify glacier-year
      combinations for geodetic mass balance applications.
    - In load mode, any duplicated legacy columns such as "YEAR.1",
      "POINT_LAT.1", or "POINT_LON.1" are removed automatically.
    - The function assumes that the mbm.data_processing module and its
      Dataset class are available and correctly configured.
    """
    if run_flag:
        # Filter data
        logging.info(
            "Running on %d glaciers:\n%s",
            len(df.GLACIER.unique()),
            df.GLACIER.unique(),
        )

        # Add a glacier-wide ID (used for geodetic MB)
        df["GLWD_ID"] = df.apply(
            lambda x: mbm.data_processing.utils.get_hash(f"{x.GLACIER}_{x.YEAR}"),
            axis=1,
        )
        df["GLWD_ID"] = df["GLWD_ID"].astype(str)

        # Create dataset
        dataset_gl = mbm.data_processing.Dataset(
            cfg=cfg,
            data=df,
            region_name=region_name,
            region_id=region_id,
            data_path=paths["csv_path"],
        )
        logging.info("Number of winter and annual samples: %d", len(df))
        logging.info(
            "Number of annual samples: %d",
            len(df[df.PERIOD == "annual"]),
        )
        logging.info(
            "Number of winter samples: %d",
            len(df[df.PERIOD == "winter"]),
        )

        # Add climate data
        logging.info("Adding climate features...")
        try:
            dataset_gl.get_climate_features(
                climate_data=paths["era5_climate_data"],
                geopotential_data=paths["geopotential_data"],
                change_units=True,
            )
        except Exception as e:
            logging.error("Failed to add climate features: %s", e)
            return None

        if add_pcsr:
            # Add radiation data
            logging.info("Adding potential clear sky radiation...")
            logging.info("Shape before adding radiation: %s", dataset_gl.data.shape)
            dataset_gl.get_potential_rad(paths["radiation_save_path"])
            logging.info("Shape after adding radiation: %s", dataset_gl.data.shape)

        # Convert to monthly resolution
        logging.info("Converting to monthly resolution...")
        if add_pcsr:
            dataset_gl.convert_to_monthly(
                meta_data_columns=cfg.metaData,
                vois_climate=vois_climate + ["pcsr"],
                vois_topographical=vois_topographical,
            )
        else:
            dataset_gl.convert_to_monthly(
                meta_data_columns=cfg.metaData,
                vois_climate=vois_climate,
                vois_topographical=vois_topographical,
            )

        # add glwd_id
        data_monthly = dataset_gl.data

        data_monthly["GLWD_ID"] = data_monthly.apply(
            lambda x: mbm.data_processing.utils.get_hash(f"{x.GLACIER}_{x.YEAR}"),
            axis=1,
        )
        data_monthly["GLWD_ID"] = data_monthly["GLWD_ID"].astype(str)

        logging.info("Number of monthly rows: %d", len(data_monthly))
        logging.info("Columns in the dataset: %s", data_monthly.columns)

        # Save processed data
        output_file = os.path.join(paths["csv_path"], output_file)
        data_monthly.to_csv(output_file, index=False)
        logging.info("Processed data saved to: %s", output_file)

        return data_monthly
    else:
        # Load preprocessed data
        try:
            input_file = os.path.join(paths["csv_path"], output_file)

            # check that input_file exists
            if not os.path.isfile(input_file):
                logging.error("Input file does not exist: %s", input_file)
                raise FileNotFoundError(f"Input file not found: {input_file}")

            data_monthly = pd.read_csv(input_file)

            filt = data_monthly.filter(["YEAR.1", "POINT_LAT.1", "POINT_LON.1"])
            data_monthly.drop(filt, inplace=True, axis=1)

            logging.info("Loaded preprocessed data.")
            logging.info("Number of monthly rows: %d", len(data_monthly))
            logging.info(
                "Number of annual rows: %d",
                len(data_monthly[data_monthly.PERIOD == "annual"]),
            )
            logging.info(
                "Number of winter rows: %d",
                len(data_monthly[data_monthly.PERIOD == "winter"]),
            )

            return data_monthly
        except FileNotFoundError as e:
            logging.error("Preprocessed data file not found: %s", e)
            return None


def get_CV_splits(
    dataloader_gl, test_split_on="YEAR", test_splits=None, random_state=0, test_size=0.2
):
    """
    Create training, test, and cross-validation splits from a glacier dataloader.

    This function partitions the data contained in a `dataloader_gl` object into
    train and test subsets based on a specified column (e.g., YEAR, GLACIER, etc.).
    It then generates cross-validation splits from the training data using the
    dataloader's built-in CV split functionality.

    The splitting can be performed in two ways:

    1) Automatic split:
       If `test_splits` is None, unique values from the column specified by
       `test_split_on` are randomly divided into training and test groups using
       sklearn's `train_test_split`.

    2) Manual split:
       If `test_splits` is provided, those values are explicitly used as the test
       set, and all remaining values are used for training.

    After splitting, the function:
      - Sets custom train/test indices inside the dataloader object.
      - Extracts feature DataFrames and target arrays for both sets.
      - Creates cross-validation folds on the training data.
      - Returns structured dictionaries describing the splits.

    Parameters
    ----------
    dataloader_gl : object
        A data loader object (e.g., mbm.data_processing.Dataset) containing a
        `.data` pandas DataFrame and methods for setting train/test indices and
        generating CV folds.

    test_split_on : str, optional (default="YEAR")
        Name of the column in `dataloader_gl.data` used to define the splits.
        Typical choices are "YEAR", "GLACIER", or any other grouping variable.

    test_splits : list or array-like, optional
        Specific values of `test_split_on` to be used as the test set.
        If None, the split is generated automatically using `test_size`.

    random_state : int, optional (default=0)
        Random seed used when generating automatic train/test splits.

    test_size : float, optional (default=0.2)
        Fraction of unique split values to use as the test set when
        `test_splits` is not provided.

    Returns
    -------
    cv_splits : list
        Cross-validation splits generated from the training set using
        `dataloader_gl.get_cv_split`.

    test_set : dict
        Dictionary describing the test set with the following keys:
        - "df_X": pandas.DataFrame of test features
        - "y": numpy.ndarray of test targets (POINT_BALANCE)
        - "meas_id": unique measurement IDs in the test set
        - "splits_vals": unique values of `test_split_on` in the test set

    train_set : dict
        Dictionary describing the training set with the following keys:
        - "df_X": pandas.DataFrame of training features
        - "y": numpy.ndarray of training targets (POINT_BALANCE)
        - "meas_id": unique measurement IDs in the training set
        - "splits_vals": unique values of `test_split_on` in the training set

    Notes
    -----
    - The function assumes that `dataloader_gl.data` contains at least the
      following columns:
        * POINT_BALANCE – target variable
        * ID – measurement identifier
        * the column specified by `test_split_on`

    - Cross-validation folds are created only on the training set using
      group-based splitting (`type_fold="group-meas-id"`), ensuring that
      measurements with the same ID remain together in the same fold.

    - The function modifies the internal state of `dataloader_gl` by calling
      `set_custom_train_test_indices`.
    """
    # Split into training and test splits with train_test_split
    if test_splits is None:
        train_splits, test_splits = train_test_split(
            dataloader_gl.data[test_split_on].unique(),
            test_size=test_size,
            random_state=random_state,
        )
    else:
        split_data = dataloader_gl.data[test_split_on].unique()
        train_splits = [x for x in split_data if x not in test_splits]

    train_indices = dataloader_gl.data[
        dataloader_gl.data[test_split_on].isin(train_splits)
    ].index
    test_indices = dataloader_gl.data[
        dataloader_gl.data[test_split_on].isin(test_splits)
    ].index

    dataloader_gl.set_custom_train_test_indices(train_indices, test_indices)

    # Get the features and targets of the training data for the indices as defined above, that will be used during the cross validation.
    df_X_train = dataloader_gl.data.iloc[train_indices]
    y_train = df_X_train["POINT_BALANCE"].values
    train_meas_id = df_X_train["ID"].unique()

    # Get test set
    df_X_test = dataloader_gl.data.iloc[test_indices]
    y_test = df_X_test["POINT_BALANCE"].values
    test_meas_id = df_X_test["ID"].unique()

    # Values split in training and test set
    train_splits = df_X_train[test_split_on].unique()
    test_splits = df_X_test[test_split_on].unique()

    # Create the CV splits based on the training dataset. The default value for the number of splits is 5.
    cv_splits = dataloader_gl.get_cv_split(n_splits=5, type_fold="group-meas-id")

    test_set = {
        "df_X": df_X_test,
        "y": y_test,
        "meas_id": test_meas_id,
        "splits_vals": test_splits,
    }
    train_set = {
        "df_X": df_X_train,
        "y": y_train,
        "splits_vals": train_splits,
        "meas_id": train_meas_id,
    }

    return cv_splits, test_set, train_set


def prepare_monthly_dfs_with_padding(
    *,
    cfg,
    df_region,
    region_name,
    region_id,
    paths,
    test_glaciers,
    vois_climate,
    vois_topographical,
    run_flag=False,
    output_file_monthly=None,
    output_file_monthly_aug=None,
    add_pcsr=False,
    from_date_aug_mmdd="0801",
):
    """
    Prepare monthly datasets and CV splits for a region using:
    (1) original FROM_DATE and
    (2) FROM_DATE shifted to a fixed MMDD (default: Aug 01).

    Parameters
    ----------
    cfg : object
        Configuration object used throughout the pipeline.
    df_region : pandas.DataFrame
        Input WGMS-style point dataset for the region.
    region_name : str
        Short region label (e.g., "FR").
    region_id : int
        RGI region id (e.g., 11 for Central Europe).
    paths : dict
        Dictionary with required data paths, e.g.:
        {
            "csv_path": "...",
            "era5_climate_data": "...nc",
            "geopotential_data": "...nc"
        }
    test_glaciers : list of str
        Glacier names to hold out as test set.
    vois_climate : list
        Climate variables of interest.
    vois_topographical : list
        Topographic variables of interest.
    run_flag : bool
        Whether to recompute monthly datasets or load from file.
    output_file_monthly, output_file_monthly_aug : str or None
        Output CSV filenames. Defaults are auto-generated from region_name.
    add_pcsr : bool
        Passed to `process_or_load_data`.
    from_date_aug_mmdd : str
        Month+day string used to override FROM_DATE (default "0801").

    Returns
    -------
    dict
        Dictionary containing:
        - data_monthly, df_train, df_test
        - data_monthly_aug, df_train_aug, df_test_aug
        - train_glaciers, missing_test_glaciers
        - months_head_pad_aug, months_tail_pad_aug
    """

    if output_file_monthly is None:
        output_file_monthly = f"{region_name}_wgms_dataset_monthly.csv"
    if output_file_monthly_aug is None:
        output_file_monthly_aug = f"{region_name}_wgms_dataset_monthly_Aug.csv"

    # ---- Monthly (original dates) ----
    data_monthly = process_or_load_data(
        run_flag=run_flag,
        df=df_region,
        paths=paths,
        cfg=cfg,
        vois_climate=vois_climate,
        vois_topographical=vois_topographical,
        region_name=region_name,
        region_id=region_id,
        add_pcsr=add_pcsr,
        output_file=output_file_monthly,
    )

    dataloader = mbm.dataloader.DataLoader(
        cfg, data=data_monthly, random_seed=cfg.seed, meta_data_columns=cfg.metaData
    )

    existing_glaciers = set(data_monthly.GLACIER.unique())
    missing_test_glaciers = [g for g in test_glaciers if g not in existing_glaciers]
    train_glaciers = sorted(existing_glaciers - set(test_glaciers))

    splits, test_set, train_set = get_CV_splits(
        dataloader,
        test_split_on="GLACIER",
        test_splits=test_glaciers,
        random_state=cfg.seed,
    )

    df_train = train_set["df_X"].copy()
    df_train["y"] = train_set["y"]

    df_test = test_set["df_X"].copy()
    df_test["y"] = test_set["y"]

    # ---- Monthly with August start ----
    df_region_aug = df_region.copy()
    year = pd.to_datetime(
        df_region_aug["FROM_DATE"].astype(str), format="%Y%m%d"
    ).dt.year
    df_region_aug["FROM_DATE"] = (year.astype(str) + from_date_aug_mmdd).astype(int)

    months_head_pad_aug, months_tail_pad_aug = (
        mbm.data_processing.utils._compute_head_tail_pads_from_df(df_region_aug)
    )

    data_monthly_aug = process_or_load_data(
        run_flag=run_flag,
        df=df_region_aug,
        paths=paths,
        cfg=cfg,
        vois_climate=vois_climate,
        vois_topographical=vois_topographical,
        region_name=region_name,
        region_id=region_id,
        add_pcsr=add_pcsr,
        output_file=output_file_monthly_aug,
    )

    dataloader_aug = mbm.dataloader.DataLoader(
        cfg, data=data_monthly_aug, random_seed=cfg.seed, meta_data_columns=cfg.metaData
    )

    splits_aug, test_set_aug, train_set_aug = get_CV_splits(
        dataloader_aug,
        test_split_on="GLACIER",
        test_splits=test_glaciers,
        random_state=cfg.seed,
    )

    df_train_aug = train_set_aug["df_X"].copy()
    df_train_aug["y"] = train_set_aug["y"]

    df_test_aug = test_set_aug["df_X"].copy()
    df_test_aug["y"] = test_set_aug["y"]

    return {
        "data_monthly": data_monthly,
        "df_train": df_train,
        "df_test": df_test,
        "data_monthly_aug": data_monthly_aug,
        "df_train_aug": df_train_aug,
        "df_test_aug": df_test_aug,
        "train_glaciers": train_glaciers,
        "missing_test_glaciers": missing_test_glaciers,
        "months_head_pad": months_head_pad_aug,
        "months_tail_pad": months_tail_pad_aug,
    }
