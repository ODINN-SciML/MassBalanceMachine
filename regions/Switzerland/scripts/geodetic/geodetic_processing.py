import numpy as np
import pandas as pd
import os
import re
import massbalancemachine as mbm
from tqdm.notebook import tqdm
import xarray as xr

from collections import defaultdict
from regions.Switzerland.scripts.config_CH import *
from regions.Switzerland.scripts.glamos import *


# -------------------------------- for MBM geodataloader -----------------------------
def prepare_geo_targets(geodetic_mb, periods_per_glacier, glacier_name=None):
    """
    Build geodetic mass-balance target vectors for one glacier or for all glaciers.

    For each glacier, this function loops over the geodetic periods listed in
    `periods_per_glacier[glacier_name]` and extracts the corresponding geodetic
    mass balance value from `geodetic_mb`.

    The function supports two possible column names for geodetic mass balance:
    - ``Bgeod`` (some inputs)
    - ``Bgeod_mwe_a`` (default for the DOI2025 preliminary file)

    Parameters
    ----------
    geodetic_mb : pandas.DataFrame
        Geodetic mass balance table. Must contain at least:
        - ``glacier_name`` (str)
        - ``Astart`` (int)
        - ``Aend`` (int)
        and one of ``Bgeod`` or ``Bgeod_mwe_a``.
    periods_per_glacier : dict[str, list[tuple[int, int]]]
        Dictionary mapping glacier name -> list of (start_year, end_year) periods.
        Each tuple defines the time window for which a geodetic target is expected.
    glacier_name : str or None, optional
        If provided, returns targets only for this glacier. If None, returns a dict
        of targets for every glacier in `periods_per_glacier`.

    Returns
    -------
    numpy.ndarray or dict[str, numpy.ndarray]
        If `glacier_name` is provided: a 1D array of geodetic targets ordered as in
        `periods_per_glacier[glacier_name]`.
        If `glacier_name` is None: a dict mapping each glacier to its target array.

    Raises
    ------
    KeyError
        If required columns are missing from `geodetic_mb` or `glacier_name` is not
        in `periods_per_glacier`.
    IndexError
        If a (glacier_name, Astart, Aend) combination does not exist in `geodetic_mb`.
        (This happens because the code accesses `.values[0]`.)
    """
    if glacier_name is not None:
        geodetic_MB_target = []
        Bgeod_key = (
            "Bgeod" if "Bgeod" in geodetic_mb.keys() else "Bgeod_mwe_a"
        )  # Handle the dV_DOI2025_allcomb_prelim.csv file
        for geodetic_period in periods_per_glacier[glacier_name]:
            mask = (
                (geodetic_mb.glacier_name == glacier_name)
                & (geodetic_mb.Astart == geodetic_period[0])
                & (geodetic_mb.Aend == geodetic_period[1])
            )
            geodetic_MB_target.append(geodetic_mb[mask][Bgeod_key].values[0])

        return np.array(geodetic_MB_target)
    else:
        return {
            glacier_name: prepare_geo_targets(
                geodetic_mb, periods_per_glacier, glacier_name=glacier_name
            )
            for glacier_name in periods_per_glacier
        }


def build_periods_per_glacier(geodetic_mb):
    """
    Build per-glacier lists of geodetic periods and associated mass balances.

    Iterates through `geodetic_mb` and constructs:
    1) `periods_per_glacier`: glacier -> list of (Astart, Aend) tuples
    2) `geoMB_per_glacier`: glacier -> list of (Bgeod_mwe_a, sigma_mwe_a) tuples

    Only periods with duration >= 5 years are retained.

    Parameters
    ----------
    geodetic_mb : pandas.DataFrame
        Geodetic mass balance table. Must contain columns:
        - ``glacier_name`` (str)
        - ``Astart`` (int)
        - ``Aend`` (int)
        - ``Bgeod_mwe_a`` (float)
        - ``sigma_mwe_a`` (float)

    Returns
    -------
    periods_per_glacier : dict[str, list[tuple[int, int]]]
        Dictionary mapping glacier -> list of (start_year, end_year) periods,
        filtered to periods >= 5 years and sorted by glacier name.
    geoMB_per_glacier : dict[str, list[tuple[float, float]]]
        Dictionary mapping glacier -> list of (geodetic_mb_mwe_a, sigma_mwe_a)
        tuples, in the same order as `periods_per_glacier[glacier]`.

    Notes
    -----
    - If multiple rows share the same (glacier_name, Astart, Aend), all will be added
      unless duplicates are removed upstream.

    Raises
    ------
    KeyError
        If required columns are missing.
    """

    periods_per_glacier = defaultdict(list)
    geoMB_per_glacier = defaultdict(list)

    # Iterate through the DataFrame rows
    for _, row in geodetic_mb.iterrows():
        glacier_name = row["glacier_name"]
        start_year = row["Astart"]
        end_year = row["Aend"]
        geoMB = row["Bgeod_mwe_a"]
        sigma = row["sigma_mwe_a"]

        # Append the (start, end) tuple to the glacier's list
        # Only if period is longer than 5 years
        if end_year - start_year >= 5:
            periods_per_glacier[glacier_name].append((start_year, end_year))
            # append geodetic MB and its uncertainty
            geoMB_per_glacier[glacier_name].append((geoMB, sigma))

    # sort by glacier_list
    periods_per_glacier = dict(sorted(periods_per_glacier.items()))
    geoMB_per_glacier = dict(sorted(geoMB_per_glacier.items()))

    return periods_per_glacier, geoMB_per_glacier


def get_geodetic_MB(cfg):
    """
    Load and filter Swiss geodetic mass balance data to match the project glacier set.

    This function:
    1) Loads glacier ID mapping (RGI v6 <-> SGI) via `get_glacier_ids(cfg)`.
    2) Loads the stake/WGMS-like dataset to determine which RGIIds are used.
    3) Loads the geodetic mass balance CSV and filters it to the matching SGI_IDs.
    4) Adds a `glacier_name` column (project short name) based on SGI_ID.
    5) Fixes invalid date encodings ending with "9999" by replacing with Sep 30 of that year.
    6) Parses `date_start` / `date_end` as datetimes and sets `Astart` / `Aend` to the
       corresponding years.
    7) Filters to glaciers that have potential clear-sky radiation data (PCSR) available
       by checking for `xr_direct_<glacier>.zarr` files.

    Parameters
    ----------
    cfg : object
        Configuration object with attribute `dataPath`. Relies on path constants from
        `config_CH`, including:
        - `path_PMB_GLAMOS_csv`
        - `path_geodetic_MB_glamos`
        - `path_pcsr`
        - `path_glacier_ids`

    Returns
    -------
    pandas.DataFrame
        Filtered geodetic mass balance DataFrame with at least:
        - ``SGI_ID``
        - ``glacier_name``
        - ``date_start`` (datetime64)
        - ``date_end`` (datetime64)
        - ``Astart`` (int)
        - ``Aend`` (int)
        plus any additional columns present in the source file.

    Notes
    -----
    - Despite the header comment in older versions, this function does not currently
      perform duplicate resolution (e.g. choosing the row where `date_end` is closest
      to month-end). If needed, that logic should be added explicitly.
    - The function renames ``A_end`` -> ``Aend`` but assumes the file already contains
      ``date_start`` and ``date_end``.

    Raises
    ------
    FileNotFoundError
        If any of the expected CSV files or directories do not exist.
    KeyError
        If required columns are missing from input tables.
    IndexError
        If SGI/RGI mapping lookups fail (e.g., missing IDs in `path_glacier_ids`).
    """

    # Load necessary data
    glacier_ids = get_glacier_ids(cfg)
    data_glamos = pd.read_csv(
        cfg.dataPath + path_PMB_GLAMOS_csv + "CH_wgms_dataset_all.csv"
    )

    # Read geodetic MB dataset
    geodetic_mb = pd.read_csv(
        cfg.dataPath + path_geodetic_MB_glamos + "dV_DOI2025_allcomb_prelim.csv"
    )

    # Get RGI IDs for the glaciers
    rgi_gl = data_glamos.RGIId.unique()
    sgi_gl = [
        glacier_ids[glacier_ids["rgi_id.v6"] == rgi]["sgi-id"].values[0]
        for rgi in rgi_gl
    ]
    geodetic_mb = geodetic_mb[geodetic_mb["SGI_ID"].isin(sgi_gl)]

    # Add glacier_name to geodetic_mb based on SGI-ID
    glacier_names = [
        glacier_ids[glacier_ids["sgi-id"] == sgi_id].index[0]
        for sgi_id in geodetic_mb["SGI_ID"].values
    ]
    geodetic_mb["glacier_name"] = glacier_names

    # Replace 'claridenL' with 'clariden'
    geodetic_mb["glacier_name"] = geodetic_mb["glacier_name"].replace(
        "claridenL", "clariden"
    )

    # Rename A_end to Aend
    geodetic_mb.rename(columns={"A_end": "Aend"}, inplace=True)

    # geodetic_mb.rename(columns={'A_end': 'Aend'}, inplace=True)
    # geodetic_mb.rename(columns={'A_start': 'Astart'}, inplace=True)

    # Function to replace 9999 with September 30
    def fix_invalid_dates(date):
        date_str = str(date)
        if date_str.endswith("9999"):
            return f"{date_str[:4]}0930"  # Replace '9999' with '0930'
        return date_str

    # Apply the function to both columns
    geodetic_mb["date_start"] = geodetic_mb["date_start"].apply(fix_invalid_dates)
    geodetic_mb["date_end"] = geodetic_mb["date_end"].apply(fix_invalid_dates)

    # Convert to datetime format
    geodetic_mb["date_start"] = pd.to_datetime(
        geodetic_mb["date_start"], format="%Y%m%d", errors="coerce"
    )
    geodetic_mb["date_end"] = pd.to_datetime(
        geodetic_mb["date_end"], format="%Y%m%d", errors="coerce"
    )

    # Manually set Astart and Aend based on date_start and date_end
    geodetic_mb["Astart"] = geodetic_mb["date_start"].dt.year
    geodetic_mb["Aend"] = geodetic_mb["date_end"].dt.year

    glDirect = np.sort(
        [
            re.search(r"xr_direct_(.*?)\.zarr", f).group(1)
            for f in os.listdir(cfg.dataPath + path_pcsr + "zarr/")
        ]
    )

    # filter to glaciers with potential clear sky radiation data
    geodetic_mb = geodetic_mb[geodetic_mb.glacier_name.isin(glDirect)]

    return geodetic_mb


def get_glacier_ids(cfg):
    """
    Load the glacier ID mapping table (short_name -> SGI / RGI identifiers).

    Reads the glacier ID CSV (defined by `path_glacier_ids`), strips whitespace from
    column names, sorts by `short_name`, and sets `short_name` as the index.

    Parameters
    ----------
    cfg : object
        Configuration object with attribute `dataPath`. Uses `path_glacier_ids`
        from `config_CH`.

    Returns
    -------
    pandas.DataFrame
        Glacier ID table indexed by ``short_name``. Expected to contain columns such as
        ``sgi-id`` and ``rgi_id.v6`` (depending on the CSV schema).

    Raises
    ------
    FileNotFoundError
        If the glacier ID CSV file does not exist.
    KeyError
        If the CSV does not contain a ``short_name`` column.
    """
    glacier_ids = pd.read_csv(cfg.dataPath + path_glacier_ids, sep=",")
    glacier_ids.rename(columns=lambda x: x.strip(), inplace=True)
    glacier_ids.sort_values(by="short_name", inplace=True)
    glacier_ids.set_index("short_name", inplace=True)

    return glacier_ids


# -------------------------------- Geodetic inputs -----------------------------
def create_geodetic_input(cfg, glacier_name, periods_per_glacier, to_seasonal=False):
    """
    Assemble MBM geodetic input features for a glacier from yearly grid parquet files.

    This function loads preprocessed per-year glacier grid files stored as Parquet
    (typically containing monthly rows), concatenates them across the full geodetic
    year span for the glacier, and adds identifiers expected by MBM:

    - ``GLWD_ID``: glacier-wide ID (hash of "<glacier>_<year>")
    - ``ID``: re-hashed to be unique across years (hash of "<old_id>_<YEAR>")

    Optionally, the monthly grid data can be aggregated to a seasonal frequency
    via `transform_df_to_seasonal`.

    Parameters
    ----------
    cfg : object
        Configuration object with attribute `dataPath`. Uses `path_glacier_grid_glamos`
        from `config_CH` to locate parquet files.
    glacier_name : str
        Glacier short name used in the folder structure and parquet filenames.
    periods_per_glacier : dict[str, list[tuple[int, int]]]
        Dictionary mapping glacier name -> list of (start_year, end_year) geodetic periods.
        The function determines the overall time span from the minimum and maximum year
        found in these tuples.
    to_seasonal : bool, optional
        If True, aggregate monthly rows to seasonal rows ("winter"/"summer") using
        `transform_df_to_seasonal`. Default is False.

    Returns
    -------
    pandas.DataFrame
        Concatenated dataframe containing all available yearly grids for the glacier,
        potentially aggregated to seasonal frequency. Adds/updates:
        - ``GLWD_ID`` (string/int hash)
        - ``ID`` (recomputed if present)

    Raises
    ------
    KeyError
        If expected columns used downstream (e.g. ``YEAR``) are missing from parquet files.
    """

    # Get the minimum and maximum geodetic years for the glacier
    min_geod_y, max_geod_y = np.min(periods_per_glacier[glacier_name]), np.max(
        periods_per_glacier[glacier_name]
    )

    df_X_geod = pd.DataFrame()
    # Assemble the blocs per year
    for year in range(min_geod_y, max_geod_y + 1):
        # Read the glacier grid file (monthly)
        file_name = f"{glacier_name}_grid_{year}.parquet"
        file_path = os.path.join(
            cfg.dataPath, path_glacier_grid_glamos, glacier_name, file_name
        )

        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found, skipping...")
            continue

        # Load parquet input glacier grid file in monthly format (pre-processed)
        df_grid_monthly = pd.read_parquet(file_path)
        df_grid_monthly.drop_duplicates(inplace=True)

        # Optional: transform to seasonal frequency
        if to_seasonal:
            df_grid_seas = transform_df_to_seasonal(df_grid_monthly)
            df_grid = df_grid_seas
        else:
            df_grid = df_grid_monthly

        # Add GLWD_ID (unique glacier-wide ID corresponding to the year)
        df_grid["GLWD_ID"] = mbm.data_processing.utils.get_hash(
            f"{glacier_name}_{year}"
        )

        # ID is not unique anymore (because of the way the monthly grids were pre-processed),
        # so recompute them:
        if "ID" in df_grid.columns:
            df_grid["ID"] = df_grid.apply(
                lambda x: mbm.data_processing.utils.get_hash(f"{x.ID}_{x.YEAR}"), axis=1
            )
        else:
            print(
                f"Warning: 'ID' column missing in {file_name}, skipping ID modification."
            )

        # Append to the final dataframe
        df_X_geod = pd.concat([df_X_geod, df_grid], ignore_index=True)

    return df_X_geod


def has_geodetic_input(cfg, glacier_name, periods_per_glacier):
    """
    Check whether a glacier has a complete set of yearly geodetic input parquet files.

    The function determines the glacier's geodetic year span from
    `periods_per_glacier[glacier_name]` and verifies that a parquet file exists for
    every year in the inclusive range [min_year, max_year].

    Parameters
    ----------
    cfg : object
        Configuration object with attribute `dataPath`. Uses `path_glacier_grid_glamos`
        from `config_CH` to locate parquet files.
    glacier_name : str
        Glacier short name used in parquet filenames and folder structure.
    periods_per_glacier : dict[str, list[tuple[int, int]]]
        Dictionary mapping glacier name -> list of (start_year, end_year) periods.

    Returns
    -------
    bool
        True if all parquet files `<glacier_name>_grid_<year>.parquet` exist for every
        year between the minimum and maximum geodetic year; otherwise False.

    Raises
    ------
    KeyError
        If `glacier_name` is not found in `periods_per_glacier`.
    """

    # Get the minimum and maximum geodetic years for the glacier
    min_geod_y, max_geod_y = np.min(periods_per_glacier[glacier_name]), np.max(
        periods_per_glacier[glacier_name]
    )

    for year in range(min_geod_y, max_geod_y + 1):
        # Check that the glacier grid file exists
        file_name = f"{glacier_name}_grid_{year}.parquet"
        file_path = os.path.join(
            cfg.dataPath, path_glacier_grid_glamos, glacier_name, file_name
        )

        if not os.path.exists(file_path):
            return False
    return True


def transform_df_to_seasonal(data_monthly):
    """
    Aggregate monthly glacier grid rows to seasonal resolution (winter/summer).

    The function assigns each monthly row to one of two seasons based on the ``MONTHS``
    column:

    - winter: Oct–Mar (``["oct","nov","dec","jan","feb","mar"]``)
    - summer: Apr–Sep (``["apr","may","jun","jul","aug","sep"]``)

    It then groups by (``ID``, ``SEASON``) and:
    - averages a predefined list of numerical columns
    - keeps one representative row for categorical columns (assumes consistency
      within each group)
    - re-attaches a season-specific MONTHS list (the full list of months per season)

    Parameters
    ----------
    data_monthly : pandas.DataFrame
        Monthly-resolution input data. Must contain at least:
        - ``ID`` : point/grid identifier
        - ``MONTHS`` : month labels as strings (e.g., "oct", "jan", ...)
        and should contain the numerical columns listed inside the function.

    Returns
    -------
    pandas.DataFrame
        Seasonal-resolution dataframe with one row per (ID, SEASON). Contains:
        - ``SEASON`` : {"winter", "summer"}
        - averaged numerical columns
        - categorical columns carried through
        - ``MONTHS`` : list of months belonging to that season


    Raises
    ------
    KeyError
        If required columns (e.g., ``ID``, ``MONTHS`` or one of the numerical columns)
        are missing.
    """
    # Aggregate to seasonal MB:
    months_winter = ["oct", "nov", "dec", "jan", "feb", "mar"]
    months_summer = ["apr", "may", "jun", "jul", "aug", "sep"]

    data_monthly["SEASON"] = np.where(
        data_monthly["MONTHS"].isin(months_winter), "winter", "summer"
    )

    numerical_cols = [
        "YEAR",
        "POINT_LON",
        "POINT_LAT",
        "POINT_BALANCE",
        "ALTITUDE_CLIMATE",
        "ELEVATION_DIFFERENCE",
        "POINT_ELEVATION",
        "N_MONTHS",
        "aspect_sgi",
        "slope_sgi",
        "hugonnet_dhdt",
        "consensus_ice_thickness",
        "millan_v",
        "t2m",
        "tp",
        "slhf",
        "sshf",
        "ssrd",
        "fal",
        "str",
        "u10",
        "v10",
        "pcsr",
    ]

    # All other non-numerical, non-grouping columns are assumed categorical
    exclude_cols = set(numerical_cols + ["ID", "MONTHS", "SEASON"])
    categorical_cols = [col for col in data_monthly.columns if col not in exclude_cols]

    # Aggregate numerical
    data_seas_num = data_monthly.groupby(["ID", "SEASON"], as_index=False)[
        numerical_cols
    ].mean()

    # Get one row per group for categoricals (assumes values per group are consistent)
    data_seas_cat = data_monthly[["ID", "SEASON"] + categorical_cols].drop_duplicates(
        subset=["ID", "SEASON"]
    )

    # Merge numerical + categorical
    data_seas = pd.merge(data_seas_num, data_seas_cat, on=["ID", "SEASON"], how="inner")

    # Add MONTHS list back in
    season_months = {"winter": months_winter, "summer": months_summer}
    data_seas["MONTHS"] = data_seas["SEASON"].map(season_months)

    return data_seas


def process_geodetic_mb_comparison(
    glacier_list,
    path_SMB_GLAMOS_csv,
    periods_per_glacier,
    geoMB_per_glacier,  # {'aletsch': [(mb, sigma), ...], ...}
    gl_area,
    test_glaciers,
    path_predictions,
    cfg,
):
    """
    Compare modeled, glaciological, and geodetic mass balances over multi-year periods.

    For each glacier and specified time period, the function:
      - computes mean and interannual variability of modeled annual mass balance (MBM),
      - computes mean and interannual variability of GLAMOS annual mass balance,
      - attaches corresponding geodetic mass balance estimates and uncertainties,
      - aggregates metadata such as glacier area, period length, and test/train flag.

    Periods are skipped if required model data or geodetic information is missing.

    Parameters
    ----------
    glacier_list : sequence of str
        List of glacier identifiers to process.
    path_SMB_GLAMOS_csv : str
        Path to the directory containing GLAMOS CSV files.
    periods_per_glacier : dict
        Mapping glacier_name -> list of (start_year, end_year) tuples defining
        comparison periods.
    geoMB_per_glacier : dict
        Mapping glacier_name -> list of (geodetic_mb, sigma) tuples corresponding
        to the periods in `periods_per_glacier`.
    gl_area : dict
        Mapping glacier_name -> glacier area (e.g. km²).
    test_glaciers : sequence of str
        List of glacier identifiers flagged as test glaciers.
    path_predictions : str
        Path to the directory containing MBM prediction Zarr files.
    cfg : object
        Configuration object used by GLAMOS helper functions.

    Returns
    -------
    pandas.DataFrame
        Table containing period-aggregated mass balance statistics with columns:
        ['MBM MB', 'GLAMOS MB', 'MBM MB std', 'GLAMOS MB std',
         'Geodetic MB', 'Geodetic MB sigma', 'GLACIER', 'Period Length',
         'Test Glacier', 'Area', 'start_year', 'end_year'].
        Rows are sorted by glacier area.
    """
    # Storage
    mbm_mb_mean, glamos_mb_mean = [], []
    mbm_mb_var, glamos_mb_var = [], []
    geodetic_mb, geodetic_sigma = [], []
    gl, gl_type, area = [], [], []
    period_len, start_year, end_year = [], [], []

    for glacier_name in tqdm(glacier_list, desc="Processing glaciers"):
        # Load GLAMOS annual balances
        glamos_file = os.path.join(
            path_SMB_GLAMOS_csv, "fix", f"{glacier_name}_fix.csv"
        )
        if os.path.exists(glamos_file):
            GLAMOS_glwmb = get_GLAMOS_glwmb(glacier_name, cfg)
            if GLAMOS_glwmb is None:
                GLAMOS_glwmb = pd.DataFrame()
        else:
            print(f"GLAMOS file not found for {glacier_name}. Using NaNs.")
            GLAMOS_glwmb = pd.DataFrame()

        periods = periods_per_glacier.get(glacier_name, [])
        geo_tuples = geoMB_per_glacier.get(glacier_name, [])

        if not periods or not geo_tuples:
            print(f"Skipping {glacier_name}: No geodetic mass balance data available.")
            continue

        # Path to model predictions
        folder_path = os.path.join(path_predictions, glacier_name)

        for i, period in enumerate(periods):
            # require matching geodetic tuple by index
            if (
                i >= len(geo_tuples)
                or not isinstance(geo_tuples[i], (tuple, list))
                or len(geo_tuples[i]) < 2
            ):
                print(
                    f"Skipping {glacier_name} {period}: missing geodetic (mb, sigma) tuple at index {i}."
                )
                continue

            # Special case skip
            if period[1] == 2021 and glacier_name == "silvretta":
                continue

            # Check input availability (your helper)
            is_missing, years_missing = check_missing_years(
                folder_path, glacier_name, period
            )
            if is_missing:
                print(
                    f"Skipping {glacier_name} {period}: Missing years: {years_missing}"
                )
                continue

            mbm_mb, glamos_mb = [], []
            for year in range(period[0], period[1] + 1):
                zarr_path = os.path.join(
                    folder_path, f"{glacier_name}_{year}_annual.zarr"
                )
                if not os.path.exists(zarr_path):
                    print(f"Warning: Missing MBM file for {glacier_name} ({year}).")
                    mbm_mb.append(np.nan)
                else:
                    # Zarr -> open_zarr (not open_dataset)
                    ds = xr.open_zarr(zarr_path)
                    mbm_mb.append(ds["pred_masked"].mean().values)
                glamos_mb.append(GLAMOS_glwmb["GLAMOS Balance"].get(year, np.nan))

            # Aggregate period stats
            mbm_mb_mean.append(np.nanmean(mbm_mb))
            mbm_mb_var.append(np.nanstd(mbm_mb))
            glamos_mb_mean.append(np.nanmean(glamos_mb))
            glamos_mb_var.append(np.nanstd(glamos_mb))

            # Geodetic (mb, sigma)
            g_mb, g_sig = geo_tuples[i][0], geo_tuples[i][1]
            geodetic_mb.append(g_mb)
            geodetic_sigma.append(g_sig)

            # Meta
            gl.append(glacier_name)
            gl_type.append(glacier_name in test_glaciers)
            period_len.append(period[1] - period[0])  # keep your original convention
            area.append(gl_area.get(glacier_name, np.nan))
            start_year.append(period[0])
            end_year.append(period[1])

    # Assemble DataFrame
    df_all = pd.DataFrame(
        {
            "MBM MB": mbm_mb_mean,
            "GLAMOS MB": glamos_mb_mean,
            "MBM MB std": mbm_mb_var,
            "GLAMOS MB std": glamos_mb_var,
            "Geodetic MB": geodetic_mb,
            "Geodetic MB sigma": geodetic_sigma,
            "GLACIER": gl,
            "Period Length": period_len,
            "Test Glacier": gl_type,
            "Area": area,
            "start_year": start_year,
            "end_year": end_year,
        }
    )

    df_all.sort_values(by="Area", inplace=True, ascending=True)
    return df_all


def check_missing_years(folder_path, glacier_name, period):
    start_year, end_year = period
    expected_years = set(range(start_year, end_year + 1))

    # Extract years from filenames
    available_years = set()
    pattern = re.compile(rf"{glacier_name}_(\d{{4}})_annual\.zarr")

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            year = int(match.group(1))
            available_years.add(year)

    missing_years = list(expected_years - available_years)
    missing_years.sort()
    if missing_years:
        return True, missing_years
    else:
        return False, []
