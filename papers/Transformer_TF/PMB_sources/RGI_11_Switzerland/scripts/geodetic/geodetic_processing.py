import numpy as np
import pandas as pd
import os
import re
import massbalancemachine as mbm
from tqdm.notebook import tqdm
import xarray as xr

from collections import defaultdict
from papers.Transformer_TF.PMB_sources.RGI_11_Switzerland.scripts.config_CH import *
from papers.Transformer_TF.PMB_sources.RGI_11_Switzerland.scripts.glamos import *


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
