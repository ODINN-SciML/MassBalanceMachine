"""
This code is taken, and refactored, and inspired from the work performed by: Kamilla Hauknes Sjursen

This method fetches the topographical features (variables of interest), for each stake measurement available,
via the OGGM library.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 21/07/2024
"""

import os

import xarray as xr
import pandas as pd

from oggm import cfg, workflow, tasks


def get_topographical_features(
    df: pd.DataFrame, output_fname: str, voi: list[str], rgi_ids: pd.Series
) -> pd.DataFrame:
    """
    Retrieves topographical features for each stake location using the OGGM library and updates the given
    DataFrame with these features.

    Args:
        df (pd.DataFrame): A DataFrame containing columns with RGI IDs, latitude, and longitude for each stake location.
        output_fname (str): The path to the output CSV file where the updated DataFrame will be saved.
        voi (list of str): A list of variables of interest (e.g., ['slope', 'aspect']) to retrieve from the gridded data.
        rgi_ids (pd.Series): A Series of RGI IDs corresponding to the stake locations in the DataFrame.

    Returns:
        pd.DataFrame: The updated DataFrame with topographical features added.

    Raises:
        ValueError: If no stakes are found for the region of interest, or if the resulting DataFrame is empty.
    """

    data = df.copy()

    # Get a list of unique RGI IDs
    rgi_ids_list = _get_unique_rgi_ids(rgi_ids)

    # Initialize the OGGM Config
    _initialize_oggm_config()

    # Initialize the OGGM Glacier Directory, given the available RGI IDs
    glacier_directories = _initialize_glacier_directories(rgi_ids_list)

    # Get all the latitude and longitude positions for all the stakes (with a
    # valid RGI ID)
    filtered_df = _filter_dataframe(df, rgi_ids_list)
    # Group stakes by RGI ID
    grouped_stakes = _group_stakes_by_rgi_id(filtered_df)

    # RGI ID: RGI123
    #    RGIId  POINT_LAT  POINT_LON
    # 0  RGI123       10.0       20.0
    # 1  RGI123       10.5       20.5

    # Load the gridded data for each glacier available in the OGGM Glacier
    # Directory
    gdirs_gridded = _load_gridded_data(glacier_directories, grouped_stakes)

    # Based on the stake location, find the nearest point on the glacier with
    # recorded topographical features
    _retrieve_topo_features(
        data, glacier_directories, gdirs_gridded, grouped_stakes, voi
    )

    # Check if the dataframe is not empty (i.e. no points were found)
    if data.empty:
        raise ValueError(
            "DataFrame is empty, no stakes were found for the region of interest. Please check if your \n"
            "RGIIDs are correct, and your coordinates are in the correct CRS."
        )

    data.to_csv(output_fname, index=False)

    return data


def _get_unique_rgi_ids(rgi_ids: pd.Series) -> list:
    """Get the list of unique RGI IDs."""
    return rgi_ids.dropna().unique().tolist()


def _initialize_oggm_config():
    """Initialize OGGM configuration."""
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["border"] = 10
    cfg.PARAMS["use_multiprocessing"] = True
    cfg.PARAMS["continue_on_error"] = True

    current_path = os.getcwd()
    workspace_path = os.path.join(current_path, "OGGM")
    cfg.PATHS["working_dir"] = workspace_path


def _initialize_glacier_directories(rgi_ids_list: list) -> list:
    """Initialize glacier directories."""
    base_url = "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5_w_data/"
    glacier_directories = workflow.init_glacier_directories(
        rgi_ids_list,
        reset=False,
        from_prepro_level=3, prepro_base_url=base_url, prepro_border=10,
    )

    workflow.execute_entity_task(
        tasks.gridded_attributes, glacier_directories, print_log=True
    )
    return glacier_directories


def _filter_dataframe(df: pd.DataFrame, rgi_ids_list: list) -> pd.DataFrame:
    """Filter the DataFrame to include only the RGI IDs of interest and select only lat/lon columns."""
    return df.loc[df["RGIId"].isin(rgi_ids_list), ["RGIId", "POINT_LAT", "POINT_LON"]]


def _group_stakes_by_rgi_id(
    filtered_df: pd.DataFrame,
) -> pd.api.typing.DataFrameGroupBy:
    """Group latitude and longitude by RGI ID."""
    return filtered_df.groupby("RGIId", sort=False)


def _load_gridded_data(
    glacier_directories: list, grouped_stakes: pd.api.typing.DataFrameGroupBy
) -> list:
    """Load gridded data for each glacier directory."""
    grouped_rgi_ids = set(grouped_stakes.groups.keys())
    return [
        xr.open_dataset(gdir.get_filepath("gridded_data")).load()
        for gdir in glacier_directories
        if gdir.rgi_id in grouped_rgi_ids
    ]


def _retrieve_topo_features(
    df: pd.DataFrame,
    glacier_directories: list,
    gdirs_gridded: list,
    grouped_stakes: pd.api.typing.DataFrameGroupBy,
    voi: list,
) -> None:
    """Find the nearest recorded point with topographical features on the glacier for each stake."""

    for i, j in zip(glacier_directories, gdirs_gridded):
        lat = grouped_stakes.get_group(i.rgi_id)[["POINT_LAT"]].values.flatten()
        lon = grouped_stakes.get_group(i.rgi_id)[["POINT_LON"]].values.flatten()

        rt = (
            j.sel(x=lon, y=lat, method="nearest")[voi]
            .to_dataframe()
            .reset_index(drop=True)
        )

        df.loc[df["RGIId"] == i.rgi_id, voi] = rt[voi]
