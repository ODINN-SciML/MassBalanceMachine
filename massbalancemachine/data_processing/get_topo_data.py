"""
This code is taken, and refactored, and inspired from the work performed by: Kamilla Hauknes Sjursen

This method fetches the topographical features (variables of interest), for each stake measurement available,
via the OGGM library.

Date Created: 21/07/2024
"""

import os
import config

import xarray as xr
import pandas as pd
import numpy as np
import pyproj

from data_processing.Product import Product
from data_processing.product_utils import rgi_id_to_folders, data_path
from data_processing.glacier_utils import create_dem_file_RGI, generate_svf_file
from data_processing.oggm_utils import (
    _initialize_oggm_config,
    _initialize_glacier_directories,
)


def get_topographical_features(
    df: pd.DataFrame,
    output_fname: str,
    voi: "list[str]",
    rgi_ids: pd.Series,
    custom_working_dir: str,
    cfg: config.Config,
) -> pd.DataFrame:
    """
    Retrieves topographical features for each stake location using the OGGM library and updates the given
    DataFrame with these features.

    Args:
        df (pd.DataFrame): A DataFrame containing columns with RGI IDs, latitude, and longitude for each stake location.
        output_fname (str): The path to the output CSV file where the updated DataFrame will be saved.
        voi (list of str): A list of variables of interest (e.g., ['slope', 'aspect']) to retrieve from the gridded data.
        rgi_ids (pd.Series): A Series of RGI IDs corresponding to the stake locations in the DataFrame.
        custom_working_dir (str): The path to the custom working directory for OGGM data.
        cfg (config.Config): Configuration instance.
    Returns:
        pd.DataFrame: The updated DataFrame with topographical features added.

    Raises:
        ValueError: If no stakes are found for the region of interest, or if the resulting DataFrame is empty.
    """

    data = df.copy()

    # Get a list of unique RGI IDs
    rgi_ids_list = _get_unique_rgi_ids(rgi_ids)

    # Initialize the OGGM Config
    _initialize_oggm_config(custom_working_dir)

    # Initialize the OGGM Glacier Directory, given the available RGI IDs
    glacier_directories = _initialize_glacier_directories(rgi_ids_list, cfg)

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

    # Add sky view factor if needed
    if "svf" in voi:
        gdirs_svf = _load_gridded_svf(glacier_directories, grouped_stakes, cfg)
        for i in range(len(gdirs_gridded)):
            assert all(gdirs_svf[i].x == gdirs_gridded[i].x) and all(
                gdirs_svf[i].y == gdirs_gridded[i].y
            )
            assert gdirs_svf[i].pyproj_srs == gdirs_gridded[i].pyproj_srs
            gdirs_gridded[i]["svf"] = gdirs_svf[i]["svf"]
        del gdirs_svf

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

    if output_fname is not None:
        data.to_csv(output_fname, index=False)

    return data


def glacier_cell_area(rgi_id: str, custom_working_dir: str, cfg: config.Config):
    """Given a `rgi_id` gets the cell area of the grid used to discretize the glacier in OGGM."""

    # Initialize the OGGM Config
    _initialize_oggm_config(custom_working_dir)

    # Initialize the OGGM Glacier Directory, given the RGI ID
    gdirs = _initialize_glacier_directories([rgi_id], cfg)

    # Get oggm data for that RGI ID
    oggm_rgis = [gdir.rgi_id for gdir in gdirs]
    if rgi_id not in oggm_rgis:
        raise ValueError(f"RGI ID {rgi_id} not found in OGGM data")
    for gdir in gdirs:
        if gdir.rgi_id == rgi_id:
            break
    with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
        ds = ds.load()
    glacier_mask = np.where(
        ds["glacier_mask"].values == 0, np.nan, ds["glacier_mask"].values
    )

    cell_area = abs(np.diff(ds.x).mean() * np.diff(ds.y).mean())
    return cell_area


def get_glacier_mask(rgi_id: str, custom_working_dir: str, cfg: config.Config):
    """Given a `rgi_id` gets glacier xarray from OGGM and masks it over the glacier outline."""

    # Initialize the OGGM Config
    _initialize_oggm_config(custom_working_dir)

    # Initialize the OGGM Glacier Directory, given the RGI ID
    gdirs = _initialize_glacier_directories([rgi_id], cfg)

    # Get oggm data for that RGI ID
    oggm_rgis = [gdir.rgi_id for gdir in gdirs]
    if rgi_id not in oggm_rgis:
        raise ValueError(f"RGI ID {rgi_id} not found in OGGM data")
    for gdir in gdirs:
        if gdir.rgi_id == rgi_id:
            break
    with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
        ds = ds.load()
    glacier_mask = np.where(
        ds["glacier_mask"].values == 0, np.nan, ds["glacier_mask"].values
    )

    # Create glacier mask
    ds = ds.assign(masked_slope=glacier_mask * ds["slope"])
    ds = ds.assign(masked_elev=glacier_mask * ds["topo"])
    ds = ds.assign(masked_aspect=glacier_mask * ds["aspect"])
    ds = ds.assign(masked_dis=glacier_mask * ds["dis_from_border"])
    ds = ds.assign(masked_hug=glacier_mask * ds["hugonnet_dhdt"])
    ds = ds.assign(masked_cit=glacier_mask * ds["consensus_ice_thickness"])
    if "millan_ice_thickness" in ds:
        ds = ds.assign(masked_mit=glacier_mask * ds["millan_ice_thickness"])
    if "millan_v" in ds:
        # Some glaciers do not have velocity data
        ds = ds.assign(masked_miv=glacier_mask * ds["millan_v"])
        ds = ds.assign(masked_mivx=glacier_mask * ds["millan_vx"])
        ds = ds.assign(masked_mivy=glacier_mask * ds["millan_vy"])

    glacier_indices = np.where(ds["glacier_mask"].values == 1)
    return ds, glacier_indices, gdir


def _get_unique_rgi_ids(rgi_ids: pd.Series) -> list:
    """Get the list of unique RGI IDs."""
    return rgi_ids.dropna().unique().tolist()


def _filter_dataframe(df: pd.DataFrame, rgi_ids_list: list) -> pd.DataFrame:
    """Filter the DataFrame to include only the RGI IDs of interest and select only lat/lon columns."""
    return df.loc[df["RGIId"].isin(rgi_ids_list), ["RGIId", "POINT_LAT", "POINT_LON"]]


def _group_stakes_by_rgi_id(
    filtered_df: pd.DataFrame,
) -> pd.api.typing.DataFrameGroupBy:
    """Group latitude and longitude by RGI ID."""
    return filtered_df.groupby("RGIId", sort=False)


def _load_gridded_svf(
    glacier_directories: list,
    grouped_stakes: pd.api.typing.DataFrameGroupBy,
    cfg,
) -> list:
    """Load sky view factor data for each glacier directory."""
    grouped_rgi_ids = set(grouped_stakes.groups.keys())
    grid_path = os.path.join(data_path, "grids")
    loaded_svf = []
    for gdir in glacier_directories:
        if gdir.rgi_id in grouped_rgi_ids:
            rgi_id = gdir.rgi_id
            path_rgi_id = os.path.join(grid_path, *rgi_id_to_folders(rgi_id))
            svf_file = os.path.join(path_rgi_id, "svf.nc")
            p_svf = Product(svf_file)
            if not p_svf.is_up_to_date():
                # Create DEM grid
                create_dem_file_RGI(cfg, rgi_id, path_rgi_id)
                # Generate sky view factor
                generate_svf_file(path_rgi_id)
                # Generate checksum so that this is not regenerated in future runs
                p_svf.gen_chk()
            svf = xr.open_dataset(os.path.join(path_rgi_id, "svf.nc"))
            loaded_svf.append(svf)
    return loaded_svf


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

    for gdir, gdir_grid in zip(glacier_directories, gdirs_gridded):

        # Coordinate transformation from WGS84 to the projection of OGGM data
        transf = pyproj.Transformer.from_proj(
            pyproj.CRS.from_user_input("EPSG:4326"),
            pyproj.CRS.from_user_input(gdir_grid.pyproj_srs),
            always_xy=True,
        )
        lon = grouped_stakes.get_group(gdir.rgi_id)[["POINT_LON"]].values.flatten()
        lat = grouped_stakes.get_group(gdir.rgi_id)[["POINT_LAT"]].values.flatten()
        x, y = transf.transform(lon, lat)

        topo_data = (
            gdir_grid.sel(
                x=xr.DataArray(x, dims="points"),
                y=xr.DataArray(y, dims="points"),
                method="nearest",
            )[voi]
            .to_dataframe()
            .reset_index(drop=True)
        )

        df.loc[df["RGIId"] == gdir.rgi_id, voi] = topo_data[voi].values
