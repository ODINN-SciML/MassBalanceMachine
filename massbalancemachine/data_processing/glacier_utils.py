import os
import numpy as np
import salem
import pyproj
import pandas as pd
import geopandas as gpd
import xarray as xr
import oggm
import venv
import subprocess
from typing import Union

import config
from data_processing.product_utils import mbm_path
from data_processing.Product import Product
from data_processing.oggm_utils import (
    _initialize_oggm_config,
    _initialize_glacier_directories,
)

venv_rtv = None


def create_glacier_grid_RGI(
    ds: xr.Dataset,
    years: list,
    glacier_indices: "tuple[np.array, np.array]",
    gdir: oggm.GlacierDirectory,
    rgi_gl: str,
    ds_svf=None,
) -> pd.DataFrame:
    """Creates a DataFrame of glacier grid data for each year

    Args:
        ds (xarray.Dataset): glacier attributes masked over its outline
        years (list): years of the data
        glacier_indices (np.array): indices of glacier mask in the OGGM grid
        gdir (oggm directory): oggm glacier directory
        rgi_gl (str): RGI Id of the glacier
    Returns:
        df_grid (pd.DataFrame): dataframe of glacier grid data, for each year
    """
    # Assuming the coordinate variables are named 'x' and 'y' in your dataset
    x_coords = ds["x"].values
    y_coords = ds["y"].values

    # Retrieve the x and y values using the glacier indices
    glacier_x_vals = x_coords[glacier_indices[1]]
    glacier_y_vals = y_coords[glacier_indices[0]]

    # Convert glacier coordinates to latitude and longitude
    # Transform stake coord to glacier system:
    transf = pyproj.Transformer.from_proj(gdir.grid.proj, salem.wgs84, always_xy=True)
    lon, lat = transf.transform(glacier_x_vals, glacier_y_vals)

    # Glacier mask as boolean array:
    gl_mask_bool = ds["glacier_mask"].values.astype(bool)

    # Create a DataFrame
    data_grid = {
        "RGIId": [rgi_gl] * len(ds.masked_elev.values[gl_mask_bool]),
        "POINT_LAT": lat,
        "POINT_LON": lon,
        "aspect": ds.masked_aspect.values[gl_mask_bool],
        "slope": ds.masked_slope.values[gl_mask_bool],
        "topo": ds.masked_elev.values[gl_mask_bool],
        "dis_from_border": ds.masked_dis.values[gl_mask_bool],
        "hugonnet_dhdt": ds.masked_hug.values[gl_mask_bool],
        "consensus_ice_thickness": ds.masked_cit.values[gl_mask_bool],
    }
    if "millan_ice_thickness" in ds:
        data_grid["millan_ice_thickness"] = ds.masked_mit.values[gl_mask_bool]
    if "masked_miv" in ds:
        # Some glaciers do not have velocity data
        data_grid["millan_v"] = ds.masked_miv.values[gl_mask_bool]
        data_grid["millan_vx"] = ds.masked_mivx.values[gl_mask_bool]
        data_grid["millan_vy"] = ds.masked_mivy.values[gl_mask_bool]
    if ds_svf is not None:
        assert all(ds_svf.x == ds.x) and all(ds_svf.y == ds.y)
        data_grid["svf"] = ds_svf.svf.values[gl_mask_bool]

    df_grid = pd.DataFrame(data_grid)
    del data_grid, ds  # Free up memory

    # Match to WGMS format:
    df_grid["POINT_ID"] = np.arange(1, len(df_grid) + 1)
    df_grid["N_MONTHS"] = 12
    df_grid["POINT_ELEVATION"] = df_grid["topo"]  # no other elevation available
    df_grid["POINT_BALANCE"] = 0  # fake PMB for simplicity (not used)
    num_rows_per_year = len(df_grid)
    # Repeat the DataFrame num_years times
    df_grid = pd.concat([df_grid] * len(years), ignore_index=True)
    # Add the 'year' and date columns to the DataFrame
    df_grid["YEAR"] = np.repeat(
        years, num_rows_per_year
    )  # 'year' column that has len(df_grid) instances of year
    df_grid["FROM_DATE"] = df_grid["YEAR"].apply(lambda x: str(x) + "1001")
    df_grid["TO_DATE"] = df_grid["YEAR"].apply(lambda x: str(x + 1) + "0930")
    df_grid["PERIOD"] = "annual"

    return df_grid


def get_region_shape_file(region: str):
    rgi_version = "62"
    shp_path = oggm.utils.get_rgi_region_file(region, version=rgi_version)
    print(f"Shapefile for region {region}: {shp_path}")
    return shp_path


def get_region_name(region: Union[str, int]):
    if isinstance(region, int):
        region = f"{region:02d}"
    rgi_version = "62"
    fp = oggm.utils.get_rgi_region_file(region, version=rgi_version)
    rgi_name = os.path.basename(fp).split("_", 1)[1].replace(".shp", "").split("_")[1]
    return rgi_name


def get_region_area_bounds(region):
    if not isinstance(region, str):
        region = f"{region:02d}"
    shp_path = get_region_shape_file(region)
    outlines = gpd.read_file(shp_path)
    minlon, minlat, maxlon, maxlat = outlines.total_bounds
    return {
        "lon": (minlon, maxlon),
        "lat": (minlat, maxlat),
    }


def get_glacier_dem(rgi_id: str, custom_working_dir: str, cfg: config.Config):
    """Given a `rgi_id` gets glacier DEM from OGGM."""

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
    return ds


def create_dem_file_RGI(cfg, rgi_id, path_rgi_id):
    out_path = os.path.abspath(os.path.join(path_rgi_id, f"dem.nc"))
    p = Product(out_path)
    if not p.is_up_to_date():
        ds = get_glacier_dem(rgi_id, "", cfg)
        lkeys = list(ds.keys())
        lkeys.remove("topo")
        ds_topo = ds.drop_vars(lkeys)
        ds_topo.to_netcdf(out_path)

        p.gen_chk()


def create_venv_rtv():
    # Define the absolute path for the RTV virtual environment
    venv_path = os.path.abspath(os.path.join(mbm_path, "venv/rvt_env/"))

    # Create the virtual environment
    venv.create(venv_path, with_pip=True)

    # Path to the Python executable in the virtual environment
    venv_python = os.path.join(venv_path, "bin", "python")  # Work only for Linux/Mac

    # Install the incompatible package
    gdal_failed = False
    gdal_version = (
        subprocess.check_output(["gdal-config", "--version"])
        .decode("utf-8")
        .replace("\n", "")
    )
    ret = subprocess.run(
        [venv_python, "-m", "pip", "install", f"GDAL=={gdal_version}"],
        capture_output=True,
    )
    if ret.returncode:
        print(ret.stderr.decode("utf-8"))
        raise Exception(
            "An error occured during installation of GDAL (see above). This usually happens when libgdal is not installed. On Debian based distros, run the following command to install it: apt-get install libgdal-dev"
        )
    else:
        print("Installed GDAL successfully")
    subprocess.run([venv_python, "-m", "pip", "install", "rvt-py"])
    subprocess.run([venv_python, "-m", "pip", "install", "xarray"])
    subprocess.run([venv_python, "-m", "pip", "install", "netCDF4"])

    return venv_path


def generate_svf_file(path_rgi_id):
    global venv_rtv
    if venv_rtv is None:
        venv_rtv = create_venv_rtv()
    path_script_rtv = os.path.abspath(
        os.path.join(mbm_path, "massbalancemachine/data_processing/sky_view_factor.py")
    )
    subprocess.run([os.path.join(venv_rtv, "bin/python"), path_script_rtv, path_rgi_id])
