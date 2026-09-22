import os
import glob
import cdsapi
import numpy as np
import zipfile
import xarray as xr
import itertools

from data_processing.glacier_utils import get_region_area_bounds


ELEV_DEP_VARS = ["air_temperature"]
# Surface variables used to build the ERA5-like fluxes and precipitation
SURFACE_FLUX_VARS = [
    "precipitation",
    "surface_downwelling_shortwave_radiation",
    "surface_upwelling_longwave_radiation",
    "surface_downwelling_longwave_radiation",  # need both to get an equivalent to surface_net_thermal_radiation
    "surface_upward_latent_heat_flux",
    "surface_upward_sensible_heat_flux",
]
AVAILABLE_VARS = [
    "geopotential_height",
    "eastward_near_surface_wind",
    "northward_near_surface_wind",
    *ELEV_DEP_VARS,
    *SURFACE_FLUX_VARS,
]  # Not exhaustive
DEFAULT_VARS = (
    AVAILABLE_VARS  # By default we set this but we could customize it in the future
)


def path_climate_data(region, ssp, gcm, var):
    """Return path of data for a given region (string or integer)."""
    if not isinstance(region, str):
        region = f"{region:02d}"
    return f".data/CMIP6/{region}/{ssp}/{gcm}/{var}/"


def ensure_climate_CMIP6(region, variables=DEFAULT_VARS, ssps=[], gcms=[]):
    if isinstance(ssps, list) or isinstance(gcms, list) or isinstance(variables, list):
        # Loop over combinations of SSP x GCM
        if not isinstance(ssps, list):
            ssps = [ssps]
        if not isinstance(gcms, list):
            gcms = [gcms]
        if not isinstance(variables, list):
            variables = [variables]
        for gcm, ssp, var in itertools.product(gcms, ssps, variables):
            ensure_climate_CMIP6(region, var, ssp, gcm)
        return

    assert ssps in [
        "historical",
        "ssp1_2_6",
        "ssp2_4_5",
        "ssp4_6_0",
        "ssp3_7_0",
        "ssp5_8_5",
    ]
    assert gcms in [
        "gfdl_esm4",
        "ukesm1_0_ll",
        "mpi_esm1_2_hr",
        "mri_esm2_0",
        "ipsl_cm6a_lr",
    ]
    assert variables in AVAILABLE_VARS

    path_data = path_climate_data(region, ssps, gcms, variables)
    os.makedirs(path_data, exist_ok=True)

    path_climate = path_data + "data.nc"
    if os.path.isfile(path_climate):
        # Data already downloaded and available locally
        return

    bounds = get_region_area_bounds(region)
    # Increase bounds with a shift larger than the one for ERA5 to be sure that the ERA5 grid is included within the CMIP6 GCM grid
    area = [
        np.ceil(bounds["lat"][1]) + 4,  # north
        np.floor(bounds["lon"][0]) - 4,  # west
        np.floor(bounds["lat"][0]) - 4,  # south
        np.ceil(bounds["lon"][1]) + 4,  # east
    ]

    path_climate_zip = path_data + "download_climate.netcdf.zip"

    c = cdsapi.Client()

    years = (
        list(map(str, range(1950, 2015)))
        if ssps == "historical"
        else list(map(str, range(2015, 2100)))
    )
    request_climate = {
        "temporal_resolution": "monthly",
        "experiment": ssps,
        "variable": variables,
        "model": gcms,
        "year": years,
        "month": list(map(lambda i: f"{i:02d}", range(1, 13))),
        "area": area,
    }
    if variables == "geopotential_height" or variables in ELEV_DEP_VARS:
        request_climate["level"] = [
            "1",
            "5",
            "10",
            "20",
            "30",
            "50",
            "70",
            "100",
            "150",
            "200",
            "250",
            "300",
            "400",
            "500",
            "600",
            "700",
            "850",
            "925",
            "1000",
        ]
    print(
        f"Downloading variable {variables} for scenario {ssps} and GCM {gcms}, please wait for the processing on the Copernicus server to finish..."
    )
    c.retrieve("projections-cmip6", request_climate, path_climate_zip)
    with zipfile.ZipFile(path_climate_zip, "r") as zip:
        zip.extractall(path_data)
    files = glob.glob(path_data + "/*")
    f = list(
        filter(
            lambda f: f.endswith(".nc") or f.endswith("download_climate.netcdf.zip"),
            files,
        )
    )
    assert len(f) == 2, f"Found {len(f)} files instead of 2: {f}"
    netcdf_file = list(filter(lambda f: f.endswith(".nc"), files))
    assert (
        len(netcdf_file) == 1
    ), f"Found {len(netcdf_file)} files instead of 1: {netcdf_file}"
    netcdf_file = netcdf_file[0]
    for e in f:
        files.remove(e)
    for e in files:
        os.remove(e)
    os.rename(netcdf_file, path_climate)
