import os
import cdsapi
import numpy as np
from oggm import utils
import zipfile
import xarray as xr

def get_region_shape_file(region:str):
    rgi_version = '6'
    shp_path = utils.get_rgi_region_file(region, version=rgi_version)
    print(f"Shapefile for region {region}: {shp_path}")
    return shp_path

def get_region_area_bounds(region):
    if not isinstance(region, str): region = f'{region:02d}'
    shp_path = get_region_shape_file(region)
    outlines = gpd.read_file(shp_path)
    minlon, minlat, maxlon, maxlat = outlines.total_bounds
    return {
        "lon": (minlon, maxlon),
        "lat": (minlat, maxlat),
    }

def path_climate_data(region):
    if not isinstance(region, str): region = f'{region:02d}'
    return f".data/{region}/"

def download_climate_ERA5(region):
    path_region = path_climate_data(region)
    os.makedirs(path_region, exist_ok=True)

    bounds = get_region_area_bounds(region)
    area = [
        np.ceil(bounds["lat"][1]), # north
        np.floor(bounds["lon"][0]), # west
        np.floor(bounds["lat"][0]), # south
        np.ceil(bounds["lon"][1]), # east
    ]
    request_climate = {
        'product_type': ['monthly_averaged_reanalysis'],
        'variable': [
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            '2m_temperature',
            'forecast_albedo',
            'snow_cover',
            'snow_density',
            'snow_depth_water_equivalent',
            'snowfall',
            'snowmelt',
            'surface_latent_heat_flux',
            'surface_net_thermal_radiation',
            'surface_sensible_heat_flux',
            'surface_solar_radiation_downwards',
            'total_precipitation',
        ],
        'year': list(map(str, range(1950,2025))),
        'month': list(map(lambda i : f'{i:02d}', range(1,13))),
        'time': ['00:00'],
        "data_format": "netcdf",
        "download_format": "zip",
        "area": area,
    }
    request_geopotential = {
        "variable": ["geopotential"],
        "data_format": "netcdf",
        "download_format": "zip",
        "area": area,
    }

    path_climate_zip = path_region + 'download_climate.netcdf.zip'
    path_climate = path_region + "era5_monthly_averaged_data.nc"
    path_geopot_zip = path_region + 'download_geopot.netcdf.zip'
    path_geopot = path_region + "era5_geopotential_pressure.nc"

    c = cdsapi.Client()

    print("Downloading climate data")
    c.retrieve('reanalysis-era5-land-monthly-means', request_climate, path_climate_zip)
    with zipfile.ZipFile(path_climate_zip, 'r') as zip:
        zip.extractall(path_region)

    dc = xr.open_dataset(path_region + 'data_stream-moda.nc')
    dc2 = dc.rename(
        {'valid_time': 'time'}
    )  # Coordinates have changed recently in the API, this is to keep compatibility with our code
    dc2.to_netcdf(path_climate)

    print("Downloading geopotential data")
    c.retrieve('reanalysis-era5-land-monthly-means', request_geopotential, path_geopot_zip)
    with zipfile.ZipFile(path_geopot_zip, 'r') as zip:
        zip.extractall(path_region)

    os.rename(path_region+f"geo.area-subset.{area[0]}.{area[3]}.{area[2]}.{area[1]}.nc", path_geopot)
