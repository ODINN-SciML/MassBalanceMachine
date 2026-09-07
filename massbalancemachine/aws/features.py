import numpy as np
import pandas as pd
import pyproj
from scipy.interpolate import griddata

from config import Config
from data_processing.get_topo_data import get_glacier_mask
from .aws_data import load_aws_monthly_precipitation


def _interpolate_glacier_topography(
    monthly_precipitation: pd.DataFrame,
    ds,
    glacier_indices,
    gdir,
) -> pd.DataFrame:
    """Interpolate glacier-grid aspect and slope at the AWS location."""
    aws_lon = monthly_precipitation["POINT_LON"].iloc[0]
    aws_lat = monthly_precipitation["POINT_LAT"].iloc[0]
    transformer = pyproj.Transformer.from_proj(
        pyproj.Proj("epsg:4326"), gdir.grid.proj, always_xy=True
    )
    aws_x, aws_y = transformer.transform(aws_lon, aws_lat)

    netcdf_x = ds.x.values
    netcdf_y = ds.y.values
    inside_netcdf = (
        netcdf_x.min() <= aws_x <= netcdf_x.max()
        and netcdf_y.min() <= aws_y <= netcdf_y.max()
    )
    if not inside_netcdf:
        monthly_precipitation[["aspect", "slope"]] = np.nan
        return monthly_precipitation

    grid_x = ds.x.values[glacier_indices[1]]
    grid_y = ds.y.values[glacier_indices[0]]
    points = np.column_stack((grid_x, grid_y))

    for variable in ("aspect", "slope"):
        values = ds[variable].values[glacier_indices]
        interpolated = griddata(points, values, (aws_x, aws_y), method="linear")
        if np.isnan(interpolated):
            interpolated = griddata(points, values, (aws_x, aws_y), method="nearest")
        monthly_precipitation[variable] = interpolated

    return monthly_precipitation


def build_features(aws_code: str):
    monthly_precipitation = load_aws_monthly_precipitation(
        aws_code, include_metadata=True
    )
    monthly_precipitation = monthly_precipitation.rename(
        columns={
            "Latitude": "POINT_LAT",
            "Longitude": "POINT_LON",
            "Elevation": "POINT_ELEVATION",
        }
    )

    rgi_id = monthly_precipitation["RGIId"].iloc[0]
    if pd.isna(rgi_id):
        raise ValueError(f"AWS {aws_code!r} has no corresponding glacier")

    cfg = Config()
    ds, glacier_indices, gdir = get_glacier_mask(rgi_id, "", cfg, mask=False)
    return _interpolate_glacier_topography(
        monthly_precipitation, ds, glacier_indices, gdir
    )


def create_aws_grid(aws_code: str) -> pd.DataFrame:
    """Create a glacier-grid-compatible monthly dataframe for one AWS.

    Each row represents one complete month of AWS data. Spatial columns use
    the same names as :func:`create_glacier_grid_RGI`, while the temporal
    columns describe the calendar month represented by the row.
    """
    features = build_features(aws_code).copy()
    features["POINT_ID"] = 1
    features["N_MONTHS"] = 1
    features["YEAR"] = features["Date"].dt.year
    # features["MONTHS"] = features["Date"].dt.month
    features["FROM_DATE"] = features["Date"].dt.strftime("%Y%m%d")
    features["TO_DATE"] = (
        features["Date"] + pd.offsets.MonthEnd(0)  # TODO: check this
    ).dt.strftime("%Y%m%d")
    features["PERIOD"] = "annual"
    return features
