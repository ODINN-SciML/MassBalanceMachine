import os
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from scipy.interpolate import griddata
from pathlib import Path

from config import Config
from data_processing.Dataset import Dataset
from data_processing.get_topo_data import get_glacier_mask
from data_processing.glacier_utils import create_dem_file_RGI, generate_svf_file
from data_processing.Product import Product
from data_processing.product_utils import rgi_id_to_folders, data_path
from .data import load_aws_monthly_precipitation


def _get_aws_grid_location(monthly_precipitation, ds, gdir):
    """Return AWS grid coordinates and whether they are inside the NetCDF."""
    transformer = pyproj.Transformer.from_proj(
        pyproj.Proj("epsg:4326"), gdir.grid.proj, always_xy=True
    )
    aws_x, aws_y = transformer.transform(
        monthly_precipitation["POINT_LON"].iloc[0],
        monthly_precipitation["POINT_LAT"].iloc[0],
    )
    inside_netcdf = (
        ds.x.values.min() <= aws_x <= ds.x.values.max()
        and ds.y.values.min() <= aws_y <= ds.y.values.max()
    )
    return aws_x, aws_y, inside_netcdf


def _interpolate_glacier_topography(
    monthly_precipitation: pd.DataFrame,
    ds,
    glacier_indices,
    gdir,
) -> pd.DataFrame:
    """Interpolate aspect and slope on the full OGGM NetCDF domain.

    The input grid must come from ``get_glacier_mask(..., mask=False)``. The
    glacier outline mask is intentionally not used here: an AWS may be outside
    the glacier while still being inside the NetCDF domain.
    """
    aws_x, aws_y, inside_netcdf = _get_aws_grid_location(
        monthly_precipitation, ds, gdir
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


def _interpolate_aws_svf(monthly_precipitation, ds, gdir, svf):
    """Interpolate cached SVF on the full OGGM NetCDF domain.

    Being outside the glacier outline is valid. Only an AWS outside the
    NetCDF coordinate domain receives ``NaN``.
    """
    aws_x, aws_y, inside_netcdf = _get_aws_grid_location(
        monthly_precipitation, ds, gdir
    )
    if not inside_netcdf:
        monthly_precipitation["svf"] = np.nan
        return monthly_precipitation

    svf_x, svf_y = np.meshgrid(svf.x.values, svf.y.values)
    points = np.column_stack((svf_x.ravel(), svf_y.ravel()))
    values = svf["svf"].values.ravel()
    interpolated = griddata(points, values, (aws_x, aws_y), method="linear")
    if np.isnan(interpolated):
        interpolated = griddata(points, values, (aws_x, aws_y), method="nearest")
    monthly_precipitation["svf"] = interpolated
    return monthly_precipitation


def _load_or_create_aws_svf(rgi_id, cfg):
    """Create or load cached DEM/SVF files for an AWS glacier."""
    grid_path = os.path.join(data_path, "grids", "Hugonnet21")
    path_rgi_id = os.path.join(grid_path, *rgi_id_to_folders(rgi_id))
    svf_file = os.path.join(path_rgi_id, "svf.nc")
    p_svf = Product(svf_file)
    if not p_svf.is_up_to_date():

        # Create DEM grid
        create_dem_file_RGI(cfg, rgi_id, path_rgi_id)

        # Generate sky view factor
        generate_svf_file(path_rgi_id)

        p_svf.gen_chk()

    return xr.open_dataset(os.path.join(path_rgi_id, "svf.nc"))


def build_features(aws_code: str, cfg, data_dir=None):
    monthly_precipitation = load_aws_monthly_precipitation(
        aws_code, data_dir=data_dir, include_metadata=True
    )
    monthly_precipitation = monthly_precipitation.rename(
        columns={
            "Latitude": "POINT_LAT",
            "Longitude": "POINT_LON",
            "Elevation": "POINT_ELEVATION",
        }
    )

    assert len(monthly_precipitation["RGIId"]) > 0, "Monthly precipitation is empty."
    rgi_id = monthly_precipitation["RGIId"].iloc[0]
    if pd.isna(rgi_id):
        raise ValueError(f"AWS {aws_code!r} has no corresponding glacier")

    # cfg = Config()
    ds, glacier_indices, gdir = get_glacier_mask(rgi_id, "", cfg, mask=False)
    features = _interpolate_glacier_topography(
        monthly_precipitation, ds, glacier_indices, gdir
    )
    if not _get_aws_grid_location(features, ds, gdir)[2]:
        features["svf"] = np.nan
        return features
    with _load_or_create_aws_svf(rgi_id, cfg) as svf:
        return _interpolate_aws_svf(features, ds, gdir, svf)


def create_aws_grid(aws_code: str, cfg, data_dir=None) -> pd.DataFrame:
    """Create a glacier-grid-compatible monthly dataframe for one AWS.

    Each row represents one complete month of AWS data. Spatial columns use
    the same names as :func:`create_glacier_grid_RGI`, while the temporal
    columns describe the calendar month represented by the row.
    """
    if data_dir is None:
        features = build_features(aws_code, cfg).copy()
    else:
        features = build_features(aws_code, cfg, data_dir=data_dir).copy()
    features["POINT_ID"] = 1
    features["N_MONTHS"] = 1
    features["POINT_BALANCE"] = 0  # fake PMB for simplicity (not used)
    features["YEAR"] = features["Date"].dt.year
    # features["MONTHS"] = features["Date"].dt.month
    features["FROM_DATE"] = features["Date"].dt.strftime("%Y%m%d")
    features["TO_DATE"] = (
        features["Date"] + pd.offsets.MonthEnd(0)  # TODO: check this
    ).dt.strftime("%Y%m%d")
    features["PERIOD"] = "annual"
    return features


def monthly_features(
    aws_codes: str,
    cfg,
    region_id: int,
    data_dir=None,
) -> pd.DataFrame:
    """Build the monthly climate/topographical feature dataframe for one AWS.

    Extracted from the AWS notebook: builds the AWS monthly grid, wraps it in
    a :class:`Dataset`, and matches ERA5-Land climate data to each row's
    ``Date`` (see :func:`create_aws_grid` for the topographical columns,
    which are already attached before this runs).

    Args:
        aws_codes (list of str): AWS station code, e.g. ``["LOM0154"]``.
        cfg: Config instance.
        region_id (int): RGI region ID.
        data_dir: Passed through to :func:`create_aws_grid`. Defaults to `None`.

    Returns:
        pd.DataFrame: The AWS monthly dataframe with climate features added.
    """
    if not isinstance(aws_codes, list):
        aws_codes = [aws_codes]

    processed_path = os.path.join(data_path, "AWS", "EEAR-Clim_processed")

    for code in aws_codes:
        feat_path = os.path.join(processed_path, f"{code}.parquet")
        p = Product(feat_path)
        if not p.is_up_to_date():

            # Climate columns
            vois_climate = [
                "t2m",
                "tp",
                "slhf",
                "sshf",
                "ssrd",
                "fal",
                "str",
                "u10",
                "v10",
                "tp_sum",
                "slhf_sum",
                "sshf_sum",
                "ssrd_sum",
                "str_sum",
            ]

            feat = create_aws_grid(code, cfg, data_dir=data_dir)

            feat["P"] = feat["P"] / 1000  # mm/day -> m/day, to match ERA5's "tp" units

            # Compute cumulative fluxes over each month
            fluxes_cols = ["P"]
            days_in_month = feat["Date"].dt.days_in_month.to_numpy()
            for variable in fluxes_cols:
                if variable in feat and f"{variable}_sum":
                    feat[f"{variable}_sum"] = feat[variable] * days_in_month

            dataset_grid = Dataset(
                cfg=cfg,
                data=feat,
                region_name="",
                region_id=region_id,
            )

            dataset_grid.get_climate_features(
                change_units=True,
                monthly=True,
                smoothing_vois={
                    "vois_climate": vois_climate,
                    "vois_other": ["ALTITUDE_CLIMATE"],
                },
            )

            df = dataset_grid.data

            df.to_parquet(feat_path, index=True)

            p.gen_chk()

    df_assembled = pd.DataFrame()
    for i, code in enumerate(aws_codes):
        feat_path = os.path.join(processed_path, f"{code}.parquet")
        df = pd.read_parquet(feat_path)
        df["ID"] = i
        df_assembled = pd.concat([df_assembled, df], ignore_index=True)

    return df_assembled
