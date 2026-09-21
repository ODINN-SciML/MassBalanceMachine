import os
from calendar import month_abbr
from functools import lru_cache
from typing import Optional
import xarray as xr
import numpy as np
import pandas as pd

import config
from data_processing.climate_projections.climate_data_download import (
    ensure_climate_CMIP6,
)
from data_processing.climate_projections.regridding import bias_corrected

# from data_processing.utils.hydro_year import months_hydro_year, _rebuild_month_index


def get_climate_features_(
    df: pd.DataFrame,
    change_units: bool,
    months_tail_pad,  # before 'oct'
    months_head_pad,  # after 'sep'
    region,
    ssp,
    gcm,
    vois_climate: list = None,
    vois_other: list = None,
    bias_correction_period=None,
) -> pd.DataFrame:

    # Check that df is aligned with region

    # Get bias corrected temperature
    t2m_corrected = bias_corrected(
        region, ssp, gcm, bias_correction_period=bias_correction_period
    )

    # Get latitudes and longitudes from the climate dataset.
    lat, lon = t2m_corrected.latitude, t2m_corrected.longitude

    # # Convert the longitudes
    # ds_180 = _adjust_longitude(ds_geopotential)

    # # Crop the geopotential height to the region of interest
    # ds_geopotential_cropped = _crop_geopotential(ds_180, lat, lon)

    # # Remove duplicates
    # ds_geopotential_cropped = ds_geopotential_cropped.drop_duplicates(dim="latitude")
    # ds_geopotential_cropped = ds_geopotential_cropped.drop_duplicates(dim="longitude")

    # # Calculate the geopotential height in meters
    # ds_geopotential_metric = _calculate_geopotential_height(ds_geopotential_cropped)

    # # Create a date range for one hydrological year
    # df = _add_date_range(df, months_tail_pad, months_head_pad)

    # # Get the climate data for the latitudes and longitudes and date ranges as
    # # specified
    # df["months_in_range"] = df["range_date"].apply(
    #     lambda rng: [d.strftime("%b").lower() for d in rng] if rng is not None else []
    # )

    # climate_df = _process_climate_data(ds_climate, df, months_tail_pad, months_head_pad)

    # # Get the geopotential height for the latitudes and longitudes as specified,
    # # for the locations of the stake measurements.
    # altitude_df = _process_altitude_data(ds_geopotential_metric, df)

    # # Combine the climate data with the altitude climate data
    # df = _combine_dataframes(df, climate_df, altitude_df)

    # # Compute the sum of the fluxes per month from the average fluxes per day
    # # Cf https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation#ERA5:datadocumentation-Meanrates/fluxesandaccumulations
    # fluxes_cols = ["tp", "slhf", "str", "sshf", "ssrd"]
    # df_cols = df.columns.values
    # month_to_id = {(month_abbr[i].lower()): i for i in range(1, 13)}
    # new_cols = {}
    # for col_df in df_cols:
    #     m = [col_df.startswith(c) for c in fluxes_cols]
    #     if any(m):
    #         assert np.sum(m) == 1
    #         flux_col = np.array(fluxes_cols)[np.array(m)][0]
    #         suffix = col_df.replace(flux_col + "_", "")
    #         id_month = str(month_to_id[suffix.replace("_", "")])
    #         # Incorrect because we retrieve the year of the hydrological year and not the true year associated to the measurement
    #         days_in_month = pd.to_datetime(
    #             df.YEAR.astype(str) + "-" + id_month + "-01"
    #         ).dt.days_in_month
    #         sum_col = flux_col + "_sum_" + suffix
    #         new_cols[sum_col] = df[col_df].values * days_in_month
    # df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # # Remove climate artifacts
    # df = smooth_era5land_by_mode(df, vois_climate, vois_other)

    # # Add a new feature to the dataframe that is the height difference between the elevation
    # # of the stake and the recorded height of the climate.
    # df = _calculate_elevation_difference(df)

    return df
