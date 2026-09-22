"""
Climate features from CMIP6 projections, bias corrected against ERA5 and put on its
grid, so that they go through the same processing as the ERA5 features.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional
import xarray as xr
import pandas as pd

from data_processing.product_utils import data_path
from data_processing.Product import Product
from data_processing.get_climate_data import (
    _file_stamp,
    _load_datasets,
    climate_features_from_datasets,
    climate_file_paths,
)
from data_processing.climate_projections.climate_data_download import (
    ensure_climate_CMIP6,
)
from data_processing.climate_projections.regridding import (
    bias_corrected_climate,
    bias_cor_suffix,
)

# Climate variables attached to the gridded products built on CMIP6 projections. The
# ERA5 ones (`gridded_utils.GRID_VOIS_CLIMATE`) also hold fal, u10 and v10, which are
# not derived from the projections.
CMIP6_GRID_VOIS_CLIMATE = [
    "t2m",
    "tp",
    "slhf",
    "sshf",
    "ssrd",
    "str",
    "tp_sum",
    "slhf_sum",
    "sshf_sum",
    "ssrd_sum",
    "str_sum",
]


def path_climate_CMIP6(region, ssp, gcm, bias_correction_period):
    """File holding the bias-corrected CMIP6 climate of a region on the ERA5 grid."""
    if not isinstance(region, str):
        region = f"{region:02d}"
    return os.path.join(
        data_path,
        "CMIP6",
        "climate",
        bias_cor_suffix(bias_correction_period),
        region,
        ssp,
        f"{gcm}.nc",
    )


def ensure_climate_features_CMIP6(region, ssp, gcm, bias_correction_period):
    """Compute and store the bias-corrected climate of a region, if not done yet.

    It is computed once rather than in every task generating gridded features: the
    bias correction needs the whole ERA5 file of the region. The time axis is
    converted to the month starts ERA5 uses, from the mid-month cftime dates of the
    GCMs, whose calendar may have no leap days.
    """
    file_path = path_climate_CMIP6(region, ssp, gcm, bias_correction_period)
    p = Product(file_path)
    if p.is_up_to_date():
        return file_path

    ssps = [ssp]
    if bias_correction_period is not None and ssp != "historical":
        # The reference period of the bias correction lies in the historical run
        ssps.append("historical")
    ensure_climate_CMIP6(region, ssps=ssps, gcms=gcm)

    ds = bias_corrected_climate(region, ssp, gcm, bias_correction_period)
    ds = ds.reset_coords(drop=True).astype("float32")
    ds = ds.assign_coords(
        time=pd.DatetimeIndex(
            [pd.Timestamp(t.year, t.month, 1) for t in ds.indexes["time"]]
        )
    )
    ds.to_netcdf(file_path)
    p.gen_chk()
    return file_path


@lru_cache(maxsize=2)
def _load_climate_CMIP6_cached(file_path, change_units, stamp):
    """Kept behind a cache for the same reason as the ERA5 data, see
    `get_climate_data._load_datasets_cached`. `stamp` is only a cache key."""
    del stamp
    with xr.open_dataset(file_path) as ds:
        ds = ds.load()
    if change_units:
        ds = ds.assign(t2m=ds["t2m"] - 273.15)
    return ds


def load_climate_CMIP6(
    region, ssp, gcm, bias_correction_period, change_units: bool = False
):
    """The bias-corrected CMIP6 climate of a region on the ERA5 grid. Shared between
    calls, so it must be treated as read-only."""
    file_path = ensure_climate_features_CMIP6(region, ssp, gcm, bias_correction_period)
    return _load_climate_CMIP6_cached(file_path, change_units, _file_stamp(file_path))


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
    output_fname: str = None,
) -> pd.DataFrame:
    """CMIP6 counterpart of `data_processing.get_climate_data.get_climate_features_`.

    The climate comes from the CMIP6 run instead of ERA5, bias corrected over
    `bias_correction_period`. Its altitude (ALTITUDE_CLIMATE) is still the ERA5
    geopotential, since the temperature was interpolated to that elevation.
    """
    ds_climate = load_climate_CMIP6(
        region, ssp, gcm, bias_correction_period, change_units
    )
    ds_geopotential = _load_datasets(*climate_file_paths(region), change_units)[1]

    return climate_features_from_datasets(
        df,
        ds_climate,
        ds_geopotential,
        output_fname,
        months_tail_pad,
        months_head_pad,
        vois_climate,
        vois_other,
    )


@dataclass(frozen=True)
class CMIP6Climate:
    """Which CMIP6 projection the gridded products take their climate from.

    Passed to the generation of the gridded products in place of the default ERA5
    climate.

    Args:
        ssp (str): SSP scenario, for example "historical", "ssp1_2_6" or "ssp5_8_5".
        gcm (str): one of the GCMs of `ensure_climate_CMIP6`.
        bias_correction_period (tuple of int): first and last year, inclusive, over
            which the projection is bias corrected against ERA5. None for no
            correction.
    """

    ssp: str
    gcm: str
    bias_correction_period: Optional[tuple] = (1961, 1990)

    def __post_init__(self):
        if self.bias_correction_period is not None:
            assert (
                len(self.bias_correction_period) == 2
            ), f"bias_correction_period should have two elements but the provided value is {self.bias_correction_period}."
            object.__setattr__(
                self, "bias_correction_period", tuple(self.bias_correction_period)
            )

    vois_climate = CMIP6_GRID_VOIS_CLIMATE

    def grid_root(self, product_source):
        """Folder holding the per-year gridded products built on these outlines and
        this projection."""
        return os.path.join(
            data_path,
            "grids_CMIP6",
            bias_cor_suffix(self.bias_correction_period),
            self.ssp,
            self.gcm,
            product_source,
        )

    def prepare(self, region_id):
        """Compute the climate of the region and load it into the cache of this
        process, so that forked workers inherit it."""
        load_climate_CMIP6(
            region_id,
            self.ssp,
            self.gcm,
            self.bias_correction_period,
            change_units=True,
        )

    def add_climate_features(self, dataset, change_units, smoothing_vois=None):
        """CMIP6 counterpart of `Dataset.get_climate_features`."""
        smoothing_vois = smoothing_vois or {}
        dataset.data = get_climate_features_(
            dataset.data,
            change_units,
            dataset.months_tail_pad,
            dataset.months_head_pad,
            dataset.region_id,
            self.ssp,
            self.gcm,
            vois_climate=smoothing_vois.get("vois_climate"),
            vois_other=smoothing_vois.get("vois_other"),
            bias_correction_period=self.bias_correction_period,
        )
