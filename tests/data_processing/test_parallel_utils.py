"""Tests of the memory-aware worker sizing and of the climate data cache it relies on."""

import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data_processing import parallel_utils
from data_processing.get_climate_data import _load_datasets, _load_datasets_cached
from data_processing.parallel_utils import GIB, worker_count


def test_worker_count_fits_the_budget():
    # 10 GiB free, 2 reserved, 2 per worker -> four workers.
    assert (
        worker_count(
            per_worker_bytes=2 * GIB,
            reserve_bytes=2 * GIB,
            max_workers=16,
            available_bytes=10 * GIB,
        )
        == 4
    )


def test_worker_count_is_capped_by_the_cores():
    assert (
        worker_count(
            per_worker_bytes=1 * GIB,
            reserve_bytes=0,
            max_workers=3,
            available_bytes=100 * GIB,
        )
        == 3
    )


def test_worker_count_never_returns_zero():
    """A machine with nothing to spare still makes progress, one glacier-year at a
    time, rather than deadlocking on an empty pool."""
    assert (
        worker_count(
            per_worker_bytes=8 * GIB,
            reserve_bytes=4 * GIB,
            max_workers=16,
            available_bytes=4 * GIB,
        )
        == 1
    )


def test_worker_count_backs_off_as_memory_is_taken():
    """The point of the whole module: the same request yields fewer workers once
    something else on the machine has taken the memory."""
    kwargs = dict(per_worker_bytes=2 * GIB, reserve_bytes=4 * GIB, max_workers=16)
    plenty = worker_count(available_bytes=28 * GIB, **kwargs)
    scarce = worker_count(available_bytes=12 * GIB, **kwargs)
    assert plenty == 12 and scarce == 4


def test_requested_worker_count_wins():
    """An explicit number is an instruction, not a hint: someone who knows their
    machine keeps the last word over the estimate."""
    assert (
        worker_count(
            requested=7,
            per_worker_bytes=8 * GIB,
            reserve_bytes=4 * GIB,
            available_bytes=4 * GIB,
        )
        == 7
    )


def test_available_memory_is_plausible():
    assert 0 < parallel_utils.available_memory() <= parallel_utils.total_memory()


def _write_fake_era5(folder):
    """Two files shaped like the ERA5 monthly data, small enough to load in a test."""
    time = pd.date_range("2000-01-01", periods=4, freq="MS")
    climate_path = os.path.join(folder, "climate.nc")
    geopotential_path = os.path.join(folder, "geopotential.nc")
    xr.Dataset(
        {"t2m": (("time", "latitude", "longitude"), np.full((4, 2, 2), 283.15))},
        coords={"time": time, "latitude": [46.0, 46.1], "longitude": [8.0, 8.1]},
    ).to_netcdf(climate_path)
    xr.Dataset(
        {"z": (("latitude", "longitude"), np.full((2, 2), 1000.0))},
        coords={"latitude": [46.0, 46.1], "longitude": [8.0, 8.1]},
    ).to_netcdf(geopotential_path)
    return climate_path, geopotential_path


@pytest.fixture
def fake_era5(tmp_path):
    _load_datasets_cached.cache_clear()
    yield _write_fake_era5(str(tmp_path))
    _load_datasets_cached.cache_clear()


def test_climate_datasets_are_served_from_the_cache(fake_era5):
    climate_path, geopotential_path = fake_era5
    first = _load_datasets(climate_path, geopotential_path)
    second = _load_datasets(climate_path, geopotential_path)
    assert first[0] is second[0] and first[1] is second[1]


def test_cached_climate_units_are_converted_exactly_once(fake_era5):
    """The unit change used to be an assignment into the loaded dataset. Serving
    that same dataset again would subtract 273.15 a second time, so the conversion
    has to happen while loading and produce a new dataset rather than edit one."""
    climate_path, geopotential_path = fake_era5
    for _ in range(3):
        ds_climate, _ = _load_datasets(climate_path, geopotential_path, True)
        assert float(ds_climate.t2m.mean()) == pytest.approx(10.0)

    # And asking for the raw file still gives the raw file.
    ds_kelvin, _ = _load_datasets(climate_path, geopotential_path, False)
    assert float(ds_kelvin.t2m.mean()) == pytest.approx(283.15)


def test_replacing_the_climate_file_invalidates_the_cache(fake_era5, tmp_path):
    climate_path, geopotential_path = fake_era5
    before, _ = _load_datasets(climate_path, geopotential_path)
    assert float(before.t2m.mean()) == pytest.approx(283.15)

    # A different number of time steps, so that the file differs in size as well as
    # in modification time and the test cannot hinge on the clock resolution.
    time = pd.date_range("2000-01-01", periods=6, freq="MS")
    xr.Dataset(
        {"t2m": (("time", "latitude", "longitude"), np.full((6, 2, 2), 290.0))},
        coords={"time": time, "latitude": [46.0, 46.1], "longitude": [8.0, 8.1]},
    ).to_netcdf(climate_path)

    after, _ = _load_datasets(climate_path, geopotential_path)
    assert float(after.t2m.mean()) == pytest.approx(290.0)
