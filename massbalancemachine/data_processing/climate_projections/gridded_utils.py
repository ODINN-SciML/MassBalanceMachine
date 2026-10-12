"""
Gridded products of glaciers whose climate comes from CMIP6 projections instead of
ERA5, see `data_processing.gridded_utils` for the ERA5 ones.
"""

import numpy as np
import pandas as pd
import xarray as xr

from data_processing.gridded_utils import (
    MONTH_TO_ID,
    _region_id_of,
    check_cell_altitudes,
    climate_cells_of_glaciers,
    create_gridded_features_RGI,
    create_gridded_features_PGO,
    create_gridded_features_GLAMOS,
    create_gridded_features_Rabatel16,
    finalize_cell_table,
)
from data_processing.glamos import SWITZERLAND_REGION_ID
from data_processing.rabatel16 import FRENCH_ALPS_REGION_ID
from data_processing.utils.years import contiguous_year_runs
from data_processing.climate_projections.get_climate_data import (
    CMIP6_GRID_VOIS_CLIMATE,
    ensure_climate_features_CMIP6,
)

# Custom outline sources: their generation function, and the RGI region of their
# glaciers when it cannot be read from the glacier ids.
_CUSTOM_OUTLINE_SOURCES = {
    "PGO": (create_gridded_features_PGO, None),
    "GLAMOS": (create_gridded_features_GLAMOS, SWITZERLAND_REGION_ID),
    "Rabatel16": (create_gridded_features_Rabatel16, FRENCH_ALPS_REGION_ID),
}

# Month token of the gridded products for every month number, see `MONTH_TO_ID`
_ID_TO_MONTH = {month_id: token for token, month_id in MONTH_TO_ID.items()}


def create_gridded_features_CMIP6(
    cfg,
    glacier_ids,
    climate,
    years,
    product_source="Hugonnet21",
    outline_kwargs=None,
    multi=True,
    num_workers=None,
):
    """Generate the per-year gridded products of glaciers with the climate of a CMIP6
    projection.

    Args:
        glacier_ids: the glaciers, identified as in `product_source`.
        climate (CMIP6Climate): the projection to take the climate from.
        years: the calendar years to generate.
        product_source (str): the outlines to build the grids on, "Hugonnet21" for
            the RGI outlines, or "PGO", "GLAMOS" or "Rabatel16".
        outline_kwargs (dict): extra arguments of the generation function of a custom
            outline source, such as the `epoch` of the GLAMOS outlines.
    """
    years = list(years)
    outline_kwargs = outline_kwargs or {}
    if product_source == "Hugonnet21":
        assert not outline_kwargs, "The RGI outlines take no outline_kwargs."
        create_gridded_features_RGI(
            cfg,
            glacier_ids,
            years=years,
            multi=multi,
            num_workers=num_workers,
            climate=climate,
        )
        return

    if product_source not in _CUSTOM_OUTLINE_SOURCES:
        raise ValueError(
            f"Unknown product source {product_source!r}. Known sources: "
            f"{sorted(['Hugonnet21', *_CUSTOM_OUTLINE_SOURCES])}."
        )
    # The custom outline sources generate the years covered by time windows. A
    # window per run of consecutive years covers exactly `years`.
    windows = [
        (f"{first}-01-01", f"{last + 1}-01-01")
        for first, last in contiguous_year_runs(years)
    ]
    create_fn, _ = _CUSTOM_OUTLINE_SOURCES[product_source]
    create_fn(
        cfg,
        {glacier_id: windows for glacier_id in glacier_ids},
        multi=multi,
        num_workers=num_workers,
        climate=climate,
        **outline_kwargs,
    )


def climate_features_of_glaciers(
    glacier_ids,
    cfg,
    ssp,
    gcm,
    years,
    product_source="Hugonnet21",
    bias_correction_period=(1961, 1990),
    features=None,
    drop_duplicate_cells=True,
    validate=False,
):
    """The monthly climate features of a set of glaciers under a CMIP6 projection,
    one row per climate cell.

    CMIP6 counterpart of `data_processing.gridded_utils.climate_features_of_glaciers`.
    It returns the values the CMIP6 gridded products carry, without generating them:
    a gridded product holds the climate of a single cell per glacier, the one covering
    most of its pixels, and that cell follows from the outline alone. It is therefore
    read from the ERA5 grids of the same outlines, and the climate of the projection,
    bias corrected against ERA5 and put on its grid, is taken at that cell.

    Args:
        glacier_ids: the glaciers whose climate to extract, identified as in
            `product_source`. Each needs an ERA5 grid of `product_source`, of any
            year.
        cfg: not used, kept for symmetry with the ERA5 function.
        ssp (str): SSP scenario, for example "historical", "ssp1_2_6" or "ssp5_8_5".
        gcm (str): the GCM, one of those of `ensure_climate_CMIP6`.
        years: the calendar years to extract.
        product_source (str): the outlines giving the climate cells, "Hugonnet21" for
            the RGI outlines, or "PGO", "GLAMOS" or "Rabatel16".
        bias_correction_period (tuple of int): first and last year, inclusive, of the
            period over which the projection is bias corrected against ERA5. None for
            no correction.
        features (list of str): the climate columns to keep. Defaults to
            `CMIP6_GRID_VOIS_CLIMATE`, everything the grids carry.
        validate (bool): additionally check that every feature is finite in every
            cell and month.

    See `data_processing.gridded_utils.climate_features_of_glaciers` for
    `drop_duplicate_cells` and the columns of the result.
    """
    del cfg
    features = list(CMIP6_GRID_VOIS_CLIMATE) if features is None else list(features)
    unknown = sorted(set(features) - set(CMIP6_GRID_VOIS_CLIMATE))
    assert not unknown, f"CMIP6 provides no {unknown}."
    years = list(years)
    glacier_ids = list(dict.fromkeys(glacier_ids))  # keep the order, drop repeated ids

    if product_source != "Hugonnet21" and product_source not in _CUSTOM_OUTLINE_SOURCES:
        raise ValueError(
            f"Unknown product source {product_source!r}. Known sources: "
            f"{sorted(['Hugonnet21', *_CUSTOM_OUTLINE_SOURCES])}."
        )
    region_id = _CUSTOM_OUTLINE_SOURCES.get(product_source, (None, None))[1]
    if region_id is None:
        region_ids = {_region_id_of(glacier_id) for glacier_id in glacier_ids}
        assert (
            len(region_ids) == 1
        ), f"The glaciers span several RGI regions: {sorted(region_ids)}."
        region_id = region_ids.pop()

    cells = climate_cells_of_glaciers(glacier_ids, product_source, region_id=region_id)
    check_cell_altitudes(cells)
    unique_cells = (
        cells.reset_index()
        .groupby(["CLIMATE_LAT", "CLIMATE_LON"], sort=True)
        .agg(
            RGI_IDS=("RGIId", lambda ids: tuple(sorted(ids))),
            ALTITUDE_CLIMATE=("ALTITUDE_CLIMATE", "first"),
        )
        .reset_index()
    )

    # The `_sum` columns are derived from their mean daily accumulation
    variables = list(
        dict.fromkeys(feature.removesuffix("_sum") for feature in features)
    )
    values = _climate_at_cells(
        region_id,
        ssp,
        gcm,
        bias_correction_period,
        variables,
        unique_cells.CLIMATE_LAT.to_numpy(),
        unique_cells.CLIMATE_LON.to_numpy(),
        years,
    )

    # One row per cell and month, as the gridded products hold them
    df = values.to_dataframe().reset_index()
    time = pd.DatetimeIndex(df.pop("time"))
    df["YEAR"] = time.year
    df["MONTHS"] = time.month.map(_ID_TO_MONTH)
    for feature in features:
        if feature.endswith("_sum"):
            df[feature] = (
                df[feature.removesuffix("_sum")].to_numpy()
                * time.days_in_month.to_numpy()
            )
    df = df.merge(unique_cells, left_on="cell", right_index=True).drop(columns="cell")
    df["RGIId"] = df.RGI_IDS.map(lambda ids: ids[0])
    df["N_GLACIERS"] = df.RGI_IDS.map(len)

    if validate:
        not_finite = [f for f in features if not np.isfinite(df[f]).all()]
        assert not not_finite, (
            f"The CMIP6 climate has non-finite values of {not_finite} at the cells "
            "of some glaciers."
        )

    return finalize_cell_table(df, cells, features, drop_duplicate_cells)


def _climate_at_cells(
    region_id, ssp, gcm, bias_correction_period, variables, lats, lons, years
):
    """The bias-corrected CMIP6 climate of some ERA5 cells over `years`, along a
    `cell` dimension, with `t2m` in °C as in the gridded products.

    The regional file is opened lazily and only the cells asked for are read, rather
    than the whole file as `load_climate_CMIP6` does.
    """
    file_path = ensure_climate_features_CMIP6(
        region_id, ssp, gcm, bias_correction_period
    )
    with xr.open_dataset(file_path) as ds:
        # The file can be in either longitude convention, as `_process_climate_data`
        # also handles when it selects the points of the grids.
        if float(ds.longitude.max()) > 180.0:
            lons = np.mod(lons, 360.0)

        missing = sorted(set(years) - set(ds["time"].dt.year.values.tolist()))
        if missing:
            raise ValueError(
                f"The {ssp} run of {gcm} has no climate for the years {missing}."
            )

        # The box spanned by the cells is read, an orthogonal selection the lazy
        # backend supports, and the cells are picked from it once in memory.
        box = (
            ds[variables]
            .sel(time=ds["time"].dt.year.isin(years))
            .sel(latitude=np.unique(lats), longitude=np.unique(lons), method="nearest")
            .load()
        )

    values = box.isel(
        latitude=xr.DataArray(_nearest(box.latitude.values, lats), dims="cell"),
        longitude=xr.DataArray(_nearest(box.longitude.values, lons), dims="cell"),
    )
    # The cells come from the ERA5 axes: a CMIP6 file on another grid would silently
    # give the climate of a neighbouring cell.
    offset = max(
        float(np.abs(values.latitude.values - lats).max()),
        float(np.abs(values.longitude.values - lons).max()),
    )
    assert offset < 1e-3, (
        f"The CMIP6 climate of region {region_id} is not on the grid of the ERA5 "
        f"cells of the glaciers (offset of {offset} degrees)."
    )

    values = values.drop_vars(["latitude", "longitude"])
    if "t2m" in values:
        values = values.assign(t2m=values["t2m"] - 273.15)
    return values


def _nearest(axis, points):
    """Index of the nearest entry of `axis` for every point."""
    return np.abs(axis[:, None] - np.asarray(points)[None, :]).argmin(axis=0)
