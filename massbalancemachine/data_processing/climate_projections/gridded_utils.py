"""
Gridded products of glaciers whose climate comes from CMIP6 projections instead of
ERA5, see `data_processing.gridded_utils` for the ERA5 ones.
"""

from data_processing.gridded_utils import (
    climate_features_from_grids,
    create_gridded_features_RGI,
    create_gridded_features_PGO,
    create_gridded_features_GLAMOS,
    create_gridded_features_Rabatel16,
)
from data_processing.glamos import SWITZERLAND_REGION_ID
from data_processing.rabatel16 import FRENCH_ALPS_REGION_ID
from data_processing.utils.years import contiguous_year_runs
from data_processing.climate_projections.get_climate_data import (
    CMIP6Climate,
    CMIP6_GRID_VOIS_CLIMATE,
)

# Custom outline sources: their generation function, and the RGI region of their
# glaciers when it cannot be read from the glacier ids.
_CUSTOM_OUTLINE_SOURCES = {
    "PGO": (create_gridded_features_PGO, None),
    "GLAMOS": (create_gridded_features_GLAMOS, SWITZERLAND_REGION_ID),
    "Rabatel16": (create_gridded_features_Rabatel16, FRENCH_ALPS_REGION_ID),
}


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
    outline_kwargs=None,
    drop_duplicate_cells=True,
    validate=False,
    multi=True,
    num_workers=None,
):
    """The monthly climate features of a set of glaciers under a CMIP6 projection,
    one row per climate cell.

    CMIP6 counterpart of `data_processing.gridded_utils.climate_features_of_glaciers`:
    the gridded products of the glaciers are generated for `years` with the climate
    of the projection, bias corrected against ERA5 and put on its grid, and the
    climate is then read back from them.

    Args:
        glacier_ids: the glaciers whose climate to extract, identified as in
            `product_source`.
        ssp (str): SSP scenario, for example "historical", "ssp1_2_6" or "ssp5_8_5".
        gcm (str): the GCM, one of those of `ensure_climate_CMIP6`.
        years: the calendar years to generate and extract.
        product_source (str): the outlines the grids are built on, "Hugonnet21" for
            the RGI outlines, or "PGO", "GLAMOS" or "Rabatel16".
        bias_correction_period (tuple of int): first and last year, inclusive, of the
            period over which the projection is bias corrected against ERA5. None for
            no correction.
        features (list of str): the climate columns to keep. Defaults to
            `CMIP6_GRID_VOIS_CLIMATE`, everything the grids carry.
        outline_kwargs (dict): extra arguments of the generation function of a custom
            outline source, such as the `epoch` of the GLAMOS outlines.
        num_workers (int): number of processes generating the grids. Left to None it
            follows the memory free on the machine, see `parallel_utils.worker_count`.

    See `data_processing.gridded_utils.climate_features_of_glaciers` for
    `drop_duplicate_cells`, `validate` and the columns of the result.
    """
    climate = CMIP6Climate(ssp, gcm, bias_correction_period)
    features = list(CMIP6_GRID_VOIS_CLIMATE) if features is None else list(features)
    years = list(years)
    glacier_ids = list(dict.fromkeys(glacier_ids))  # keep the order, drop repeated ids

    create_gridded_features_CMIP6(
        cfg,
        glacier_ids,
        climate,
        years,
        product_source=product_source,
        outline_kwargs=outline_kwargs,
        multi=multi,
        num_workers=num_workers,
    )

    region_id = _CUSTOM_OUTLINE_SOURCES.get(product_source, (None, None))[1]
    return climate_features_from_grids(
        glacier_ids,
        years,
        product_source,
        features,
        drop_duplicate_cells=drop_duplicate_cells,
        validate=validate,
        grid_root=climate.grid_root(product_source),
        region_id=region_id,
    )
