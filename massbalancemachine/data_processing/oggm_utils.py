import os
import numpy as np
import rasterio
import shapely
import xarray as xr
from oggm import workflow, tasks
from oggm import cfg as oggmCfg

import config


def _initialize_oggm_config(custom_working_dir):
    """Initialize OGGM configuration."""
    oggmCfg.initialize(logging_level="WARNING")
    oggmCfg.PARAMS["border"] = 10
    oggmCfg.PARAMS["use_multiprocessing"] = True
    oggmCfg.PARAMS["continue_on_error"] = True
    if len(custom_working_dir) == 0:
        current_path = os.getcwd()
        oggmCfg.PATHS["working_dir"] = os.path.join(current_path, "OGGM")
    else:
        oggmCfg.PATHS["working_dir"] = custom_working_dir


def _initialize_glacier_directories(rgi_ids_list: list, cfg: config.Config) -> list:
    """Initialize glacier directories."""
    base_url = cfg.base_url_w5e5 if cfg.prepro_level >= 3 else cfg.base_url_l2
    glacier_directories = workflow.init_glacier_directories(
        rgi_ids_list,
        reset=False,
        from_prepro_level=cfg.prepro_level,
        prepro_base_url=base_url,
        prepro_border=10,
    )

    workflow.execute_entity_task(
        tasks.gridded_attributes, glacier_directories, print_log=False
    )
    return glacier_directories


def check_elevation_raster(path: str):
    """Refuse a raster that cannot be a DEM.

    OGGM reads the first band of a user DEM as elevations. Given a rendered image of a
    DEM - grey levels stretched to 0-255 in 8-bit bands - it fails on the nodata value
    of `define_glacier_region`, but `continue_on_error` swallows that and the run only
    breaks later on a missing `gridded_data.nc`, far from the cause. The same image
    converted to another data type would not fail at all: the glaciers would lie
    between 0 and 255 m and every feature derived from the DEM would be wrong.
    """
    with rasterio.open(path) as src:
        dtype = src.dtypes[0]
        assert src.count == 1 and dtype != "uint8", (
            f"{path} does not look like an elevation raster: it has {src.count} "
            f"band(s) of {dtype} with colour interpretation "
            f"{[c.name for c in src.colorinterp]}. This is typically a rendered image "
            "of the DEM rather than the DEM itself; a single band of elevations is "
            "expected."
        )


def _define_glacier_region_with_dem(gdirs, dem_source, dem_file):
    """Run `tasks.define_glacier_region` on `gdirs` with the requested DEM.

    `dem_source` is the name of a DEM supported by OGGM ('COPDEM30', 'COPDEM90',
    'SRTM', ...). `dem_file` optionally overrides it with a user-supplied raster:
    either a single path applied to every glacier, or a {glacier_id: path} mapping.
    OGGM reads the user raster through the *global* `cfg.PATHS['dem_file']`, so a
    per-glacier mapping has to be applied one directory at a time.
    """
    if dem_file is None:
        workflow.execute_entity_task(
            tasks.define_glacier_region, gdirs, source=dem_source
        )
        return

    paths = set(dem_file.values()) if isinstance(dem_file, dict) else {dem_file}
    for path in paths:
        check_elevation_raster(path)

    prev_dem_file = oggmCfg.PATHS.get("dem_file", "")
    try:
        if isinstance(dem_file, dict):
            missing = [gdir.rgi_id for gdir in gdirs if gdir.rgi_id not in dem_file]
            assert not missing, (
                "A per-glacier DEM mapping was given but it has no entry for "
                f"{missing}. Provide a DEM for every glacier or pass a single path."
            )
            # One glacier at a time: cfg.PATHS['dem_file'] is global, so a parallel
            # run would apply whichever path was set last to all of them.
            for gdir in gdirs:
                oggmCfg.PATHS["dem_file"] = dem_file[gdir.rgi_id]
                tasks.define_glacier_region(gdir, source="USER")
        else:
            oggmCfg.PATHS["dem_file"] = dem_file
            workflow.execute_entity_task(
                tasks.define_glacier_region, gdirs, source="USER"
            )
    finally:
        oggmCfg.PATHS["dem_file"] = prev_dem_file


# Variables of OGGM's gridded data that `_initialize_custom_glacier_directories` removes
# once it has replaced the glacier mask by the true outline
_CUSTOM_DROPPED_VARIABLES = [
    "glacier_ext",
    "glacier_ext_erosion",
    "ice_divides",
    "dis_from_border",
    "topo_valid_mask",
]


def _custom_gdir_complete(gdir) -> bool:
    """Whether a custom glacier directory went through the whole of
    `_initialize_custom_glacier_directories`. A run interrupted after OGGM's tasks
    leaves a `gridded_data` file with the mask of the convex hull and the variables the
    last step removes, which the gridded features cannot be built from."""
    if not gdir.has_file("gridded_data"):
        return False
    with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
        return not any(v in ds for v in _CUSTOM_DROPPED_VARIABLES)


def _initialize_custom_glacier_directories(
    df,
    splitdf,
    dem_source: str = "COPDEM30",
    dem_file=None,
    reset: bool = True,
    id_column: str = "RGIId",
) -> list:
    """Build OGGM glacier directories from a custom (non-RGI) inventory.

    Args:
        df: RGI-v7-shaped frame of dissolved outlines, one row per glacier.
        splitdf: the *un-dissolved* frame. A glacier of a custom inventory is often
            stored as several polygons (isolated blocks of ice); `df` carries their
            convex hull so that the OGGM grid covers them all, and the true mask is
            rebuilt here from every polygon of `splitdf`.
        dem_source: name of a DEM supported by OGGM ('COPDEM30', 'SRTM', ...).
        dem_file: optional user-supplied raster overriding `dem_source`, either one
            path for every glacier or a {glacier_id: path} mapping.
        reset: wipe and rebuild the directories. When False, glaciers that already
            have a `gridded_data` file are left alone, so a persistent working
            directory does not re-download every DEM on each run.
        id_column: the column of `splitdf` holding the glacier id.
    """
    oggmCfg.PARAMS["use_rgi_area"] = False  # recompute area from geometry
    oggmCfg.PARAMS["use_intersects"] = False
    # oggmCfg.PARAMS['border'] = 10

    gdirs = workflow.init_glacier_directories(
        df,
        reset=reset,
        force=reset,
    )

    todo = gdirs if reset else [g for g in gdirs if not _custom_gdir_complete(g)]
    if todo:
        # Define the local map projection and get the DEM
        _define_glacier_region_with_dem(todo, dem_source, dem_file)

        # Compute glacier masks, slope, and aspect
        workflow.execute_entity_task(tasks.glacier_masks, todo)

        workflow.execute_entity_task(tasks.gridded_attributes, todo)

        for gdir in todo:
            rgi_id = gdir.rgi_id
            with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
                ds = ds.load()
            mask = np.zeros_like(ds.glacier_mask.values)
            true_geom = splitdf[splitdf[id_column] == rgi_id]
            for i in range(true_geom.shape[0]):
                out = gdir.grid.region_of_interest(geometry=true_geom.iloc[i].geometry)
                mask = mask | out
            ds["glacier_mask"].values = mask
            ds = ds.drop(_CUSTOM_DROPPED_VARIABLES)
            # Overwrite netcdf
            save_path = gdir.get_filepath("gridded_data")
            ds.to_netcdf(save_path)

    # Every opening of a directory rewrites its outlines with the area of the convex
    # hull, so the true area goes back in each time, once the grid (whose resolution
    # follows the area) is defined
    for gdir in gdirs:
        _write_true_area(gdir, splitdf[splitdf[id_column] == gdir.rgi_id])

    return gdirs


def _write_true_area(gdir, polygons):
    """Write the area of the union of `polygons`, the outline of the glacier, into
    the outlines of its custom directory, which OGGM gave the area of their convex
    hull. It is the area the elevation-band flowlines are scaled to
    (`gdir.rgi_area_km2`)."""
    outlines = gdir.read_shapefile("outlines")
    true_geom = shapely.union_all(polygons.geometry.to_crs(outlines.crs).values)
    outlines["Area"] = true_geom.area / 1e6
    gdir.write_shapefile(outlines, "outlines")
    # rgi_area_km2 is a lazy property: forget the area it read before
    gdir.__dict__.pop("_lazy_rgi_area_km2", None)


def _glacier_name(rgi_ids_list: list, cfg: config.Config, custom_working_dir=""):

    # Initialize the OGGM Config
    _initialize_oggm_config(custom_working_dir)
    glacier_directories = _initialize_glacier_directories(rgi_ids_list, cfg)
    return {gdir.rgi_id: gdir.name for gdir in glacier_directories}
