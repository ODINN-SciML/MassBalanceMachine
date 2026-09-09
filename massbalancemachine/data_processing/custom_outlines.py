"""
Build OGGM glacier directories, and therefore MassBalanceMachine glacier grids,
from an arbitrary set of glacier outlines instead of the RGI.

The gridded feature pipeline normally rests on the RGI: OGGM ships pre-processed
glacier directories for every RGI entity and MassBalanceMachine only has to read
them. A study calibrated against a geodetic mass balance cannot always do that,
because a geodetic rate is referenced to the glacier area of *its own epoch*: a
1957-1999 GLAMOS window belongs to the Swiss inventory of 1973, not to the RGI 6.2
outlines, which for the Alps date from 2003.

This module holds the source-agnostic half of that job - turning a GeoDataFrame of
outlines into glacier directories carrying a DEM, a mask, a slope and an aspect.
What is specific to one outline dataset (where the files live, how the glaciers are
named, which DEM epoch to pair them with) belongs in that dataset's own module, such
as `data_processing.pgo` or `data_processing.glamos`, which describe themselves with
a `CustomOutlineSpec` and call `build_custom_gdirs` here.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional, Union

import geopandas as gpd
from pyproj import CRS
import oggm.utils

from data_processing.oggm_utils import (
    _initialize_oggm_config,
    _initialize_custom_glacier_directories,
)
from data_processing.product_utils import data_path


@dataclass
class CustomOutlineSpec:
    """Everything `build_custom_gdirs` needs to know about one outline dataset.

    Args:
        name: short name of the dataset ("PGO", "GLAMOS"). It is also the folder
            under `.data/grids/` in which the gridded products are stored, and the
            one under `.data/oggm/` holding the glacier directories.
        id_column: column of the outlines carrying the glacier identifier. That
            identifier becomes `gdir.rgi_id` and the key of every gridded product,
            so it should be the dataset's own id (the SGI id "B36-26", say) rather
            than a synthetic RGI-looking one.
        o1_region, o2_region: RGI first- and second-order region codes. OGGM needs
            them to locate region-wide datasets, and they are not inferred from the
            geometry.
        src_date: date of the outlines, "YYYY-MM-DD HH:MM:SS". OGGM only reads the
            year, and uses it as the date the glacier geometry refers to.
        bgndate: same date in the RGI's own "YYYYMMDD" form. Defaults to OGGM's
            "20009999" placeholder, which is what the PGO grids were built with.
        dem_source: name of a DEM supported by OGGM ("COPDEM30", "COPDEM90",
            "SRTM", ...). Worth pairing with the outline epoch: SRTM and NASADEM
            both sample a February 2000 ice surface.
        dem_file: optional user-supplied raster, overriding `dem_source`. Either one
            path used for every glacier, or a {glacier_id: path} mapping. OGGM reads
            it through the global `cfg.PATHS['dem_file']`, so a mapping is applied
            one glacier at a time.
        working_dir: OGGM working directory. Defaults to `.data/oggm/<name>`, which
            is persistent: unlike a temporary directory it lets a second run reuse
            the directories and the DEMs downloaded by the first.
        reset: wipe and rebuild every glacier directory. Leave False unless the DEM
            choice or the outlines themselves changed - the `Product` cache does not
            track those, so it will not notice on its own.
    """

    name: str
    id_column: str = "RGIId"
    o1_region: str = "11"
    o2_region: str = "01"
    src_date: str = "2019-01-01 00:00:00"
    bgndate: str = "20009999"
    dem_source: str = "COPDEM30"
    dem_file: Optional[Union[str, dict]] = None
    working_dir: Optional[str] = None
    reset: bool = False

    def resolved_working_dir(self) -> str:
        if self.working_dir is not None:
            return self.working_dir
        return os.path.join(data_path, "oggm", self.name)

    def grid_root(self) -> str:
        """Folder holding the per-year gridded products of this dataset."""
        return os.path.join(data_path, "grids", self.name)

    def fingerprint(self) -> dict:
        """The fields that change what the glacier directories and the grids
        contain. `working_dir` and `reset` say where and how the work is done, not
        what comes out of it, so they are left out.
        """
        return {
            "id_column": self.id_column,
            "o1_region": self.o1_region,
            "o2_region": self.o2_region,
            "src_date": self.src_date,
            "bgndate": self.bgndate,
            "dem_source": self.dem_source,
            "dem_file": self.dem_file,
        }


SPEC_FILE = "mbm_outline_spec.json"


def assert_spec_matches(
    spec: CustomOutlineSpec, folder: str, overwrite: bool = False
) -> None:
    """Refuse to reuse `folder` if it was filled from different outlines.

    Nothing about a stored glacier directory or a gridded parquet records which
    outlines or which DEM produced it, and `Product` only checks that a file exists.
    Two variants of one dataset - the SGI 1973 and SGI 2016 inventories, say, whose
    glaciers carry the *same* ids - would therefore silently serve each other's
    results, which is the exact mistake that pairing a geodetic window with the wrong
    epoch's geometry is meant to avoid. So the defining fields are written next to
    the products the first time, and compared on every later run.

    `overwrite` records the new specification instead of comparing, for a caller that
    is about to rebuild the folder from scratch anyway.
    """
    path = os.path.join(folder, SPEC_FILE)
    fingerprint = spec.fingerprint()
    if os.path.exists(path) and not overwrite:
        with open(path) as f:
            stored = json.load(f)
        differing = {
            k: (stored.get(k), v) for k, v in fingerprint.items() if stored.get(k) != v
        }
        if differing:
            details = "\n".join(
                f"  {k}: stored {old!r}, requested {new!r}"
                for k, (old, new) in sorted(differing.items())
            )
            raise ValueError(
                f"{folder} already holds products of the {spec.name!r} outlines built "
                f"with a different specification:\n{details}\n"
                "Give this variant its own CustomOutlineSpec.name so the two can live "
                "side by side, or delete that folder to rebuild it."
            )
        return
    os.makedirs(folder, exist_ok=True)
    with open(path, "w") as f:
        json.dump(fingerprint, f, indent=4, sort_keys=True)


def _utm_crs_of(outlines: gpd.GeoDataFrame) -> CRS:
    """UTM projection covering `outlines`, taken from their CRS when they already
    are in one, and otherwise inferred from the centroid of the whole set."""
    crs = CRS(outlines.crs)
    if crs.utm_zone is not None:
        return crs
    centroid_lon = outlines.geometry.centroid.x.mean()
    centroid_lat = outlines.geometry.centroid.y.mean()
    zone_number = int((centroid_lon + 180) / 6) + 1
    return CRS.from_dict(
        {"proj": "utm", "zone": zone_number, "south": centroid_lat < 0}
    )


def build_custom_gdirs(outlines: gpd.GeoDataFrame, spec: CustomOutlineSpec):
    """Turn a set of custom outlines into OGGM glacier directories.

    A glacier of a custom inventory is often stored as several polygons - isolated
    blocks of ice sharing one identifier. Those are dissolved into one entity, and
    the entity's geometry is then replaced by its convex hull so that the OGGM grid
    covers every block; the true mask is rebuilt from the individual polygons inside
    `_initialize_custom_glacier_directories`, which is why the un-dissolved frame is
    returned alongside the directories.

    Args:
        outlines: the glacier outlines, carrying `spec.id_column`.
        spec: description of the outline dataset.

    Returns:
        gdirs (list): the OGGM glacier directories, `gdir.rgi_id` being the value of
            `spec.id_column`.
        splitdf (gpd.GeoDataFrame): the un-dissolved outlines in RGI-like form, one
            row per polygon, used to rebuild the true glacier mask and to match the
            outlines against another inventory.
    """
    id_column = spec.id_column
    assert (
        id_column in outlines.columns
    ), f"The outlines have no column {id_column!r} to identify glaciers with."

    # Initialize the OGGM Config
    working_dir = spec.resolved_working_dir()
    os.makedirs(working_dir, exist_ok=True)
    # Reusing directories built from other outlines would go unnoticed otherwise
    assert_spec_matches(spec, working_dir, overwrite=spec.reset)
    _initialize_oggm_config(working_dir)

    # Retrieve projection
    utm_crs = _utm_crs_of(outlines)
    outlines = outlines.to_crs(utm_crs)

    # Cook dataframe into an RGI-compatible GeoDataFrame. `cook_rgidf` mints its own
    # RGI-looking ids; assigning the dataset's own id over them is what makes
    # `gdir.rgi_id` the native identifier.
    rgidf = oggm.utils.cook_rgidf(
        outlines,
        o1_region=spec.o1_region,
        o2_region=spec.o2_region,
        bgndate=spec.bgndate,
        assign_column_values={id_column: id_column},
    )
    rgidf["utm_zone"] = utm_crs.utm_zone

    # Drop samples for which there is no associated glacier ID
    rgidf = rgidf[rgidf[id_column].notna()]

    # Merge entries that share the same glacier ID
    merged_df = rgidf.dissolve(by=id_column).reset_index()

    # Reproject and compute area in km². `cook_rgidf` leaves Area at -9999 and
    # GlacierDirectory would copy that straight into rgi_area_km2.
    geom_utm = merged_df.geometry.to_crs(utm_crs)
    merged_df["Area"] = geom_utm.area / 1e6

    # Replace the geometry by the convex hull of the set of geometries to get a grid that covers all of the entries
    # That geometry is overwritten in _initialize_custom_glacier_directories by the union of all geometries since OGGM does not handle MultiPolygon
    merged_df["geometry"] = merged_df.geometry.convex_hull

    df = merged_df.rename(
        columns={
            id_column: "rgi_id",
            "O1Region": "o1region",
            "O2Region": "o2region",
        }
    )
    df["src_date"] = spec.src_date

    # Run OGGM and overwrite geometry
    gdirs = _initialize_custom_glacier_directories(
        df,
        rgidf,
        dem_source=spec.dem_source,
        dem_file=spec.dem_file,
        reset=spec.reset,
        id_column=id_column,
    )

    return gdirs, rgidf
