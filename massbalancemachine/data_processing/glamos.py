"""
GLAMOS geodetic mass balance and the Swiss Glacier Inventory outlines it refers to.

GLAMOS publishes every DEM-difference pair it has for Swiss glaciers
(https://doi.glamos.ch/data/volumechange), each one a glacier-wide geodetic mass
balance `Bgeod` over a window that can start as early as the 1850s. That target is
referenced to the glacier area of its own epoch, so a pre-2000 window must be paired
with the Swiss Glacier Inventory (SGI) contemporary with it rather than with the RGI
6.2 outlines, which for the Alps date from 2003. This module provides both halves:
the outlines, and the target derived from the volume-change table.

Glaciers are identified by their **SGI id** ("B36-26"), not by an RGI id. The Swiss
inventory and the RGI do not cut the ice into the same glaciers - Claridenfirn is one
SGI entity that RGI 6.2 splits into four - so the crosswalk built by
`table_RGI62_to_GLAMOS` is deliberately many RGI ids to one SGI entity. As a
consequence the gridded products of this source carry `RGIId == "B36-26"`: that column
is the group key of the grid, not something to join against an RGI table.
"""

import os
import zipfile
import urllib.request

import pandas as pd
import geopandas as gpd

from data_processing.custom_outlines import CustomOutlineSpec
from data_processing.product_utils import data_path
from data_processing.Product import Product
from data_processing.glacier_utils import get_region_shape_file

# GLAMOS covers Switzerland only, which is entirely inside RGI region 11 (Central
# Europe), second-order region 11-01 (Alps).
SWITZERLAND_REGION_ID = 11
SWITZERLAND_SUBREGION = "01"

# The SGI shapefiles carry a compound CRS (LV95 + LN02 height) that geopandas reads
# as such, so the horizontal CRS has to be set explicitly.
SGI_CRS = 2056

GLAMOS_CSV_URL = "https://doi.glamos.ch/data/volumechange/volumechange.csv"


def glamos_folder():
    """Directory holding the cached GLAMOS inputs, always the same place so that a
    later run finds what an earlier one downloaded."""
    return os.path.join(data_path, "GLAMOS")


def glamos_volume_change_file(download: bool = True):
    """Path to the GLAMOS volume-change table, downloading it if necessary."""
    path = os.path.join(glamos_folder(), "volumechange.csv")
    if not os.path.exists(path) and download:
        os.makedirs(glamos_folder(), exist_ok=True)
        print(f"downloading {GLAMOS_CSV_URL}")
        urllib.request.urlretrieve(GLAMOS_CSV_URL, path)
    return path


def sgi_shapefile(epoch: int = 1973, download: bool = True):
    """Path to the SGI shapefile of one inventory epoch, downloading it if
    necessary. GLAMOS publishes 1850, 1931, 1973, 2010 and 2016."""
    url = f"https://doi.glamos.ch/data/inventory/inventory_sgi{epoch}_r1976.zip"
    folder = os.path.join(glamos_folder(), f"inventory_sgi{epoch}_r1976")
    path = os.path.join(folder, f"SGI_{epoch}.shp")
    if not os.path.exists(path) and download:
        os.makedirs(glamos_folder(), exist_ok=True)
        zpath = os.path.join(glamos_folder(), os.path.basename(url))
        if not os.path.exists(zpath):
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, zpath)
        zipfile.ZipFile(zpath).extractall(folder)
    return path


def load_glamos_volume_change(path: str = None):
    """Read the GLAMOS volume-change table.

    Line 7 of the file is a units row with an empty SGI-ID and is dropped. Dates are
    `yyyymmdd`, but about 17 % of the pre-2000 ones encode an unknown day as
    `yyyy9999`, which `pd.to_datetime` cannot parse; only the year is used here, so
    it is read off the string directly.
    """
    df = pd.read_csv(path or glamos_volume_change_file(), skiprows=6)
    df = df[df["SGI-ID"].notna()].copy()
    df["Name"] = df.Name.astype(str).str.strip()
    for c in [
        "A_start",
        "A_end",
        "dV",
        "dh_mean",
        "Bgeod",
        "sigma",
        "covered",
        "rho_dv",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for src, dst in [("date_start", "y0"), ("date_end", "y1")]:
        df[dst] = df[src].astype("int64").astype(str).str[:4].astype(int)
    df["dur"] = df.y1 - df.y0
    df["A_mean"] = (df.A_start + df.A_end) / 2
    return df


def load_sgi_outlines(epoch: int = 1973, sgi_ids_to_keep=None):
    """Read the SGI outlines of one epoch, one entity per SGI id.

    A glacier is stored as several rows when it is split into isolated blocks of
    ice, so the rows are dissolved into one entity per id. The returned frame is in
    EPSG:4326 with the SGI id in a column named "SGI", ready for
    `custom_outlines.build_custom_gdirs`.
    """
    sgi = gpd.read_file(sgi_shapefile(epoch)).set_crs(SGI_CRS, allow_override=True)
    sgi = sgi.dissolve("SGI").reset_index()[["SGI", "geometry"]]
    sgi["area_sgi"] = sgi.area / 1e6
    if sgi_ids_to_keep is not None:
        missing = set(sgi_ids_to_keep).difference(sgi.SGI)
        assert (
            not missing
        ), f"The SGI {epoch} inventory has no entity {sorted(missing)}."
        sgi = sgi[sgi.SGI.isin(sgi_ids_to_keep)]
    return sgi.to_crs("EPSG:4326")


def glamos_outline_spec(epoch: int = 1973, dem_source: str = "SRTM", **overrides):
    """Description of the SGI outlines for the generic custom-outline machinery.

    The DEM defaults to SRTM: its February 2000 acquisition is the same epoch as the
    NASADEM shipped with the RGI level-3 directories, so a calibration run on the SGI
    geometry and a reconstruction run on the RGI one see the same ice surface and
    differ only in the outline.
    """
    spec = dict(
        name="GLAMOS",
        id_column="SGI",
        o1_region=f"{SWITZERLAND_REGION_ID:02d}",
        o2_region=SWITZERLAND_SUBREGION,
        src_date=f"{epoch}-01-01 00:00:00",
        bgndate=f"{epoch}0101",
        dem_source=dem_source,
    )
    spec.update(overrides)
    return CustomOutlineSpec(**spec)


def select_glamos_windows(
    glamos,
    max_year: int = 2000,
    min_window_years: int = 10,
    min_covered: float = 95,
    min_year: int = None,
):
    """Pick one geodetic window per SGI entity.

    A glacier can have dozens of overlapping GLAMOS windows, so one has to be chosen.
    The rule is **the longest window, ties broken on the lowest reported sigma**:
    the mismatch between the survey dates and the calendar years the model integrates
    is a fixed offset in years, so its relative weight falls as the window grows, and
    the signal-to-noise of a DEM difference improves with the elapsed time.

    Args:
        max_year: every window must end at or before this year. This is what makes a
            study pre-2000, and what keeps the calibration out of the period a later
            evaluation uses.
        min_window_years: shorter windows carry too much survey-date noise.
        min_covered: minimum percentage of the glacier the DEM difference covers.
        min_year: earliest year a window may start at. Set it to the first year of
            the climate forcing, otherwise the windows built on 1850s-1930s maps are
            selected and cannot be modelled.
    """
    candidates = glamos.loc[
        (glamos.y1 <= max_year)
        & (glamos.dur >= min_window_years)
        & (glamos.covered >= min_covered)
    ]
    if min_year is not None:
        candidates = candidates.loc[candidates.y0 >= min_year]
    return (
        candidates.sort_values(["dur", "sigma"], ascending=[False, True])
        .drop_duplicates("SGI-ID")
        .set_index("SGI-ID")
    )


def geodetic_target_GLAMOS(
    max_year: int = 2000,
    min_window_years: int = 10,
    min_covered: float = 95,
    min_year: int = None,
    sgi_ids_to_keep=None,
):
    """GLAMOS geodetic targets, one window per SGI entity.

    Returns a dataframe shaped like the one `data_processing.pgo.geodetic_target_PGO`
    returns, so that both can drive the same code path in `GeoDataLoader`:

        RGIId                 the SGI entity id, e.g. "B36-26"
        FROM_DATE, TO_DATE    start and end of the geodetic window
        mwe_per_year          Bgeod, m w.e. per year
        sigma_mwe_per_year    its 1-sigma uncertainty, m w.e. per year

    GLAMOS already reports a rate in m w.e. per year, so unlike the PGO table no unit
    conversion is applied here. Only the year of a survey date is known reliably (see
    `load_glamos_volume_change`), so the window is taken to run from the 1st of
    January of `y0` to the 1st of January of `y1` - the same convention OGGM's
    `mb_calibration_from_scalar_mb` applies to a `ref_period`, which averages over
    `np.arange(y0, y1)`.
    """
    glamos = load_glamos_volume_change()
    chosen = select_glamos_windows(
        glamos,
        max_year=max_year,
        min_window_years=min_window_years,
        min_covered=min_covered,
        min_year=min_year,
    )
    if sgi_ids_to_keep is not None:
        chosen = chosen.loc[chosen.index.intersection(list(sgi_ids_to_keep))]

    df = chosen.reset_index().rename(columns={"SGI-ID": "RGIId"})
    df["FROM_DATE"] = pd.to_datetime(df.y0.astype(str) + "-01-01")
    df["TO_DATE"] = pd.to_datetime(df.y1.astype(str) + "-01-01")
    df["mwe_per_year"] = df.Bgeod
    df["sigma_mwe_per_year"] = df.sigma
    return df


def table_RGI62_to_GLAMOS(
    epoch: int = 1973,
    region_id=SWITZERLAND_REGION_ID,
    min_frac_rgi: float = 0.5,
):
    """Crosswalk from RGI 6.2 ids to the SGI entity of one inventory epoch.

    The match is made by **area overlap**, never by name: GLAMOS carries 1,422
    distinct names and fuzzy matching cannot separate `Feegletscher` from
    `Feegletscher N`, gives four identical answers for the Clariden ids, and fails
    outright on Gietro, Murtel and Basodino.

    The mapping is many RGI ids to one SGI entity, which is the real relation:
    Claridenfirn is one SGI glacier that RGI 6.2 splits into four, and Findel and
    Adler were one entity in 1973 and have since separated into three RGI ids.

    Which epoch is used is not cosmetic: on SGI2016 the Adler stakes map to a
    different entity than on SGI1973, because the ice divide moved. The inventory
    must be the one contemporary with the calibration window.

    Args:
        min_frac_rgi: an RGI outline is accepted as belonging to an SGI entity only
            if at least this fraction of it lies inside, which removes
            largest-overlap matches that are really no match at all.

    Returns a dataframe with columns RGIId, custom_id (the SGI id), area_rgi,
    area_sgi, frac_rgi, frac_sgi - `custom_id` matching the column name
    `table_RGI62_to_PGO` uses, so `GeoDataLoader` reads both the same way.
    """
    if not isinstance(region_id, str):
        region_id = f"{region_id:02d}"
    save_path = os.path.abspath(
        os.path.join(
            data_path, "grids", "GLAMOS", f"RGI62_to_GLAMOS_sgi{epoch}_{region_id}.csv"
        )
    )
    p = Product(save_path)
    if p.is_up_to_date():
        return pd.read_csv(save_path)

    sgi = load_sgi_outlines(epoch=epoch).to_crs(SGI_CRS)
    rgi = gpd.read_file(get_region_shape_file(region_id)).to_crs(sgi.crs)
    rgi = rgi[["RGIId", "geometry"]].copy()
    rgi["area_rgi"] = rgi.area / 1e6
    sgi["area_sgi"] = sgi.area / 1e6

    inter = gpd.overlay(
        rgi[["RGIId", "area_rgi", "geometry"]],
        sgi[["SGI", "area_sgi", "geometry"]],
        how="intersection",
    )
    inter["ov"] = inter.area / 1e6

    matches = inter.sort_values("ov", ascending=False).drop_duplicates("RGIId").copy()
    matches["frac_rgi"] = matches.ov / matches.area_rgi
    matches["frac_sgi"] = matches.ov / matches.area_sgi
    matches = matches.loc[
        matches.frac_rgi >= min_frac_rgi,
        ["RGIId", "SGI", "area_rgi", "area_sgi", "frac_rgi", "frac_sgi"],
    ].rename(columns={"SGI": "custom_id"})

    matches.to_csv(save_path, index=False)
    p.gen_chk()
    return matches
