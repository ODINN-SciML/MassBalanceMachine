"""
Geodetic mass balance of Austrian glaciers from Fischer (2011), and the Austrian
Glacier Inventory outlines it refers to.

Fischer, A.: Comparison of direct and geodetic mass balances on a multi-annual time
scale, The Cryosphere, 5, 107-124, doi:10.5194/tc-5-107-2011, 2011.

Table 3 of the paper gives the cumulative specific geodetic balance of five Austrian
glaciers over 15 periods between DEMs, from 1953 to 2006. They are kept as constants
in `FISCHER11_PERIODS`: the paper publishes no data file. Only the periods between
two consecutive DEMs are kept, not the "total" rows, which are built from the same
DEMs and therefore carry no independent information. Übergossene Alm (HK) is left
out: its area in Table 1 (1.636 km²) matches none of the three entities the
inventories cut it into, nor their union (2.47 km² in 1969), so the area its balance
is referenced to cannot be rebuilt.

The paper divides the volume change by the glacier area at the *start* of a period
(its Eq. 2), so a period is gridded on the inventory of its first DEM when there is
one, and on the first inventory, GI 1 (1969), otherwise; see `outline_epoch_of_period`.
The inventories are those of Fischer et al. (2015) on PANGAEA
(doi:10.1594/PANGAEA.844988), and two of them are used:

    GI 1   1969         outlines of the aerial survey of August-September 1969
    GI 2   1996-2002    one survey year per mountain range, in the `Year` column:
                        1997 for the Ötztal Alps, 1998 for the Granatspitzgruppe,
                        2002 for the Silvretta

An inventory is identified by its **epoch**, 1969 or 1998, the year PANGAEA names it
after. Every period ending before 2000 is gridded on GI 1; only the periods starting
in 1997 and Jamtalferner 2002-2006 need GI 2.

Glaciers are identified by their **inventory number** ("2125" for Hintereisferner),
which is the same in every inventory. As for GLAMOS, the gridded products of this
source therefore carry `RGIId == "2125"`, and the crosswalk built by
`table_RGI62_to_Fischer11` is many RGI ids to one inventory entity.

The paper converts volume changes to water equivalent with a density of
850 kg m-3, the one of this workflow, so no conversion is needed.

A period runs over whole hydrological years: the DEMs are acquired at the end of the
ablation season, between mid-August and early October, and a period between the
DEMs of `y0` and `y1` is taken to cover the hydrological years `y0 + 1` to `y1`, from
the 1st of October of `y0` to the 1st of October of `y1`. That is the number of years
`N` Table 3 divides by.
"""

import os
import urllib.request
import zipfile

import geopandas as gpd
import pandas as pd

from data_processing.custom_outlines import CustomOutlineSpec, match_rgi62_by_overlap
from data_processing.product_utils import data_path
from data_processing.Product import Product
from data_processing.glacier_utils import get_region_shape_file
from data_processing.utils.years import years_outside

# Austria is entirely inside RGI region 11 (Central Europe), second-order region
# 11-01 (Alps).
AUSTRIA_REGION_ID = 11
AUSTRIA_SUBREGION = "01"

# Projected CRS the inventories are brought to before measuring areas: every mountain
# range is delivered in its own Gauss-Krüger zone (EPSG:31254 or 31255), and MGI /
# Austria Lambert covers the whole country.
AUSTRIA_CRS = 31287

# Epoch, archive and PANGAEA dataset of each inventory. The archives are fetched
# through the persistent handle of the AWI repository PANGAEA links to.
GI_RELEASES = {
    1969: ("GI_1", "https://hdl.handle.net/10013/epic.45270.d001"),  # PANGAEA.844983
    1998: ("GI_2", "https://hdl.handle.net/10013/epic.45273.d001"),  # PANGAEA.844984
}
FIRST_EPOCH = min(GI_RELEASES)

# Name and GI 2 survey year of the glaciers of the paper, keyed by inventory number.
# `load_fischer11_outlines` checks the years against the GI 2 shapefiles.
FISCHER11_GLACIERS = {
    "2125": ("Hintereisferner", 1997),
    "2129": ("Kesselwandferner", 1997),
    "2133": ("Vernagtferner", 1997),
    "13019": ("Jamtalferner", 2002),
    "5097": ("Stubacher Sonnblickkees", 1998),
}

# Table 3 of Fischer (2011): (inventory number, year of the first DEM, year of the
# second DEM, cumulative geodetic balance in m w.e.), with the acquisition method of
# both DEMs from Table 2: TP terrestrial photogrammetry, AP aerial photogrammetry, AL
# airborne laser scanning. Table 2 lists no 1991 DEM for Hintereisferner although
# Table 3 uses one; it is taken to be aerial photogrammetry like every DEM of the
# glacier between 1969 and 1997.
FISCHER11_PERIODS = [
    ("2125", 1953, 1964, -7.5, "TP", "TP"),
    ("2125", 1964, 1967, 1.3, "TP", "TP"),
    ("2125", 1967, 1969, -3.9, "TP", "AP"),
    ("2125", 1969, 1979, 2.7, "AP", "AP"),
    ("2125", 1979, 1991, -13.1, "AP", "AP"),
    ("2125", 1991, 1997, -4.4, "AP", "AP"),
    ("2125", 1997, 2006, -10.1, "AP", "AL"),
    ("2129", 1969, 1971, 0.6, "AP", "AP"),
    ("2129", 1971, 1997, -3.5, "AP", "AP"),
    ("2129", 1997, 2006, -5.0, "AP", "AL"),
    ("13019", 1996, 2002, -2.0, "AP", "AP"),
    ("13019", 2002, 2006, -5.0, "AP", "AL"),
    ("2133", 1969, 1997, -2.6, "AP", "AP"),
    ("2133", 1997, 2006, -10.9, "AP", "AL"),
    ("5097", 1969, 1998, -3.5, "AP", "AP"),
]
# "total" rows of Table 3, which the periods of a glacier add up to
FISCHER11_TOTALS = {"2125": -35.0, "2129": -7.9, "13019": -7.0, "2133": -13.5}

# Error budget of the paper (end of Section 6), for a period of 10 years: m w.e. per
# year from the processes the geodetic method mixes into the balance - densification,
# seasonal snow on a DEM, basal melt - and m of elevation per DEM, cumulated over the
# period. Terrestrial photogrammetry and laser scanning are 1.0 and 0.001 m per year
# over 10 years in the paper; aerial photogrammetry is not in the budget and is taken
# from Section 5.1 (better than 0.71 m). The paper adds the terms linearly and folds
# the 10 % of its density uncertainty into the rounding, which gives its 2.7 m w.e.
# per year for two terrestrial DEMs and 0.702 for two laser scans; see
# `period_sigma_mwe_per_year`.
PROCESS_SIGMA_MWE_PER_YEAR = {
    "densification": 0.1,
    "seasonal_snow": 0.1,
    "basal_melt": 0.5,
}
DEM_SIGMA_M = {"TP": 10.0, "AP": 0.71, "AL": 0.01}


def outline_epoch_of_period(start_year: int, gi2_year: int):
    """Inventory epoch a period starting with the DEM of `start_year` is gridded on.

    The paper refers a balance to the area at the first DEM, so the period is gridded
    on the inventory surveyed that year: GI 2 when `start_year` is the GI 2 survey year
    of the glacier (`gi2_year`), GI 1 when it is 1969. Every other period has no
    inventory of its own and falls back to GI 1.
    """
    return 1998 if start_year == gi2_year else FIRST_EPOCH


def period_sigma_mwe_per_year(n_years: int, dem_start: str, dem_end: str):
    """Uncertainty of the mean annual balance over a period of `n_years` between a DEM
    acquired with `dem_start` and one acquired with `dem_end`, in m w.e. per year.

    It is the error budget of Fischer (2011), whose terms are added linearly as in the
    paper: the processes of `PROCESS_SIGMA_MWE_PER_YEAR`, taken as rates, plus the
    elevation error of both DEMs spread over the period,

        sigma = sum(PROCESS_SIGMA_MWE_PER_YEAR) + (DEM_SIGMA_M[dem_start]
                                                   + DEM_SIGMA_M[dem_end]) / n_years

    which gives the 2.7 of the paper for two terrestrial DEMs 10 years apart and its
    0.702 for two laser scans. The DEM term dominates the historical periods: 7.4 over
    the three years 1964-1967 of Hintereisferner, 0.75 over the 28 years 1969-1997 of
    Vernagtferner. The basal melt term, 0.5 m w.e. per year, sets a floor of 0.7 m w.e.
    per year under every period.
    """
    assert n_years > 0, f"A period of {n_years} years has no mass balance."
    for method in (dem_start, dem_end):
        assert method in DEM_SIGMA_M, (
            f"Unknown DEM acquisition method {method!r}, "
            f"expected one of {sorted(DEM_SIGMA_M)}."
        )
    return float(
        sum(PROCESS_SIGMA_MWE_PER_YEAR.values())
        + (DEM_SIGMA_M[dem_start] + DEM_SIGMA_M[dem_end]) / n_years
    )


def fischer11_periods():
    """Every period of `FISCHER11_PERIODS` with its dates, rate, uncertainty and the
    inventory epoch it is gridded on, one row per period.

    Columns: RGIId (inventory number), name, y0, y1, n_years, cumulative_mwe,
    dem_start, dem_end, FROM_DATE, TO_DATE, mwe_per_year, sigma_mwe_per_year,
    outline_epoch.
    """
    df = pd.DataFrame(
        FISCHER11_PERIODS,
        columns=["RGIId", "y0", "y1", "cumulative_mwe", "dem_start", "dem_end"],
    )
    df.insert(1, "name", df.RGIId.map(lambda g: FISCHER11_GLACIERS[g][0]))
    df["n_years"] = df.y1 - df.y0
    df["FROM_DATE"] = pd.to_datetime(df.y0.astype(str) + "-10-01")
    df["TO_DATE"] = pd.to_datetime(df.y1.astype(str) + "-10-01")
    df["mwe_per_year"] = df.cumulative_mwe / df.n_years
    df["sigma_mwe_per_year"] = [
        period_sigma_mwe_per_year(r.n_years, r.dem_start, r.dem_end)
        for r in df.itertuples()
    ]
    df["outline_epoch"] = [
        outline_epoch_of_period(r.y0, FISCHER11_GLACIERS[r.RGIId][1])
        for r in df.itertuples()
    ]
    return df


def geodetic_target_Fischer11(
    epoch: int = FIRST_EPOCH,
    max_year: int = None,
    min_year: int = None,
    min_period_years: int = 1,
    glacier_ids_to_keep=None,
    allowed_years=None,
    max_outside_years: int = 0,
    multi_period: bool = True,
    tie_break: str = "sigma",
):
    """Geodetic targets of Fischer (2011), one row per period between two DEMs.

    Returns a dataframe shaped like the one `data_processing.glamos.geodetic_target_GLAMOS`
    returns, so that both drive the same code path in `GeoDataLoader`:

        RGIId                 the inventory number, e.g. "2125"
        FROM_DATE, TO_DATE    1st of October of the years of the two DEMs
        cumulative_mwe        the balance of Table 3 over the period, m w.e.
        mwe_per_year          that balance divided by the number of years
        sigma_mwe_per_year    its uncertainty, see `period_sigma_mwe_per_year`
        outline_epoch         the inventory the period is gridded on

    By default a glacier carries every eligible period; they follow each other, the
    DEM that ends one starting the next, and never overlap. Without `multi_period` a
    single period is kept per glacier, chosen as GLAMOS chooses its window (see
    `glamos.select_glamos_windows`): **the longest one, ties broken on the lowest
    sigma**, or on the most recent period with `tie_break="recent"`.

    Args:
        epoch: keep the periods gridded on this inventory, 1969 or 1998 (see
            `outline_epoch_of_period`), since the grids of one run are built on a
            single inventory. None keeps them all, for inspection.
        max_year: every period must end at or before the DEM of this year.
        min_year: every period must start at or after the DEM of this year. Set it
            to the first year of the climate forcing.
        min_period_years: shortest period to keep. The DEM term of the uncertainty
            already weighs down short periods, so none is dropped by default.
        glacier_ids_to_keep: inventory numbers to restrict the target to.
        allowed_years: calendar years a period may touch, for instance the years of
            one side of a train/validation split, up to `max_outside_years` years of
            tolerance. A period is fixed by its DEMs and cannot be cut, as for GLAMOS.
        multi_period: keep every eligible period of a glacier rather than one.
        tie_break: how the single period is chosen among the longest ones, "sigma"
            or "recent". Ignored with `multi_period`.
    """
    tie_break_keys = {
        "sigma": ["sigma_mwe_per_year"],
        "recent": ["y1", "sigma_mwe_per_year"],
    }
    assert (
        tie_break in tie_break_keys
    ), f"tie_break must be one of {sorted(tie_break_keys)}, not {tie_break!r}."
    assert (
        epoch is None or epoch in GI_RELEASES
    ), f"No Austrian inventory of epoch {epoch}, only {sorted(GI_RELEASES)}."
    df = fischer11_periods()
    keep = df.n_years >= min_period_years
    if epoch is not None:
        keep &= df.outline_epoch == epoch
    if max_year is not None:
        keep &= df.y1 <= max_year
    if min_year is not None:
        keep &= df.y0 >= min_year
    if glacier_ids_to_keep is not None:
        keep &= df.RGIId.isin([str(g) for g in glacier_ids_to_keep])
    if allowed_years is not None:
        allowed = set(allowed_years)
        keep &= pd.Series(
            [
                len(years_outside(r.FROM_DATE, r.TO_DATE, allowed)) <= max_outside_years
                for r in df.itertuples()
            ],
            index=df.index,
        )
    df = df[keep]
    if not multi_period:
        keys = ["n_years"] + tie_break_keys[tie_break]
        df = df.sort_values(
            keys, ascending=[k == "sigma_mwe_per_year" for k in keys]
        ).drop_duplicates("RGIId")
    return df.sort_values(["RGIId", "FROM_DATE"]).reset_index(drop=True)


def fischer11_folder():
    """Directory holding the downloaded inventories, always the same place so that a
    later run finds what an earlier one downloaded."""
    return os.path.join(data_path, "Fischer11")


def gi_folder(epoch: int = FIRST_EPOCH, download: bool = True):
    """Folder holding the shapefiles of one inventory, one per mountain range,
    downloading and extracting the archive if necessary."""
    assert (
        epoch in GI_RELEASES
    ), f"No Austrian inventory of epoch {epoch}, only {sorted(GI_RELEASES)}."
    archive, url = GI_RELEASES[epoch]
    folder = os.path.join(fischer11_folder(), archive)
    if not os.path.isdir(folder) and download:
        zpath = os.path.join(fischer11_folder(), f"{archive}.zip")
        if not os.path.exists(zpath):
            os.makedirs(fischer11_folder(), exist_ok=True)
            print(f"downloading {url}")
            # Through a temporary name, so that an interrupted download is not
            # mistaken for the archive on the next run
            urllib.request.urlretrieve(url, zpath + ".part")
            os.replace(zpath + ".part", zpath)
        zipfile.ZipFile(zpath).extractall(folder + ".part")
        os.replace(folder + ".part", folder)
    return folder


def load_gi_outlines(epoch: int = FIRST_EPOCH, glacier_ids_to_keep=None):
    """Read one Austrian inventory, one entity per inventory number.

    The inventory comes as one shapefile per mountain range, each in its own
    Gauss-Krüger zone, so every file is reprojected before they are put together.
    Three quirks of the files are handled here:

    - an alternative delineation of a glacier is stored under its number times 1000
      and overlaps it (Hintereisferner without its dead ice, 2125000, next to 2125);
      those rows are dropped.
    - a glacier split into several polygons can be stored as several rows sharing
      its number (Griesbachjoch Kees in GI 2); they are dissolved into one entity.
      The same holds, wrongly, for 18009, which GI 2 gives to two glaciers of
      different ranges.
    - a few polygons are self-intersecting; they are repaired.

    The returned frame is in EPSG:4326 with the inventory number, as a string, in a
    column named "GI_ID", ready for `custom_outlines.build_custom_gdirs`, along with
    the glacier name, its mountain range, the survey year (GI 2 only) and its area in
    km².
    """
    folder = gi_folder(epoch)
    shapefiles = sorted(f for f in os.listdir(folder) if f.endswith(".shp"))
    assert shapefiles, f"{folder} holds no shapefile."
    frames = [
        gpd.read_file(os.path.join(folder, f))
        .to_crs(AUSTRIA_CRS)
        .assign(range=os.path.splitext(f)[0])
        for f in shapefiles
    ]
    gi = pd.concat(frames, ignore_index=True)
    numbers = set(gi.nr)
    variant = (gi.nr % 1000 == 0) & (gi.nr // 1000).isin(numbers)
    gi = gi[~variant].copy()
    gi["geometry"] = gi.geometry.buffer(0)
    if "Year" not in gi:
        gi["Year"] = float(epoch)
    gi["GI_ID"] = gi.nr.astype(str)
    gi = (
        gi.dissolve(
            "GI_ID",
            aggfunc={"Gletschern": "first", "range": "first", "Year": "first"},
        )
        .reset_index()
        .rename(columns={"Gletschern": "name", "Year": "survey_year"})
    )
    gi["area_gi"] = gi.area / 1e6

    if epoch != FIRST_EPOCH:
        years = gi.set_index("GI_ID").survey_year
        for glacier_id, (name, gi2_year) in FISCHER11_GLACIERS.items():
            assert years.get(glacier_id) == gi2_year, (
                f"GI {epoch} dates {name} ({glacier_id}) to {years.get(glacier_id)}, "
                f"not {gi2_year} as FISCHER11_GLACIERS says."
            )

    if glacier_ids_to_keep is not None:
        missing = set(glacier_ids_to_keep).difference(gi.GI_ID)
        assert not missing, f"The GI {epoch} inventory has no entity {sorted(missing)}."
        gi = gi[gi.GI_ID.isin(glacier_ids_to_keep)]
    return gi[["GI_ID", "name", "range", "survey_year", "area_gi", "geometry"]].to_crs(
        "EPSG:4326"
    )


def fischer11_outline_spec(
    epoch: int = FIRST_EPOCH, dem_source: str = "SRTM", **overrides
):
    """Description of the Austrian inventory outlines for the generic custom-outline
    machinery.

    The DEM defaults to SRTM, as for GLAMOS: its February 2000 acquisition is the same
    epoch as the NASADEM shipped with the RGI level-3 directories, so a calibration
    run on the inventory geometry and a reconstruction run on the RGI one see the same
    ice surface and differ only in the outline.

    Both epochs share the name "Fischer11", so the grids of one run are built on a
    single inventory; `custom_outlines.assert_spec_matches` refuses to mix them. Pass
    another `name` to keep the grids of both side by side.
    """
    spec = dict(
        name="Fischer11",
        id_column="GI_ID",
        o1_region=f"{AUSTRIA_REGION_ID:02d}",
        o2_region=AUSTRIA_SUBREGION,
        src_date=f"{epoch}-01-01 00:00:00",
        bgndate=f"{epoch}0101",
        dem_source=dem_source,
    )
    spec.update(overrides)
    return CustomOutlineSpec(**spec)


def table_RGI62_to_Fischer11(
    epoch: int = FIRST_EPOCH,
    region_id=AUSTRIA_REGION_ID,
    min_frac_rgi: float = 0.5,
):
    """Crosswalk from RGI 6.2 ids to the entities of one Austrian inventory.

    The match is made by area overlap, see `custom_outlines.match_rgi62_by_overlap`.
    The inventory numbers are the same in every inventory, so the crosswalk of one
    epoch is valid for the other wherever no glacier split in between.

    Returns a dataframe with columns RGIId, custom_id (the inventory number),
    area_rgi, area_gi, frac_rgi, frac_gi, `custom_id` matching the column name
    `table_RGI62_to_GLAMOS` uses, so `GeoDataLoader` reads both the same way.
    """
    if not isinstance(region_id, str):
        region_id = f"{region_id:02d}"
    save_path = os.path.abspath(
        os.path.join(
            data_path,
            "grids",
            "Fischer11",
            f"RGI62_to_Fischer11_gi{epoch}_{region_id}.csv",
        )
    )
    p = Product(save_path)
    if p.is_up_to_date():
        return pd.read_csv(save_path, dtype={"custom_id": str})

    matches = match_rgi62_by_overlap(
        load_gi_outlines(epoch=epoch).to_crs(AUSTRIA_CRS),
        "GI_ID",
        gpd.read_file(get_region_shape_file(region_id)),
        min_frac_rgi=min_frac_rgi,
    ).rename(columns={"area_custom": "area_gi", "frac_custom": "frac_gi"})

    matches.to_csv(save_path, index=False)
    p.gen_chk()
    return matches
