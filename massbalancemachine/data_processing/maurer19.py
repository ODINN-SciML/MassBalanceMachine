"""
Geodetic mass balance of Himalayan glaciers over 1975-2000 from Maurer et al. (2019),
gridded on the 1975 outlines of the authors.

Maurer, J. M., Schaefer, J. M., Rupper, S. and Corley, A.: Acceleration of ice loss
across the Himalayas over the past 40 years, Sci. Adv., 5, eaav7266,
doi:10.1126/sciadv.aav7266, 2019.

The paper gives the geodetic balance of about 650 glaciers larger than 3 km², from
Spiti Lahaul to Bhutan (75°E to 93°E), over 1975-2000 and 2000-2016. The first period
is the target of this source. Its DEMs are KH-9 Hexagon ones, acquired between 1973
and 1976 for the most part and in winter, differenced with the ASTER trend of
2000-2016 sampled at the start of 2000 (Materials and Methods, "Trend fitting"). The
second period, 1040 glaciers including all of the first, is a linear trend fitted
through the ASTER DEMs of 2000 to 2017 and is read with `period="2000-2016"`, for
evaluation.

**Data.** Both products are archived at NSIDC by the authors and need a NASA
Earthdata login, read from `~/.netrc` by `earthaccess`:

    HMA_GlacierAvg_dH v1   doi:10.5067/93BANQZIG1KD   one point per glacier and per
                           period with its geodetic balance and uncertainty, read by
                           `load_maurer19_table`
    HMA_Glacier_dH v1      doi:10.5067/GGGSQ06ZR0R8   one netCDF per glacier and per
                           period with the thickness change grid, the glacier masks
                           at both ends of the period and the years of the DEMs

They are downloaded to `.data/Maurer19`, see `download_maurer19_avg` and
`download_maurer19_gridded`.

**Geometry.** The authors edited the RGI 5.0 outlines to the glacier extent of 1975,
2000 and 2016. The 1975 and 2000 outlines are published as the `glacierStartMask` and
`glacierEndMask` grids of the 1975-2000 netCDF, 30 m in EPSG:4326, and the grids of
this source are built on the 1975 one, polygonized by `load_maurer19_outlines`. The
2000 outline is the one the 2000-2016 netCDF starts from too, rasterized on another
grid (areas within 0.2 %), and serves to evaluate that period
(`load_maurer19_outlines(epoch=2000)`). The paper divides the volume change by the
mean of the 1975 and 2000 areas, excluding slopes above 45°; that mean area is
recovered as the volume change over the mean thickness change
(`area_mean_km2` of `maurer19_periods`), to compare with the area of the 1975
outline (`area_1975_km2` of `load_maurer19_outlines`). No DEM of 1975 is published,
so the outlines are paired with SRTM, the February 2000 surface every DEM of the
paper is co-registered to, see `maurer19_outline_spec`.

Glaciers are identified by their **RGI 5.0 id** ("RGI50-15.02201"), which the
products carry. As for GLAMOS, the gridded products of this source therefore carry
`RGIId == "RGI50-15.02201"`, and the crosswalk built by `table_RGI62_to_Maurer19`
matches the RGI 6.2 outlines to the 1975 ones by overlap. The glaciers span three RGI
regions, 15 (South Asia East, 447 glaciers), 14 (South Asia West, 162) and 13
(Central Asia, 40, on the Tibetan side of the range), so, unlike the other custom
outlines, the region is set glacier by glacier.

The paper converts volume changes to water equivalent with a density of
850 kg m-3, the one of this workflow, so no conversion is needed. The uncertainty of
a balance is the one of the product, which already holds a 10 % area uncertainty and
a density uncertainty of 60 kg m-3.

**Lake-terminating glaciers.** About a seventh of the glaciers ended in a proglacial
lake, where they also lose ice by calving and thermal undercutting, which a
temperature-index model does not represent. They are identified with the inventory of
lake-terminating glaciers of High Mountain Asia of Luo and Liu (2025),
doi:10.5281/zenodo.17369580, and can be left out of the target with
`geodetic_target_Maurer19(exclude_lake_terminating=True)`, see
`lake_terminating_maurer19`.

A 1975-2000 period runs from the first of the month of the earliest Hexagon DEM of
the glacier to `END_DATE`, the 1st of January 2000, see `hexagon_start_date`. A
2000-2016 period runs over the years of its ASTER DEMs, from the 1st of January 2000
to the 1st of January 2017 for every glacier, see `aster_period_dates`.
"""

import glob
import os
import re
import shutil
import urllib.request

import earthaccess
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import shapely
import xarray as xr
from oggm.utils import get_rgi_dir
from rasterio.transform import from_origin

from data_processing.custom_outlines import CustomOutlineSpec, match_rgi62_by_overlap
from data_processing.product_utils import get_data_path
from data_processing.Product import Product
from data_processing.glacier_utils import get_region_shape_file
from data_processing.utils.periods import select_dem_periods

# RGI first-order regions the glaciers of the paper lie in, according to their RGI 5.0
# ids: South Asia East and West, and Central Asia for the glaciers on the Tibetan side
MAURER19_REGION_IDS = (13, 14, 15)

# NSIDC products, the period of the target and the later one used for evaluation
AVG_SHORT_NAME = "HMA_GlacierAvg_dH"
GRIDDED_SHORT_NAME = "HMA_Glacier_dH"
NSIDC_VERSION = "1"
PERIOD = "1975-2000"
LATE_PERIOD = "2000-2016"
PERIODS = (PERIOD, LATE_PERIOD)

# Year of the outlines of the target, for OGGM, and the mask of the 1975-2000 netCDF
# holding the outline of each epoch
OUTLINES_YEAR = 1975
OUTLINE_MASKS = {1975: "glacierStartMask", 2000: "glacierEndMask"}
# The ASTER trend of 2000-2016 is sampled at the start of 2000
END_DATE = pd.Timestamp("2000-01-01")
# Hexagon acquisitions end in 1980; any later year of `demYears` is an ASTER one
LAST_HEXAGON_YEAR = 1980

# Inventory of lake-terminating glaciers of High Mountain Asia, 1990 and 2022 (Luo and
# Liu, 2025, CC-BY 4.0). A glacier of its 1990 layer is "Type 1" when it still ends in
# its lake in 2022, "Type 3" when it has lost contact with it by then, and "Type 2"
# when it only reaches a lake after 1990.
LTG_DOI = "10.5281/zenodo.17369580"
LTG_URL = "https://zenodo.org/api/records/17369580/files/HMA_LTG.gpkg/content"
LTG_GLACIER_LAYER = "glacier_1990"
# The glaciers ending in a lake over the 1975-2000 period, as far as 1990 tells, and
# all the types of the inventory, those reaching a lake after 1990 included: the
# default of `geodetic_target_Maurer19`, since the calibrated glaciers are also
# evaluated over 2000-2016
LAKE_TERMINATING_1990 = ("Type 1", "Type 3")
LAKE_TERMINATING_ALL = ("Type 1", "Type 2", "Type 3")

# Density of the paper, the one of this workflow: no conversion is needed
ICE_DENSITY = 850.0

# Equal-area projection covering the Himalaya, in which the areas are measured: the
# glaciers span the UTM zones 43 to 46
HIMALAYA_CRS = "ESRI:102025"  # Asia North Albers Equal Area Conic

# Columns of the averaged product (user guide of HMA_GlacierAvg_dH, Table 1)
AVG_COLUMNS = [
    "id",
    "pctDeb",
    "VolChj",
    "VolChjSig",
    "meanElevCh",
    "meanElev_1",
    "geoMassBal",
    "geoMassB_1",
    "pctCov",
    "demYears",
]
# Spelled differently by the shapefile of one period: "percentCov" in 2000-2016
AVG_COLUMN_ALIASES = {"percentcov": "pctCov"}


def maurer19_folder():
    """Directory holding the downloaded products, always the same place so that a
    later run finds what an earlier one downloaded."""
    return os.path.join(get_data_path(), "Maurer19")


def _earthdata_login():
    auth = earthaccess.login(strategy="netrc")
    if auth is None or not auth.authenticated:
        raise RuntimeError(
            "The Maurer19 products are served by NSIDC behind a NASA Earthdata login. "
            "Create an account at https://urs.earthdata.nasa.gov and write the line\n"
            "    machine urs.earthdata.nasa.gov login <user> password <password>\n"
            "to ~/.netrc (chmod 600)."
        )


def _download_links(links, folder):
    """Download `links` to `folder`, going through a temporary folder so that an
    interrupted download is not mistaken for a file on the next run."""
    part = folder + ".part"
    os.makedirs(part, exist_ok=True)
    os.makedirs(folder, exist_ok=True)
    earthaccess.download(links, local_path=part)
    for link in links:
        name = os.path.basename(link)
        assert os.path.exists(
            os.path.join(part, name)
        ), f"The download of {link} did not produce {name}."
        os.replace(os.path.join(part, name), os.path.join(folder, name))
    shutil.rmtree(part)


def maurer19_avg_folder():
    return os.path.join(maurer19_folder(), AVG_SHORT_NAME)


def maurer19_gridded_folder():
    return os.path.join(maurer19_folder(), GRIDDED_SHORT_NAME)


def download_maurer19_avg(period: str = PERIOD):
    """Path to the shapefile of the averaged product for `period`, one of `PERIODS`,
    downloading the granule of that period if necessary."""
    assert period in PERIODS, f"Maurer19 has no period {period}, only {PERIODS}."
    folder = maurer19_avg_folder()
    pattern = os.path.join(folder, f"*{period}*.shp")
    if not glob.glob(pattern):
        _earthdata_login()
        granules = earthaccess.search_data(
            short_name=AVG_SHORT_NAME, version=NSIDC_VERSION
        )
        links = [
            link
            for g in granules
            for link in g.data_links()
            if period in os.path.basename(link)
        ]
        assert links, f"NSIDC holds no {AVG_SHORT_NAME} file of {period}."
        _download_links(links, folder)
    shapefiles = glob.glob(pattern)
    assert (
        len(shapefiles) == 1
    ), f"Expected one shapefile of {period}, not {shapefiles}."
    return shapefiles[0]


def gridded_file_name(glacier_id: str):
    """Name of the netCDF of the gridded product for `glacier_id`, "RGI50-15.02201"
    being stored as "HMA_Glacier_dH_1975-2000_RGI50_15_02201.nc"."""
    return f"{GRIDDED_SHORT_NAME}_{PERIOD}_{re.sub(r'[-.]', '_', glacier_id)}.nc"


def gridded_catalog():
    """{file name: download link} of every netCDF of the gridded product for `PERIOD`,
    which holds the outlines of both 1975 and 2000.

    The product has no 1975-2000 file for a few glaciers of the averaged one (4 of
    649), which have no 1975 outline, see `geodetic_target_Maurer19`. The list is
    read from NSIDC once and kept in `maurer19_gridded_folder()`.
    """
    path = os.path.join(maurer19_gridded_folder(), f"catalog_{PERIOD}.csv")
    if not os.path.exists(path):
        _earthdata_login()
        granules = earthaccess.search_data(
            short_name=GRIDDED_SHORT_NAME,
            version=NSIDC_VERSION,
            granule_name=f"{GRIDDED_SHORT_NAME}_{PERIOD}_*",
            count=-1,
        )
        links = sorted(
            {
                link
                for g in granules
                for link in g.data_links()
                if link.endswith(".nc") and f"_{PERIOD}_" in os.path.basename(link)
            }
        )
        assert links, f"NSIDC holds no {GRIDDED_SHORT_NAME} netCDF of {PERIOD}."
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Through a temporary name, so that an interrupted write is not mistaken for
        # the list on the next run
        pd.DataFrame(
            {"name": [os.path.basename(link) for link in links], "link": links}
        ).to_csv(path + ".part", index=False)
        os.replace(path + ".part", path)
    catalog = pd.read_csv(path)
    return dict(zip(catalog.name, catalog.link))


def has_1975_outline(glacier_ids):
    """Whether the gridded product holds the netCDF, and so the 1975 outline, of every
    glacier of `glacier_ids`, as a boolean array."""
    catalog = gridded_catalog()
    return np.array([gridded_file_name(g) in catalog for g in glacier_ids], dtype=bool)


def download_maurer19_gridded(glacier_ids):
    """Paths to the netCDF of the gridded product of every glacier of `glacier_ids`,
    downloading the missing ones. Only the netCDF of `PERIOD` is fetched, not the
    GeoTIFF and the metadata of the same granule."""
    folder = maurer19_gridded_folder()
    paths = {g: os.path.join(folder, gridded_file_name(g)) for g in glacier_ids}
    missing = {
        os.path.basename(p): g for g, p in paths.items() if not os.path.exists(p)
    }
    if missing:
        catalog = gridded_catalog()
        absent = sorted(g for name, g in missing.items() if name not in catalog)
        assert not absent, (
            f"NSIDC holds no {GRIDDED_SHORT_NAME} netCDF of {PERIOD}, and so no 1975 "
            f"outline, for {absent}."
        )
        _earthdata_login()
        print(f"Maurer19: downloading {len(missing)} netCDF of {GRIDDED_SHORT_NAME}")
        _download_links([catalog[name] for name in sorted(missing)], folder)
    return paths


def parse_dem_years(dem_years):
    """The years of `demYears`, a list of years in text, as floats. The separators
    of the text do not matter."""
    return [float(y) for y in re.findall(r"\d{4}(?:\.\d+)?", str(dem_years))]


def hexagon_start_date(years):
    """Start of a period whose DEMs were acquired in `years`: the first of the month
    of the earliest Hexagon DEM.

    A year with a fraction is a date and is rounded to the nearest first of a month;
    a whole year is taken as the 1st of January of that year, which is at most a few
    months off since the Hexagon images were taken in winter.
    """
    hexagon = [y for y in years if y < LAST_HEXAGON_YEAR + 1]
    assert hexagon, f"No Hexagon year among {years}."
    first = min(hexagon)
    year = int(np.floor(first))
    month = int(np.round((first - year) * 12))
    return pd.Timestamp(year=year, month=1, day=1) + pd.DateOffset(months=month)


def aster_period_dates(years):
    """Start and end of a 2000-2016 period whose ASTER DEMs span `years`: the first of
    the month of the first and of the last year, the 1st of January for whole years
    ([2000, 2017] for every glacier of the product)."""
    assert len(years) >= 2, f"A trend needs two DEM years, not {years}."

    def first_of_month(y):
        year = int(np.floor(y))
        return pd.Timestamp(year=year, month=1, day=1) + pd.DateOffset(
            months=int(np.round((y - year) * 12))
        )

    return first_of_month(min(years)), first_of_month(max(years))


def load_maurer19_table(period: str = PERIOD):
    """The averaged product of `period`, one row per glacier.

    Columns: RGIId (the RGI 5.0 id), mwe_per_year (`geoMassBal`, m w.e. per year),
    sigma_mwe_per_year (`geoMassB_1`), pctCov (percentage of the glacier with data),
    pctDeb (percentage covered in debris), dem_years (the years of `demYears`) and
    area_mean_km2, the mean of the areas at both ends of the period the balance
    refers to (1975 and 2000, or 2000 and 2016). That
    area is the volume change over the mean thickness change, which the product
    rounds to 0.01 m per year: it is 2 % off for a thinning of 0.25 m per year, and
    NaN where the thinning rounds to zero.

    Rows without a balance are left out: the 2000-2016 shapefile holds one, every
    field the text "'NaN'".
    """
    avg = gpd.read_file(download_maurer19_avg(period))
    balance = pd.to_numeric(avg.geoMassBal.str.strip("'\""), errors="coerce")
    avg = avg[balance.notna()]
    # The shapefile spells some columns differently from the user guide ("volChj")
    canonical = {c.lower(): c for c in AVG_COLUMNS} | AVG_COLUMN_ALIASES
    avg = avg.rename(columns=lambda c: canonical.get(c.lower(), c))
    missing = set(AVG_COLUMNS).difference(avg.columns)
    assert (
        not missing
    ), f"The {AVG_SHORT_NAME} shapefile has no column {sorted(missing)}."
    assert avg.id.is_unique, f"The {AVG_SHORT_NAME} shapefile repeats glaciers."
    # Every value is stored as text
    if "geo_mb_num" in avg:
        assert np.allclose(
            avg.geoMassBal.astype(float), avg.geo_mb_num.astype(float)
        ), "geo_mb_num is documented as a copy of geoMassBal."
    df = pd.DataFrame(
        {
            # Stored as text between quotes: "'RGI50-13.04767'"
            "RGIId": avg.id.str.strip().str.strip("'\""),
            "mwe_per_year": avg.geoMassBal.astype(float),
            "sigma_mwe_per_year": avg.geoMassB_1.astype(float),
            "pctCov": avg.pctCov.astype(float),
            "pctDeb": avg.pctDeb.astype(float),
            "dem_years": avg.demYears.map(parse_dem_years),
            # m3 per year over m per year. The thickness change is rounded to
            # 0.01 m per year, so the area is only known to 0.005 / |meanElevCh|,
            # and not at all where that rounds to zero.
            "area_mean_km2": avg.VolChj.astype(float)
            / avg.meanElevCh.astype(float).replace(0, np.nan)
            / 1e6,
        }
    )
    assert df.RGIId.str.match(
        r"^RGI50-\d{2}\.\d{5}$"
    ).all(), "Every glacier of the averaged product should carry an RGI 5.0 id."
    return df.sort_values("RGIId").reset_index(drop=True)


def maurer19_periods(table=None, period: str = PERIOD):
    """The period of every glacier with its dates, rate and uncertainty, one row per
    glacier.

    Columns: RGIId (the RGI 5.0 id), name (the same id, the product names no glacier),
    y0, y1, n_years, FROM_DATE, TO_DATE, mwe_per_year, cumulative_mwe,
    sigma_mwe_per_year, area_mean_km2, pctCov and pctDeb. `n_years` counts the
    months of the period, over 12, as `gridded_utils.geodetic_window_weights` does.

    Args:
        table: the averaged product, read by `load_maurer19_table` when not given.
        period: 1975-2000 (`hexagon_start_date` to `END_DATE`), or 2000-2016
            (`aster_period_dates`).
    """
    df = load_maurer19_table(period) if table is None else table.copy()
    if period == PERIOD:
        df["FROM_DATE"] = df.dem_years.map(hexagon_start_date)
        df["TO_DATE"] = END_DATE
    else:
        dates = df.dem_years.map(aster_period_dates)
        df["FROM_DATE"] = pd.to_datetime(dates.str[0])
        df["TO_DATE"] = pd.to_datetime(dates.str[1])
    months = (df.TO_DATE.dt.year - df.FROM_DATE.dt.year) * 12 + (
        df.TO_DATE.dt.month - df.FROM_DATE.dt.month
    )
    df["n_years"] = months / 12
    df["y0"] = df.FROM_DATE.dt.year
    df["y1"] = df.TO_DATE.dt.year
    df["cumulative_mwe"] = df.mwe_per_year * df.n_years
    df["name"] = df.RGIId
    return df[
        [
            "RGIId",
            "name",
            "y0",
            "y1",
            "n_years",
            "FROM_DATE",
            "TO_DATE",
            "mwe_per_year",
            "cumulative_mwe",
            "sigma_mwe_per_year",
            "area_mean_km2",
            "pctCov",
            "pctDeb",
        ]
    ]


def geodetic_target_Maurer19(
    max_year: int = None,
    min_year: int = None,
    min_period_years: int = 1,
    glacier_ids_to_keep=None,
    allowed_years=None,
    max_outside_years: int = 0,
    min_coverage: float = None,
    require_1975_outline: bool = True,
    table=None,
    verbose: bool = True,
    period: str = PERIOD,
    exclude_lake_terminating: bool = False,
    lake_terminating_types=LAKE_TERMINATING_ALL,
):
    """Geodetic targets of Maurer et al. (2019) over 1975-2000, one row per glacier,
    or over 2000-2016 with `period="2000-2016"`, for evaluation.

    Returns a dataframe shaped like the one `data_processing.glamos.geodetic_target_GLAMOS`
    returns, so that both drive the same code path in `GeoDataLoader`:

        RGIId                 the RGI 5.0 id, e.g. "RGI50-15.02201"
        FROM_DATE, TO_DATE    first of the month of the earliest Hexagon DEM, and
                              the 1st of January 2000
        mwe_per_year          the geodetic balance of the product, m w.e. per year
        cumulative_mwe        that rate times the length of the period
        sigma_mwe_per_year    its uncertainty, as given by the product

    Every glacier has a single period, so the periods are selected by
    `utils.periods.select_dem_periods`, which documents the other arguments, only to
    filter them.

    Args:
        min_coverage: keep only the glaciers with data over at least this percentage
            of their area (`pctCov`). The rest of a glacier is filled with the mean
            thickness change of the glaciers within 50 km at the same elevation.
        require_1975_outline: leave out the glaciers the gridded product holds no
            1975 outline for (4 of 649), which cannot be gridded on the geometry of
            their period, see `has_1975_outline`.
        table: the averaged product, read by `load_maurer19_table` when not given.
        verbose: report the glaciers left out for lack of an outline.
        period: one of `PERIODS`. The outlines of both epochs come from the
            1975-2000 netCDF, so `require_1975_outline` applies to both.
        exclude_lake_terminating: leave out the glaciers ending in a proglacial lake,
            see `lake_terminating_maurer19`.
        lake_terminating_types: the types of the inventory left out, by default all
            of them (`LAKE_TERMINATING_ALL`), the glaciers reaching a lake after 1990
            ("Type 2") included. `LAKE_TERMINATING_1990` only leaves out those in
            contact with their lake in 1990.
    """
    df = maurer19_periods(table, period)
    if min_coverage is not None:
        df = df[df.pctCov >= min_coverage]
    if glacier_ids_to_keep is not None:
        # Before the outline check, so that it only reports the glaciers asked for
        df = df[df.RGIId.isin([str(g) for g in glacier_ids_to_keep])]
    if require_1975_outline:
        has_outline = has_1975_outline(df.RGIId)
        if verbose and not has_outline.all():
            print(
                f"Maurer19: {(~has_outline).sum()} glacier(s) without a {PERIOD} netCDF "
                f"in {GRIDDED_SHORT_NAME}, and so without outlines, are left out: "
                f"{sorted(df.RGIId[~has_outline])}"
            )
        df = df[has_outline]
    if exclude_lake_terminating:
        flagged = lake_terminating_maurer19()
        flagged = set(flagged.RGIId[flagged.ltg_type.isin(lake_terminating_types)])
        lake = df.RGIId.isin(flagged)
        if verbose:
            print(
                f"Maurer19: {lake.sum()} lake-terminating glacier(s) "
                f"({', '.join(lake_terminating_types)}) are left out."
            )
        df = df[~lake]
    return select_dem_periods(
        df,
        max_year=max_year,
        min_year=min_year,
        min_period_years=min_period_years,
        glacier_ids_to_keep=glacier_ids_to_keep,
        allowed_years=allowed_years,
        max_outside_years=max_outside_years,
    )


def _coordinate(ds, names):
    for name in names:
        if name in ds.variables:
            return np.asarray(ds[name].values, dtype=float).ravel()
    raise KeyError(f"The netCDF has none of the coordinates {names}.")


def mask_to_polygon(path: str, variable: str = "glacierStartMask"):
    """The outline drawn by the binary grid `variable` of a netCDF of the gridded
    product, as one (multi)polygon in EPSG:4326.

    The grid is regular in longitude and latitude, and its cells are polygonized with
    the transform built from the two coordinate vectors.
    """
    with xr.open_dataset(path) as ds:
        lat = _coordinate(ds, ["latitude", "lat"])
        lon = _coordinate(ds, ["longitude", "lon"])
        mask = np.asarray(ds[variable].values)
    mask = np.squeeze(mask)
    if mask.shape == (len(lon), len(lat)) and len(lon) != len(lat):
        mask = mask.T
    assert mask.shape == (
        len(lat),
        len(lon),
    ), f"{variable} of {path} has shape {mask.shape}, not ({len(lat)}, {len(lon)})."
    mask = np.nan_to_num(mask) > 0
    # Rows from north to south, columns from west to east
    if lat[0] < lat[-1]:
        mask, lat = mask[::-1], lat[::-1]
    if lon[0] > lon[-1]:
        mask, lon = mask[:, ::-1], lon[::-1]
    dlon = np.median(np.diff(lon))
    dlat = -np.median(np.diff(lat))
    assert np.allclose(np.diff(lon), dlon, rtol=1e-3) and np.allclose(
        -np.diff(lat), dlat, rtol=1e-3
    ), f"The grid of {path} is not regular."
    transform = from_origin(lon[0] - dlon / 2, lat[0] + dlat / 2, dlon, dlat)
    polygons = [
        shapely.geometry.shape(geom)
        for geom, value in rasterio.features.shapes(
            mask.astype(np.uint8), mask=mask, transform=transform
        )
        if value == 1
    ]
    assert polygons, f"{variable} of {path} holds no glacier cell."
    return shapely.union_all(polygons)


def _second_order_regions(outlines: gpd.GeoDataFrame):
    """RGI 6.2 second-order region ("15-01") of every outline: among the regions of
    its first-order region, the one nearest to its representative point, which is
    the one containing it unless it lies just outside the coarse region polygons."""
    o2 = gpd.read_file(
        os.path.join(
            get_rgi_dir(version="62"), "00_rgi62_regions", "00_rgi62_O2Regions.shp"
        )
    ).to_crs(HIMALAYA_CRS)
    points = outlines.geometry.to_crs(HIMALAYA_CRS).representative_point()
    codes = []
    for o1, point in zip(outlines.o1_region, points):
        candidates = o2[o2.RGI_CODE.str.startswith(f"{o1}-")]
        assert len(candidates), f"RGI 6.2 has no second-order region in region {o1}."
        codes.append(candidates.RGI_CODE.iloc[candidates.distance(point).argmin()])
    return [c.split("-")[1] for c in codes]


def load_maurer19_outlines(glacier_ids_to_keep=None, epoch: int = OUTLINES_YEAR):
    """The outlines of `epoch`, 1975 or 2000, of the glaciers of the averaged product,
    one entity per RGI 5.0 id, polygonized from the `glacierStartMask` or the
    `glacierEndMask` of the 1975-2000 netCDF (`OUTLINE_MASKS`).

    The returned frame is in EPSG:4326 with the RGI 5.0 id in a column named
    "MAURER19_ID", ready for `custom_outlines.build_custom_gdirs`, along with the RGI
    first- and second-order regions of the glacier ("15", "01") and the area of the
    outline in km², `area_1975_km2` or `area_2000_km2`.

    Args:
        glacier_ids_to_keep: the glaciers to return. Only their netCDF are downloaded.
            None returns every glacier the gridded product holds an outline for.
        epoch: 1975, the outlines of the target, or 2000, those of the start of the
            2000-2016 period.
    """
    assert (
        epoch in OUTLINE_MASKS
    ), f"Maurer19 has no outline of {epoch}, only {sorted(OUTLINE_MASKS)}."
    glacier_ids = list(load_maurer19_table().RGIId)
    if glacier_ids_to_keep is None:
        glacier_ids = [
            g for g, ok in zip(glacier_ids, has_1975_outline(glacier_ids)) if ok
        ]
    else:
        missing = set(glacier_ids_to_keep).difference(glacier_ids)
        assert not missing, f"Maurer19 has no glacier {sorted(missing)}."
        glacier_ids = sorted(set(glacier_ids_to_keep))
    paths = download_maurer19_gridded(glacier_ids)
    outlines = gpd.GeoDataFrame(
        {"MAURER19_ID": glacier_ids},
        geometry=[mask_to_polygon(paths[g], OUTLINE_MASKS[epoch]) for g in glacier_ids],
        crs="EPSG:4326",
    )
    outlines["o1_region"] = outlines.MAURER19_ID.str.slice(6, 8)
    assert (
        outlines.o1_region.astype(int).isin(MAURER19_REGION_IDS).all()
    ), f"Maurer19 glaciers outside the regions {MAURER19_REGION_IDS}."
    outlines["o2_region"] = _second_order_regions(outlines)
    area = f"area_{epoch}_km2"
    outlines[area] = outlines.geometry.to_crs(HIMALAYA_CRS).area / 1e6
    return outlines[["MAURER19_ID", "o1_region", "o2_region", area, "geometry"]]


def maurer19_outline_spec(
    dem_source: str = "SRTM", epoch: int = OUTLINES_YEAR, **overrides
):
    """Description of the outlines of `epoch` for the generic custom-outline machinery.

    The DEM defaults to SRTM: its February 2000 acquisition is the surface every DEM
    of the paper is co-registered to, and the one Fischer11 and GLAMOS are gridded
    on. No DEM of 1975 is published with the paper.

    The regions are left to the `o1_region` and `o2_region` columns of the outlines,
    since the glaciers span three RGI regions. The outlines of 2000 are named
    "Maurer19_2000", so that their products do not mix with those of 1975.
    """
    assert (
        epoch in OUTLINE_MASKS
    ), f"Maurer19 has no outline of {epoch}, only {sorted(OUTLINE_MASKS)}."
    spec = dict(
        name="Maurer19" if epoch == OUTLINES_YEAR else f"Maurer19_{epoch}",
        id_column="MAURER19_ID",
        o1_region=None,
        o2_region=None,
        src_date=f"{epoch}-01-01 00:00:00",
        bgndate=f"{epoch}0101",
        dem_source=dem_source,
    )
    spec.update(overrides)
    return CustomOutlineSpec(**spec)


def ltg_file():
    """Path to the inventory of lake-terminating glaciers of Luo and Liu (2025),
    downloaded to `.data/Maurer19/HMA_LTG` on first use."""
    path = os.path.join(maurer19_folder(), "HMA_LTG", "HMA_LTG.gpkg")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading {LTG_URL}")
        # Through a temporary name, so that an interrupted download is not mistaken
        # for the inventory on the next run
        urllib.request.urlretrieve(LTG_URL, path + ".part")
        os.replace(path + ".part", path)
    return path


def lake_terminating_maurer19(min_frac_ltg: float = 0.5, min_frac_maurer: float = 0.25):
    """The glaciers of Maurer et al. (2019) that the inventory of Luo and Liu (2025)
    lists as lake-terminating, with their type.

    A glacier of the 1990 layer of the inventory is matched to the 1975 outline it
    overlaps most, if at least `min_frac_ltg` of it lies inside and it covers at least
    `min_frac_maurer` of that outline, so that a small tributary ending in a lake does
    not flag a large glacier. Of the 645 glaciers with a 1975 outline, 85 match a
    glacier ending in a lake in 1990 (65 of "Type 1", 20 of "Type 3") and 25 one that
    only reaches a lake after 1990 ("Type 2"). The inventory starts in 1990, while
    proglacial lakes were growing: some of these glaciers did not end in a lake yet in
    1975.

    The result is cached in `.data/Maurer19/HMA_LTG`.

    Returns a dataframe with one row per matched glacier: RGIId (the RGI 5.0 id),
    ltg_type ("Type 1", "Type 2" or "Type 3"), ltg_rgi_id (the RGI 7.0 glacier
    complex of the inventory), frac_ltg and frac_maurer, the overlap over the area of
    the inventory glacier and of the 1975 outline.
    """
    path = os.path.join(
        maurer19_folder(),
        "HMA_LTG",
        f"lake_terminating_maurer19_{min_frac_ltg}_{min_frac_maurer}.csv",
    )
    p = Product(path)
    if p.is_up_to_date():
        return pd.read_csv(path)

    outlines = load_maurer19_outlines().to_crs(HIMALAYA_CRS)
    outlines["area_maurer"] = outlines.area
    ltg = gpd.read_file(ltg_file(), layer=LTG_GLACIER_LAYER)
    ltg["geometry"] = shapely.force_2d(ltg.geometry.values)
    ltg = ltg.to_crs(HIMALAYA_CRS)
    ltg["area_ltg"] = ltg.area
    inter = gpd.overlay(
        outlines[["MAURER19_ID", "area_maurer", "geometry"]],
        ltg[["Type", "rgi_id", "area_ltg", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    inter["overlap"] = inter.area
    inter["frac_ltg"] = inter.overlap / inter.area_ltg
    inter["frac_maurer"] = inter.overlap / inter.area_maurer
    matches = (
        inter[(inter.frac_ltg >= min_frac_ltg) & (inter.frac_maurer >= min_frac_maurer)]
        .sort_values("overlap", ascending=False)
        .drop_duplicates("MAURER19_ID")
        .rename(
            columns={"MAURER19_ID": "RGIId", "Type": "ltg_type", "rgi_id": "ltg_rgi_id"}
        )
    )
    matches = pd.DataFrame(
        matches[["RGIId", "ltg_type", "ltg_rgi_id", "frac_ltg", "frac_maurer"]]
    ).sort_values("RGIId")
    matches.to_csv(path, index=False)
    p.gen_chk()
    return matches.reset_index(drop=True)


def table_RGI62_to_Maurer19(region_id=15, min_frac_rgi: float = 0.5):
    """Crosswalk from RGI 6.2 ids to the glaciers of Maurer et al. (2019).

    The match is made by area overlap with the 1975 outlines, see
    `custom_outlines.match_rgi62_by_overlap`, which needs the netCDF of every glacier.
    Only regions 13 to 15 can match.

    Returns a dataframe with columns RGIId, custom_id (the RGI 5.0 id), area_rgi,
    area_custom, frac_rgi, frac_custom, `custom_id` matching the column name
    `table_RGI62_to_GLAMOS` uses, so `GeoDataLoader` reads both the same way.
    """
    columns = [
        "RGIId",
        "custom_id",
        "area_rgi",
        "area_custom",
        "frac_rgi",
        "frac_custom",
    ]
    if int(region_id) not in MAURER19_REGION_IDS:
        return pd.DataFrame(columns=columns)
    region_id = f"{int(region_id):02d}"
    save_path = os.path.abspath(
        os.path.join(
            get_data_path(), "grids", "Maurer19", f"RGI62_to_Maurer19_{region_id}.csv"
        )
    )
    p = Product(save_path)
    if p.is_up_to_date():
        return pd.read_csv(save_path)

    outlines = load_maurer19_outlines()
    outlines = outlines[outlines.o1_region == region_id]
    if len(outlines) == 0:
        matches = pd.DataFrame(columns=columns)
    else:
        matches = match_rgi62_by_overlap(
            outlines.to_crs(HIMALAYA_CRS),
            "MAURER19_ID",
            gpd.read_file(get_region_shape_file(region_id)),
            min_frac_rgi=min_frac_rgi,
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    matches.to_csv(save_path, index=False)
    p.gen_chk()
    return matches
