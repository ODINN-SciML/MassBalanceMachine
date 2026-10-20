"""
Geodetic mass balance of Norwegian glaciers from the glacier database of NVE,
gridded on their RGI 6.2 outlines.

The table is an extract of the geodetic balances NVE holds, compiled by L. M. Andreassen
on 2026-10-15 (`RAW_FILE`). It is not public: the file is kept at one fixed place,
`.data/NVE/`, and is not downloaded. It gathers 286 periods on 185 glaciers
(Bre-ID of NVE) from about thirty reports and papers, from 1942 to 2020: the long
periods from the 1950s-60s maps to the laser scans of 2008-2013 of Andreassen et al.
(2020), and, before 2000, the photogrammetric surveys of the glaciers with a
mass-balance programme, of Hardangerjøkulen (1961-1995) and of Folgefonna
(1959-1997). Andreassen16 holds a subset of it, the 21 periods of Table 4 of
Andreassen et al. (2016), corrected to whole mass-balance years.

**Versions.** `generation_no` tells the current value of a period (0) from the older
ones it replaced (1), e.g. the 2016 reanalysis of Ålfotbreen 1988-1997 (+0.87 m w.e.
per year) and the value of 2004 (+0.50). Only generation 0 is kept. Overlapping
periods of one glacier remain, such as Engabreen 1968-2001 and 1969-2001, or the
1966-2008 periods spanning shorter ones, and are resolved when the target is selected
(`geodetic_target_NVE`).

**Values.** The rate is `mass_bal_avg_year`, m w.e. per year, which is given for
every period. The total `mass_bal` is only given for a third of them and carries the
wrong sign on Harbardsbreen 1966-1996, Langfjordjøkelen 2008-2018, Strupbreen
1953-2010 and Bondhusbrea 1997-2007, so it is not used: `cumulative_mwe` is the rate
times the number of years. Most periods before 2000 are not corrected for the
ablation between the survey dates and the end of the mass-balance year ("ikke
justert for ulike datoer"), so they are taken to cover the mass-balance years between
the surveys, from the 1st of October of `survey_year1` to the 1st of October of
`survey_year2`, as for Andreassen16. `survey_year1` follows the glaciological year
where it differs from the survey date (Engabreen 1969, mapped in 1968). One period
carries a wrong year, see `SURVEY_YEAR_FIXES`.

**Density.** The volume changes are converted with a density of 850, 860 or
900 kg m-3 (`density_conv_factor`), while this workflow uses 850. Rates and
uncertainties converted with a single bulk density are multiplied by 850 / that
density, as for Hagg12 and Rabatel16. Storsteinsfjellbreen 1960-1993 used 600 above
1350 m and 900 below (716 overall), which is not a bulk density, and four periods
give none: both are kept as published.

**Uncertainty.** Only 62 current periods report one (`mass_bal_avg_year_uncer`),
7 of them as 0. A period without a positive uncertainty is given `DEFAULT_SIGMA_MWE`
in m w.e. over the period, spread over its years: 2 m w.e. when a photogrammetric map
or a ground survey is involved, as the ones reported for periods of photogrammetric
maps before 2000 (0.5-3.2, median 1.8) and those of Table 4 of Andreassen et al.
(2016), about 2; 0.5 m w.e. between two laser scans, the most common of the reported
ones. The balances of Hardangerjøkulen 1961-1995 are rounded to the metre of water
equivalent over the period, 0.03 m w.e. per year.

**Several glaciers in one value.** Some balances were computed for a group of
glaciers and stored under one of their Bre-IDs, which the comment of the period
tells: the eight outlets of Langfjordjøkelen 1966-1994, Strupbreen and Koppangsbreen
1978-1985 and 1985-1998, the fifteen glaciers around Strupbreen 1953-2010, and
pairs or triples of outlets of Folgefonna 1959-2013. Vetlefjordbreen 1966-1993 is
the whole Jostefonn, which no comment says: its reference is "Volumendringer på
Jostefonn 1966-93", and its area of 1966, 12.6 km², is the sum of those the table
gives for the outlets of Jostefonn, not that of Vetlefjordbreen (2.4). `AGGREGATED_PERIODS` lists these
groups, period by period, since a Bre-ID can be a group on one period and itself on
another (Strupbreen 1952-2010). Every distinct group is an entity of its own, the
RGI outlines of its glaciers dissolved into one. A group with a glacier the table
gives no position for cannot be built and its period is left out: the fifteen
glaciers of Strupbreen 1953-2010. Ramnabergbreen's neighbour 2963 1961-1995
"probably also includes 2961" (no position either) and is kept as 2963.

**Geometry.** The table gives the area at both surveys but not the outlines. Each
glacier is first matched to one RGI 6.2 outline, from NVE's inventory of Landsat
scenes of 1999 to 2006, the one holding the position of the glacier (`utm_east`,
`utm_north`; the columns `latitude` and `logitude` are swapped). All 185 positions
fall in an RGI outline, a different one each.

The periods whose first survey is before `GAO_BEFORE_YEAR` (1985) are gridded on the
outlines the old maps were measured on, the Glacier Area Outline (GAO) 1947-1985 of
Winsvold et al. (2014), digitised from the first edition of the N50 maps
(`gao_file`, downloaded from NVA, CC BY-NC-SA 4.0). That file gives neither the
Bre-ID nor the ice divides: a polygon is a whole ice mass, Jostedalsbreen one of
543 km² holding 48 glaciers of the table. Each ice mass is split between the RGI 6.2
outlines it touches: the ice inside an RGI outline goes to it, and the ice beyond them,
where the glaciers have retreated since, to the nearest one (`gao_basins`, which needs
only public data). A glacier's GAO outline is the basin of its RGI outline. Four small
glaciers (2602, 2773, 2774, 2776) are on no N50 map, so their periods before 1985
are left out. The other periods are gridded on the RGI 6.2 outlines. The inventory of
a period is its `outlines` column ("GAO" or "RGI"), and its entity is gridded on
each inventory its periods use (`load_nve_outlines`).

The ice divides of the old maps do not always match those of the inventory: on
Hardangerjøkulen the 1961-1995 periods give the areas of 1995 of Austra
Leirebottsskåka (11.9 km²) and of 2973 (2.3) as 1.7 and 0.6 times those of their GAO
basins, while the ice cap as a whole agrees, and the 1961-2010 periods agree with
the basins within 5%. The ratio of the mean area of the surveys to the area of the
entity on the outlines of the period is kept as `area_ratio`, and the periods for
which it exceeds `max_area_ratio` either way are left out of the target by default.

**Calving.** Austdalsbreen (2478) calves into a regulated lake, see Andreassen16: its
periods are left out by default.

Glaciers are identified by "NVE-" and their Bre-ID ("NVE-2297"), and a group by the
Bre-IDs of its glaciers joined by "+" ("NVE-3113+3115+3117"). The crosswalk built by
`table_RGI62_to_NVE` is many RGI ids to one entity, and an RGI outline lies
in its own entity and in every group holding it.
"""

import functools
import json
import os
import urllib.request
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from data_processing.andreassen16 import NORWAY_REGION_ID, OUTLINES_YEAR
from data_processing.custom_outlines import CustomOutlineSpec
from data_processing.glacier_utils import get_region_shape_file
from data_processing.product_utils import get_data_path
from data_processing.utils.periods import best_period_chain, select_dem_periods

# The extract of the NVE glacier database, in `.data/NVE/`. The table header
# is on the third row of its only sheet.
RAW_FILE = "geodetic_mb_bredatabasen_NVE_2026-10-15_lma.xlsx"
HEADER_ROW = 2

# Prefix of the glacier ids of this source, followed by one Bre-ID or several joined
# by "+"
ID_PREFIX = "NVE-"

# Periods whose year does not match the survey date: (Bre-ID, survey_year1,
# survey_year2) to the corrected years. Bre-ID 152 was surveyed on 2010-10-02, as its
# neighbours 153 to 158 of the same paper, and the table counts 58 years.
SURVEY_YEAR_FIXES = {(152, 1952, 2008): (1952, 2010)}

# Periods computed for a group of glaciers: (Bre-ID the value is stored under,
# survey_year1, survey_year2) to the Bre-IDs of the group, from the comments of the
# table (and the reference and areas for Jostefonn, see the module docstring)
LANGFJORDJOKELEN = (49, 50, 51, 52, 53, 54, 55, 56)
STRUPBREEN_KOPPANGSBREEN = (200, 205)
STRUPBREEN_15 = tuple(range(195, 210))
JOSTEFONN = (2141, 2142, 2143, 2145, 2146, 2147, 2148, 2149, 2150, 2152)
GROUP_NAMES = {
    LANGFJORDJOKELEN: "Langfjordjøkelen",
    STRUPBREEN_KOPPANGSBREEN: "Strupbreen and Koppangsbreen",
    STRUPBREEN_15: "Strupbreen and neighbours",
    JOSTEFONN: "Jostefonn",
}
FOLGEFONNA_GROUPS = {
    3115: (3113, 3115, 3117),
    3120: (3120, 3122),
    3125: (3124, 3125),
    3133: (3130, 3133),
}
AGGREGATED_PERIODS = {
    (52, 1966, 1994): LANGFJORDJOKELEN,
    (200, 1953, 2010): STRUPBREEN_15,
    (200, 1978, 1985): STRUPBREEN_KOPPANGSBREEN,
    (200, 1985, 1998): STRUPBREEN_KOPPANGSBREEN,
    (2148, 1966, 1993): JOSTEFONN,
    **{
        (stored, y0, y1): members
        for stored, members in FOLGEFONNA_GROUPS.items()
        for y0, y1 in ((1959, 1997), (1997, 2007), (2007, 2013))
    },
}

# Glaciers losing mass by calving: Austdalsbreen ends in a hydropower lake
CALVING_GLACIERS = {2478}

# Density of this workflow, and the single bulk densities of the table, which are
# converted to it
ICE_DENSITY = 850.0
BULK_DENSITIES = (850.0, 860.0, 900.0)

# Uncertainty in m w.e. over a period without one, see the module docstring: between
# two laser scans, and otherwise
DEFAULT_SIGMA_MWE = {"laser": 0.5, "other": 2.0}
LASER_METHODS = {"l", "L"}

# Largest ratio, either way, between the mean area of the surveys and the area of the
# outlines of the entity, for the period to be a target by default
MAX_AREA_RATIO = 1.5

# Glacier Area Outline (GAO) 1947-1985 (Winsvold et al., 2014), the NVA record of the
# dataset and its file. NVA hands out a signed link to the file at `GAO_LINK_URL`.
GAO_RECORD = "0185ca7d0a25-5fa9f372-5b58-463b-baa8-cf1d0885b248"
GAO_FILE_ID = "d19d9af8-7757-45a2-a808-4ef43f353196"
GAO_LINK_URL = (
    f"https://api.nva.unit.no/publication/{GAO_RECORD}/filelink/{GAO_FILE_ID}"
)
GAO_ARCHIVE = "cryoclim_GAO_NO_1947_1985_UTM_33N"
GAO_CRS = "EPSG:25833"

# Periods whose first survey is before this year are gridded on the GAO outlines
GAO_BEFORE_YEAR = 1985

# Year the GAO outlines are dated to for OGGM, which only reads it as the date of the
# geometry: the first survey of most of the periods gridded on them
GAO_OUTLINES_YEAR = 1966

# Spacing in metres of the points along the RGI outlines whose Voronoi cells give the
# nearest basin of the GAO ice beyond them
GAO_DIVIDE_SPACING = 25

# The outline inventories a period can be gridded on, see the module docstring
OUTLINE_INVENTORIES = ("RGI", "GAO")


def nve_data_folder():
    """Directory holding the NVE table, always the same one under `.data/`."""
    return os.path.join(get_data_path(), "NVE")


def nve_file():
    """Path to the NVE table, which must have been copied there: it is not public."""
    path = os.path.join(nve_data_folder(), RAW_FILE)
    assert os.path.exists(path), (
        f"The NVE geodetic table is missing: copy {RAW_FILE}, provided by NVE, to "
        f"{nve_data_folder()}."
    )
    return path


def entity_id(members) -> str:
    """Glacier id of the entity made of the glaciers `members` (Bre-IDs)."""
    return ID_PREFIX + "+".join(str(m) for m in sorted(members))


def entity_members(glacier_id: str):
    """Bre-IDs of the glaciers of the entity `glacier_id`."""
    assert glacier_id.startswith(ID_PREFIX), f"{glacier_id} is no NVE id."
    return tuple(int(m) for m in glacier_id[len(ID_PREFIX) :].split("+"))


def read_nve_table():
    """The table as published, one row per period, with the swapped coordinate
    columns renamed (`lon`, `lat`) and the years of `SURVEY_YEAR_FIXES` corrected."""
    df = pd.read_excel(nve_file(), header=HEADER_ROW)
    df = df.rename(columns={"latitude": "lon", "logitude": "lat"})
    for (gid, y0, y1), (fy0, fy1) in SURVEY_YEAR_FIXES.items():
        rows = (
            (df.glacier_id == gid) & (df.survey_year1 == y0) & (df.survey_year2 == y1)
        )
        assert rows.sum() == 1, f"No single period {gid} {y0}-{y1} to correct."
        df.loc[rows, ["survey_year1", "survey_year2"]] = [fy0, fy1]
    return df


@functools.cache
def _rgi_of_breid():
    """{Bre-ID: RGI 6.2 id of the outline holding the position of the glacier}."""
    df = read_nve_table().drop_duplicates("glacier_id")
    points = gpd.GeoDataFrame(
        df[["glacier_id"]],
        geometry=gpd.points_from_xy(df.utm_east, df.utm_north),
        crs="EPSG:25833",
    )
    rgi = gpd.read_file(get_region_shape_file(f"{NORWAY_REGION_ID:02d}"))
    joined = gpd.sjoin(
        points.to_crs(rgi.crs), rgi[["RGIId", "geometry"]], predicate="within"
    )
    assert joined.glacier_id.is_unique, "A glacier lies in several RGI outlines."
    missing = set(df.glacier_id).difference(joined.glacier_id)
    assert not missing, f"Bre-IDs outside every RGI 6.2 outline: {sorted(missing)}."
    return dict(zip(joined.glacier_id, joined.RGIId))


def gao_file():
    """Path to the shapefile of the GAO outlines 1947-1985 in `.data/NVE/`, downloading
    and extracting the archive from NVA if necessary."""
    folder = os.path.join(nve_data_folder(), GAO_ARCHIVE)
    if not os.path.isdir(folder):
        zpath = folder + ".zip"
        if not os.path.exists(zpath):
            os.makedirs(nve_data_folder(), exist_ok=True)
            with urllib.request.urlopen(GAO_LINK_URL) as response:
                url = json.load(response)["id"]
            print(f"downloading {GAO_ARCHIVE}.zip from NVA")
            # Through a temporary name, so that an interrupted download is not
            # mistaken for the archive on the next run
            urllib.request.urlretrieve(url, zpath + ".part")
            os.replace(zpath + ".part", zpath)
        zipfile.ZipFile(zpath).extractall(folder + ".part")
        os.replace(folder + ".part", folder)
    return os.path.join(folder, f"{GAO_ARCHIVE}.shp")


def outline_inventory(y0: int) -> str:
    """Inventory of the outlines a period whose first survey is in `y0` is gridded on,
    one of `OUTLINE_INVENTORIES`."""
    return "GAO" if y0 < GAO_BEFORE_YEAR else "RGI"


def _polygonal(geometry):
    """The polygons of a geometry, which an overlay can leave with lines and points."""
    parts = [p for p in shapely.get_parts(geometry) if p.geom_type == "Polygon"]
    return shapely.union_all(parts)


def _nearest_outline(ice, outlines):
    """Split `ice` between `outlines` (a GeoSeries indexed by RGI id): the ice inside an
    outline goes to it, and the ice beyond every outline to the nearest one, through
    the Voronoi cells of points every `GAO_DIVIDE_SPACING` metres along the outlines,
    merged per outline. Returns {RGI id: its part}, without the empty ones."""
    if len(outlines) == 1:
        return {outlines.index[0]: _polygonal(ice)}
    points = outlines.boundary.segmentize(GAO_DIVIDE_SPACING).get_coordinates()
    points = points[~points.duplicated()]
    cells = shapely.voronoi_polygons(
        shapely.MultiPoint(points.to_numpy()), extend_to=ice.envelope
    )
    owner = gpd.sjoin(
        gpd.GeoDataFrame(geometry=list(shapely.get_parts(cells)), crs=GAO_CRS),
        gpd.GeoDataFrame(
            {"RGIId": points.index},
            geometry=gpd.points_from_xy(points.x, points.y),
            crs=GAO_CRS,
        ),
        predicate="contains",
    )
    nearest = owner.dissolve("RGIId").geometry
    beyond = ice.difference(outlines.union_all())
    parts = {}
    for rgi_id, outline in outlines.items():
        part = outline.intersection(ice)
        if rgi_id in nearest.index:
            part = part.union(nearest[rgi_id].intersection(beyond))
        part = _polygonal(part)
        if not part.is_empty:
            parts[rgi_id] = part
    return parts


def gao_basins(rgi_ids):
    """{RGI 6.2 id: the GAO ice of 1947-1985 in its basin}, in `GAO_CRS`, for the RGI
    outlines `rgi_ids` of region 08 touching GAO ice.

    Each ice mass of GAO is split between the RGI outlines it touches: the ice inside
    an outline goes to it, the ice beyond them to the nearest one. The basin of an
    outline is its share of every ice mass it touches. It only depends on the outline
    and its neighbours, not on the other ids asked for, and only on public data, so
    the GAO outline of an entity can be rebuilt from the RGI ids of its glaciers
    without the NVE table. The basins do not overlap.
    """
    rgi = gpd.read_file(get_region_shape_file(f"{NORWAY_REGION_ID:02d}"))
    rgi = rgi.set_index("RGIId").geometry.to_crs(GAO_CRS).make_valid()
    wanted = rgi[sorted(set(rgi_ids))]
    # The polygons of one ice mass split by the edge of a map sheet are joined
    gao = gpd.read_file(gao_file()).to_crs(GAO_CRS).geometry.make_valid()
    masses = gpd.GeoSeries(shapely.get_parts(_polygonal(gao.union_all())), crs=GAO_CRS)
    masses = masses[masses.intersects(wanted.union_all())]

    parts = {}
    for mass in masses:
        for rgi_id, part in _nearest_outline(mass, rgi[rgi.intersects(mass)]).items():
            parts.setdefault(rgi_id, []).append(part)
    return {r: shapely.union_all(parts[r]) for r in wanted.index if r in parts}


@functools.cache
def _gao_basins():
    """`gao_basins` of the RGI outlines of the glaciers of the table."""
    return gao_basins(_rgi_of_breid().values())


def _gao_outline(members):
    """GAO outline of the entity made of the glaciers `members` (Bre-IDs), the union
    of the basins of their RGI outlines, in `GAO_CRS`. None when one of them is on no
    N50 map."""
    rgi_of, basins = _rgi_of_breid(), _gao_basins()
    if not all(rgi_of[m] in basins for m in members):
        return None
    return shapely.union_all([basins[rgi_of[m]] for m in members])


def _default_sigma(method1, method2):
    laser = method1 in LASER_METHODS and method2 in LASER_METHODS
    return DEFAULT_SIGMA_MWE["laser" if laser else "other"]


def nve_periods():
    """Every current period of the table that can be built on its outlines, with its
    entity, dates, rate, uncertainty and areas, one row per period.

    Columns: RGIId (entity id), name, breid (the Bre-ID the value is stored under),
    members (the Bre-IDs of the entity), y0, y1, n_years, FROM_DATE and TO_DATE (the
    1st of October of y0 and y1), survey_from and survey_to (the survey dates),
    mwe_per_year, cumulative_mwe, sigma_mwe_per_year (all at 850 kg m-3),
    sigma_reported (whether the table gives it), density (of the table),
    area_y0_km2, area_y1_km2, area_mean_km2 (the mean of the ones given), outlines
    (the inventory the period is gridded on, "GAO" or "RGI"), area_rgi (of the RGI
    outlines of the entity), area_outlines (of the entity on the outlines of the
    period), area_ratio (area_mean_km2 / area_outlines) and reference.

    The periods of a group with a glacier of unknown position, and those before 1985
    of a glacier on no N50 map, are left out, see the module docstring.
    """
    raw = read_nve_table()
    raw = raw[raw.generation_no == 0]
    rgi_of = _rgi_of_breid()
    rgi_area = (
        gpd.read_file(get_region_shape_file(f"{NORWAY_REGION_ID:02d}"))
        .set_index("RGIId")
        .Area
    )

    rows = []
    for r in raw.itertuples():
        y0, y1 = int(r.survey_year1), int(r.survey_year2)
        members = AGGREGATED_PERIODS.get((r.glacier_id, y0, y1), (r.glacier_id,))
        if not set(members) <= set(rgi_of):
            continue
        area_rgi = sum(rgi_area[rgi_of[m]] for m in members)
        inventory = outline_inventory(y0)
        if inventory == "GAO":
            outline = _gao_outline(members)
            if outline is None:
                continue
            area_outlines = outline.area / 1e6
        else:
            area_outlines = area_rgi
        density = r.density_conv_factor
        factor = ICE_DENSITY / density if density in BULK_DENSITIES else 1.0
        n_years = y1 - y0
        reported = r.mass_bal_avg_year_uncer > 0
        sigma = (
            r.mass_bal_avg_year_uncer
            if reported
            else _default_sigma(r.survey_method_year1, r.survey_method_year2) / n_years
        )
        areas = [a for a in (r.area_survey_year1, r.area_survey_year2) if a > 0]
        rows.append(
            dict(
                RGIId=entity_id(members),
                name=GROUP_NAMES.get(
                    members,
                    r.glacier_name if isinstance(r.glacier_name, str) else None,
                ),
                breid=r.glacier_id,
                members=tuple(sorted(members)),
                y0=y0,
                y1=y1,
                n_years=n_years,
                FROM_DATE=pd.Timestamp(f"{y0}-10-01"),
                TO_DATE=pd.Timestamp(f"{y1}-10-01"),
                survey_from=pd.Timestamp(r.dt_survey_year1_date),
                survey_to=pd.Timestamp(r.dt_survey_year2_date),
                mwe_per_year=r.mass_bal_avg_year * factor,
                sigma_mwe_per_year=sigma * factor,
                sigma_reported=bool(reported),
                density=density,
                area_y0_km2=r.area_survey_year1,
                area_y1_km2=r.area_survey_year2,
                area_mean_km2=np.mean(areas) if areas else np.nan,
                outlines=inventory,
                area_rgi=area_rgi,
                area_outlines=area_outlines,
                reference=r.geod_reference,
            )
        )
    df = pd.DataFrame(rows)
    df["cumulative_mwe"] = df.mwe_per_year * df.n_years
    df["area_ratio"] = df.area_mean_km2 / df.area_outlines
    return df.sort_values(["RGIId", "FROM_DATE", "TO_DATE"]).reset_index(drop=True)


def geodetic_target_NVE(
    max_year: int = None,
    min_year: int = None,
    min_period_years: int = 1,
    glacier_ids_to_keep=None,
    allowed_years=None,
    max_outside_years: int = 0,
    multi_period: bool = True,
    tie_break: str = "sigma",
    max_area_ratio: float = MAX_AREA_RATIO,
    exclude_calving: bool = True,
    periods: dict = None,
):
    """Geodetic targets of the NVE table, one row per period between two surveys.

    Returns a dataframe shaped like the one `data_processing.glamos.geodetic_target_GLAMOS`
    returns, so that both drive the same code path in `GeoDataLoader`:

        RGIId                 the entity id, e.g. "NVE-2297" or "NVE-3113+3115+3117"
        FROM_DATE, TO_DATE    1st of October of the years of the two surveys
        mwe_per_year          the geodetic balance, m w.e. per year at 850 kg m-3
        cumulative_mwe        that rate times the number of years
        sigma_mwe_per_year    its uncertainty, one sigma

    along with the other columns of `nve_periods`.

    The periods are first selected by `utils.periods.select_dem_periods`, which
    documents the arguments. As for Belart20 the periods of an entity overlap
    (1966-2008 and 1994-2008), so with `multi_period` an entity keeps the best chain of
    non-overlapping periods, see `utils.periods.best_period_chain`, and without it the
    longest period, ties going to the lowest sigma.

    Args:
        max_area_ratio: periods whose mean surveyed area differs from the area of
            their entity on the outlines of the period by more than this factor,
            either way, are left out; a period without an area is kept. None keeps
            them all.
        exclude_calving: leave out the glaciers of `CALVING_GLACIERS`, whose geodetic
            balance includes a calving loss.
        periods: {entity id: [[y0, y1], ...]} names the periods to keep instead, by the
            years of their two surveys, as for Belart20: for instance the ones a
            train/validation split assigns to each side. `glacier_ids_to_keep` still
            applies; the other selection arguments must keep their defaults, and a
            named period is kept whatever its area ratio.
    """
    if periods is not None:
        assert (max_year, min_year, min_period_years, allowed_years) == (
            None,
            None,
            1,
            None,
        ), "periods names the periods: leave the other selection arguments out."
        df = nve_periods()
        wanted = {(g, int(y0), int(y1)) for g, ys in periods.items() for y0, y1 in ys}
        df = df[[(g, y0, y1) in wanted for g, y0, y1 in zip(df.RGIId, df.y0, df.y1)]]
        found = set(zip(df.RGIId, df.y0, df.y1))
        assert found == wanted, f"No NVE period {sorted(wanted - found)}."
        assert len(df) == len(wanted), "Two NVE periods share their survey years."
        if exclude_calving:
            calving = [
                g for g, m in zip(df.RGIId, df.members) if CALVING_GLACIERS & set(m)
            ]
            assert not calving, f"Periods named on calving glaciers: {calving}."
        if glacier_ids_to_keep is not None:
            df = df[df.RGIId.isin(glacier_ids_to_keep)]
        return df.sort_values(["RGIId", "FROM_DATE"]).reset_index(drop=True)

    df = nve_periods()
    if exclude_calving:
        df = df[[not CALVING_GLACIERS.intersection(m) for m in df.members]]
    if max_area_ratio is not None:
        ratio = df.area_ratio
        df = df[
            ratio.isna() | ((ratio <= max_area_ratio) & (ratio >= 1 / max_area_ratio))
        ]
    df = select_dem_periods(
        df,
        max_year=max_year,
        min_year=min_year,
        min_period_years=min_period_years,
        glacier_ids_to_keep=glacier_ids_to_keep,
        allowed_years=allowed_years,
        max_outside_years=max_outside_years,
        multi_period=multi_period,
        tie_break=tie_break,
    )
    if multi_period and len(df) > 0:
        df = pd.concat([best_period_chain(g) for _, g in df.groupby("RGIId")])
    return df.sort_values(["RGIId", "FROM_DATE"]).reset_index(drop=True)


def _entity_rgi_ids(glacier_ids):
    """One row per (entity, RGI 6.2 outline of one of its glaciers)."""
    rgi_of = _rgi_of_breid()
    return pd.DataFrame(
        [(g, rgi_of[m]) for g in glacier_ids for m in entity_members(g)],
        columns=["NVE_ID", "RGIId"],
    ).drop_duplicates()


def load_nve_outlines(glacier_ids_to_keep=None, inventory: str = "RGI"):
    """Read the outlines of the entities in one inventory, the outlines of a group
    dissolved into one.

    With `inventory="RGI"` these are the RGI 6.2 outlines, with "GAO" the basins of
    the GAO ice of 1947-1985 (see the module docstring), of the entities with a period
    gridded on them.

    The returned frame is in EPSG:4326 with the entity id in a column named
    "NVE_ID", ready for `custom_outlines.build_custom_gdirs`, along with the
    RGI first- and second-order regions of the entity ("08", "02"), its name, the RGI
    ids it is made of and its area in km² (`area_km2`), summed from the RGI areas for
    RGI. Every entity of `nve_periods` with a period on `inventory` is returned by
    default.
    """
    assert inventory in OUTLINE_INVENTORIES, f"No outline inventory {inventory!r}."
    periods = nve_periods()
    ids = sorted(periods[periods.outlines == inventory].RGIId.unique())
    if glacier_ids_to_keep is not None:
        missing = set(glacier_ids_to_keep).difference(ids)
        assert not missing, f"NVE has no glacier {sorted(missing)} on {inventory}."
        ids = sorted(set(glacier_ids_to_keep))
    pairs = _entity_rgi_ids(ids)
    rgi = gpd.read_file(get_region_shape_file(f"{NORWAY_REGION_ID:02d}"))
    rgi = pairs.merge(rgi, on="RGIId")
    rgi = gpd.GeoDataFrame(rgi, geometry="geometry", crs="EPSG:4326")
    o2 = rgi.groupby("NVE_ID").O2Region.nunique()
    assert (o2 == 1).all(), f"Entities across O2 regions: {list(o2[o2 > 1].index)}."
    outlines = rgi.dissolve(
        "NVE_ID",
        aggfunc={"Area": "sum", "O1Region": "first", "O2Region": "first"},
    ).reset_index()
    outlines["rgi_ids"] = outlines.NVE_ID.map(
        pairs.groupby("NVE_ID").RGIId.apply(lambda s: tuple(sorted(s)))
    )
    outlines = outlines.rename(columns={"Area": "area_km2"})
    if inventory == "GAO":
        gao = gpd.GeoSeries(
            [_gao_outline(entity_members(g)) for g in outlines.NVE_ID], crs=GAO_CRS
        )
        outlines["area_km2"] = gao.area.to_numpy() / 1e6
        outlines = outlines.set_geometry(gao.to_crs(outlines.crs).to_numpy())
    outlines["o1_region"] = outlines.O1Region.astype(int).map("{:02d}".format)
    outlines["o2_region"] = outlines.O2Region.astype(int).map("{:02d}".format)
    names = periods.drop_duplicates("RGIId").set_index("RGIId").name
    outlines["name"] = outlines.NVE_ID.map(names)
    return (
        outlines[
            [
                "NVE_ID",
                "o1_region",
                "o2_region",
                "name",
                "rgi_ids",
                "area_km2",
                "geometry",
            ]
        ]
        .sort_values("NVE_ID")
        .reset_index(drop=True)
        .to_crs("EPSG:4326")
    )


def nve_outline_spec(dem_source: str = "COPDEM30", inventory: str = "RGI", **overrides):
    """Description of the Norwegian outlines of one inventory (see
    `load_nve_outlines`) for the generic custom-outline machinery. The grids on the
    RGI outlines are in the `NVE` tree, those on the GAO outlines in `NVE_GAO`.

    As for Andreassen16, the DEM defaults to the Copernicus GLO-30 DEM, since SRTM
    stops at 60°N, and the second-order region is left to the `o2_region` column of
    the outlines.
    """
    assert inventory in OUTLINE_INVENTORIES, f"No outline inventory {inventory!r}."
    year = OUTLINES_YEAR if inventory == "RGI" else GAO_OUTLINES_YEAR
    spec = dict(
        name="NVE" if inventory == "RGI" else f"NVE_{inventory}",
        id_column="NVE_ID",
        o1_region=f"{NORWAY_REGION_ID:02d}",
        o2_region=None,
        src_date=f"{year}-01-01 00:00:00",
        bgndate=f"{year}0101",
        dem_source=dem_source,
    )
    spec.update(overrides)
    return CustomOutlineSpec(**spec)


def table_RGI62_to_NVE(region_id=NORWAY_REGION_ID):
    """Crosswalk from RGI 6.2 ids to the entities of the NVE table.

    The outlines of this source are the RGI 6.2 ones, so the crosswalk is read off the
    positions of the glaciers rather than matched by overlap: every RGI outline lies
    entirely in its entity (`frac_rgi` = 1), and in every group holding it, so an RGI
    id can have several rows. Only region 08 holds any.

    Returns a dataframe with columns RGIId, custom_id (the entity id), area_rgi,
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
    if int(region_id) != NORWAY_REGION_ID:
        return pd.DataFrame(columns=columns)
    pairs = _entity_rgi_ids(sorted(nve_periods().RGIId.unique()))
    area = (
        gpd.read_file(get_region_shape_file(f"{NORWAY_REGION_ID:02d}"))
        .set_index("RGIId")
        .Area
    )
    table = pairs.rename(columns={"NVE_ID": "custom_id"})
    table["area_rgi"] = table.RGIId.map(area)
    table["area_custom"] = table.groupby("custom_id").area_rgi.transform("sum")
    table["frac_rgi"] = 1.0
    table["frac_custom"] = table.area_rgi / table.area_custom
    return table[columns].sort_values(["RGIId", "custom_id"]).reset_index(drop=True)
