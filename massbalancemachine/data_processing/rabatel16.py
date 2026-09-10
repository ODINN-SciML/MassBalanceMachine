"""
Glacier outlines and DEM of the French Alps used with the geodetic mass balance of
Rabatel et al. (2016).

The outlines are the 1985-86 inventory of the French Alps, delineated on Landsat
scenes acquired between 1985 and 1988 (one date per glacier in the `date` column of
the shapefile). The DEM is the photogrammetric IGN DEM the paper names "1979": 25 m
there, 50 m in the copy used here, and built from aerial photographs taken between
1975 and 1985 depending on the massif, so its epoch is not the same everywhere. It is
not the public BD ALTI, which IGN keeps updated from recent surveys. Compared with
Copernicus GLO-30 on ice-free terrain, its elevations agree to within a few metres and
its georeferencing is off by 20 to 50 m depending on the massif, less than a pixel,
which is left uncorrected. The grids of this source are built on that geometry rather
than on the RGI 6.2 outlines, which for the Alps date from 2003, since a geodetic rate
is referenced to the glacier area of its own epoch.

The target comes from the table of annual glacier-wide mass balances that ALPGM
(Bolibar et al., 2020) redistributes. That table merges the series of Rabatel et al.
(2016), 1984-2014, with GLACIOCLIM field series, as its author confirmed, and only the
former are kept: the 27 glaciers whose mean over 1984-2014 reproduces the mean mass
balance of Table 1 of the paper (see `keep_rabatel16_series`). Every value covers one
hydrological year, from October to September. The target is their sum over a period
chosen by the user, with the uncertainty of 0.3 m w.e. per year that Rabatel et al.
(2016) give over 1983-2014 carried over to that period; see
`geodetic_target_Rabatel16`.

Glaciers are identified by their **GLIMS id** ("G006985E45951N"), the only attribute
of the shapefile that is both filled and unique: the WGI code is shared by
neighbouring glaciers and sometimes missing. As for GLAMOS, the gridded products of
this source therefore carry `RGIId == "G006985E45951N"`, and the crosswalk built by
`table_RGI62_to_Rabatel16` is many RGI ids to one GLIMS entity.

The raw outlines and DEM are not published, so unlike GLAMOS they are not downloaded
and live outside of `.data/`, see `Rabatel16_folder`. What is derived from them, such
as the DEM with its missing values declared (`rabatel16_dem_file`), is written under
`.data/Rabatel16`, and the ALPGM tables are downloaded to `.data/Rabatel16/ALPGM`.
"""

import difflib
import os
import re
import socket
import unicodedata
import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_dilation

from data_processing.custom_outlines import CustomOutlineSpec, match_rgi62_by_overlap
from data_processing.product_utils import data_path
from data_processing.Product import Product
from data_processing.glacier_utils import get_region_shape_file

# The French Alps are entirely inside RGI region 11 (Central Europe), second-order
# region 11-01 (Alps).
FRENCH_ALPS_REGION_ID = 11
FRENCH_ALPS_SUBREGION = "01"

# Year most of the outlines were delineated in, see the module docstring
OUTLINES_YEAR = 1985

# Missing value of the DEM written to `.data/Rabatel16`: the one the raw PCIDSK layer
# declares without using it, and the one OGGM assumes for a raster that declares none
DEM_NODATA = -9999.0

# The ALPGM repository is pinned to one commit, so that every machine downloads the
# same tables.
ALPGM_COMMIT = "55061173681ce73ac3822df8e61543107b478276"
ALPGM_URL = (
    f"https://raw.githubusercontent.com/JordiBolibar/ALPGM/{ALPGM_COMMIT}/glacier_data"
)
ALPGM_FILES = {
    # annual glacier-wide mass balances, one row per glacier and one column per year
    "smb": "smb/SMB_w_years_temporal.csv",
    # the same glaciers in the same order, with their coordinates
    "glaciers": "GLIMS/GLIMS_temporal_32_1950.csv",
}

# Uncertainty Rabatel et al. (2016) give for the mean annual balance over their study
# period, "1983-2014" (caption of Table 1): the hydrological years from 1983-84 to
# 2013-14, which the ALPGM table labels 1984 to 2014. See `period_sigma_mwe_per_year`
# for another period.
REFERENCE_SIGMA_MWE_PER_YEAR = 0.3
REFERENCE_PERIOD = (1984, 2014)
# Section 4.1 of the paper: uncertainty of an annual balance, on average, and the part
# of it coming from the mean geodetic balance, shared by every year of a glacier
ANNUAL_SIGMA_MWE = 0.22
GEODETIC_MEAN_SIGMA_MWE_PER_YEAR = 0.12

# Mean glacier-wide mass balance over 1983-2014 of the 30 glaciers of Table 1 of Rabatel
# et al. (2016), in m w.e. per year, keyed by the GLIMS id of their 1985 outline
RABATEL16_TABLE1_MEAN_MB = {
    "G006988E45987N": ("Tour", -0.73),
    "G006985E45951N": ("Argentière", -0.84),
    "G006992E45913N": ("Talèfre", -0.97),
    "G006784E45784N": ("Tré la Tête", -1.05),
    "G006880E45521N": ("Savinaz", -0.76),
    "G006874E45531N": ("Gurraz", -0.72),
    "G006988E45515N": ("Sassière", -0.85),
    "G006881E45423N": ("Grande Motte", -0.98),
    "G007144E45371N": ("Mulinet", -0.80),
    "G007143E45363N": ("Grand Méan", -0.91),
    "G006780E45369N": ("Arcelin", -0.81),
    "G006763E45344N": ("Pelve", -0.94),
    "G006756E45318N": ("Arpont", -0.86),
    "G006745E45302N": ("Mahure", -0.85),
    "G006904E45335N": ("Vallonnet", -0.71),
    "G006629E45295N": ("Gébroulaz", -0.70),
    "G007117E45255N": ("Baounet", -1.26),
    "G007067E45221N": ("Rochemelon", -1.13),
    "G006159E45160N": ("Saint-Sorlin", -1.04),
    "G006149E45143N": ("Quirlies", -1.03),
    "G006226E45003N": ("Mont de Lans", -0.53),
    "G006252E45009N": ("Girose", -0.59),
    "G006274E44992N": ("Selle", -0.76),
    "G006339E45007N": ("Lautaret", -0.86),
    "G006439E44956N": ("Casset", -0.94),
    "G006382E44944N": ("Blanc", -0.89),
    "G006347E44919N": ("Vallon Pilatte", -0.90),
    "G006275E44878N": ("Rouies", -0.61),
    "G006367E44861N": ("Sélé", -1.09),
    "G006334E44865N": ("Pilatte", -1.08),
}
# Table 1 gives two decimals: a series of the paper reproduces it to 0.005
TABLE1_TOLERANCE_MWE_PER_YEAR = 0.01


def Rabatel16_folder():
    hostname = socket.gethostname()
    # We don't want to publish these files for the moment :)
    if hostname == "ige-osugb1-p48":
        return "/home/gossarda/Téléchargements/geodetic_Rabatel16/"
    elif "bigfoot" in hostname:
        return "/home/gossarda/geodetic_Rabatel16/"
    elif hostname == "63bceb1ea564":
        return "/workspace/geodetic_Rabatel16/"
    elif hostname == "ige-calcul1" or hostname == "ige-calcul3":
        return "/home/gossarda/geodetic_Rabatel16/"
    else:
        raise ValueError(f"Unknown host {hostname}")


def rabatel16_outlines_file():
    return os.path.join(
        Rabatel16_folder(), "Glacier_1985-86_FR", "GLIMS_glaciers_1985.shp"
    )


def rabatel16_raw_dem_file():
    return os.path.join(Rabatel16_folder(), "MNT_IGN_Alpes_50m_UTM_80s.pix")


def rabatel16_data_folder():
    """Directory holding what is derived from the raw Rabatel16 files, always the same
    place so that a later run finds what an earlier one wrote."""
    return os.path.join(data_path, "Rabatel16")


def write_dem_with_nodata(
    src_path: str, dst_path: str, missing_value: float = 0, edge_cells: int = 2
):
    """Copy the first band of a DEM to a GeoTIFF, turning `missing_value` into a
    declared missing value.

    The raw IGN DEM fills everything it does not cover - Italy, Switzerland, the rest
    of its bounding box - with 0 m and declares no missing value rasterio can see, so
    OGGM would take those cells for real terrain at sea level. The cells bordering
    that fill are no better: they hold a blend of the terrain and of the 0 m, about
    two thirds of the true elevation (2050 m next to 3200 m along the Rochemelon
    ridge). Half of the first ring of cells is affected, and 1 % of the second, where
    the raster repeats a row or a column; from the third ring on the values are as
    sound as the interior. The `edge_cells` rings of cells around every missing cell
    are therefore declared missing too, otherwise the filling below would spread
    those values over the missing part of a glacier.

    Once declared missing, the cells are filled by OGGM when it builds a glacier
    directory (`process_dem`): by linear interpolation where valid cells surround
    them, and with the nearest valid value beyond the edge of the data, which is the
    case along the national border.
    """
    with rasterio.open(src_path) as src:
        dem = src.read(1).astype("float32")
        profile = dict(
            driver="GTiff",
            width=src.width,
            height=src.height,
            count=1,
            dtype="float32",
            crs=src.crs,
            transform=src.transform,
            nodata=DEM_NODATA,
            compress="deflate",
            predictor=3,
            tiled=True,
        )
    assert not (
        dem == DEM_NODATA
    ).any(), f"{src_path} already holds {DEM_NODATA}, which cannot mark missing cells."
    missing = dem == missing_value
    if edge_cells > 0:
        missing = binary_dilation(
            missing, structure=np.ones((3, 3), dtype=bool), iterations=edge_cells
        )
    dem[missing] = DEM_NODATA
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(dem, 1)


def rabatel16_dem_file(prepare: bool = True):
    """Path to the IGN DEM with its missing values declared, writing it from the raw
    DEM if necessary, see `write_dem_with_nodata`."""
    path = os.path.join(rabatel16_data_folder(), "MNT_IGN_Alpes_50m_UTM_80s.tif")
    if prepare:
        p = Product(path)
        if not p.is_up_to_date():
            print(f"writing {path} from {rabatel16_raw_dem_file()}")
            write_dem_with_nodata(rabatel16_raw_dem_file(), path)
            p.gen_chk()
    return path


def load_rabatel16_outlines(glacier_ids_to_keep=None):
    """Read the 1985-86 outlines of the French Alps, one entity per GLIMS id.

    The returned frame is in EPSG:4326 with the GLIMS id in a column named
    "GLIMS_ID", ready for `custom_outlines.build_custom_gdirs`, along with the glacier
    name, its massif and the date of the scene it was delineated on.
    """
    outlines = gpd.read_file(rabatel16_outlines_file())
    assert (
        outlines.GLIMS_ID.is_unique
    ), "The Rabatel16 outlines hold several rows for one GLIMS id."
    outlines = outlines.rename(columns={"Glacier_2": "name", "Massif": "massif"})
    outlines["date"] = pd.to_datetime(outlines.date, format="%d-%m-%Y")
    outlines = outlines[["GLIMS_ID", "name", "massif", "date", "geometry"]]
    if glacier_ids_to_keep is not None:
        missing = set(glacier_ids_to_keep).difference(outlines.GLIMS_ID)
        assert not missing, f"The Rabatel16 outlines have no entity {sorted(missing)}."
        outlines = outlines[outlines.GLIMS_ID.isin(glacier_ids_to_keep)]
    return outlines.to_crs("EPSG:4326")


def rabatel16_outline_spec(**overrides):
    """Description of the Rabatel16 outlines for the generic custom-outline machinery.

    The DEM defaults to the IGN DEM with its missing values declared, which this does
    not write: `create_gridded_features_Rabatel16` does, when it is needed. Pass
    `dem_file=None, dem_source="SRTM"` (or any other DEM OGGM supports) to build the
    grids on another surface, together with another `name` so that the two variants
    do not share their products, see `custom_outlines.assert_spec_matches`.
    """
    spec = dict(
        name="Rabatel16",
        id_column="GLIMS_ID",
        o1_region=f"{FRENCH_ALPS_REGION_ID:02d}",
        o2_region=FRENCH_ALPS_SUBREGION,
        src_date=f"{OUTLINES_YEAR}-01-01 00:00:00",
        bgndate=f"{OUTLINES_YEAR}0101",
        dem_file=rabatel16_dem_file(prepare=False),
    )
    spec.update(overrides)
    return CustomOutlineSpec(**spec)


def alpgm_file(kind: str, download: bool = True):
    """Path to one of the ALPGM tables of `ALPGM_FILES`, downloading it if necessary."""
    relative = ALPGM_FILES[kind]
    path = os.path.join(rabatel16_data_folder(), "ALPGM", os.path.basename(relative))
    if not os.path.exists(path) and download:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = f"{ALPGM_URL}/{relative}"
        print(f"downloading {url}")
        # Through a temporary name, so that an interrupted download is not mistaken
        # for the table on the next run
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def _normalized_name(name: str):
    """Glacier name without accents, articles, numbers and punctuation, "Glacier de la
    Selle_1" and "de la Selle 1" both giving "selle"."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    words = re.split(r"[^a-z]+", ascii_name.lower())
    stop_words = {"glacier", "de", "la", "du", "des", "le", "les", "l", "d"}
    return " ".join(w for w in words if w and w not in stop_words)


def assert_rows_describe_same_glaciers(names, other_names):
    """Check that two lists of glacier names spelled differently are in the same order.

    ALPGM links its mass balance table to its glacier table by row, while the two spell
    the names in their own way ("Sarennes" and "de Sarenne 1"). Names alone cannot be
    matched reliably, but a shifted row can be detected: the name of every row must be
    closer to the name of the same row in the other list than to any other.
    """
    assert len(names) == len(
        other_names
    ), f"{len(names)} glaciers on one side, {len(other_names)} on the other."
    names = [_normalized_name(n) for n in names]
    other_names = [_normalized_name(n) for n in other_names]
    for i, name in enumerate(names):
        similarity = [
            difflib.SequenceMatcher(None, name, other).ratio() for other in other_names
        ]
        assert int(np.argmax(similarity)) == i, (
            f"Row {i} is {name!r} on one side and {other_names[i]!r} on the other, but "
            f"{other_names[int(np.argmax(similarity))]!r} is a closer name: the two "
            "tables do not list the glaciers in the same order."
        )


def load_rabatel16_smb():
    """The annual glacier-wide mass balances of Rabatel et al. (2016) redistributed by
    ALPGM, in long form, without the GLACIOCLIM series ALPGM merges them with (see
    `keep_rabatel16_series`).

    Glaciers are identified by the GLIMS id of the **1985 outline containing the
    coordinates** ALPGM gives for them, not by the GLIMS id ALPGM lists. The ids come
    from later inventories and one of them is wrong: "de la Selle 1" carries the id of
    its 0.6 km² neighbour Selle 2, while its coordinates, name and area are those of
    Selle 1.

    Returns a dataframe with columns GLIMS_ID, name (the ALPGM spelling), year (the
    hydrological year, which ends in September of that year) and smb, in m w.e.
    """
    smb = pd.read_csv(alpgm_file("smb"), sep=";", encoding="latin-1")
    glaciers = pd.read_csv(alpgm_file("glaciers"), sep=";", encoding="latin-1")
    assert_rows_describe_same_glaciers(list(smb.Glacier), list(glaciers.Glacier))

    outlines = load_rabatel16_outlines()
    points = gpd.GeoDataFrame(
        {"row": range(len(glaciers))},
        geometry=gpd.points_from_xy(glaciers.x_coord, glaciers.y_coord),
        crs="EPSG:4326",
    )
    inside = gpd.sjoin(points, outlines, predicate="within")
    counts = inside.row.value_counts().reindex(points.row, fill_value=0)
    assert (counts == 1).all(), (
        "Every ALPGM glacier must lie inside exactly one 1985 outline, not "
        f"{dict(zip(glaciers.Glacier[counts != 1], counts[counts != 1]))}."
    )
    glims_ids = inside.set_index("row").GLIMS_ID.sort_index()
    assert glims_ids.is_unique, "Two ALPGM glaciers lie inside the same 1985 outline."

    smb = smb.assign(GLIMS_ID=glims_ids.to_numpy(), name=smb.Glacier.str.strip())
    smb = smb.drop(columns="Glacier").melt(
        id_vars=["GLIMS_ID", "name"], var_name="year", value_name="smb"
    )
    smb = smb.dropna(subset=["smb"]).astype({"year": int, "smb": float})
    smb = keep_rabatel16_series(smb)
    return smb.sort_values(["GLIMS_ID", "year"]).reset_index(drop=True)


def keep_rabatel16_series(
    smb: pd.DataFrame,
    table1=RABATEL16_TABLE1_MEAN_MB,
    tolerance: float = TABLE1_TOLERANCE_MWE_PER_YEAR,
):
    """Keep the glaciers whose series is the one of Rabatel et al. (2016).

    The ALPGM table merges the remote-sensing series of the paper with GLACIOCLIM field
    series. By construction of the method, the mean of a series of the paper over
    `REFERENCE_PERIOD` is the mean glacier-wide mass balance of its glacier in Table 1,
    so a glacier is kept only if its mean over that period reproduces it to within
    `tolerance`. This leaves out the glaciers the paper does not study (Mer de Glace,
    Sarennes) and those for which ALPGM holds the field series instead (Saint-Sorlin,
    Argentière, Gébroulaz, whose means differ from Table 1 by 0.10 to 0.16 m w.e. per
    year).

    Args:
        smb: annual balances in long form, with columns GLIMS_ID, year and smb.
        table1: {GLIMS id: (name, mean mass balance over the reference period)}.
    """
    missing = set(table1).difference(smb.GLIMS_ID)
    assert not missing, (
        "Glaciers of Table 1 of Rabatel et al. (2016) are missing from the ALPGM table: "
        f"{sorted(table1[g][0] for g in missing)}."
    )
    in_reference = smb[smb.year.between(*REFERENCE_PERIOD)]
    mean = in_reference.groupby("GLIMS_ID").smb.mean()
    reference = pd.Series({g: mb for g, (_, mb) in table1.items()})
    matches = (mean.reindex(reference.index) - reference).abs() <= tolerance
    kept = set(matches[matches].index)

    left_out = smb[~smb.GLIMS_ID.isin(kept)].groupby("GLIMS_ID").name.first()
    if len(left_out):
        reasons = [
            f"{name} ("
            + (
                "not in Table 1"
                if glims_id not in table1
                else f"mean {mean[glims_id]:.2f} against {table1[glims_id][1]:.2f}"
            )
            + ")"
            for glims_id, name in left_out.items()
        ]
        print(
            f"Rabatel16: {len(left_out)} glacier(s) of the ALPGM table whose series is "
            "not the one of Rabatel et al. (2016) are left out: " + ", ".join(reasons)
        )
    return smb[smb.GLIMS_ID.isin(kept)]


def period_sigma_mwe_per_year(n_years: int):
    """Uncertainty of the mean annual balance over a period of `n_years` hydrological
    years.

    Rabatel et al. (2016) give `REFERENCE_SIGMA_MWE_PER_YEAR` for the mean over their
    whole period, `REFERENCE_PERIOD`. An annual balance is uncertain by
    `ANNUAL_SIGMA_MWE`, of which `GEODETIC_MEAN_SIGMA_MWE_PER_YEAR` comes from the mean
    geodetic balance every series is normalised to, and is therefore shared by all the
    years of a glacier. Only the rest,

        sigma_year = sqrt(ANNUAL_SIGMA_MWE**2 - GEODETIC_MEAN_SIGMA_MWE_PER_YEAR**2)

    about 0.18 m w.e., varies from one year to the next, and averages out over a period
    like 1 / sqrt(n_years). Anchoring the uncertainty to the published value over the
    reference period gives

        sigma(n_years)**2 = REFERENCE_SIGMA_MWE_PER_YEAR**2
                            + sigma_year**2 * (1 / n_years - 1 / n_reference)

    The shared part dominates, so the uncertainty hardly depends on the length of the
    period: 0.302 over 16 years, 0.309 over 5, 0.351 for a single year, and 0.3 over the
    reference period itself.
    """
    assert n_years > 0, f"A period of {n_years} years has no mass balance."
    n_reference = REFERENCE_PERIOD[1] - REFERENCE_PERIOD[0] + 1
    sigma_year_squared = ANNUAL_SIGMA_MWE**2 - GEODETIC_MEAN_SIGMA_MWE_PER_YEAR**2
    return float(
        np.sqrt(
            REFERENCE_SIGMA_MWE_PER_YEAR**2
            + sigma_year_squared * (1 / n_years - 1 / n_reference)
        )
    )


def period_smb_to_window(smb: pd.DataFrame, start_year: int, end_year: int):
    """Sum the annual glacier-wide balances of every glacier over one period.

    The period runs over the hydrological years `start_year` to `end_year` included,
    i.e. from the 1st of October of `start_year - 1` to the 1st of October of
    `end_year`. A glacier enters the target only if its series covers every one of
    those years: a sum over an incomplete series would not be the balance of the
    period.

    Returns one row per glacier with the sum over the period, `cumulative_mwe`, and
    the rate the pipeline compares the model with, `mwe_per_year`, which is that sum
    divided by the number of years. `geodetic_window_weights` turns the monthly
    predictions into the same mean annual rate. Its uncertainty depends on the length
    of the period, see `period_sigma_mwe_per_year`.
    """
    assert (
        start_year <= end_year
    ), f"The period {start_year}-{end_year} ends before it starts."
    n_years = end_year - start_year + 1
    in_period = smb[smb.year.between(start_year, end_year)]
    per_glacier = in_period.groupby(["GLIMS_ID", "name"]).agg(
        n_years=("year", "nunique"), cumulative_mwe=("smb", "sum")
    )
    complete = per_glacier.n_years == n_years
    if not complete.all():
        incomplete = per_glacier[~complete].reset_index()
        print(
            f"Rabatel16: {len(incomplete)} glacier(s) without a balance for every "
            f"hydrological year of {start_year}-{end_year} are left out: "
            + ", ".join(
                f"{r.name} ({r.n_years} years)" for r in incomplete.itertuples()
            )
        )
    df = per_glacier[complete].reset_index().rename(columns={"GLIMS_ID": "RGIId"})
    df["FROM_DATE"] = pd.Timestamp(f"{start_year - 1}-10-01")
    df["TO_DATE"] = pd.Timestamp(f"{end_year}-10-01")
    df["mwe_per_year"] = df.cumulative_mwe / n_years
    df["sigma_mwe_per_year"] = period_sigma_mwe_per_year(n_years)
    return df.sort_values("RGIId").reset_index(drop=True)


def geodetic_target_Rabatel16(
    start_year: int,
    end_year: int,
    glacier_ids_to_keep=None,
):
    """Mass balance of French glaciers over the hydrological years `start_year` to
    `end_year`, summed from their annual glacier-wide balances (see
    `period_smb_to_window`).

    Returns a dataframe shaped like the one `data_processing.glamos.geodetic_target_GLAMOS`
    returns, so that both drive the same code path in `GeoDataLoader`, with one row per
    glacier:

        RGIId                 the GLIMS id of the 1985 outline, e.g. "G006985E45951N"
        FROM_DATE, TO_DATE    1st of October of start_year - 1, of end_year
        cumulative_mwe        sum of the annual balances over the period, m w.e.
        mwe_per_year          that sum divided by the number of years, m w.e. per year
        sigma_mwe_per_year    its uncertainty, m w.e. per year: 0.3 over 1984-2014 and
                              slightly more over a shorter period, see
                              `period_sigma_mwe_per_year`

    Args:
        start_year, end_year: first and last hydrological year of the period, each
            ending in September of that year. The grids are built on the 1985
            outlines, so the further the period is from 1985, the more the glacier it
            describes differs from the one modelled.
        glacier_ids_to_keep: GLIMS ids to restrict the target to.
    """
    smb = load_rabatel16_smb()
    if glacier_ids_to_keep is not None:
        smb = smb[smb.GLIMS_ID.isin(list(glacier_ids_to_keep))]
    return period_smb_to_window(smb, start_year, end_year)


def table_RGI62_to_Rabatel16(
    region_id=FRENCH_ALPS_REGION_ID,
    min_frac_rgi: float = 0.5,
):
    """Crosswalk from RGI 6.2 ids to the GLIMS entities of the 1985-86 outlines.

    The match is made by area overlap, see `custom_outlines.match_rgi62_by_overlap`.
    Only the RGI glaciers of the French Alps can match, since the outlines cover
    nothing else.

    Returns a dataframe with columns RGIId, custom_id (the GLIMS id), area_rgi,
    area_custom, frac_rgi, frac_custom, `custom_id` matching the column name
    `table_RGI62_to_GLAMOS` uses, so `GeoDataLoader` reads both the same way.
    """
    if not isinstance(region_id, str):
        region_id = f"{region_id:02d}"
    save_path = os.path.abspath(
        os.path.join(
            data_path, "grids", "Rabatel16", f"RGI62_to_Rabatel16_{region_id}.csv"
        )
    )
    p = Product(save_path)
    if p.is_up_to_date():
        return pd.read_csv(save_path)

    outlines = gpd.read_file(rabatel16_outlines_file())
    matches = match_rgi62_by_overlap(
        outlines,
        "GLIMS_ID",
        gpd.read_file(get_region_shape_file(region_id)),
        min_frac_rgi=min_frac_rgi,
    )

    matches.to_csv(save_path, index=False)
    p.gen_chk()
    return matches
