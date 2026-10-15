# --- standard library ---
import logging

# --- third-party ---
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

# --- project/local ---
import massbalancemachine as mbm

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


# ---------------- Stake point processing ---------------- #
def add_rgi_ids_to_df(df_all_raw, glacier_outline_fname):
    """
    Attach RGIId to each stake point by intersecting with glacier outlines; fill gaps by nearest outline.

    Steps:
    1) Selects core WGMS-like columns from `df_all_raw`.
    2) Uses `mbm.data_processing.utils.get_rgi` to assign RGIId by intersection.
    3) For unmatched points, assigns the closest glacier outline polygon's RGIId.

    Parameters
    ----------
    df_all_raw : pandas.DataFrame
        Stake dataset with columns including POINT_LAT/POINT_LON and GLACIER metadata.
    glacier_outline_fname : str
        Path to a glacier outline file readable by GeoPandas (e.g., shapefile, geopackage).

    Returns
    -------
    pandas.DataFrame
        DataFrame with a populated `RGIId` column.

    Raises
    ------
    FileNotFoundError
        If `glacier_outline_fname` cannot be read.
    """
    # Select relevant columns
    df_pmb = df_all_raw[
        [
            "YEAR",
            "POINT_ID",
            "GLACIER",
            "FROM_DATE",
            "TO_DATE",
            "POINT_LAT",
            "POINT_LON",
            "POINT_ELEVATION",
            "POINT_BALANCE",
            "PERIOD",
        ]
    ].copy()

    # Load the glacier outlines
    glacier_outline = gpd.read_file(glacier_outline_fname)

    # Add RGI IDs through intersection
    df_pmb = mbm.data_processing.utils.get_rgi(
        data=df_pmb, glacier_outlines=glacier_outline
    )

    # Handle unmatched points
    no_match_df = df_pmb[df_pmb["RGIId"].isna()]
    geometry = [
        Point(lon, lat)
        for lon, lat in zip(no_match_df["POINT_LON"], no_match_df["POINT_LAT"])
    ]
    points_gdf = gpd.GeoDataFrame(
        no_match_df, geometry=geometry, crs=glacier_outline.crs
    )

    for index in tqdm(no_match_df.index, desc="Finding closest RGIId"):
        point = points_gdf.loc[index]["geometry"]
        polygon_index = glacier_outline.distance(point).sort_values().index[0]
        closest_rgi = glacier_outline.loc[polygon_index].RGIId
        df_pmb.at[index, "RGIId"] = closest_rgi

    return df_pmb
