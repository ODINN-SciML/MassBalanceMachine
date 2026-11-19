# --- standard library ---
import logging

# --- third-party ---
import numpy as np
import pandas as pd
import pyproj
from scipy.spatial.distance import cdist

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


# ---------------- Quality control and normalization on stake points ---------------- #
def remove_close_points(df_gl):
    """
    Merge stake points that are closer than 10 meters within each (YEAR, PERIOD) group.

    For each year and for each period present in `df_gl`, this function:
    - converts lat/lon to LAEA coordinates (EPSG:3035) via `latlon_to_laea`
    - computes pairwise distances
    - for clusters of points within 10 m, assigns the mean POINT_BALANCE to the kept point
    - drops the redundant points

    Parameters
    ----------
    df_gl : pandas.DataFrame
        Stake dataset containing columns: `YEAR`, `PERIOD`, `POINT_LAT`, `POINT_LON`,
        and `POINT_BALANCE`.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe with close duplicates merged/dropped. Rows of all
        periods are kept; filter periods (e.g. drop summer) before calling.

    Notes
    -----
    The function logs how many points were dropped. It may add temporary `x`, `y`
    columns during processing.
    """
    df_gl_cleaned = pd.DataFrame()
    for year in df_gl.YEAR.unique():
        for period in df_gl.PERIOD.unique():
            df_gl_y = df_gl[(df_gl.YEAR == year) & (df_gl.PERIOD == period)]
            if len(df_gl_y) <= 1:
                df_gl_cleaned = pd.concat([df_gl_cleaned, df_gl_y])
                continue

            # Calculate distances to other points
            df_gl_y["x"], df_gl_y["y"] = latlon_to_laea(
                df_gl_y["POINT_LAT"], df_gl_y["POINT_LON"]
            )

            distance = cdist(df_gl_y[["x", "y"]], df_gl_y[["x", "y"]], "euclidean")

            # Merge close points
            merged_indices = set()
            for i in range(len(df_gl_y)):
                if i in merged_indices:
                    continue  # Skip already merged points

                # Find close points (distance < 10m)
                close_indices = np.where(distance[i, :] < 10)[0]
                close_indices = [idx for idx in close_indices if idx != i]

                if close_indices:
                    mean_MB = df_gl_y.iloc[close_indices + [i]].POINT_BALANCE.mean()

                    # Assign mean balance to the first point
                    df_gl_y.loc[df_gl_y.index[i], "POINT_BALANCE"] = mean_MB

                    # Mark other indices for removal
                    merged_indices.update(close_indices)

            # Drop surplus points
            indices_to_drop = list(merged_indices)
            df_gl_y = df_gl_y.drop(df_gl_y.index[indices_to_drop])

            # Append cleaned DataFrame
            df_gl_cleaned = pd.concat([df_gl_cleaned, df_gl_y])

    # Final output
    df_gl_cleaned.reset_index(drop=True, inplace=True)
    points_dropped = len(df_gl) - len(df_gl_cleaned)
    print(f"--- Number of points dropped: {points_dropped}")
    # log.info(f"--- Number of points dropped: {points_dropped}")
    if points_dropped > 0:
        return df_gl_cleaned
    # Nothing merged: return the input unchanged, with the same temporary
    # x/y columns as above so that callers can always drop them
    return df_gl.assign(x=np.nan, y=np.nan)


def latlon_to_laea(lat, lon):
    """
    Transform WGS84 latitude/longitude to LAEA Europe (EPSG:3035) coordinates.

    Parameters
    ----------
    lat : array-like
        Latitudes in degrees (EPSG:4326).
    lon : array-like
        Longitudes in degrees (EPSG:4326).

    Returns
    -------
    tuple
        (easting, northing) in meters (EPSG:3035).
    """
    # Define the transformer: WGS84 to ETRS89 / LAEA Europe
    transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:3035")

    # Perform the transformation
    easting, northing = transformer.transform(lat, lon)
    return easting, northing


def rename_stakes_by_elevation(df_pmb_topo):
    """
    Reassign stake POINT_IDs per glacier by sorting stakes by mean elevation.

    For each glacier, the function computes mean elevation per POINT_ID,
    sorts by elevation, and renames POINT_ID to `<glacier>_<rank>`.

    Parameters
    ----------
    df_pmb_topo : pandas.DataFrame
        Stake dataset containing `GLACIER`, `POINT_ID`, and `POINT_ELEVATION`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with updated `POINT_ID` values.
    """
    for glacierName in df_pmb_topo.GLACIER.unique():
        gl_data = df_pmb_topo[df_pmb_topo.GLACIER == glacierName]
        stakeIDS = gl_data.groupby("POINT_ID")[
            ["POINT_LAT", "POINT_LON", "POINT_ELEVATION"]
        ].mean()
        stakeIDS.reset_index(inplace=True)
        # Change the ID according to elevation
        new_ids = stakeIDS[["POINT_ID", "POINT_ELEVATION"]].sort_values(
            by="POINT_ELEVATION"
        )
        new_ids["POINT_ID_new"] = [f"{glacierName}_{i}" for i in range(len(new_ids))]
        for i, row in new_ids.iterrows():
            df_pmb_topo.loc[
                (df_pmb_topo.GLACIER == glacierName)
                & (df_pmb_topo.POINT_ID == row.POINT_ID),
                "POINT_ID",
            ] = row.POINT_ID_new
    return df_pmb_topo


def check_multiple_rgi_ids(df):
    """
    Check whether any GLACIER name maps to more than one RGIId.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing columns `GLACIER` and `RGIId`.

    Returns
    -------
    bool
        True if at least one glacier has multiple unique RGIId values, else False.
    """
    rgi_per_glacier = df.groupby("GLACIER")["RGIId"].nunique()
    glaciers_with_multiple_rgi = rgi_per_glacier[rgi_per_glacier > 1]
    if not glaciers_with_multiple_rgi.empty:
        return True
    else:
        return False


def check_point_ids_contain_glacier(dataframe):
    """
    Validate that each POINT_ID contains the corresponding GLACIER name substring.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Must contain columns `GLACIER` and `POINT_ID`.

    Returns
    -------
    tuple[bool, pandas.DataFrame or None]
        (is_valid, invalid_rows). invalid_rows is None if valid.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    if "GLACIER" not in dataframe.columns or "POINT_ID" not in dataframe.columns:
        raise ValueError("The dataframe must contain 'GLACIER' and 'POINT_ID' columns.")

    # Check condition
    invalid_rows = dataframe[
        ~dataframe.apply(lambda row: row["GLACIER"] in row["POINT_ID"], axis=1)
    ]

    # Report
    if invalid_rows.empty:
        print("All POINT_IDs correctly contain their respective GLACIER names.")
        return True, None
    else:
        print(
            f"Found {len(invalid_rows)} rows where POINT_ID does not contain the GLACIER name."
        )
        return False, invalid_rows
