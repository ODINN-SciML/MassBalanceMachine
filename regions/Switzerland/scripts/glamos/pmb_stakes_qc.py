# --- standard library ---
import os
import re
import logging
from pathlib import Path
from collections import defaultdict
from typing import Iterable, Optional, Tuple

# --- third-party ---
import numpy as np
import pandas as pd
import xarray as xr
import pyproj
from scipy.spatial.distance import cdist
from tqdm import tqdm

# --- project/local ---
import massbalancemachine as mbm
from regions.Switzerland.scripts.config_CH import *
from regions.Switzerland.scripts.utils import *
from regions.Switzerland.scripts.geo_data.geodata import LV03_to_WGS84

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


# --------------------------------------- Quality control and normalization on stake points ------------------------------------------- #
def remove_close_points(df_gl):
    """
    Merge stake points that are closer than 10 meters within each (YEAR, PERIOD) group.

    For each year and for each period ('annual', 'winter'), this function:
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
        Cleaned dataframe with close duplicates merged/dropped.

    Notes
    -----
    The function logs how many points were dropped. It may add temporary `x`, `y`
    columns during processing.
    """
    df_gl_cleaned = pd.DataFrame()
    for year in df_gl.YEAR.unique():
        for period in ["annual", "winter"]:
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
    log.info(f"--- Number of points dropped: {points_dropped}")
    return df_gl_cleaned if points_dropped > 0 else df_gl


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


def clean_rgi_ids(df):
    """
    Apply hard-coded corrections to RGIId assignments for known problematic glaciers.

    Depending on glacier name, this function may:
    - drop rows where the RGIId does not match the known valid id
    - replace RGIId with a known valid id
    - remove all rows for a glacier entirely

    Parameters
    ----------
    df : pandas.DataFrame
        Stake dataset containing columns `GLACIER` and `RGIId`.

    Returns
    -------
    pandas.DataFrame
        The same DataFrame (mutated in-place) with corrections applied.

    Notes
    -----
    This function currently mutates the DataFrame in-place (drops rows).
    """
    corrections = {
        # Format: 'GLACIER': {'valid_rgi': 'RGI60-XX.XXXXX', 'action': 'drop|replace'}
        "albigna": {"valid_rgi": "RGI60-11.02285", "action": "drop"},
        "adler": {"valid_rgi": "RGI60-11.02764", "action": "drop"},
        "allalin": {"valid_rgi": "RGI60-11.02704", "action": "drop"},
        "basodino": {"valid_rgi": "RGI60-11.01987", "action": "drop"},
        "blauschnee": {"action": "remove_glacier"},
        "corvatsch": {"valid_rgi": "RGI60-11.01962", "action": "drop"},
        "damma": {"valid_rgi": "RGI60-11.01246", "action": "drop"},
        "findelen": {"valid_rgi": "RGI60-11.02773", "action": "drop"},
        "hohlaub": {"valid_rgi": "RGI60-11.02679", "action": "drop"},
        "gries": {"valid_rgi": "RGI60-11.01876", "action": "drop"},
        "limmern": {"valid_rgi": "RGI60-11.00918", "action": "drop"},
        "ofental": {"action": "remove_glacier"},
        "orny": {"valid_rgi": "RGI60-11.02775", "action": "replace"},
        "otemma": {"valid_rgi": "RGI60-11.02801", "action": "drop"},
        "plattalva": {"valid_rgi": "RGI60-11.00892", "action": "replace"},
        "plainemorte": {"valid_rgi": "RGI60-11.02072", "action": "drop"},
        "rhone": {"valid_rgi": "RGI60-11.01238", "action": "drop"},
        "sanktanna": {"valid_rgi": "RGI60-11.01367", "action": "drop"},
        "sexrouge": {"valid_rgi": "RGI60-11.02244", "action": "drop"},
        "silvretta": {"valid_rgi": "RGI60-11.00804", "action": "drop"},
        "tsanfleuron": {"valid_rgi": "RGI60-11.02249", "action": "drop"},
        "unteraar": {"action": "remove_glacier"},
    }

    for glacier, details in corrections.items():
        if details["action"] == "drop":
            df.drop(
                df[(df.GLACIER == glacier) & (df.RGIId != details["valid_rgi"])].index,
                inplace=True,
            )
        elif details["action"] == "replace":
            df.loc[df.GLACIER == glacier, "RGIId"] = details["valid_rgi"]
        elif details["action"] == "remove_glacier":
            df.drop(df[df.GLACIER == glacier].index, inplace=True)

    return df


# ------------------------- Correct for wrong elevations: -------------------------------- #
def find_mismatch_by_year(
    df_gl: pd.DataFrame,
    path_xr_grids: str,
    var_name: str = "masked_elev",
    lon_name: str = "lon",
    lat_name: str = "lat",
    year_col: str = "YEAR",
    glacier_col: str = "GLACIER",
    threshold: float = 500.0,
    file_pattern: str = "{glacier}_{year}.zarr",  # pattern of your files
    strict: bool = False,  # if True, raise if any glacier-year file is missing
):
    """
    Identify stake points whose recorded elevation strongly disagrees with a glacier-year DEM.

    The function groups ``df_gl`` by (``glacier_col``, ``year_col``) and, for each group,
    opens the corresponding Zarr dataset located at::

        <path_xr_grids>/<file_pattern>

    It then samples the DEM (or other raster variable) at each point location using
    nearest-neighbor selection and flags points where::

        abs(POINT_ELEVATION - DEM_value) >= threshold

    Parameters
    ----------
    df_gl : pandas.DataFrame
        Input point table. Must contain columns:
        - ``POINT_LON`` : float, longitude in degrees (WGS84)
        - ``POINT_LAT`` : float, latitude in degrees (WGS84)
        - ``POINT_ELEVATION`` : float, point elevation in meters
        - ``year_col`` (default ``"YEAR"``)
        - ``glacier_col`` (default ``"GLACIER"``)

        The index is preserved; returned indices refer to the original ``df_gl`` index.
    path_xr_grids : str
        Directory containing per-glacier-per-year Zarr datasets.
    var_name : str, optional
        Variable to sample from the Zarr dataset (default ``"masked_elev"``).
        This should represent the DEM elevation if you want an elevation check.
    lon_name : str, optional
        Name of the longitude coordinate in the Zarr datasets (default ``"lon"``).
    lat_name : str, optional
        Name of the latitude coordinate in the Zarr datasets (default ``"lat"``).
    year_col : str, optional
        Name of the year column in ``df_gl`` (default ``"YEAR"``).
    glacier_col : str, optional
        Name of the glacier identifier column in ``df_gl`` (default ``"GLACIER"``).
    threshold : float, optional
        Absolute difference threshold in meters above which a point is considered a mismatch
        (default 500.0).
    file_pattern : str, optional
        Filename pattern used to locate the Zarr dataset for each group.
        Must include ``{glacier}`` and ``{year}`` placeholders.
        Default: ``"{glacier}_{year}.zarr"``.
    strict : bool, optional
        If True, raise ``FileNotFoundError`` when a required glacier-year Zarr is missing.
        If False, missing glacier-year datasets are skipped (default False).

    Returns
    -------
    mismatch_idx : pandas.Index
        Index values from ``df_gl`` corresponding to mismatching points.
        Empty if no mismatches are found.
    mismatch_df : pandas.DataFrame
        Subset of ``df_gl`` containing mismatching rows, with two extra columns:
        - ``DEM_elv`` : sampled DEM values at the point locations
        - ``elev_diff`` : ``POINT_ELEVATION - DEM_elv``

        The result is sorted by ``elev_diff`` ascending. Empty if no mismatches are found.

    Raises
    ------
    KeyError
        If required columns are missing from ``df_gl`` or if the Zarr dataset lacks the
        specified coordinates/variable.
    FileNotFoundError
        If ``path_xr_grids`` does not exist, or if ``strict=True`` and any required
        glacier-year Zarr dataset is missing.

    Notes
    -----
    - Sampling uses nearest-neighbor selection (`xarray.DataArray.sel(..., method="nearest")`).
      This assumes your points and the Zarr grids share the same CRS (typically WGS84 lon/lat).
    - DEM values can be NaN outside the glacier mask; such rows are dropped before thresholding.
    - Zarr datasets are cached per (glacier, year) within the call to avoid re-opening repeatedly.

    Examples
    --------
    >>> idx, mm = find_mismatch_by_year(df, "/path/to/zarrs", threshold=300)
    >>> mm.head()
    """
    required_cols = {"POINT_LON", "POINT_LAT", "POINT_ELEVATION", year_col, glacier_col}
    missing = required_cols - set(df_gl.columns)
    if missing:
        raise KeyError(f"df_gl is missing required columns: {sorted(missing)}")

    base = Path(path_xr_grids)
    if not base.exists():
        raise FileNotFoundError(f"Directory not found: {base}")

    all_mismatch_idx = []
    mismatch_frames = []

    # Cache opened datasets per (glacier, year) to avoid re-opening
    ds_cache = {}

    # Work on current index space (no reset), so returned indices match df_gl
    grouped = df_gl.groupby([glacier_col, year_col], sort=False)

    for (glacier, year), df_group in grouped:
        # Build path for this glacier-year
        fname = file_pattern.format(glacier=str(glacier), year=int(year))
        zarr_path = base / fname

        if not zarr_path.exists():
            msg = f"Missing DEM for glacier '{glacier}', year {year}: {zarr_path}"
            if strict:
                raise FileNotFoundError(msg)
            else:
                # Skip gracefully
                # print(msg)
                continue

        # Open or reuse the dataset
        key = (glacier, int(year))
        if key not in ds_cache:
            ds_cache[key] = xr.open_zarr(str(zarr_path))
        ds = ds_cache[key]

        # Sanity checks
        if lon_name not in ds.coords or lat_name not in ds.coords:
            raise KeyError(
                f"Dataset {zarr_path} missing coords '{lon_name}'/'{lat_name}'. "
                "Rename coords or pass lon_name/lat_name correctly."
            )
        if var_name not in ds.variables:
            raise KeyError(f"Variable '{var_name}' not found in dataset {zarr_path}.")

        # Vectorized nearest sampling for this group
        lons = xr.DataArray(df_group["POINT_LON"].to_numpy(), dims="points")
        lats = xr.DataArray(df_group["POINT_LAT"].to_numpy(), dims="points")
        xr_elev_da = ds[var_name].sel(
            {lon_name: lons, lat_name: lats}, method="nearest"
        )
        xr_elev = xr_elev_da.to_numpy()

        pt_elev = df_group["POINT_ELEVATION"].to_numpy()
        elev_diff = pt_elev - xr_elev

        # Assemble a small result frame aligned to original indices
        res = df_group.copy()
        res["DEM_elv"] = xr_elev
        res["elev_diff"] = elev_diff

        # Drop NaNs (on either DEM or diff)
        res = res.dropna(subset=["DEM_elv", "elev_diff"])

        # Flag mismatches (abs diff >= threshold)
        mismatch_mask = np.abs(res["elev_diff"]) >= threshold

        if mismatch_mask.any():
            # Collect indices and rows
            these_idx = res.index[mismatch_mask]
            all_mismatch_idx.append(these_idx)
            mismatch_frames.append(res.loc[these_idx])

    if len(all_mismatch_idx) == 0:
        # Nothing found
        return pd.Index([], dtype=df_gl.index.dtype), pd.DataFrame(
            columns=list(df_gl.columns) + ["DEM_elv", "elev_diff"]
        )

    mismatch_idx = (
        all_mismatch_idx[0].append(all_mismatch_idx[1:])
        if len(all_mismatch_idx) > 1
        else all_mismatch_idx[0]
    )
    mismatch_df = pd.concat(mismatch_frames, axis=0).sort_values(
        by="elev_diff", ascending=True
    )

    return mismatch_idx, mismatch_df


def reconcile_points_by_year(
    df: pd.DataFrame,
    path_xr_grids: str,
    var_name: str = "masked_elev",
    lon_name: str = "lon",
    lat_name: str = "lat",
    year_col: str = "YEAR",
    glacier_col: str = "GLACIER",
    point_elev_col: str = "POINT_ELEVATION",
    threshold: float = 500.0,
    file_pattern: str = "{glacier}_{year}.zarr",
    replace_glaciers: Optional[Iterable[str]] = None,  # e.g., {"aletsch"}
    strict: bool = False,
    verbose: bool = True,
):
    """
    Clean or correct stake-point elevations by comparing them to glacier-year DEM values.

    This function is a "repair-or-drop" companion to :func:`find_mismatch_by_year`.
    It groups the input table by (``glacier_col``, ``year_col``) and for each group:

    1) Resolves the DEM Zarr path:
       - Tries the exact glacier-year file: ``<glacier>_<year>.zarr`` (via ``file_pattern``).
       - If missing, falls back to the *earliest* available ``<glacier>_YYYY.zarr`` in the
         directory (fallback).
       - If no DEMs exist for a glacier:
         - increments a missing counter
         - raises if ``strict=True``, otherwise skips the group.

    2) Samples the DEM variable at point locations (nearest-neighbor).

    3) Flags mismatches where::

           abs(point_elev_col - DEM_value) >= threshold

       For mismatching rows:
       - if glacier name is in ``replace_glaciers`` (case-insensitive), replace the point
         elevation with the DEM value;
       - otherwise drop the mismatching rows.

    Parameters
    ----------
    df : pandas.DataFrame
        Input point table. Must contain columns:
        - ``glacier_col`` (default ``"GLACIER"``)
        - ``year_col`` (default ``"YEAR"``)
        - ``POINT_LON`` and ``POINT_LAT`` (WGS84 lon/lat)
        - ``point_elev_col`` (default ``"POINT_ELEVATION"``)

        The function resets the index internally to ensure clean dropping/replacement.
    path_xr_grids : str
        Directory containing per-glacier-per-year Zarr datasets.
    var_name : str, optional
        Variable to sample from each Zarr dataset (default ``"masked_elev"``).
    lon_name : str, optional
        Longitude coordinate name in the Zarr datasets (default ``"lon"``).
    lat_name : str, optional
        Latitude coordinate name in the Zarr datasets (default ``"lat"``).
    year_col : str, optional
        Column name holding the year (default ``"YEAR"``).
    glacier_col : str, optional
        Column name holding glacier identifier (default ``"GLACIER"``).
    point_elev_col : str, optional
        Column name holding point elevation to reconcile (default ``"POINT_ELEVATION"``).
    threshold : float, optional
        Absolute difference threshold in meters used to define mismatches (default 500.0).
    file_pattern : str, optional
        Pattern used to locate the glacier-year Zarr dataset.
        Must include ``{glacier}`` and ``{year}`` placeholders.
    replace_glaciers : iterable of str or None, optional
        Glaciers (names) for which mismatching points are corrected (elevation replaced)
        rather than dropped. Matching is case-insensitive.
        Example: ``{"aletsch", "rhone"}``.
    strict : bool, optional
        If True, raise when no DEM exists for a glacier (i.e., no ``<glacier>_YYYY.zarr`` files).
        If False, missing DEM groups are skipped (default False).
    verbose : bool, optional
        If True, prints fallback usage and a per-glacier summary of dropped/replaced counts.

    Returns
    -------
    df_clean : pandas.DataFrame
        Cleaned DataFrame after dropping mismatches and/or replacing elevations. Index is reset.
    df_mismatch : pandas.DataFrame
        DataFrame of all mismatching rows (before drop/replace) with two additional columns:
        - ``DEM_elv`` : sampled DEM value
        - ``elev_diff`` : ``point_elev_col - DEM_elv``

        Sorted by ``elev_diff`` ascending. Empty if no mismatches.
    summary : pandas.DataFrame
        Per-glacier summary table with columns:
        - ``GLACIER``
        - ``dropped`` : number of points removed due to mismatch
        - ``replaced`` : number of points whose elevation was replaced by DEM
        - ``fallback_groups_used`` : number of (glacier, year) groups that used fallback DEMs
        - ``missing_dem_groups`` : number of groups skipped due to no DEMs existing for the glacier

    Raises
    ------
    KeyError
        If required columns are missing in the input DataFrame or if Zarr datasets do not
        contain required coordinates/variables.
    FileNotFoundError
        If ``path_xr_grids`` does not exist, or if ``strict=True`` and a glacier has no DEMs.

    Notes
    -----
    - Sampling uses nearest-neighbor selection; this assumes CRS consistency between points
      and grids (typically WGS84 lon/lat).
    - If the DEM contains NaNs (e.g., outside glacier mask), those rows are ignored for
      mismatch detection (dropped before thresholding).
    - The fallback strategy picks the *earliest year* available for the glacier, which is
      robust but may introduce temporal mismatch. Consider choosing the closest year instead
      if you want temporal consistency.

    Examples
    --------
    >>> df_clean, df_mm, summary = reconcile_points_by_year(
    ...     df_points, "/path/to/zarrs", threshold=300, replace_glaciers={"aletsch"}
    ... )
    >>> summary
    """
    # --- validations ---
    required = {glacier_col, year_col, "POINT_LON", "POINT_LAT", point_elev_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Input df is missing columns: {sorted(missing)}")

    base = Path(path_xr_grids)
    if not base.exists():
        raise FileNotFoundError(f"Directory not found: {base}")

    replace_set = set(g.lower() for g in (replace_glaciers or set()))

    # Work on a unique-index copy so drops/replacements are precise
    df_clean = df.reset_index(drop=True).copy()

    # Caches and accumulators
    ds_cache: dict[Tuple[str, int], xr.Dataset] = {}
    mismatch_frames = []
    dropped_indices = []
    drop_counts = defaultdict(int)
    replace_counts = defaultdict(int)
    missing_dem_groups = defaultdict(int)
    fallback_counts = defaultdict(int)

    def _find_existing_dem_path(
        base_dir: Path, glacier_name: str, requested_year: int, patt: str
    ) -> Tuple[Optional[Path], Optional[int], bool]:
        """
        Return (path, used_year, used_fallback).
          - Exact match -> (exact_path, requested_year, False)
          - Else earliest glacier_YYYY.zarr -> (fallback_path, year, True)
          - Else -> (None, None, False)
        """
        exact_name = patt.format(glacier=str(glacier_name), year=int(requested_year))
        exact_path = base_dir / exact_name
        if exact_path.exists():
            return exact_path, int(requested_year), False

        rgx = re.compile(
            rf"^{re.escape(str(glacier_name))}_(\d{{4}})\.zarr$", re.IGNORECASE
        )
        candidates = []
        for entry in base_dir.iterdir():
            if not entry.name.lower().endswith(".zarr"):
                continue
            m = rgx.match(entry.name)
            if not m:
                continue
            try:
                y = int(m.group(1))
                candidates.append((y, entry))
            except ValueError:
                continue
        if not candidates:
            return None, None, False
        candidates.sort(key=lambda t: t[0])  # earliest year first
        y_min, path_min = candidates[0]
        return path_min, y_min, True

    # Iterate by (GLACIER, YEAR) groups; indices remain aligned with df_clean
    for (glacier, year), grp in df_clean.groupby([glacier_col, year_col], sort=False):
        # Resolve DEM path with fallback if needed
        zarr_path, used_year, used_fallback = _find_existing_dem_path(
            base_dir=base,
            glacier_name=str(glacier),
            requested_year=int(year),
            patt=file_pattern,
        )

        if zarr_path is None:
            if verbose:
                print(f"[WARN] No DEMs found at all for glacier='{glacier}' in {base}")
            missing_dem_groups[str(glacier)] += 1
            if strict:
                raise FileNotFoundError(
                    f"Missing DEM for glacier '{glacier}' (no files like {glacier}_YYYY.zarr)"
                )
            continue

        if used_fallback and verbose:
            print(
                f"[INFO] Fallback DEM for glacier='{glacier}': requested {year} -> using {used_year} ({zarr_path.name})"
            )
            fallback_counts[str(glacier)] += 1

        key = (str(glacier), int(used_year))
        if key not in ds_cache:
            ds_cache[key] = xr.open_zarr(str(zarr_path))
        ds = ds_cache[key]

        # sanity checks
        if lon_name not in ds.coords or lat_name not in ds.coords:
            raise KeyError(
                f"{zarr_path} missing coords '{lon_name}'/'{lat_name}'. "
                "Rename dataset coords or pass correct names."
            )
        if var_name not in ds.variables:
            raise KeyError(f"Variable '{var_name}' not found in {zarr_path}.")

        # vectorized nearest sampling
        lons = xr.DataArray(grp["POINT_LON"].to_numpy(), dims="points")
        lats = xr.DataArray(grp["POINT_LAT"].to_numpy(), dims="points")
        xr_elev = (
            ds[var_name]
            .sel({lon_name: lons, lat_name: lats}, method="nearest")
            .to_numpy()
        )

        pt_elev = grp[point_elev_col].to_numpy()
        elev_diff = pt_elev - xr_elev

        res = grp.copy()
        res["DEM_elv"] = xr_elev
        res["elev_diff"] = elev_diff

        # Remove rows where DEM or diff is NaN before thresholding
        res = res.dropna(subset=["DEM_elv", "elev_diff"])

        # mismatches for this (glacier, year)
        mask = np.abs(res["elev_diff"]) >= threshold
        if not mask.any():
            continue

        # rows to act on (indices in df_clean)
        mm = res.loc[mask]
        mismatch_frames.append(mm)

        gkey = str(glacier)
        if gkey.lower() in replace_set:
            # Replace POINT_ELEVATION with DEM value for those rows
            df_clean.loc[mm.index, point_elev_col] = mm["DEM_elv"].values
            replace_counts[gkey] += len(mm)
        else:
            # Drop mismatched rows
            dropped_indices.extend(mm.index.tolist())
            drop_counts[gkey] += len(mm)

    # Apply drops once
    if dropped_indices:
        df_clean = df_clean.drop(index=dropped_indices)

    # Build mismatch table
    if mismatch_frames:
        df_mismatch = pd.concat(mismatch_frames, axis=0).sort_values("elev_diff")
    else:
        df_mismatch = pd.DataFrame(columns=list(df.columns) + ["DEM_elv", "elev_diff"])

    # Tidy
    df_clean = df_clean.sort_index().reset_index(drop=True)

    # Summary dataframe
    glaciers = sorted(
        set(
            list(drop_counts.keys())
            + list(replace_counts.keys())
            + list(missing_dem_groups.keys())
            + list(fallback_counts.keys())
        )
    )
    summary = pd.DataFrame(
        {
            "GLACIER": glaciers,
            "dropped": [drop_counts[g] for g in glaciers],
            "replaced": [replace_counts[g] for g in glaciers],
            "fallback_groups_used": [fallback_counts[g] for g in glaciers],
            "missing_dem_groups": [missing_dem_groups[g] for g in glaciers],
        }
    )

    # Print per-glacier info
    if verbose and len(summary):
        print("\n=== Reconcile summary (per glacier) ===")
        for _, row in summary.iterrows():
            g = row["GLACIER"]
            d = int(row["dropped"])
            r = int(row["replaced"])
            f = int(row["fallback_groups_used"])
            m = int(row["missing_dem_groups"])
            msg = f"{g}: removed {d} point(s)"
            if r:
                msg += f", replaced {r} point(s)"
            if f:
                msg += f", fallback DEM groups: {f}"
            if m:
                msg += f", missing DEM groups: {m}"
            print(msg)

    return df_clean, df_mismatch, summary


def first_year_per_glacier(path_xr_grids: str) -> pd.DataFrame:
    """
    Find the earliest available DEM year per glacier from a directory of Zarr datasets.

    This scans ``path_xr_grids`` for entries matching the naming convention::

        <glacier>_<year>.zarr

    where ``<year>`` is a 4-digit integer. Glacier names may include underscores; the year is
    interpreted as the final underscore-separated token before ``.zarr``.

    Parameters
    ----------
    path_xr_grids : str
        Directory to scan for Zarr datasets.

    Returns
    -------
    pandas.DataFrame
        Table with one row per glacier and columns:
        - ``glacier`` : str, glacier name parsed from the filename
        - ``first_year`` : int, earliest year found for that glacier
        - ``first_year_path`` : str, absolute path to the corresponding Zarr entry

        The output is sorted by ``glacier`` then ``first_year``.

    Raises
    ------
    FileNotFoundError
        If ``path_xr_grids`` does not exist.

    Notes
    -----
    - Zarr stores are often directories; this function accepts any filesystem entry ending
      in ``.zarr``.
    - Files that do not match the expected pattern are ignored.

    Examples
    --------
    >>> first = first_year_per_glacier("/path/to/zarrs")
    >>> first.loc[first["glacier"] == "aletsch"]
    """
    p = Path(path_xr_grids)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {p}")

    # Match anything up to the last underscore, then 4-digit year, then .zarr
    # e.g. "aletsch_1951.zarr", "some_glacier_name_2008.zarr"
    pat = re.compile(r"^(?P<glacier>.+)_(?P<year>\d{4})\.zarr$", re.IGNORECASE)

    # Keep min year and path per glacier
    best = {}  # glacier -> (year, full_path)
    for entry in p.iterdir():
        # Zarr datasets are often directories; accept both dir and file if present
        if not entry.name.lower().endswith(".zarr"):
            continue
        m = pat.match(entry.name)
        if not m:
            continue
        glacier = m.group("glacier")
        try:
            year = int(m.group("year"))
        except ValueError:
            continue

        if glacier not in best or year < best[glacier][0]:
            best[glacier] = (year, str(entry.resolve()))

    if not best:
        # No valid matches found
        return pd.DataFrame(columns=["glacier", "first_year", "first_year_path"])

    rows = [
        {"glacier": g, "first_year": y, "first_year_path": path}
        for g, (y, path) in best.items()
    ]
    df = (
        pd.DataFrame(rows).sort_values(["glacier", "first_year"]).reset_index(drop=True)
    )
    return df
