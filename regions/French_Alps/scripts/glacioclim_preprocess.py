import pandas as pd
import numpy as np
from datetime import datetime
import pyproj
import xarray as xr
import zipfile
import logging
from tqdm import tqdm
from pathlib import Path

from regions.French_Alps.scripts.config_FR import *
from regions.French_Alps.scripts.helpers import *

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


# --- Preprocess --- #
def extract_glacioclim_files(path):
    """
    Extract GLACIOCLIM zipfiles into organized directories.
    """
    path = Path(path)
    glacioclim_dir = path.parent

    seasons = ["annual", "summer", "winter"]

    path.mkdir(parents=True, exist_ok=True)

    emptyfolder(path)

    for glacier_dir in glacioclim_dir.glob("*Glacier*"):
        glacier_name = glacier_dir.name
        print(f"\nProcessing {glacier_name}")

        for season in seasons:
            season_dir = glacier_dir / season
            if not season_dir.exists():
                print(f"  Skipping {season} - directory not found")
                continue

            zip_files = list(season_dir.glob("*.zip"))
            print(f"  Found {len(zip_files)} zip files in {season}")

            for zip_path in zip_files:
                extract_dir = path / glacier_name / season / zip_path.stem
                extract_dir.mkdir(parents=True, exist_ok=True)

                try:
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                        print(f"    Extracted {zip_path.name} to {extract_dir}")
                except Exception as e:
                    print(f"    Error extracting {zip_path.name}: {str(e)}")


def format_date(date_str):
    """
    Convert date strings like '3. 10. 2004' to 'yyyymmdd' format
    """
    date_str = str(date_str).strip()

    # Split by dots and remove spaces
    parts = [p.strip() for p in date_str.split(".") if p.strip()]

    # Extract parts and add leading zeros where needed
    day = parts[0].zfill(2)
    month = parts[1].zfill(2)
    year = parts[2]

    # Return in yyyymmdd format
    return f"{year}{month}{day}"


def extract_sarennes_data(all_sheets):
    """
    Extract winter, summer, and annual mass balance data from Sarennes Excel sheets
    Each sheet corresponds to a year and contains multiple stake measurements
    """
    sarennes_dfs = {}

    # Process each sheet (year)
    for sheet_name, df in all_sheets.items():

        year = sheet_name

        # Find rows with end_dates (these rows contain stake measurements)
        end_date_rows = df[(pd.notna(df["Unnamed: 13"])) & (df["Unnamed: 13"] != 0)]

        # Process each season (winter, summer, annual)
        for season_idx, season in enumerate(["winter", "summer", "annual"]):
            # Column mapping for balance values
            balance_col = f"Unnamed: {17 + season_idx}"

            # Date columns mapping based on season
            if season == "winter":
                from_date_col = "Unnamed: 11"  # start date
                to_date_col = "Unnamed: 12"  # spring date
            elif season == "summer":
                from_date_col = "Unnamed: 12"  # spring date
                to_date_col = "Unnamed: 13"  # end date
            else:  # annual
                from_date_col = "Unnamed: 11"  # start date
                to_date_col = "Unnamed: 13"  # end date

            # Create rows for this season
            rows = []
            for i, (_, row) in enumerate(end_date_rows.iterrows(), 1):
                # Skip if no balance value
                balance_value = row.get(balance_col)
                if pd.isna(balance_value) or balance_value == 0:
                    continue

                # Create data row
                rows.append(
                    {
                        "POINT_ID": f"sarennes_complete_{season}_{year}_{i}",
                        "x_lambert3": row["Unnamed: 14"],
                        "y_lambert3": row["Unnamed: 15"],
                        "POINT_ELEVATION": row["Unnamed: 16"],
                        "FROM_DATE": format_date(row.get(from_date_col)),
                        "TO_DATE": format_date(row.get(to_date_col)),
                        "POINT_BALANCE": row[balance_col],
                        "GLACIER": "sarennes",
                        "PERIOD": season,
                        "GLACIER_ZONE": "complete",
                    }
                )

            # Create DataFrame if we have rows
            if rows:
                key = f"sarennes_complete_{season}_{year}"
                sarennes_dfs[key] = pd.DataFrame(rows)

    return sarennes_dfs


def extract_blanc_data(blanc_data):
    """
    Extract annual mass-balance stake series for Glacier Blanc from a wide-format table.

    The expected input layout is the "Blanc" Excel sheet format where:
    - Row 0 contains X coordinates for each stake (one stake per column)
    - Row 1 contains Y coordinates for each stake
    - Row 2 contains elevations (Z) for each stake
    - Rows 3..(n-2) contain annual mass-balance values by year (years in first column)
    - The last column is ignored (often metadata or comments)
    - A column named "balise" is skipped if present

    The function creates one output row per (stake, year) with WGMS-like fields and
    hydrological-year date bounds:
    - FROM_DATE = Oct 1 of (year-1)
    - TO_DATE   = Sep 30 of (year)

    Parameters
    ----------
    blanc_data : pandas.DataFrame
        DataFrame containing the Glacier Blanc sheet content in the layout described above.

    Returns
    -------
    pandas.DataFrame
        Long-format table with columns:
        - POINT_ID
        - x_lambert3, y_lambert3
        - POINT_ELEVATION
        - FROM_DATE, TO_DATE (YYYYMMDD strings)
        - POINT_BALANCE (float)
        - GLACIER ('glacier_blanc')
        - PERIOD ('annual')
        - GLACIER_ZONE ('complete')

    Notes
    -----
    - Balance values equal to 'ND' (case-insensitive) are skipped.
    - Decimal commas are converted to decimal points before casting to float.
    """
    rows = []

    # Get coordinate rows (X, Y, Z are in rows 0, 1, 2)
    x_row = blanc_data.iloc[0]
    y_row = blanc_data.iloc[1]
    z_row = blanc_data.iloc[2]

    # Get year rows (from row 3 onwards)
    data_rows = blanc_data.iloc[3:-1]  # exclude first 3 and last row

    # Iterate through each stake (column)
    for col_idx, stake_id in enumerate(blanc_data.columns[:-1]):  # exclude last column

        if stake_id == "balise":
            continue

        # Get coordinates for this stake
        x_coord = x_row.iloc[col_idx]
        y_coord = y_row.iloc[col_idx]
        elevation = z_row.iloc[col_idx]

        # Iterate through each year for this stake
        for year_idx in range(len(data_rows)):
            year = data_rows.iloc[year_idx, 0]  # First column contains the year
            balance_value = data_rows.iloc[year_idx, col_idx]

            # Skip if balance value is ND
            if str(balance_value).upper() == "ND":
                continue

            # Clean balance value (replace comma with dot for decimals)
            try:
                balance_clean = float(str(balance_value).replace(",", "."))
            except:
                continue

            # Create hydrological year dates (Oct 1 to Sep 30)
            from_date = pd.to_datetime(f"{year-1}-10-01").strftime("%Y%m%d")
            to_date = pd.to_datetime(f"{year}-09-30").strftime("%Y%m%d")

            # Create data row
            rows.append(
                {
                    "POINT_ID": f"glacier_blanc_complete_annual_{year}_{stake_id}",
                    "x_lambert3": x_coord,
                    "y_lambert3": y_coord,
                    "POINT_ELEVATION": elevation,
                    "FROM_DATE": from_date,
                    "TO_DATE": to_date,
                    "POINT_BALANCE": balance_clean,
                    "GLACIER": "glacier_blanc",
                    "PERIOD": "annual",
                    "GLACIER_ZONE": "complete",
                }
            )
    return pd.DataFrame(rows)


def lamberttoWGS84(df, lambert_type="III"):
    """Converts from x & y Lambert III (EPSG:27563) or Lambert II (EPSG:27562) to lat/lon WGS84 (EPSG:4326) coordinate system"""

    if lambert_type == "II":
        transformer = pyproj.Transformer.from_crs(
            "EPSG:27562", "EPSG:4326", always_xy=True
        )
    else:
        transformer = pyproj.Transformer.from_crs(
            "EPSG:27563", "EPSG:4326", always_xy=True
        )

    # Transform to Latitude and Longitude (WGS84)
    lon, latitude = transformer.transform(df.x_lambert3, df.y_lambert3)

    df["lat"] = latitude
    df["lon"] = lon
    df.drop(["x_lambert3", "y_lambert3"], axis=1, inplace=True)
    return df


def lambert_transform(df):
    """
    Transform coordinates from Lambert to WGS84

    Columns are named differently depending on the glacier
    """
    transformed_df = df.copy()

    # Coordinate transformation
    for key, value in transformed_df.items():
        if key.startswith(("mdg", "Argentiere")):
            value = value.rename(
                columns={"x_lambert2e": "x_lambert3", "y_lambert2e": "y_lambert3"}
            )
            transformed_df[key] = lamberttoWGS84(value, "II")
        # 3 years in the Saint-Sorlin are falsely named lambert2.
        elif (
            key == "stso_winter_smb_accu_2019"
            or "stso_winter_smb_accu_2019"
            or "stso_winter_smb_accu_2019"
        ):
            value = value.rename(
                columns={"x_lambert2e": "x_lambert3", "y_lambert2e": "y_lambert3"}
            )
            transformed_df[key] = lamberttoWGS84(value)
        else:
            transformed_df[key] = lamberttoWGS84(value)

        lat_check = value["lat"].between(45, 46).all()
        lon_check = value["lon"].between(6, 7.5).all()

        if not (lat_check and lon_check):
            print(f"\nWarning for {key}:")
            if not lat_check:
                print(
                    f"Latitude range: {value['lat'].min():.4f} to {value['lat'].max():.4f}"
                )
            if not lon_check:
                print(
                    f"Longitude range: {value['lon'].min():.4f} to {value['lon'].max():.4f}"
                )
    return transformed_df


def transform_WGMS_df(df, key):
    """
    Transform df into WGMS format
    """

    new_df = df.copy()

    # Extract glacier name (everything before first '_')
    glacier_name = key.split("_")[0]

    # Determine period and balance type from key
    if "winter" in key:
        period = "winter"
        balance_col = "winter_smb"
    elif "summer" in key:
        period = "summer"
        balance_col = "summer_smb"
    elif "annual" in key:
        period = "annual"
        balance_col = "annual_smb"
    else:
        print("ERROR")

    # Create POINT_ID
    new_df["POINT_ID"] = (
        key
        + "_"
        + new_df["profile_name"].astype(str)
        + "_"
        + "setup"
        + new_df["stake_year_setup"].astype(str)
        + "_"
        + new_df["stake_number"].astype(str)
    )

    # Create dates
    new_df["FROM_DATE"] = (
        new_df["year_start"].astype(str)
        + new_df["month_start"].astype(str).str.zfill(2)
        + new_df["day_start"].astype(str).str.zfill(2)
    ).astype(int)

    new_df["TO_DATE"] = (
        new_df["year_end"].astype(str)
        + new_df["month_end"].astype(str).str.zfill(2)
        + new_df["day_end"].astype(str).str.zfill(2)
    ).astype(int)

    # Create final DataFrame with required columns
    final_df = pd.DataFrame(
        {
            "POINT_ID": new_df["POINT_ID"],
            "POINT_LAT": new_df["lat"],
            "POINT_LON": new_df["lon"],
            "POINT_ELEVATION": new_df["altitude"],
            "FROM_DATE": new_df["FROM_DATE"],
            "TO_DATE": new_df["TO_DATE"],
            "POINT_BALANCE": new_df[balance_col],
            "GLACIER": glacier_name,
            "PERIOD": period,
            "GLACIER_ZONE": new_df["profile_name"],
        }
    )

    return final_df


def check_period_consistency(df):
    """
    Checks if date ranges make sense for annual, winter, and summer periods.
    Returns dataframes with inconsistent periods.
    """
    df_check = df.copy()

    # Convert dates to datetime objects
    df_check["FROM_DATE_DT"] = pd.to_datetime(df_check["FROM_DATE"], format="%Y%m%d")
    df_check["TO_DATE_DT"] = pd.to_datetime(df_check["TO_DATE"], format="%Y%m%d")

    df_check["MONTH_DIFF"] = (
        (df_check["TO_DATE_DT"].dt.year - df_check["FROM_DATE_DT"].dt.year) * 12
        + df_check["TO_DATE_DT"].dt.month
        - df_check["FROM_DATE_DT"].dt.month
    )

    # Define expected ranges
    ranges = {
        "annual": (9, 15),
        "winter": (4, 9),
        "summer": (3, 8),
    }

    inconsistent_dfs = {}

    for period, (min_m, max_m) in ranges.items():
        period_df = df_check[df_check["PERIOD"] == period]
        inconsistent = period_df[
            (period_df["MONTH_DIFF"] < min_m) | (period_df["MONTH_DIFF"] > max_m)
        ]

        total = len(period_df)
        n_bad = len(inconsistent)

        if total == 0:
            pct = 0.0
        else:
            pct = n_bad / total * 100

        print(
            f"{period.capitalize()} periods: {n_bad} out of {total} ({pct:.1f}%) are inconsistent"
        )

        inconsistent_dfs[period] = inconsistent

    return (
        inconsistent_dfs["annual"],
        inconsistent_dfs["winter"],
        inconsistent_dfs["summer"],
    )


def flag_elevation_mismatch(df, threshold=400):
    """
    Flag rows where POINT_ELEVATION differs from DEM elevation ('topo')
    by more than a given threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'POINT_ELEVATION' and 'topo'.
    threshold : float, optional
        Maximum allowed absolute elevation difference (meters).
        Default is 400 m.

    Returns
    -------
    df_out : pandas.DataFrame
        Copy of input dataframe with:
        - 'elev_diff' : POINT_ELEVATION - topo
        - 'elev_mismatch' : True if abs(diff) > threshold
    mismatches : pandas.DataFrame
        Subset of rows where mismatch is True.
    """
    df_out = df.copy()

    df_out["elev_diff"] = df_out["POINT_ELEVATION"] - df_out["topo"]
    df_out["elev_mismatch"] = df_out["elev_diff"].abs() > threshold

    mismatches = df_out[df_out["elev_mismatch"]]

    print(
        f"{len(mismatches)} out of {len(df_out)} points "
        f"({len(mismatches)/len(df_out)*100:.2f}%) exceed ±{threshold} m elevation difference."
    )

    return df_out, mismatches


# ---- SVF functions ----
def add_svf_from_rgi_zarr(
    df_pmb_topo,
    path_masked_grids,
    rgi_col="RGIId",
    lon_col="POINT_LON",
    lat_col="POINT_LAT",
    svf_var="svf",
    out_col="svf",
):
    """
    Add sky-view factor (SVF) to a PMB point dataframe by nearest-neighbor sampling
    from per-glacier Zarr grids named <RGIId>.zarr.

    Parameters
    ----------
    df_pmb_topo : pandas.DataFrame
        Must contain columns [rgi_col, lon_col, lat_col].
    path_masked_grids : str
        Directory containing <RGIId>.zarr datasets.
    rgi_col, lon_col, lat_col : str
        Column names in df_pmb_topo.
    svf_var : str
        Variable name in the Zarr dataset (default "svf").
    out_col : str
        Output column name in the dataframe.

    Returns
    -------
    pandas.DataFrame
        Copy of df_pmb_topo (filtered to glaciers that exist on disk) with added `out_col`.
    """

    # Which zarr files exist?
    zarr_ids = {f[:-5] for f in os.listdir(path_masked_grids) if f.endswith(".zarr")}

    # Keep only rows whose RGIId exists on disk
    df = df_pmb_topo[df_pmb_topo[rgi_col].isin(zarr_ids)].copy()

    # Initialize output
    df[out_col] = np.nan

    # Group by glacier (RGIId) and sample in bulk
    for rgi_id, group in df.groupby(rgi_col):
        file_path = os.path.join(path_masked_grids, f"{rgi_id}.zarr")
        try:
            # Zarr: open with open_zarr (your example), works well here
            ds = xr.open_zarr(file_path)

            # nearest-neighbor sample
            lon = group[lon_col].to_numpy()
            lat = group[lat_col].to_numpy()

            stake = ds[svf_var].sel(
                lon=xr.DataArray(lon, dims="points"),
                lat=xr.DataArray(lat, dims="points"),
                method="nearest",
            )

            # write back
            df.loc[group.index, out_col] = stake.to_numpy()

        except Exception as e:
            # keep going; column stays NaN for that glacier
            print(f"[add_svf_from_rgi_zarr] {rgi_id}: {type(e).__name__}: {e}")
            continue

    return df


def plot_missing_svf_for_all_glaciers(
    df_with_svf,
    path_masked_xr,
    rgi_col="RGIId",
    lon_col="POINT_LON",
    lat_col="POINT_LAT",
    svf_col="svf",  # column in df
    svf_var="svf",  # var name in zarr
    plot_valid_points=True,
    save_dir=None,
):
    """
    Plot SVF rasters and overlay stake points with missing SVF values for each glacier.

    For each glacier (grouped by `rgi_col`) where at least one point has missing SVF
    (`df_with_svf[svf_col]` is NaN), the function:
    - opens the corresponding Zarr dataset `<RGIId>.zarr` in `path_masked_xr`
    - plots the SVF raster (`svf_var`) on its lon/lat grid
    - overlays the missing-SVF points in red
    - optionally overlays valid points (non-NaN SVF) in white for context
    - either shows each figure interactively or saves PNGs to `save_dir`

    Parameters
    ----------
    df_with_svf : pandas.DataFrame
        Point dataset containing at least columns:
        - `rgi_col` (e.g., 'RGIId')
        - `lon_col`, `lat_col` (point coordinates)
        - `svf_col` (SVF values, may contain NaNs)
    path_masked_xr : str
        Directory containing per-glacier Zarr datasets named `<RGIId>.zarr`.
    rgi_col, lon_col, lat_col : str, optional
        Column names in `df_with_svf` for glacier id and coordinates.
    svf_col : str, optional
        Column name in `df_with_svf` holding SVF values.
    svf_var : str, optional
        Variable name inside the Zarr datasets (default 'svf').
    plot_valid_points : bool, optional
        If True, plot non-missing SVF points as well (white markers).
    save_dir : str or None, optional
        If provided, figures are saved as `<RGIId>_svf_missing.png` into this folder.
        If None, figures are displayed via `plt.show()`.

    Returns
    -------
    None

    Notes
    -----
    This is a diagnostic plot. Missing SVF often indicates that points fall outside
    the valid/masked SVF area (e.g., off-glacier pixels) or that glacier IDs / CRS
    do not match the raster grids.
    """
    df_missing = df_with_svf[df_with_svf[svf_col].isna()].copy()
    glaciers = sorted(df_missing[rgi_col].unique())

    print(f"Plotting {len(glaciers)} glaciers with missing SVF points.")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for rgi_id in glaciers:
        zpath = os.path.join(path_masked_xr, f"{rgi_id}.zarr")
        if not os.path.exists(zpath):
            print(f"[skip] missing zarr: {rgi_id}")
            continue

        ds = xr.open_zarr(zpath)
        da = ds[svf_var]

        miss = df_missing[df_missing[rgi_col] == rgi_id]
        lon_m = miss[lon_col].to_numpy()
        lat_m = miss[lat_col].to_numpy()

        if plot_valid_points:
            ok = df_with_svf[
                (df_with_svf[rgi_col] == rgi_id) & (~df_with_svf[svf_col].isna())
            ]
            lon_ok = ok[lon_col].to_numpy()
            lat_ok = ok[lat_col].to_numpy()

        fig, ax = plt.subplots(figsize=(7, 6))

        # SVF raster
        da.plot(ax=ax, x="lon", y="lat", robust=True, add_colorbar=True)

        # Missing points (red)
        ax.scatter(
            lon_m,
            lat_m,
            s=22,
            c="red",
            edgecolor="k",
            linewidth=0.3,
            label=f"missing svf (n={len(miss)})",
        )

        # Valid points (white) for context
        if plot_valid_points and len(ok) > 0:
            ax.scatter(
                lon_ok,
                lat_ok,
                s=12,
                c="white",
                edgecolor="k",
                linewidth=0.2,
                alpha=0.7,
                label=f"valid svf (n={len(ok)})",
            )

        ax.set_title(f"{rgi_id} — SVF with missing points")
        ax.legend(loc="upper right")

        # Zoom to raster extent
        ax.set_xlim(float(da["lon"].min()), float(da["lon"].max()))
        ax.set_ylim(float(da["lat"].min()), float(da["lat"].max()))

        plt.tight_layout()

        if save_dir is not None:
            out = os.path.join(save_dir, f"{rgi_id}_svf_missing.png")
            plt.savefig(out, dpi=200)
            plt.close(fig)
        else:
            plt.show()


def _nearest_index_1d(coord_vals, query_vals):
    """
    Return nearest indices in a sorted 1D coord array for query values.
    Works for increasing or decreasing coords.
    """
    coord_vals = np.asarray(coord_vals)
    q = np.asarray(query_vals)

    # ensure increasing for searchsorted
    increasing = coord_vals[0] < coord_vals[-1]
    if not increasing:
        coord_inc = coord_vals[::-1]
        idx = np.searchsorted(coord_inc, q)
        idx = np.clip(idx, 1, len(coord_inc) - 1)
        left = coord_inc[idx - 1]
        right = coord_inc[idx]
        choose_right = (q - left) > (right - q)
        out = idx - 1 + choose_right
        # map back to original (reversed) indices
        out = (len(coord_vals) - 1) - out
        return out.astype(int)

    idx = np.searchsorted(coord_vals, q)
    idx = np.clip(idx, 1, len(coord_vals) - 1)
    left = coord_vals[idx - 1]
    right = coord_vals[idx]
    choose_right = (q - left) > (right - q)
    out = idx - 1 + choose_right
    return out.astype(int)


def _fill_nearest_valid(values2d, iy, ix, max_radius=30):
    """
    For a single point index (iy, ix), if values2d[iy,ix] is NaN,
    search outward in square windows until a non-NaN is found.
    Return that value (nearest by Euclidean distance in index space).
    """
    if not np.isnan(values2d[iy, ix]):
        return values2d[iy, ix]

    ny, nx = values2d.shape

    for r in range(1, max_radius + 1):
        y0 = max(0, iy - r)
        y1 = min(ny, iy + r + 1)
        x0 = max(0, ix - r)
        x1 = min(nx, ix + r + 1)

        window = values2d[y0:y1, x0:x1]
        mask = ~np.isnan(window)
        if not mask.any():
            continue

        # indices of valid cells in the window
        vy, vx = np.where(mask)

        # convert to full-array indices
        vy_full = vy + y0
        vx_full = vx + x0

        # choose nearest valid by distance in index space
        dy = vy_full - iy
        dx = vx_full - ix
        k = np.argmin(dy * dy + dx * dx)

        return values2d[vy_full[k], vx_full[k]]

    return np.nan


def add_svf_nearest_valid(
    df_pmb_topo,
    path_masked_grids,
    rgi_col="RGIId",
    lon_col="POINT_LON",
    lat_col="POINT_LAT",
    svf_var="svf",
    out_col="svf",
    max_radius=30,
):
    """
    Add SVF from per-glacier zarr. For points that land on NaN (masked),
    fill using the nearest non-NaN SVF pixel within `max_radius` cells.
    """
    zarr_ids = {f[:-5] for f in os.listdir(path_masked_grids) if f.endswith(".zarr")}
    df = df_pmb_topo[df_pmb_topo[rgi_col].isin(zarr_ids)].copy()
    df[out_col] = np.nan

    for rgi_id, group in df.groupby(rgi_col):
        zpath = os.path.join(path_masked_grids, f"{rgi_id}.zarr")
        try:
            ds = xr.open_zarr(zpath)
            da = ds[svf_var]

            # assume 2D field on (lat, lon) with 1D coords lat/lon
            lons = da["lon"].values
            lats = da["lat"].values
            arr = da.values  # (lat, lon) typically

            qlon = group[lon_col].to_numpy()
            qlat = group[lat_col].to_numpy()

            ix = _nearest_index_1d(lons, qlon)
            iy = _nearest_index_1d(lats, qlat)

            sampled = arr[iy, ix].astype(float)

            # fill NaNs by nearest valid pixel
            nan_idx = np.where(np.isnan(sampled))[0]
            if len(nan_idx) > 0:
                for j in nan_idx:
                    sampled[j] = _fill_nearest_valid(
                        arr, int(iy[j]), int(ix[j]), max_radius=max_radius
                    )

            df.loc[group.index, out_col] = sampled

        except Exception as e:
            print(f"[add_svf_nearest_valid] {rgi_id}: {type(e).__name__}: {e}")
            continue

    return df


def plot_glacier_svf_with_points(
    df_with_svf,
    path_masked_xr,
    rgi_col="RGIId",
    lon_col="POINT_LON",
    lat_col="POINT_LAT",
    svf_col="svf",  # column in df
    svf_var="svf",  # variable in zarr
    save_dir=None,
    dpi=200,
):
    """
    Plot SVF raster maps and overlay point SVF values for each glacier.

    For each glacier found in `df_with_svf[rgi_col]`, the function:
    - opens the per-glacier Zarr dataset `<RGIId>.zarr` from `path_masked_xr`
    - plots the SVF raster (`svf_var`) on its lon/lat grid
    - overlays stake points colored by their assigned SVF values in `df_with_svf[svf_col]`
    - optionally marks points still missing SVF (NaNs) as red 'x'
    - either displays figures interactively or saves them to disk

    Parameters
    ----------
    df_with_svf : pandas.DataFrame
        Point dataset with columns:
        - `rgi_col` : glacier id matching Zarr filenames (e.g., 'RGIId')
        - `lon_col`, `lat_col` : point coordinates (WGS84 lon/lat)
        - `svf_col` : assigned SVF values (float; may include NaNs)
    path_masked_xr : str
        Directory containing per-glacier Zarr datasets named `<RGIId>.zarr`.
    rgi_col, lon_col, lat_col : str, optional
        Column names in `df_with_svf`.
    svf_col : str, optional
        Column holding SVF values to color the points.
    svf_var : str, optional
        Variable name inside the Zarr datasets (default 'svf').
    save_dir : str or None, optional
        If provided, save figures as `<RGIId>_svf_points.png` in this folder.
        If None, figures are displayed via `plt.show()`.
    dpi : int, optional
        Resolution for saved figures.

    Returns
    -------
    None

    Notes
    -----
    The raster colorbar (background) and point colorbar (assigned SVF) may not match
    exactly if the raster is masked or uses different scaling; the point colorbar
    reflects values actually assigned to the stake locations.
    """
    glaciers = sorted(df_with_svf[rgi_col].dropna().unique())

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    print(f"Plotting {len(glaciers)} glaciers.")

    for rgi_id in glaciers:
        gl_name = df_with_svf[df_with_svf[rgi_col] == rgi_id]["GLACIER"].iloc[0]
        zpath = os.path.join(path_masked_xr, f"{rgi_id}.zarr")
        if not os.path.exists(zpath):
            print(f"[skip] missing zarr: {rgi_id}")
            continue

        try:
            ds = xr.open_zarr(zpath)
            da = ds[svf_var]

            g = df_with_svf[df_with_svf[rgi_col] == rgi_id]
            lon = g[lon_col].to_numpy()
            lat = g[lat_col].to_numpy()
            vals = g[svf_col].to_numpy()

            ok = ~np.isnan(vals)
            miss = np.isnan(vals)

            fig, ax = plt.subplots(figsize=(7.5, 6.2))

            # Background raster
            im = da.plot(ax=ax, x="lon", y="lat", robust=True, add_colorbar=True)

            # Overlay: points colored by assigned SVF
            if ok.any():
                sc = ax.scatter(
                    lon[ok],
                    lat[ok],
                    c=vals[ok],
                    s=28,
                    edgecolor="k",
                    linewidth=0.3,
                )
                # colorbar for point values (separate from raster)
                cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("SVF (assigned to points)")

            # If anything still missing, show as red x
            if miss.any():
                ax.scatter(
                    lon[miss],
                    lat[miss],
                    s=50,
                    marker="x",
                    c="red",
                    linewidth=1.2,
                    label=f"still missing (n={miss.sum()})",
                )
                ax.legend(loc="upper right")

            ax.set_title(f"{gl_name}-{rgi_id} — SVF raster + stake-point SVF")
            ax.set_xlim(float(da["lon"].min()), float(da["lon"].max()))
            ax.set_ylim(float(da["lat"].min()), float(da["lat"].max()))
            plt.tight_layout()

            if save_dir is not None:
                out = os.path.join(save_dir, f"{rgi_id}_svf_points.png")
                plt.savefig(out, dpi=dpi)
                plt.close(fig)
            else:
                plt.show()

        except Exception as e:
            print(f"[error] {gl_name}-{rgi_id}: {type(e).__name__}: {e}")
            continue
