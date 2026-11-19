import os
import shutil
import pandas as pd
import pyproj
import zipfile
from pathlib import Path


# --- Preprocess --- #
def emptyfolder(path):
    """Removes all files and subdirectories in the given folder."""
    if os.path.exists(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)  # Remove file
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Remove folder and all contents
            except Exception as e:
                print(f"Error removing {item_path}: {e}")
    else:
        os.makedirs(path, exist_ok=True)  # Ensure directory exists


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
        # 3 years in the Saint-Sorlin are falsely named lambert2 but are Lambert III.
        # Any other file with *_lambert2e columns fails in the else branch.
        elif key in (
            "stso_winter_smb_accu_2018",
            "stso_winter_smb_accu_2019",
            "stso_winter_smb_accu_2020",
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
