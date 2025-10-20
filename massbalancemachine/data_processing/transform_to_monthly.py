"""
This method transforms annual and seasonal data records into to monthly data records, each with
their respective months, e.g., a record for a period winter will be melted in the monthly data records consisting of
the following months: 'oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr' (depending on the FROM and TO date).
We account for a variable period for the SMB target data, which gives more flexibility.

Date Created: 21/07/2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def transform_to_monthly(
    df: pd.DataFrame,
    meta_data_columns: "list[str]",
    vois_climate: "list[str]",
    vois_topographical: "list[str]",
    output_fname: str,
) -> pd.DataFrame:
    """
    Converts the DataFrame to a monthly format based on climate-related columns.

    Args:
        df (pd.DataFrame): Input DataFrame with date ranges and climate-related columns + all other columns
            and data from the previous steps.
        meta_data_columns (list[str]): List of metadata columns.
        vois_climate (list[str]): List of climate variable prefixes.
        vois_topographical (list[str]): List of topographical variable prefixes.
        output_fname (str): Name of the output CSV file.

    Returns:
        pd.DataFrame: Original DataFrame (for consistency with previous implementation).
    """

    # Convert dates to datetime
    df = _convert_dates_to_datetime(df)

    # Generate monthly ranges based on the FROM and TO dates available
    df = _generate_monthly_ranges(df)

    # Add a unique ID column to the dataframe to identify columns of the same data range
    df = _add_id_column(df)

    # Explode the dataframe based on the date range
    df_exploded = _explode_dataframe(df)

    # Get the column names for the new dataframe
    column_names = _get_column_names(meta_data_columns, vois_topographical)

    # Create the final dataframe with the new exploded climate data
    result_df = _create_result_dataframe(df_exploded, column_names, vois_climate)

    result_df.to_csv(output_fname, index=False)

    return result_df


def _convert_dates_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'FROM_DATE' and 'TO_DATE' columns to datetime format."""
    df["FROM_DATE"] = pd.to_datetime(df["FROM_DATE"], format="%Y%m%d")
    df["TO_DATE"] = pd.to_datetime(df["TO_DATE"], format="%Y%m%d")
    return df


def _round_to_start_of_month(date):
    """
    Round date to the nearest start of the month.
    If day < 15, round down to start of current month.
    If day >= 15, round up to start of next month.
    """
    if date.day < 15 & date.day > 1:
        print(date - pd.offsets.MonthBegin())
        return date - pd.offsets.MonthBegin()
    elif date.day == 1:
        return date
    else:
        return date + pd.offsets.MonthBegin()


def _generate_monthly_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Generate monthly ranges and convert to month names."""
    df["FROM_DATE_RND"] = df["FROM_DATE"].apply(_round_to_start_of_month)
    df["TO_DATE_RND"] = df["TO_DATE"].apply(_round_to_start_of_month)

    df["MONTHS"] = df.apply(
        lambda row: pd.date_range(
            start=row["FROM_DATE_RND"],
            end=row["TO_DATE_RND"],
            freq="MS",
            inclusive="left",
        )
        .strftime("%b")
        .str.lower()
        .tolist(),
        axis=1,
    )

    def tag_hydro_year(months: List[str]) -> List[str]:
        """
        Tag month tokens to disambiguate padding around the Oct→Sep hydrological core.

        Rules:
        - Leading (tail) padding: starting 'aug'/'sep' should be tagged 'aug_'/'sep_'
        if they appear before the first 'oct' (or if no 'oct' exists, they are still tagged if at the start).
        - Trailing (head) padding: ending 'oct'/'nov' should be tagged 'oct_'/'nov_'
        if they appear after the last 'sep' (or if no 'sep' exists, they are still tagged if at the end).
        - Middle months are untouched.
        """
        base = [str(m).strip().lower() for m in months]
        n = len(base)
        if n == 0:
            return base

        # Find anchors (may be missing)
        try:
            first_oct = base.index("oct")
        except ValueError:
            first_oct = None

        try:
            last_sep = n - 1 - base[::-1].index("sep")
        except ValueError:
            last_sep = None

        tagged = base[:]  # copy

        # --- Tag leading tail padding: starting run of aug/sep before first oct (or if no oct, still at start) ---
        i = 0
        while (
            i < n and base[i] in ("aug", "sep") and (first_oct is None or i < first_oct)
        ):
            tagged[i] = base[i] + "_"
            i += 1

        # --- Tag trailing head padding: ending run of oct/nov after last sep (or if no sep, still at end) ---
        j = n - 1
        while (
            j >= 0 and base[j] in ("oct", "nov") and (last_sep is None or j > last_sep)
        ):
            tagged[j] = base[j] + "_"
            j -= 1

        return tagged

    # Apply the tagging function to the MONTHS column
    df["MONTHS"] = df["MONTHS"].apply(tag_hydro_year)
    df["N_MONTHS"] = df["MONTHS"].apply(len) - 1
    df = df.drop(columns=["FROM_DATE_RND", "TO_DATE_RND"])
    return df


def _add_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ID column to keep track of records when melted."""
    df["ID"] = np.arange(len(df))
    return df


def _explode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Explode the DataFrame based on the 'MONTHS' column."""
    df_exploded = df.explode("MONTHS")
    return df_exploded.reset_index(drop=True)


def _get_column_names(
    meta_data_columns: "list[str]", vois_topographical: "list[str]"
) -> "list[str]":
    """Get the list of column names to keep in the final DataFrame."""
    column_names = [
        "YEAR",
        "POINT_LON",
        "POINT_LAT",
        "POINT_BALANCE",
        "ALTITUDE_CLIMATE",
        "ELEVATION_DIFFERENCE",
        "POINT_ELEVATION",
    ]
    column_names.extend(meta_data_columns)
    column_names.extend(vois_topographical)
    return column_names


def _get_climate_values(
    row: pd.Series, vois_climate: "list[str]", column_names: "list[str]"
) -> np.ndarray:
    """Get climate values for a specific row and month."""

    cols = [f'{voi}_{row["MONTHS"]}' for voi in vois_climate]

    all_cols = column_names + cols
    return row[all_cols].values


def _create_result_dataframe(
    df_exploded: pd.DataFrame,
    column_names: "list[str]",
    vois_climate: "list[str]",
    chunk_size=10000,
) -> pd.DataFrame:
    """Create the final result DataFrame."""
    apply_func = lambda row: _get_climate_values(row, vois_climate, column_names)
    if chunk_size > 0:
        # Split call to apply in chunks
        # This is useful when working with large dataframes to avoid having OOM errors
        climate_records = []
        for start in range(0, df_exploded.shape[0], chunk_size):
            chunk = df_exploded.iloc[start : start + chunk_size]
            chunk_records = chunk.apply(apply_func, axis=1).tolist()
            climate_records.extend(chunk_records)
    else:
        climate_records = df_exploded.apply(apply_func, axis=1).tolist()
    final_column_names = column_names + vois_climate
    return pd.DataFrame(climate_records, columns=final_column_names)
