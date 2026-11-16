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

from data_processing.utils.hydro_year import (
    tag_hydro_year,
    _compute_months_before_after,
    months_hydro_year,
    SPINUP_TOKEN_PREFIX,
)

# Flag of the spin-up rows of the surface type, in the stake tables
# (the spin-up of `transform_to_monthly`) and the geodetic grids
# (`data_processing.gridded_utils`)
SPINUP_COLUMN = "SFC_SPINUP"


def transform_to_monthly(
    df: pd.DataFrame,
    meta_data_columns: "list[str]",
    vois_climate: "list[str]",
    vois_topographical: "list[str]",
    output_fname: str,
    output_format: str,
    spinup_month_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Converts the DataFrame to a monthly format based on climate-related columns.

    With `spinup_month_list`, the tokens of the climate columns of `df` in time order
    (spin-up tokens, padded months, hydrological year, see
    `utils.hydro_year.spinup_tokens`), every measurement also gets the spin-up rows of
    the surface type of the TIlike model (`model.sfc_type`, `TILikeSfcModel`), whose snow cover is the
    cumulative balance since the 1st of October: the months from the 1st of October
    preceding its first month to that month. A measurement starting in its
    hydrological year (e.g. a summer balance) is padded from the 1st of October of
    that year, one starting in the months padded before it ('aug_', 'sep_') from the
    1st of October of the year before ('_oct', ..., '_jul'). The rows are flagged in
    the column `SFC_SPINUP`, have the ID and the other metadata of the measurement
    (`N_MONTHS` counts its own months only), and their climate is read from the same
    row of `df`, at the location of the stake, like its other months. They come
    after the rows of the measurements. A measurement whose spin-up precedes the
    climate data (missing values) is not padded.

    Args:
        df (pd.DataFrame): Input DataFrame with date ranges and climate-related columns + all other columns
            and data from the previous steps.
        meta_data_columns (list[str]): List of metadata columns.
        vois_climate (list[str]): List of climate variable prefixes.
        vois_topographical (list[str]): List of topographical variable prefixes.
        output_fname (str): Name of the output CSV file.
        spinup_month_list (list[str]): Add the spin-up rows of the surface type.

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
    spinup = spinup_month_list is not None
    if spinup:
        df_exploded = pd.concat(
            (
                df_exploded.assign(**{SPINUP_COLUMN: False}),
                _explode_dataframe(
                    df.assign(
                        MONTHS=df["MONTHS"].map(
                            lambda m: _spinup_months(m, spinup_month_list)
                        )
                    )
                )
                .dropna(subset=["MONTHS"])
                .assign(**{SPINUP_COLUMN: True}),
            ),
            ignore_index=True,
        )

    # Check that we don't have inconsistent values in MONTHS, every entry should contain a month tag
    assert all(
        [any(m2 in m for m2 in months_hydro_year) for m in df_exploded.MONTHS.unique()]
    ), "MONTHS column in exploded dataframe contains inconsistent values. This is likely because of a nan in MONTHS."

    # Get the column names for the new dataframe
    column_names = _get_column_names(meta_data_columns, vois_topographical)
    if spinup:
        column_names.append(SPINUP_COLUMN)

    # Create the final dataframe with the new exploded climate data
    result_df = _create_result_dataframe(df_exploded, column_names, vois_climate)
    if spinup:
        result_df = _drop_spinup_without_climate(result_df, vois_climate)

    if output_fname is not None:
        if output_format == "csv":
            result_df.to_csv(output_fname, index=False)
        elif output_format == "parquet":
            result_df.to_parquet(output_fname, index=False)
        else:
            print("output format must be csv or parquet")
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
    if date.day < 15 and date.day > 1:
        return date - pd.offsets.MonthBegin()
    elif date.day == 1:
        return date
    else:
        return date + pd.offsets.MonthBegin()


def _generate_monthly_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Generate monthly ranges and convert to month names."""
    df["FROM_DATE_RND"] = df["FROM_DATE"].apply(_round_to_start_of_month)
    df["TO_DATE_RND"] = df["TO_DATE"].apply(_round_to_start_of_month)

    df = _compute_months_before_after(df)
    df["MONTHS"] = df.apply(tag_hydro_year, axis=1)
    df["N_MONTHS"] = df["MONTHS"].apply(len)
    df = df.drop(
        columns=[
            "FROM_DATE_RND",
            "TO_DATE_RND",
            "HY_YEAR",
            "HY_START",
            "HY_END",
            "MONTHS_BEFORE",
            "MONTHS_AFTER",
        ]
    )
    return df


def _spinup_months(months: "list[str]", month_list: "list[str]") -> "list[str]":
    """The tokens of `month_list` from the 1st of October preceding the first month of
    `months`, a measurement, to that month: 'oct' for a month of the hydrological
    year, '_oct' (the year before) for a month padded before it."""
    first = months[0]
    start = "oct" if first in months_hydro_year else SPINUP_TOKEN_PREFIX + "oct"
    if start not in month_list:
        # No month padded before the hydrological year: no spin-up token
        assert first in months_hydro_year, f"No spin-up tokens before {first}."
    return month_list[month_list.index(start) : month_list.index(first)]


def _drop_spinup_without_climate(df: pd.DataFrame, vois_climate: "list[str]"):
    """`df` without the spin-up rows of the measurements whose spin-up has a month
    without climate data, see `get_climate_data._process_climate_data`."""
    missing = df[SPINUP_COLUMN].astype(bool) & df[vois_climate].isna().all(axis=1)
    if not missing.any():
        return df
    ids = df.ID[missing].unique()
    print(
        f"Surface type: {len(ids)} measurements are not padded, their spin-up "
        "precedes the climate data."
    )
    return df[~(df[SPINUP_COLUMN].astype(bool) & df.ID.isin(ids))].reset_index(
        drop=True
    )


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
    return list(set(column_names))  # Delete duplicate column names


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
