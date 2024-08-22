"""
This method transforms annual and seasonal data records into to monthly data records, each with
their respective months, e.g., a record for a period winter will be melted in the monthly data records consisting of
the following months: 'oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr' (depending on the FROM and TO date).
We account for a variable period for the SMB target data, which gives more flexibility.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 21/07/2024
"""

import pandas as pd
import numpy as np


def transform_to_monthly(
    df: pd.DataFrame,
    vois_climate: "list[str]",
    vois_topographical: "list[str]",
    output_fname: str,
) -> pd.DataFrame:
    """
    Converts the DataFrame to a monthly format based on climate-related columns.

    Args:
        df (pd.DataFrame): Input DataFrame with date ranges and climate-related columns + all other columns
            and data from the previous steps.
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
    column_names = _get_column_names(vois_topographical)

    # Create the final dataframe with the new exploded climate data
    result_df = _create_result_dataframe(df_exploded, column_names,
                                         vois_climate)

    result_df.to_csv(output_fname, index=False)

    return result_df


def _convert_dates_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'FROM_DATE' and 'TO_DATE' columns to datetime format."""
    df["FROM_DATE"] = pd.to_datetime(df["FROM_DATE"], format="%Y%m%d")
    df["TO_DATE"] = pd.to_datetime(df["TO_DATE"], format="%Y%m%d")
    return df


def _generate_monthly_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Generate monthly ranges and convert to month names."""
    df["MONTHS"] = df.apply(
        lambda row: pd.date_range(
            start=row["FROM_DATE"], end=row["TO_DATE"], freq="MS").strftime(
                "%b").str.lower().tolist(),
        axis=1,
    )
    df["N_MONTHS"] = df["MONTHS"].apply(len) - 1
    return df


def _add_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ID column to keep track of records when melted."""
    df["ID"] = np.arange(len(df))
    return df


def _explode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Explode the DataFrame based on the 'MONTHS' column."""
    df_exploded = df.explode("MONTHS")
    return df_exploded.reset_index(drop=True)


def _get_column_names(vois_topographical: "list[str]") -> "list[str]":
    """Get the list of column names to keep in the final DataFrame."""
    column_names = [
        "MONTHS",
        "ID",
        "RGIId",
        "POINT_ID",
        "YEAR",
        "N_MONTHS",
        "PERIOD",  
        "POINT_LON",
        "POINT_LAT",
        "POINT_BALANCE",
        "ALTITUDE_CLIMATE",
        "ELEVATION_DIFFERENCE",
    ]
    column_names.extend(vois_topographical)
    return column_names


def _get_climate_values(row: pd.Series, vois_climate: "list[str]",
                        column_names: "list[str]") -> np.ndarray:
    """Get climate values for a specific row and month."""
    cols = [f'{voi}_{row["MONTHS"]}' for voi in vois_climate]
    all_cols = column_names + cols
    return row[all_cols].values


def _create_result_dataframe(df_exploded: pd.DataFrame,
                             column_names: "list[str]",
                             vois_climate: "list[str]") -> pd.DataFrame:
    """Create the final result DataFrame."""
    climate_records = df_exploded.apply(
        lambda row: _get_climate_values(row, vois_climate, column_names),
        axis=1).tolist()
    final_column_names = column_names + vois_climate
    return pd.DataFrame(climate_records, columns=final_column_names)
