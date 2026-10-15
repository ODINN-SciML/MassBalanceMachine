import pandas as pd


def datetime_obj(value):
    """
    Convert an integer or string in YYYYMMDD format to a pandas Timestamp.

    Parameters
    ----------
    value : int or str
        Date encoded as YYYYMMDD (e.g., 20171015 or "20171015").

    Returns
    -------
    pandas.Timestamp
        Corresponding datetime object.

    Notes
    -----
    This helper assumes that the input always has exactly eight digits
    representing year, month, and day in that order.

    Raises
    ------
    ValueError
        If the input cannot be parsed into a valid date.
    """
    date = str(value)
    year = date[:4]
    month = date[4:6]
    day = date[6:8]
    return pd.to_datetime(month + "-" + day + "-" + year)


def transform_dates(df_or):
    """
    Correct and standardize GLAMOS stake measurement dates.

    The original GLAMOS data sometimes contains missing or inconsistent
    measurement dates. This function:

    1) Converts ``date0`` and ``date1`` columns from YYYYMMDD format to
       proper pandas datetime objects.
    2) Creates hydrological-year-based replacement dates:
       - ``date_fix0`` = October 1 of the measurement year
       - ``date_fix1`` = September 30 of the following year
    3) Restores the original ``date0`` and ``date1`` columns to WGMS-style
       YYYYMMDD string format.

    Parameters
    ----------
    df_or : pandas.DataFrame
        Raw GLAMOS DataFrame containing at least the columns:
        - ``date0`` : start date (YYYYMMDD format)
        - ``date1`` : end date (YYYYMMDD format)

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with additional columns:
        - ``date_fix0`` : pandas.Timestamp
        - ``date_fix1`` : pandas.Timestamp
        and with ``date0``/``date1`` standardized to string YYYYMMDD format.

    Notes
    -----
    - The function operates on a copy of the input DataFrame.
    - Fixed dates are based purely on the year of ``date0`` and are not
      derived from the original day/month values.

    Raises
    ------
    KeyError
        If required columns ``date0`` or ``date1`` are missing.
    """
    df = df_or.copy()

    # Ensure 'date0' and 'date1' are datetime objects
    df["date0"] = df["date0"].apply(lambda x: datetime_obj(x))
    df["date1"] = df["date1"].apply(lambda x: datetime_obj(x))

    # Initialize new columns with NaT (not np.nan, since we'll use datetime later)
    df["date_fix0"] = pd.NaT
    df["date_fix1"] = pd.NaT

    # Assign fixed dates using .loc to avoid chained assignment warning
    for i in range(len(df)):
        year = df.loc[i, "date0"].year
        df.loc[i, "date_fix0"] = pd.Timestamp(f"{year}-10-01")
        df.loc[i, "date_fix1"] = pd.Timestamp(f"{year + 1}-09-30")

    # Format original dates for WGMS
    df["date0"] = df["date0"].apply(lambda x: x.strftime("%Y%m%d"))
    df["date1"] = df["date1"].apply(lambda x: x.strftime("%Y%m%d"))

    return df


def clean_winter_dates(df_raw):
    """
    Correct invalid winter stake measurement date ranges.

    In some GLAMOS records, winter measurements have identical years for
    ``FROM_DATE`` and ``TO_DATE`` (or even identical dates). This function
    corrects such cases by setting the start date to the beginning of the
    hydrological year (October 1 of the previous calendar year).

    After correction, the function validates that all winter periods span
    exactly one year; otherwise an error is raised.

    Parameters
    ----------
    df_raw : pandas.DataFrame
        DataFrame containing at least the columns:
        - ``PERIOD`` : string, e.g. "winter" or "annual"
        - ``FROM_DATE`` : date string in YYYYMMDD format
        - ``TO_DATE`` : date string in YYYYMMDD format
        - ``GLACIER`` : glacier identifier (used only for error messages)

    Returns
    -------
    pandas.DataFrame
        The same DataFrame with corrected ``FROM_DATE`` values for winter
        periods when necessary.

    Notes
    -----
    - This function modifies the input DataFrame in-place.
    - Only rows with ``PERIOD == "winter"`` are affected.

    Raises
    ------
    ValueError
        If, after correction, any winter record does not span exactly one
        calendar year.
    KeyError
        If required columns are missing.
    """
    # For some winter measurements the FROM_DATE is the same year as the TO_DATE (even same date)
    # Correct it by setting it to beginning of hydrological year:
    for index, row in df_raw.iterrows():
        if row["PERIOD"] == "winter":
            df_raw.loc[index, "FROM_DATE"] = (
                str(pd.to_datetime(row["TO_DATE"], format="%Y%m%d").year - 1) + "1001"
            )
    for i, row in df_raw.iterrows():
        if (
            pd.to_datetime(row["TO_DATE"], format="%Y%m%d").year
            - pd.to_datetime(row["FROM_DATE"], format="%Y%m%d").year
            != 1
        ):
            # throw error if not corrected
            raise ValueError(
                "Date correction failed:",
                row["GLACIER"],
                row["PERIOD"],
                row["FROM_DATE"],
                pd.to_datetime(row["FROM_DATE"], format="%Y%m%d").year,
                row["TO_DATE"],
                pd.to_datetime(row["TO_DATE"], format="%Y%m%d").year,
            )
    return df_raw
