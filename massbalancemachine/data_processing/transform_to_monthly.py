"""
This script transforms annual and seasonal data records into to monthly data records, each with
their respective months, e.g., a record for a period winter will be melted in the monthly data records consisting of
the following months: 'oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr' (depending on the FROM and TO date).
We account for a variable period for the SMB target data, which gives more flexibility.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 10/07/2024
"""

import re
import pandas as pd
import numpy as np


def prepare_dataset(df, vois_climate):
    """
    Prepares a dataset by formatting dates, generating monthly ranges, and masking climate columns that are not in
    the date ranges as specified.

    Args:
        - df (pandas.DataFrame): The input DataFrame containing the data to be processed. It must include
        'FROM_DATE' and 'TO_DATE' columns formatted as strings (YYYYMMDD).
        - vois_climate (list of str): A list of variable of interest (VOI) prefixes to match climate-related columns in the DataFrame.

    Returns:
        - df (pandas.DataFrame): The processed DataFrame with the following modifications:
            - 'FROM_DATE' and 'TO_DATE' columns converted to datetime objects.
            - 'MONTHS' column added, containing lists of month names (abbreviated to three lower-case letters) within the date range for each row.
            - 'N_MONTHS' column added, containing the number of months in the 'MONTHS' column for each row.
            - 'ID' column added, containing unique identifiers for each row.
            - Climate columns not corresponding to the months within the date range for each row are set to NaN.
    """
    df['FROM_DATE'] = pd.to_datetime(df['FROM_DATE'], format='%Y%m%d')
    df['TO_DATE'] = pd.to_datetime(df['TO_DATE'], format='%Y%m%d')

    # Generate monthly ranges and convert to month names
    df['MONTHS'] = df.apply(lambda row: pd.date_range(start=row['FROM_DATE'], end=row['TO_DATE'], freq='MS').strftime(
        '%b').str.lower().tolist(), axis=1)
    df['N_MONTHS'] = df['MONTHS'].apply(len)

    # Add ID column, that keeps track what records belong to each other when melted
    df['ID'] = np.arange(len(df))

    # Find climate columns matching the pattern
    pattern = '|'.join([f'{voi}_[a-zA-Z]*' for voi in vois_climate])
    vois_climate_columns = [col for col in df.columns if re.match(pattern, col)]

    # Function to create a mask of columns to set NaN
    def create_nan_mask(row, columns):
        # Columns to keep based on the current row's MONTHS
        keep_columns = [col for col in columns if any(col.endswith(month) for month in row['MONTHS'])]
        # Create a mask where True indicates the column should be set to NaN
        mask = [col not in keep_columns for col in columns]
        return mask

    # Apply the mask to each row in a vectorized way
    masks = df['MONTHS'].apply(lambda months: create_nan_mask({'MONTHS': months}, vois_climate_columns))

    # Convert the masks to a DataFrame
    mask_df = pd.DataFrame(masks.tolist(), index=df.index, columns=vois_climate_columns)

    # Apply the mask to set the values to NaN
    df[vois_climate_columns] = df[vois_climate_columns].mask(mask_df)

    return df


# TODO: This code has a lot of potential to be optimized (vectorising?)
def convert_to_monthly(df, vois_climate, vois_topographical, output_fname):
    """
    Converts the DataFrame to a monthly format based on climate-related columns.

    Args:
        - df (pandas.DataFrame): Input DataFrame with date ranges and climate-related columns.
        - vois_climate (list of str): List of climate variable prefixes.

    Returns:
        - df (pandas.DataFrame): Transformed DataFrame in a monthly format.
    """

    # Helper function to process each row
    def process_row(row):
        # Get the months for the current row
        months = row['MONTHS']

        # Identify columns corresponding to the months in the current row
        columns_to_keep = [col for col in df.columns if any(col.endswith(month) for month in months)]

        # Reshape the data for the current row to a 2D array (months x climate variables)
        reshaped_data = np.reshape(row[columns_to_keep].values, (len(months), len(vois_climate)), order='F')

        # Create a MultiIndex for the new DataFrame, these are the columns to keep in the new dataframe
        index_names = ['MONTH', 'ID', 'YEAR', 'N_MONTHS', 'POINT_BALANCE']
        index_names.extend(vois_topographical)

        indices = [
            months,
            [row['ID']],
            [str(row['YEAR'])],
            [str(row['N_MONTHS'])],
            [str(row['POINT_BALANCE'])]
        ]

        for voi_topo in vois_topographical:
            indices.append([str(row[voi_topo])])

        index_multi = pd.MultiIndex.from_product(
            indices,
            names = index_names
        )

        # Create a DataFrame using the reshaped data and the MultiIndex
        monthly_df = pd.DataFrame(data=reshaped_data, index=index_multi, columns=vois_climate)
        return monthly_df

    # Apply the helper function to each row and concatenate the results
    monthly_dfs = df.apply(process_row, axis=1)
    concatenated_df = pd.concat(monthly_dfs.tolist()).reset_index()

    concatenated_df.to_csv(output_fname, index=False)

    return concatenated_df

