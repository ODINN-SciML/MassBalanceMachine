"""
This script transforms annual and seasonal data records into to monthly data records, each with
their respective months, i.e., a record for winter will be melted in the monthly data record consisting of
the following months: 'oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr'.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/06/2024
"""

import pandas as pd
import numpy as np


# TODO: Refactor/rewrite this code for optimization/efficiency reasons
def convert_to_monthly(df, output_fname, vois_columns_climate, vois_topo_columns, smb_column_names, column_name_year):
    # Define a dictionary for the seasonal data
    seasons = {
        'annual': {'column': 'ba_stratigraphic', 'n_months': 12},
        'winter': {'column': 'bw_stratigraphic', 'n_months': 7},
        'summer': {'column': 'bs_stratigraphic', 'n_months': 5}
    }

    winter_months = ['oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr']
    summer_months = ['may', 'jun', 'jul', 'aug', 'sep']

    winter_climate_columns = [voi for voi in sum(vois_columns_climate.values(), []) if voi[-3:] in winter_months]
    summer_climate_columns = [voi for voi in sum(vois_columns_climate.values(), []) if voi[-3:] in summer_months]

    df_reshaped = pd.DataFrame()

    for season, info in seasons.items():
        # Select relevant columns
        list_climate_columns = sum(vois_columns_climate.values(), [])
        combined_columns_to_keep = list_climate_columns + vois_topo_columns + smb_column_names + column_name_year
        data = df[combined_columns_to_keep]

        # Remove records with NaN values for the respective surface mass balances
        data = data[data[info['column']].notna()].reset_index(drop=True)

        # Assign SMB and drop smb_types
        data['SMB'] = data[info['column']]
        data.drop(smb_column_names, axis=1, inplace=True)

        # Adjust climate columns based on season
        if season == 'winter':
            data.loc[:, summer_climate_columns] = np.nan
        elif season == 'summer':
            data.loc[:, winter_climate_columns] = np.nan

        data['n_months'] = info['n_months']
        data['id'] = np.arange(len(data))

        # Reshape dataset monthly
        months = winter_months + summer_months if season == 'annual' else winter_months if season == 'winter' else summer_months

        reshaped = reshape_dataset_monthly(
            data,
            vois_topo_columns + column_name_year + ['n_months', 'id', 'SMB'],
            vois_columns_climate,
            months
        )

        # Merge melted dataframe with merged_df based on ID and month
        df_reshaped = pd.concat([df_reshaped, reshaped], ignore_index=True)

    df_reshaped.to_csv(output_fname, index=False)

    return df_reshaped


# TODO: Optimize/refactor this code for efficiency reasons
def reshape_dataset_monthly(df, id_vars, variables, months_order):
    """
    Reshapes the dataset monthly based on specified variables and months order.

    Parameters:
        - df (DataFrame): Input dataframe containing variables to reshape.
        - id_vars (list): List of columns to keep as IDs.
        - variables (dict): Dictionary mapping variables to their respective columns.
        - months_order (list): Order of months for reshaping.

    Returns:
        - merged_df (DataFrame): Reshaped dataframe with variables melted and merged.
    """
    merged_df = None  # Initialize merged_df as None

    # Iterate over each variable to reshape
    for var in variables:
        # Select columns related to the current variable and ID columns
        cols = [col for col in df.columns if col.startswith(var) or col in id_vars]
        df_var = df[cols]

        # Rename columns to remove prefixes and keep ID columns intact
        df_var = df_var.rename(columns=lambda col: col.split('_')[-1] if col not in id_vars else col)

        # Melt the dataframe to reshape it based on months
        df_melted = df_var.melt(id_vars=id_vars, var_name='month', value_name=var)

        # Convert 'month' column to categorical with specified order
        df_melted['month'] = pd.Categorical(df_melted['month'], categories=months_order, ordered=True)

        # Merge melted dataframe with merged_df based on ID and month
        if merged_df is None:
            merged_df = df_melted
        else:
            merged_df = merged_df.merge(df_melted, on=id_vars + ['month'], how='left')

        # Drop rows where both variable and month are NaN
        merged_df.dropna(subset=[var, 'month'], how='all', inplace=True)

    # Sort the merged dataframe by ID and month
    merged_df.sort_values(by=id_vars + ['month'], inplace=True)

    return merged_df
