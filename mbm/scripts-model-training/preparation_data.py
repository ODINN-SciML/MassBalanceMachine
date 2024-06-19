"""
This script contains functions for preparing and splitting data for machine learning models.

Functions:
1. make_train_test_split(dataset, n_splits):
   - Prepares training and validation data using GroupKFold cross-validation based on glacier IDs.
   - Returns train and validation dataframes along with arrays of features and targets.

2. create_train_test_data(df, seasons, vois_climate_columns, vois_topo_columns, smb_types, misc_columns, random_seed, num_samples=None):
   - Creates training and testing datasets split by seasons (winter, summer, annual).
   - Reshapes datasets monthly and saves them as CSV files.
   - Returns a dictionary containing training and testing datasets for each season.

3. reshape_dataset_monthly(df, id_vars, variables, months_order):
   - Reshapes the dataset monthly based on specified variables and months order.
   - Returns a reshaped DataFrame.

4. create_model_data(df, seasons, vois_climate_columns, vois_topo_columns, smb_types, misc_columns, random_seed, num_samples=None):
   - Creates combined training and testing datasets for all seasons.
   - Saves combined datasets as CSV files.
   - Returns a dictionary containing all seasonal and combined datasets.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/06/2024
"""

import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split


def make_train_test_split(dataset, n_splits):
    """
    Prepares training and validation data using GroupKFold cross-validation based on glacier IDs.

    Parameters:
    - dataset (DataFrame): Input dataset containing features and targets.
    - n_splits (int): Number of folds for GroupKFold cross-validation.

    Returns:
    - df_X_train (DataFrame): DataFrame containing training features.
    - df_y_train (DataFrame): DataFrame containing training targets.
    - X_train (array): Array of training features.
    - y_train (array): Array of training targets.
    - splits (list): List of tuples containing indices for training and validation splits.
    """
    # Select features for training
    df_X_train = dataset.drop(['yr', 'SMB'], axis=1)

    # Move id and n_months to the end of the dataframe (these are used as metadata)
    metadata_columns = ['id', 'n_months', 'month']
    df_X_train = df_X_train[[col for col in df_X_train.columns if col not in metadata_columns] + metadata_columns]

    # Select the targets for training
    df_y_train = dataset[['SMB']]

    # Get arrays of features+metadata and targets
    X_train, y_train = df_X_train.values, df_y_train.values

    # Get glacier IDs from training dataset
    glacier_ids = np.array(dataset['id'].values)

    # Use GroupKFold for splitting by glacier ID
    group_kf = GroupKFold(n_splits=n_splits)
    splits = list(group_kf.split(X_train, y_train, glacier_ids))

    return df_X_train, df_y_train, X_train, y_train, splits


def create_train_test_data(df, seasons, vois_climate_columns, vois_topo_columns, smb_types, misc_columns,
                           random_seed, num_samples=None):
    """
    Creates training and testing datasets split by seasons (winter, summer, annual).

    Parameters:
    - df (DataFrame): Input dataframe containing all data.
    - seasons (dict): Dictionary specifying seasons and their respective metadata (e.g., months).
    - vois_climate_columns (dict): Dictionary mapping climate variables to their respective columns.
    - vois_topo_columns (list): List of topographical variables.
    - smb_types (list): List of surface mass balance types.
    - misc_columns (list): List of additional metadata columns.
    - random_seed (int): Random seed for reproducibility.
    - num_samples (int, optional): Number of samples to use from the dataframe.

    Returns:
    - datasets (dict): Dictionary containing training and testing datasets for each season.
    """
    if num_samples is not None:
        df = df.sample(n=num_samples)

    datasets = {}

    winter_months = ['oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr']
    summer_months = ['may', 'jun', 'jul', 'aug', 'sep']

    winter_climate_columns = [voi for voi in sum(vois_climate_columns.values(), []) if voi[-3:] in winter_months]
    summer_climate_columns = [voi for voi in sum(vois_climate_columns.values(), []) if voi[-3:] in summer_months]

    for season, info in seasons.items():
        # Select relevant columns
        list_climate_columns = sum(vois_climate_columns.values(), [])
        combined_columns_to_keep = list_climate_columns + vois_topo_columns + smb_types + misc_columns
        data = df[combined_columns_to_keep]

        # Remove records with NaN values for the respective surface mass balances
        data = data[data[info['column']].notna()].reset_index(drop=True)

        # Assign SMB and drop smb_types
        data['SMB'] = data[info['column']]
        data.drop(smb_types, axis=1, inplace=True)

        # Adjust climate columns based on season
        if season == 'winter':
            data.loc[:, summer_climate_columns] = np.nan
        elif season == 'summer':
            data.loc[:, winter_climate_columns] = np.nan

        # Divide the dataset into training and testing
        train_data, test_data = train_test_split(
            data,
            test_size=0.3,
            random_state=random_seed,
            shuffle=True
        )

        # Add number of months and IDs to each dataframe
        train_data['n_months'] = info['n_months']
        test_data['n_months'] = info['n_months']
        train_data['id'] = np.arange(len(train_data))
        test_data['id'] = np.arange(len(test_data))

        # Reshape dataset monthly
        months = winter_months + summer_months if season == 'annual' else winter_months if season == 'winter' else summer_months
        train_data = reshape_dataset_monthly(
            train_data,
            vois_topo_columns + misc_columns + ['n_months', 'id', 'SMB'],
            vois_climate_columns,
            months
        )
        test_data = reshape_dataset_monthly(
            test_data,
            vois_topo_columns + misc_columns + ['n_months', 'id', 'SMB'],
            vois_climate_columns,
            months
        )

        # Store datasets in CSV files
        train_data.to_csv(f'.././data/files/monthly/{season}_train_data.csv', index=False)
        test_data.to_csv(f'.././data/files/monthly/{season}_test_data.csv', index=False)

        datasets[season] = {
            'train': train_data,
            'test': test_data,
        }

        # Print basic statistics of the training and testing datasets
        print(
            f"Amount of entries in train/test for {season} surface mass balances: {train_data.shape[0]}/{test_data.shape[0]}, train: {season}_train, and test: {season}_test")

        del train_data, test_data, data
        gc.collect()

    del df
    gc.collect()

    return datasets


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

        # Clean up temporary dataframes to save memory
        del df_melted, df_var
        gc.collect()

    # Sort the merged dataframe by ID and month
    merged_df.sort_values(by=id_vars + ['month'], inplace=True)

    return merged_df


def create_model_data(df, seasons, vois_climate_columns, vois_topo_columns, smb_types, misc_columns, random_seed,
                      num_samples=None):
    """
    Creates combined training and testing datasets for all seasons.

    Parameters:
    - df (DataFrame): Input dataframe containing all data.
    - seasons (dict): Dictionary specifying seasons and their respective metadata (e.g., months).
    - vois_climate_columns (dict): Dictionary mapping climate variables to their respective columns.
    - vois_topo_columns (list): List of topographical variables.
    - smb_types (list): List of surface mass balance types.
    - misc_columns (list): List of additional metadata columns.
    - random_seed (int): Random seed for reproducibility.
    - num_samples (int, optional): Number of samples to use from the dataframe.

    Returns:
    - datasets (dict): Dictionary containing training and testing datasets for each season and combined dataset.
    """
    # Create training and testing datasets for each season
    datasets = create_train_test_data(df, seasons, vois_climate_columns, vois_topo_columns, smb_types, misc_columns,
                                      random_seed, num_samples)

    # Concatenate training and testing datasets for all seasons into a combined dataset
    datasets['all'] = {
        'train': pd.concat([datasets[season]['train'] for season in seasons], ignore_index=True),
        'test': pd.concat([datasets[season]['test'] for season in seasons], ignore_index=True)
    }

    # Save combined datasets as CSV files
    datasets['all']['train'].to_csv(f'.././data/files/monthly/all_train_data.csv', index=False)
    datasets['all']['test'].to_csv(f'.././data/files/monthly/all_test_data.csv', index=False)

    return datasets
