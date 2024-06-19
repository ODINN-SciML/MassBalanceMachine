"""
This script contains functions for preparing data for machine learning models.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/06/2024
"""

import numpy as np
import pandas as pd
import dask.dataframe as dd
import sklearn
import gc


def make_train_test_split(dataset, n_splits):
    # Select features for training
    df_X_train = dataset.drop(['yr', 'SMB'], axis=1)

    # Move id and n_months to the end of the dataframe (these are used as metadata)
    # Columns to move to the end
    metadata_columns = ['id', 'n_months', 'month']

    # Reindex the DataFrame, moving the specified columns to the end
    df_X_train = df_X_train[[col for col in df_X_train.columns if col not in metadata_columns] + metadata_columns]

    # Select the targets for training
    df_y_train = dataset[['SMB']]

    # Get arrays of features+metadata and targets
    X_train, y_train = df_X_train.values, df_y_train.values

    # Get glacier IDs from training dataset (in the order of which they appear in training dataset).
    # gp_s is an array with shape equal to the shape of X_train_s and y_train_s.
    glacier_ids = np.array(dataset['id'].values)

    # Use five folds
    group_kf_s = sklearn.model_selection.GroupKFold(n_splits=n_splits)

    # Split into folds according to group by glacier ID.
    # For each unique glacier ID, indices in gp_s indicate which rows in X_train_s and y_train_s belong to the glacier.
    splits = list(group_kf_s.split(X_train, y_train, glacier_ids))

    return df_X_train, df_y_train, X_train, y_train, splits


def create_train_test_data(df, seasons, vois_climate_columns, vois_topo_columns, smb_types, misc_columns,
                           random_seed, num_samples=None):

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

        # Divide the dataset into 70/30 split for training and testing
        train_data, test_data = sklearn.model_selection.train_test_split(
            data,
            test_size=0.3,
            random_state=random_seed,
            shuffle=True
        )

        # Add number of months to each dataframe
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
    merged_df = None

    for var in variables:
        cols = [col for col in df.columns if col.startswith(var) or col in id_vars]
        df_var = df[cols]

        df_var = df_var.rename(columns=lambda col: col.split('_')[-1] if col not in id_vars else col)

        df_melted = df_var.melt(id_vars=id_vars, var_name='month', value_name=var)
        df_melted['month'] = pd.Categorical(df_melted['month'], categories=months_order, ordered=True)

        if merged_df is None:
            merged_df = df_melted
        else:
            merged_df = merged_df.merge(df_melted, on=id_vars + ['month'], how='left')

        merged_df.dropna(subset=[var, 'month'], how='all', inplace=True)

        del df_melted, df_var
        gc.collect()

    merged_df.sort_values(by=id_vars + ['month'], inplace=True)

    return merged_df


def create_model_data(df, seasons, vois_climate_columns, vois_topo_columns, smb_types, misc_columns, random_seed, num_samples=None):
    datasets = create_train_test_data(df, seasons, vois_climate_columns, vois_topo_columns, smb_types, misc_columns,
                                      random_seed, num_samples)

    datasets['all'] = {
        'train': pd.concat([datasets[season]['train'] for season in seasons], ignore_index=True),
        'test': pd.concat([datasets[season]['test'] for season in seasons], ignore_index=True)
    }

    datasets['all']['train'].to_csv(f'.././data/files/monthly/all_train_data.csv', index=False)
    datasets['all']['test'].to_csv(f'.././data/files/monthly/all_test_data.csv', index=False)

    return datasets
