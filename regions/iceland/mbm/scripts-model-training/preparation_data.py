"""
This script contains functions for preparing data for machine learning models.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/06/2024
"""


import numpy as np
import pandas as pd
import sklearn


def create_train_test_data(df, seasons, datasets, random_seed):
    # Process each season
    for season, info in seasons.items():
        # Remove records with NaN values for the respective surface mass balances
        data = df[df[info['column']].notna()].reset_index(drop=True)

        # Divide the dataset into 70/30 split for training and testing
        train_data, test_data = sklearn.model_selection.train_test_split(data, test_size=0.3, random_state=random_seed,
                                                                         shuffle=True)

        # Add number of months to each dataframe
        train_data['n_months'] = info['n_months']
        test_data['n_months'] = info['n_months']

        # Store the datasets in the dictionary
        datasets[season] = (train_data, test_data)

        # Print basic statistics of the training and testing datasets
        print(f"Amount of entries in train/test for {season} surface mass balances: {train_data.shape[0]}/{test_data.shape[0]}, train: df_{season}_train, and test: df_{season}_test")


def prepare_dfs(data, smb_type, data_type, temp_columns, prec_columns, topo_cols, cols=None):
    if cols is None: cols = []

    columns_to_keep = []

    tmp_temp_summer_cols = temp_columns[7:].append(temp_columns[0])
    tmp_temp_winter_cols = temp_columns[0:7]

    tmp_prec_summer_cols = prec_columns[7:].append(prec_columns[0])
    tmp_prec_winter_cols = prec_columns[0:7]

    match data_type:
        case 'annual':
            columns_to_keep = list(set(temp_columns + prec_columns + topo_cols + [smb_type, 'n_months'] + cols))
        case 'winter':
            data[tmp_temp_summer_cols] = np.nan
            data[tmp_prec_summer_cols] = np.nan
            columns_to_keep = list(set(topo_cols + [smb_type, 'n_months'] + cols))
        case 'summer':
            data[tmp_temp_winter_cols] = np.nan
            data[tmp_prec_winter_cols] = np.nan
            columns_to_keep = list(set(topo_cols + [smb_type, 'n_months'] + cols))

    filtered_data = data[columns_to_keep]

    filtered_data = filtered_data.rename(columns={smb_type: 'SMB'})

    return filtered_data


def make_train_test_split(model, random_seed):
    # Select features for training -> t2m, tp, elevation, slope, aspect and height_difference
    df_train_X = model['train'].drop(['SMB'], axis=1)

    # Select the target variables -> Surface Mass Balance
    df_train_y = model['train'][['SMB']]

    # Get arrays of features and targets
    X_train, y_train = df_train_X.values, df_train_y.values

    # Use five folds for cross validation
    k_fold = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=random_seed)
    splits = list(k_fold.split(X_train, y_train))

    return df_train_X, X_train, y_train , splits


def create_model_data(seasons, models, datasets, temp_columns, prec_columns, topo_columns, name_model):
    for season, info in seasons.items():
        train = prepare_dfs(datasets[season][0], info['column'], season, temp_columns, prec_columns, topo_columns)
        test = prepare_dfs(datasets[season][1], info['column'], season, temp_columns, prec_columns, topo_columns)

        models[name_model][season] = {
            'train': train,
            'test': test
        }

    models[name_model]['all'] = {}

    # Concatenate train and test data for all seasons
    all_train = pd.concat([models[name_model][season]['train'] for season in seasons], ignore_index=True)
    all_test = pd.concat([models[name_model][season]['test'] for season in seasons], ignore_index=True)

    # Assign concatenated data to 'all' under 'model_1'
    models[name_model]['all'] = {'train': all_train, 'test': all_test}


