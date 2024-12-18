from scripts.helpers import *
import massbalancemachine as mbm
import config
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, train_test_split, GroupShuffleSplit
import xarray as xr


def getMonthlyDataLoaderOneGl(glacierName, vois_climate, voi_topographical):
    # Load stakes data from GLAMOS
    data_glamos = pd.read_csv(path_PMB_GLAMOS_csv + 'CH_wgms_dataset.csv')

    rgi_df = pd.read_csv(path_rgi, sep=',')
    rgi_df.rename(columns=lambda x: x.strip(), inplace=True)
    rgi_df.set_index('short_name', inplace=True)

    # Get dataloader:
    rgi_gl = rgi_df.loc[glacierName]['rgi_id.v6']
    data_gl = data_glamos[data_glamos.RGIId == rgi_gl]

    dataset_gl = mbm.Dataset(data=data_gl,
                             region_name='CH',
                             data_path=path_PMB_GLAMOS_csv)

    # Add climate data:
    # Specify the files of the climate data, that will be matched with the coordinates of the stake data
    era5_climate_data = path_ERA5_raw + 'era5_monthly_averaged_data.nc'
    geopotential_data = path_ERA5_raw + 'era5_geopotential_pressure.nc'

    # Match the climate features, from the ERA5Land netCDF file, for each of the stake measurement dataset
    dataset_gl.get_climate_features(climate_data=era5_climate_data,
                                    geopotential_data=geopotential_data,
                                    change_units=True)

    # Add potential clear sky radiation:
    dataset_gl.get_potential_rad(path_direct_save)

    # For each record, convert to a monthly time resolution
    dataset_gl.convert_to_monthly(
        meta_data_columns=config.META_DATA,
        vois_climate=vois_climate + ['pcsr'],  # add potential radiation
        vois_topographical=voi_topographical)

    # Create a new DataLoader object with the monthly stake data measurements.
    dataloader_gl = mbm.DataLoader(data=dataset_gl.data,
                                   random_seed=config.SEED,
                                   meta_data_columns=config.META_DATA)

    return dataloader_gl


def getCVSplits(dataloader_gl, test_split_on='YEAR', test_splits=None):
    # Split into training and test splits with train_test_split
    if test_splits is None:
        train_splits, test_splits = train_test_split(
            dataloader_gl.data[test_split_on].unique(),
            test_size=0.2,
            random_state=config.SEED)
    else:
        split_data = dataloader_gl.data[test_split_on].unique()
        train_splits = [x for x in split_data if x not in test_splits]
    train_indices = dataloader_gl.data[dataloader_gl.data[test_split_on].isin(
        train_splits)].index
    test_indices = dataloader_gl.data[dataloader_gl.data[test_split_on].isin(
        test_splits)].index

    dataloader_gl.set_custom_train_test_indices(train_indices, test_indices)

    # Get the features and targets of the training data for the indices as defined above, that will be used during the cross validation.
    df_X_train = dataloader_gl.data.iloc[train_indices]
    y_train = df_X_train['POINT_BALANCE'].values
    train_meas_id = df_X_train['ID'].unique()

    # Get test set
    df_X_test = dataloader_gl.data.iloc[test_indices]
    y_test = df_X_test['POINT_BALANCE'].values
    test_meas_id = df_X_test['ID'].unique()

    # Values split in training and test set
    train_splits = df_X_train[test_split_on].unique()
    test_splits = df_X_test[test_split_on].unique()

    # Create the CV splits based on the training dataset. The default value for the number of splits is 5.
    splits = dataloader_gl.get_cv_split(n_splits=5, type_fold='group-meas-id')

    test_set = {
        'df_X': df_X_test,
        'y': y_test,
        'meas_id': test_meas_id,
        'splits_vals': test_splits
    }
    train_set = {
        'df_X': df_X_train,
        'y': y_train,
        'splits_vals': train_splits,
        'meas_id': train_meas_id,
    }

    return splits, test_set, train_set


def plot_gsearch_results(grid, params):
    """
    Params: 
        grid: A trained GridSearchCV object.
    """
    ## Results from grid search
    results = grid.cv_results_
    grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    means_train = results['mean_train_score']
    stds_train = results['std_train_score']

    ## Getting indexes of values per hyper-parameter
    masks = []
    masks_names = list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_' + p_k].data == p_v))

    #params=grid.cv_results_['params']
    #print(params)

    width = len(grid.best_params_.keys()) * 5

    ## Ploting results
    fig, ax = plt.subplots(1,
                           len(params),
                           sharex='none',
                           sharey='all',
                           figsize=(width, 5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i + 1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        y_2 = np.array(means_train[best_index])
        e_2 = np.array(stds_train[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
        ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^', label='train')
        ax[i].set_xlabel(p.upper())
        ax[i].grid()

    plt.legend()
    plt.show()


def getDfAggregatePred(test_set, y_pred_agg, all_columns):
    # Aggregate predictions to annual or winter:
    df_pred = test_set['df_X'][all_columns].copy()
    df_pred['target'] = test_set['y']
    grouped_ids = df_pred.groupby('ID').agg({
        'target': 'mean',
        'YEAR': 'first',
        'POINT_ID': 'first'
    })
    grouped_ids['pred'] = y_pred_agg
    grouped_ids['PERIOD'] = test_set['df_X'][all_columns].groupby(
        'ID')['PERIOD'].first()
    grouped_ids['GLACIER'] = grouped_ids['POINT_ID'].apply(
        lambda x: x.split('_')[0])

    return grouped_ids


def GlacierWidePred(custom_model, df_grid_monthly, type_pred='annual'):
    if type_pred == 'annual':
        # Make predictions on whole glacier grid
        features_grid, metadata_grid = custom_model._create_features_metadata(
            df_grid_monthly, config.META_DATA)

        # Make predictions aggregated to measurement ID:
        y_pred_grid_agg = custom_model.aggrPredict(metadata_grid,
                                                   config.META_DATA,
                                                   features_grid)

        # Aggregate predictions to annual:
        grouped_ids_annual = df_grid_monthly.groupby('ID').agg({
            'YEAR':
            'mean',
            'POINT_LAT':
            'mean',
            'POINT_LON':
            'mean'
        })
        grouped_ids_annual['pred'] = y_pred_grid_agg

        return grouped_ids_annual

    elif type_pred == 'winter':
        # winter months from October to April
        winter_months = ['oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr']
        df_grid_winter = df_grid_monthly[df_grid_monthly.MONTHS.isin(
            winter_months)]

        # Make predictions on whole glacier grid
        features_grid, metadata_grid = custom_model._create_features_metadata(
            df_grid_winter, config.META_DATA)

        # Make predictions aggregated to measurement ID:
        y_pred_grid_agg = custom_model.aggrPredict(metadata_grid,
                                                   config.META_DATA,
                                                   features_grid)

        # Aggregate predictions for winter:
        grouped_ids_winter = df_grid_monthly.groupby('ID').agg({
            'YEAR':
            'mean',
            'POINT_LAT':
            'mean',
            'POINT_LON':
            'mean'
        })
        grouped_ids_winter['pred'] = y_pred_grid_agg

        return grouped_ids_winter

    else:
        raise ValueError(
            "Unsupported prediction type. Only 'annual' and 'winter' are currently supported."
        )


def cumulativeMB(df_pred,
                 test_gl,
                 ids_year_dict):
    df_pred_gl = df_pred[df_pred['GLACIER'] == test_gl]

    dfCumMB_all = pd.DataFrame(columns=[
        'months',
        'cum_MB',
        'year',
        'ID',
        'monthNb',
    ])
    month_abbr_hydr = config.month_abbr_hydr.copy()
    for ID in df_pred_gl['ID'].unique():
        df_pred_stake = df_pred_gl[df_pred_gl['ID'] == ID]

        if df_pred_stake.MONTHS.iloc[-1] == 'sep':
            # rename last element
            df_pred_stake.MONTHS.iloc[-1] = 'sep_'
            month_abbr_hydr['sep_'] = 13

        dfCumMB = pd.DataFrame({
            'months': df_pred_stake.MONTHS,
            'cum_MB': df_pred_stake.y_pred.cumsum(),
            'ID': np.tile(ID, len(df_pred_stake.MONTHS))
        })
        dfCumMB.set_index('ID', inplace=True)
        dfCumMB['year'] = dfCumMB.index.map(ids_year_dict)
        # reset index
        dfCumMB.reset_index(inplace=True)

        # Complete missing months (NaN):
        missing_months = Diff(list(df_pred_stake.MONTHS),
                              list(month_abbr_hydr.keys()))
        missingRows = pd.DataFrame(columns=dfCumMB.columns)
        missingRows['months'] = missing_months
        missingRows['ID'] = ID
        missingRows['year'] = dfCumMB['year'].unique()[0]

        # Concatenate missing rows
        dfCumMB = pd.concat([dfCumMB, missingRows], axis=0)
        dfCumMB['monthNb'] = dfCumMB['months'].apply(
            lambda x: month_abbr_hydr[x])

        # Sort by their monthNB
        dfCumMB = dfCumMB.sort_values(by='monthNb')
        dfCumMB_all = pd.concat([dfCumMB_all, dfCumMB], axis=0)

    return dfCumMB_all






