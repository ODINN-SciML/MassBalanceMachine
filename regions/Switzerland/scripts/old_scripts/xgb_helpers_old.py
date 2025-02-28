from scripts.helpers import *
import massbalancemachine as mbm
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, train_test_split, GroupShuffleSplit
import xarray as xr
import geopandas as gpd


def getMonthlyDataLoaderOneGl(glacierName, vois_climate, voi_topographical,
                              cfg: mbm.Config):
    # Load stakes data from GLAMOS
    data_glamos = pd.read_csv(path_PMB_GLAMOS_csv + 'CH_wgms_dataset.csv')

    rgi_df = pd.read_csv(path_rgi, sep=',')
    rgi_df.rename(columns=lambda x: x.strip(), inplace=True)
    rgi_df.set_index('short_name', inplace=True)

    # Get dataloader:
    rgi_gl = rgi_df.loc[glacierName]['rgi_id.v6']
    data_gl = data_glamos[data_glamos.RGIId == rgi_gl]

    dataset_gl = mbm.Dataset(cfg=cfg,
                             data=data_gl,
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
        meta_data_columns=cfg.metaData,
        vois_climate=vois_climate + ['pcsr'],  # add potential radiation
        vois_topographical=voi_topographical)

    # Create a new DataLoader object with the monthly stake data measurements.
    dataloader_gl = mbm.DataLoader(cfg=cfg,
                                   data=dataset_gl.data,
                                   meta_data_columns=cfg.metaData)

    return dataloader_gl





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







def cumulativeMB(df_pred, test_gl, ids_year_dict, month_abbr_hydr):
    df_pred_gl = df_pred[df_pred['GLACIER'] == test_gl]

    dfCumMB_all = pd.DataFrame(columns=[
        'months',
        'cum_MB',
        'year',
        'ID',
        'monthNb',
    ])
    month_abbr_hydr = month_abbr_hydr.copy()
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






