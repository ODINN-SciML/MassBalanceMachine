from scripts.helpers import *
import massbalancemachine as mbm
import config
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, train_test_split, GroupShuffleSplit


def getMonthlyDataLoader(glacierName, vois_climate, voi_topographical):
    # Load stakes data from GLAMOS
    data_glamos = pd.read_csv(path_PMB_GLAMOS_csv + 'CH_wgms_dataset.csv')
    
    rgi_df = pd.read_csv(path_rgi, sep=',')
    rgi_df.rename(columns=lambda x: x.strip(), inplace=True)
    rgi_df.set_index('short_name', inplace=True)

    # Get dataloader:
    rgi_gl = rgi_df.loc[glacierName]['rgi_id.v6']
    data_gl = data_glamos[data_glamos.RGIId == rgi_gl]
    
    # change mm w.e. to m w.e.
    data_gl['POINT_BALANCE'] = data_gl['POINT_BALANCE'] / 1000
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

    # For each record, convert to a monthly time resolution
    dataset_gl.convert_to_monthly(meta_data_columns=config.META_DATA,
                                  vois_climate=vois_climate,
                                  vois_topographical=voi_topographical)

    # Create a new DataLoader object with the monthly stake data measurements.
    dataloader_gl = mbm.DataLoader(data=dataset_gl.data,
                                   random_seed=config.SEED,
                                   meta_data_columns=config.META_DATA)

    return dataloader_gl


def getCVSplits(dataloader_gl):
    # Split into training and test years with train_test_split
    train_years, test_years = train_test_split(
        dataloader_gl.data.YEAR.unique(),
        test_size=0.2,
        random_state=config.SEED)

    train_indices = dataloader_gl.data[dataloader_gl.data.YEAR.isin(
        train_years)].index
    test_indices = dataloader_gl.data[dataloader_gl.data.YEAR.isin(
        test_years)].index

    dataloader_gl.set_custom_train_test_indices(train_indices, test_indices)

    # Get the features and targets of the training data for the indices as defined above, that will be used during the cross validation.
    df_X_train = dataloader_gl.data.iloc[train_indices]
    y_train = df_X_train['POINT_BALANCE'].values
    train_meas_id = df_X_train['ID'].unique()

    # Get test set
    df_X_test = dataloader_gl.data.iloc[test_indices]
    y_test = df_X_test['POINT_BALANCE'].values
    test_meas_id = df_X_test['ID'].unique()
    
    # Years in training and test set
    train_years = df_X_train.YEAR.unique()
    test_years = df_X_test.YEAR.unique()
    
    # Create the CV splits based on the training dataset. The default value for the number of splits is 5.
    splits = dataloader_gl.get_cv_split(n_splits=5, type_fold='group-meas-id')
    

    test_set = {
        'df_X': df_X_test,
        'y': y_test,
        'meas_id': test_meas_id,
        'years': test_years
    }
    train_set = {
        'df_X': df_X_train,
        'y': y_train,
        'years': train_years,
        'meas_id': train_meas_id,
    }

    return splits, test_set, train_set
