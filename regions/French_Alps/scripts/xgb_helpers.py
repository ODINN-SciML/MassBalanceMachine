import os
import logging
import pandas as pd
import massbalancemachine as mbm
from sklearn.model_selection import GroupKFold, KFold, train_test_split, GroupShuffleSplit
import geopandas as gpd
import xarray as xr

from scripts.config_FR import *


def process_or_load_data_glacioclim(run_flag, df, paths, cfg, vois_climate,
                         vois_topographical, 
                         output_file = 'FR_wgms_dataset_monthly_full.csv'):
    """
    Process or load the data based on the RUN flag.
    """
    if run_flag:
        logging.info("Number of annual and seasonal samples: %d",
                     len(df))

        # Filter data
        logging.info("Running on %d glaciers:\n%s",
                     len(df.GLACIER.unique()),
                     df.GLACIER.unique())

        # Create dataset
        dataset_gl = mbm.Dataset(cfg=cfg,
                                 data=df,
                                 region_name='FR',
                                 data_path=paths['csv_path'])
        logging.info("Number of winter, summer and annual samples: %d",
                     len(df))
        logging.info("Number of annual samples: %d",
                     len(df[df.PERIOD == 'annual']))
        logging.info("Number of winter samples: %d",
                     len(df[df.PERIOD == 'winter']))

        # Add climate data
        logging.info("Adding climate features...")
        try:
            dataset_gl.get_climate_features(
                climate_data=paths['era5_climate_data'],
                geopotential_data=paths['geopotential_data'],
                change_units=True)
        except Exception as e:
            logging.error("Failed to add climate features: %s", e)
            return None
        """
        # Add radiation data
        logging.info("Adding potential clear sky radiation...")
        logging.info("Shape before adding radiation: %s",
                     dataset_gl.data.shape)
        dataset_gl.get_potential_rad(paths['radiation_save_path'])
        logging.info("Shape after adding radiation: %s", dataset_gl.data.shape)
        """

        # Convert to monthly resolution
        logging.info("Converting to monthly resolution...")
        dataset_gl.convert_to_monthly(meta_data_columns=cfg.metaData,
                                      vois_climate=vois_climate, # + ['pcsr']
                                      vois_topographical=vois_topographical)

        # Create DataLoader
        dataloader_gl = mbm.DataLoader(cfg,
                                       data=dataset_gl.data,
                                       random_seed=cfg.seed,
                                       meta_data_columns=cfg.metaData)
        logging.info("Number of monthly rows: %d", len(dataloader_gl.data))
        logging.info("Columns in the dataset: %s", dataloader_gl.data.columns)

        # Save processed data
        output_file = os.path.join(paths['csv_path'],
                                   output_file)
        dataloader_gl.data.to_csv(output_file, index=False)
        logging.info("Processed data saved to: %s", output_file)

        return dataloader_gl
    else:
        # Load preprocessed data
        try:
            input_file = os.path.join(paths['csv_path'],
                                      output_file)
            data_monthly = pd.read_csv(input_file)
            dataloader_gl = mbm.DataLoader(cfg,
                                           data=data_monthly,
                                           random_seed=cfg.seed,
                                           meta_data_columns=cfg.metaData)
            logging.info("Loaded preprocessed data.")
            logging.info("Number of monthly rows: %d", len(dataloader_gl.data))
            logging.info(
                "Number of annual rows: %d",
                len(dataloader_gl.data[dataloader_gl.data.PERIOD == 'annual']))
            logging.info(
                "Number of winter rows: %d",
                len(dataloader_gl.data[dataloader_gl.data.PERIOD == 'winter']))

            return dataloader_gl
        except FileNotFoundError as e:
            logging.error("Preprocessed data file not found: %s", e)
            return None


def get_CV_splits(dataloader_gl,
                test_split_on='YEAR',
                test_splits=None,
                random_state=0,
                test_size=0.2):
    # Split into training and test splits with train_test_split
    if test_splits is None:
        train_splits, test_splits = train_test_split(
            dataloader_gl.data[test_split_on].unique(),
            test_size=test_size,
            random_state=random_state)
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
    cv_splits = dataloader_gl.get_cv_split(n_splits=5, type_fold='group-meas-id')

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

    return cv_splits, test_set, train_set


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


def get_gl_area():
    # Load glacier metadata
    rgi_df = pd.read_csv(path_glacier_ids, sep=',')
    rgi_df.rename(columns=lambda x: x.strip(), inplace=True)
    rgi_df.sort_values(by='short_name', inplace=True)
    rgi_df.set_index('short_name', inplace=True)

    # Load the shapefile
    shapefile_path = "../../../data/GLAMOS/topo/SGI2020/SGI_2016_glaciers_copy.shp"
    gdf_shapefiles = gpd.read_file(shapefile_path)

    gl_area = {}

    for glacierName in rgi_df.index:
        if glacierName == 'clariden':
            rgi_shp = rgi_df.loc[
                'claridenL',
                'rgi_id_v6_2016_shp'] if 'claridenL' in rgi_df.index else None
        else:
            rgi_shp = rgi_df.loc[glacierName, 'rgi_id_v6_2016_shp']

        # Skip if rgi_shp is not found
        if pd.isna(rgi_shp) or rgi_shp is None:
            continue

        # Ensure matching data types
        rgi_shp = str(rgi_shp)
        gdf_mask_gl = gdf_shapefiles[gdf_shapefiles.RGIId.astype(str) ==
                                     rgi_shp]

        # If a glacier is found, get its area
        if not gdf_mask_gl.empty:
            gl_area[glacierName] = gdf_mask_gl.Area.iloc[
                0]  # Use .iloc[0] safely

    return gl_area


def correct_for_biggest_grid(df, group_columns, value_column="value"):
    """
    Assign the most frequent value in the specified column to all rows in each group
    if there are more than one unique value in the column within the group.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        group_columns (list): The columns to group by (e.g., YEAR, MONTHS).
        value_column (str): The name of the column to check and replace.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """

    def process_group(group):
        # Check if the column has more than one unique value in the group
        if group[value_column].nunique() > 1:
            # Find the most frequent value
            most_frequent_value = group[value_column].mode()[0]
            # Replace all values with the most frequent value
            group[value_column] = most_frequent_value
        return group

    # Apply the function to each group
    return df.groupby(group_columns).apply(process_group).reset_index(
        drop=True)


def correct_vars_grid(df_grid_monthly,
                      c_prec=1.434,
                      t_off=0.617,
                      temp_grad=-6.5 / 1000,
                      dpdz=1.5 / 10000):
    # Correct climate grids:
    for voi in [
            't2m', 'tp', 'slhf', 'sshf', 'ssrd', 'fal', 'str', 'u10', 'v10',
            'ALTITUDE_CLIMATE'
    ]:
        df_grid_monthly = correct_for_biggest_grid(
            df_grid_monthly,
            group_columns=["YEAR", "MONTHS"],
            value_column=voi)

    # New elevation difference with corrected altitude climate (same for all cells of big glacier):
    df_grid_monthly['ELEVATION_DIFFERENCE'] = df_grid_monthly[
        "POINT_ELEVATION"] - df_grid_monthly["ALTITUDE_CLIMATE"]

    # Apply T & P correction
    df_grid_monthly['t2m_corr'] = df_grid_monthly['t2m'] + (
        df_grid_monthly['ELEVATION_DIFFERENCE'] * temp_grad)
    df_grid_monthly['tp_corr'] = df_grid_monthly['tp'] * c_prec
    df_grid_monthly['t2m_corr'] += t_off

    # Apply elevation correction factor
    df_grid_monthly['tp_corr'] += df_grid_monthly['tp_corr'] * (
        df_grid_monthly['ELEVATION_DIFFERENCE'] * dpdz)

    return df_grid_monthly