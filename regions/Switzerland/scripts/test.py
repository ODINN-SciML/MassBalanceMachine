glaciers = [
    re.split('_grid.csv', f)[0]
    for f in os.listdir('../../../data/GLAMOS/glacier-wide/grid/')
]
satellite_years = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
satellite_glaciers = [
    'adler', 'aletsch', 'allalin', 'basodino', 'claridenL', 'claridenU',
    'findelen', 'gries', 'hohlaub', 'limmern', 'oberaar', 'plattalva', 'rhone',
    'sanktanna', 'schwarzbach', 'schwarzberg'
]
# Feature columns:
feature_columns = [
    'ELEVATION_DIFFERENCE'
] + list(vois_climate) + list(vois_topographical) + ['pcsr']
all_columns = feature_columns + config.META_DATA + config.NOT_METADATA_NOT_FEATURES

RUN = True
if RUN:
    emptyfolder('results/nc/var_normal/')
    for glacierName in tqdm(satellite_glaciers, desc='Glaciers', position=0):
        print(f'----------------------\nProcessing glacier: {glacierName}')
        # Get stake measurements for the glacier
        rgi_gl = rgi_df.loc[glacierName]['rgi_id.v6']
        data_gl = data_glamos[data_glamos.RGIId == rgi_gl]
        dataset_gl = mbm.Dataset(data=data_gl,
                                 region_name='CH',
                                 data_path=path_PMB_GLAMOS_csv)

        # Get OGGM glacier mask and grid
        ds, glacier_indices, gdir = dataset_gl.get_glacier_mask(
            custom_working_dir)

        # Create pandas dataframe of glacier grid
        # df_grid_annual = dataset_gl.create_glacier_grid(custom_working_dir)
        # df_grid_annual["PERIOD"] = "annual"
        # df_grid_annual['GLACIER'] = glacierName
        # print('\nNumber of total (yearly) measurements:', len(df_grid_annual))

        # Get glacier grid (preprocessed):
        df_grid_monthly = pd.read_csv(path_glacier_grid +
                                      f'{glacierName}_grid.csv')
        df_grid_monthly['GLACIER'] = glacierName
        df_grid_monthly['POINT_ELEVATION'] = df_grid_monthly['topo']
        df_grid_monthly.drop_duplicates(inplace=True)  # remove duplicates
        df_grid_monthly = correctTP(df_grid_monthly)  # Correct T & P for altitude
        df_grid_monthly = df_grid_monthly[all_columns]

        # Add cumulative monthly SMB:
        df_grid_monthly = cumulativeMonthly(df_grid_monthly, custom_model)

        dataloader = mbm.DataLoader(data=df_grid_monthly,
                                    meta_data_columns=config.META_DATA)

        # get years with stake measurements:
        # years_stakes = data_gl['YEAR'].unique()
        # print('Years with stake measurements:', years_stakes)
        # print('\nNumber of years: {}, from {} to {}'.format(
        #     len(years_stakes), years_stakes[0], years_stakes[-1]))

        # Make glacier wide predictions over grid:
        grouped_ids_annual = GlacierWidePred(custom_model,
                                             glacierName,
                                             vois_climate,
                                             vois_topographical,
                                             c_prec,
                                             t_off,
                                             type_pred='annual')
        grouped_ids_winter = GlacierWidePred(custom_model,
                                             glacierName,
                                             vois_climate,
                                             vois_topographical,
                                             c_prec,
                                             t_off,
                                             type_pred='winter')

        # Save to netcdf for mapping:
        # path_lv95 = f"results/nc/var_corr/{glacierName}/lv95/"  # Swiss coordinate system
        # path_wgs84 = f"results/nc/var_corr/{glacierName}/wgs84/"  # Laton coordinate system
        path_lv95 = f"results/nc/var_normal/{glacierName}/lv95/"  # Swiss coordinate system
        path_wgs84 = f"results/nc/var_normal/{glacierName}/wgs84/"  # Laton coordinate system

        emptyfolder(path_lv95)
        emptyfolder(path_wgs84)

        for year in tqdm(satellite_years,
                         desc='Years',
                         leave=False,
                         position=1):
            # Annual SMB:
            pred_y_annual = grouped_ids_annual[grouped_ids_annual.YEAR ==
                                               year].drop(['YEAR'], axis=1)

            ds_pred_annual_latlon, ds_pred_annual_xy = predXarray(
                ds, gdir, pred_y_annual)
            save_to_netcdf(ds_pred_annual_latlon, path_wgs84,
                           f"{glacierName}_{year}.nc")
            save_to_netcdf(ds_pred_annual_xy, path_lv95,
                           f"{glacierName}_{year}.nc")

            # Winter SMB:
            pred_y_winter = grouped_ids_winter[grouped_ids_winter.YEAR ==
                                               year].drop(['YEAR'], axis=1)
            ds_pred_winter_latlon, ds_pred_winter_xy = predXarray(
                ds, gdir, pred_y_winter)
            save_to_netcdf(ds_pred_winter_latlon, path_wgs84,
                           f"{glacierName}_{year}_w.nc")
            save_to_netcdf(ds_pred_winter_xy, path_lv95,
                           f"{glacierName}_{year}_w.nc")

            # Save monthly grids:
            df_grid_monthly_gl_y = df_grid_monthly[
                (df_grid_monthly['GLACIER'] == glacierName)
                & (df_grid_monthly['YEAR'] == year)]

            for i, month in enumerate(month_abbr_hydr.keys()):
                if month == 'sep_':
                    continue
                monthNb = month_abbr_hydr[month]
                df_grid_gl_m = df_grid_monthly_gl_y[
                    df_grid_monthly_gl_y['MONTHS'] == month]

                # Get in format for map:
                df_grid_gl_m = df_grid_gl_m.groupby('ID').agg({
                    'YEAR':
                    'mean',
                    'POINT_LAT':
                    'mean',
                    'POINT_LON':
                    'mean',
                    'pred':
                    'mean',
                    'cum_pred':
                    'mean'
                })
                pred_y = df_grid_gl_m.drop(['YEAR'], axis=1)
                ds_pred_latlon, ds_pred_xy = predXarray(ds,
                                                        gdir,
                                                        pred_y,
                                                        pred_var='cum_pred')
                # ds_pred_latlon = GaussianFilter(ds_pred_latlon, sigma=0.5)

                # save grids
                save_to_netcdf(ds_pred_xy, path_lv95,
                               f"{glacierName}_{year}_{monthNb}.nc")
                save_to_netcdf(ds_pred_latlon, path_wgs84,
                               f"{glacierName}_{year}_{monthNb}.nc")
