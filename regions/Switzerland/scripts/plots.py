import matplotlib.pyplot as plt
import seaborn as sns
from cmcrameri import cm
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import config
from scripts.helpers import *

colors = get_cmap_hex(cm.batlow, 2)
color_xgb = colors[0]
# color_tim = colors[1]
color_tim = '#c51b7d'


def visualiseSplits(y_test, y_train, splits, colors=[color_xgb, color_tim]):
    # Visualise the cross validation splits
    fig, ax = plt.subplots(1, 6, figsize=(20, 5))
    ax[0].hist(y_train, bins=20, color=colors[0], density=False, alpha=0.5)
    ax[0].set_title('Train & Test PMB')
    ax[0].hist(y_test, bins=20, color=colors[1], density=False, alpha=0.5)
    ax[0].set_ylabel('Frequency')
    for i, (train_idx, val_idx) in enumerate(splits):
        # Check that there is no overlap between the training, val and test IDs
        ax[i + 1].hist(y_train[train_idx],
                       bins=20,
                       color=colors[0],
                       density=False,
                       alpha=0.5)
        ax[i + 1].hist(y_train[val_idx],
                       bins=20,
                       color=colors[1],
                       density=False,
                       alpha=0.5)
        ax[i + 1].set_title('CV train Fold ' + str(i + 1))
        ax[i + 1].set_xlabel('[m w.e.]')
    plt.tight_layout()


def predVSTruth(ax, grouped_ids, scores, hue = 'GLACIER', palette = None):
    legend_xgb = "\n".join(
        ((r"$\mathrm{RMSE_{xgb}}=%.3f$," % (scores["rmse"], )),
         (r"$\mathrm{MSE_{xgb}}=%.3f,$ " % (scores["mse"], )),
         (r"$\mathrm{MAE_{xgb}}=%.3f,$ " % (scores["mae"], )),
         (r"$\mathrm{\rho_{xgb}}=%.3f$" % (scores["pearson_corr"], ))))
    
    marker_xgb = 'o'
    sns.scatterplot(
        grouped_ids,
        x="target",
        y="pred",
        palette=palette,
        hue = hue,
        ax=ax,
        # alpha=0.8,
        color=color_xgb,
        marker=marker_xgb)

    ax.set_ylabel('Predicted PMB [m w.e.]', fontsize=20)
    ax.set_xlabel('Observed PMB [m w.e.]', fontsize=20)

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.03,
            0.98,
            legend_xgb,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=20, bbox=props)
    if hue is not None:
        ax.legend()
    else:
        ax.legend([], [], frameon=False)
    # diagonal line
    pt = (0, 0)
    ax.axline(pt, slope=1, color="grey", linestyle="-", linewidth=0.2)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.grid()
    
    # Set ylimits to be the same as xlimits
    ax.set_xlim(-15, 6)
    ax.set_ylim(-15, 6)
    plt.tight_layout()

def plotMeanPred(grouped_ids, ax):
    mean = grouped_ids.groupby('YEAR')['target'].mean().values
    std = grouped_ids.groupby('YEAR')['target'].std().values
    years = grouped_ids.YEAR.unique()
    ax.fill_between(
        years,
        mean - std,
        mean + std,
        color="orange",
        alpha=0.3,
    )
    ax.plot(years, mean, color="orange", label="mean target")
    ax.scatter(years, mean, color="orange", marker='x')
    ax.plot(years,
            grouped_ids.groupby('YEAR')['pred'].mean().values,
            color=color_xgb,
            label="mean pred",
            linestyle='--')
    ax.scatter(years,
               grouped_ids.groupby('YEAR')['pred'].mean().values,
               color=color_xgb,
               marker='x')
    ax.fill_between(
        years,
        grouped_ids.groupby('YEAR')['pred'].mean().values -
        grouped_ids.groupby('YEAR')['pred'].std().values,
        grouped_ids.groupby('YEAR')['pred'].mean().values +
        grouped_ids.groupby('YEAR')['pred'].std().values,
        color=color_xgb,
        alpha=0.3,
    )
    # rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    mae, rmse, pearson_corr = mean_absolute_error(
        grouped_ids.groupby('YEAR')['pred'].mean().values,
        mean), mean_squared_error(
            grouped_ids.groupby('YEAR')['pred'].mean().values,
            mean,
            squared=False), np.corrcoef(
                grouped_ids.groupby('YEAR')['pred'].mean().values, mean)[0, 1]
    legend_xgb = "\n".join((r"$\mathrm{MAE}=%.3f, \mathrm{RMSE}=%.3f,$ " % (
        mae,
        rmse,
    ), (r"$\mathrm{\rho}=%.3f$" % (pearson_corr, ))))
    ax.text(0.03,
            0.98,
            legend_xgb,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=20)


def FIPlot(best_estimator, feature_columns, vois_climate):
    FI = best_estimator.feature_importances_
    cmap = cm.devon
    color_palette_glaciers = get_cmap_hex(cmap, len(FI) + 5)
    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 1, 1)
    feature_importdf = pd.DataFrame(data={
        "variables": feature_columns,
        "feat_imp": FI
    })

    feature_importdf['variables'] = feature_importdf['variables'].apply(
        lambda x: vois_long_name[x] + f' ({x})'
        if x in vois_long_name.keys() else x)

    feature_importdf.sort_values(by="feat_imp", ascending=True, inplace=True)
    sns.barplot(feature_importdf,
                x='feat_imp',
                y='variables',
                dodge=False,
                ax=ax,
                palette=color_palette_glaciers)

    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Feature')


def plotGlAttr(ds, cmap=cm.batlow):
    # Plot glacier attributes
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    ds.masked_slope.plot(ax=ax[0], cmap=cmap)
    ax[0].set_title('Slope')
    ds.masked_elev.plot(ax=ax[1], cmap=cmap)
    ax[1].set_title('Elevation')
    ds.masked_aspect.plot(ax=ax[2], cmap=cmap)
    ax[2].set_title('Aspect')
    ds.masked_dis.plot(ax=ax[3], cmap=cmap)
    ax[3].set_title('Dis from border')
    plt.tight_layout()


def plotGlGrid(df_grid_annual, data_gl):
    # Plot glacier oggm grid and stakes
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    df_grid_annual_one_year = df_grid_annual[df_grid_annual.YEAR == 2006]
    ax.scatter(df_grid_annual_one_year.POINT_LON,
               df_grid_annual_one_year.POINT_LAT,
               s=1,
               label='OGGM grid',
               color=color_xgb)
    ax.scatter(data_gl.POINT_LON,
               data_gl.POINT_LAT,
               s=8,
               label='stakes',
               marker='x',
               color=color_tim)
    ax.legend()
    ax.set_title(
        f'OGGM grid and GLAMOS stakes for {df_grid_annual.GLACIER.iloc[0]}')


def plotNumMeasPerYear(data_gl, glacierName):
    # Plot number of measurements per year
    # Number of measurements per glacier per year:
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    num_gl_yr = data_gl.groupby(['YEAR',
                                 'PERIOD']).size().unstack().reset_index()
    num_gl_yr.plot(x='YEAR',
                   kind='bar',
                   stacked=True,
                   ax=ax,
                   colormap='flare_r')
    ax.set_ylabel('Number of measurements')
    ax.set_title(f'Number of measurements per year: {glacierName}',
                 fontsize=14)
    plt.tight_layout()


def plotGridSearchParams(custom_xgboost, param_grid, best_params):
    dfCVResults = pd.DataFrame(custom_xgboost.param_search.cv_results_)
    mask_raisonable = dfCVResults['mean_train_score'] >= -10
    dfCVResults_ = dfCVResults[mask_raisonable]
    fig = plt.figure(figsize=(15, 5))
    for i, param in enumerate(param_grid.keys()):

        dfParam = dfCVResults_.groupby(f'param_{param}')[[
            'split0_test_score', 'split1_test_score', 'split2_test_score',
            'split3_test_score', 'split4_test_score', 'mean_test_score',
            'std_test_score', 'rank_test_score', 'split0_train_score',
            'split1_train_score', 'split2_train_score', 'split3_train_score',
            'split4_train_score', 'mean_train_score', 'std_train_score'
        ]].mean()

        mean_test = abs(dfParam[[f'split{i}_test_score'
                                 for i in range(5)]].mean(axis=1))
        std_test = abs(dfParam[[f'split{i}_test_score'
                                for i in range(5)]].std(axis=1))

        mean_train = abs(dfParam[[f'split{i}_train_score'
                                  for i in range(5)]].mean(axis=1))
        std_train = abs(dfParam[[f'split{i}_train_score'
                                 for i in range(5)]].std(axis=1))

        # plot mean values with std
        ax = plt.subplot(1, len(param_grid.keys()), i + 1)
        ax.scatter(x=mean_test.index,
                   y=mean_test.values,
                   marker='x',
                   color=color_tim)
        ax.plot(mean_test.index,
                mean_test,
                color=color_tim,
                label='validation')
        ax.fill_between(mean_test.index,
                        mean_test - std_test,
                        mean_test + std_test,
                        alpha=0.2,
                        color=color_tim)

        ax.scatter(x=mean_train.index,
                   y=mean_train.values,
                   marker='x',
                   color=color_xgb)
        ax.plot(mean_train.index, mean_train, color=color_xgb, label='train')
        ax.fill_between(mean_train.index,
                        mean_train - std_train,
                        mean_train + std_train,
                        alpha=0.2,
                        color=color_xgb)
        # add vertical line of best param
        ax.axvline(best_params[param], color='red', linestyle='--')

        ax.set_ylabel(f'{config.LOSS} {loss_units[config.LOSS]}')
        ax.set_title(param)
        ax.legend()

    plt.suptitle('Grid search results')
    plt.tight_layout()


def plotGridSearchScore(custom_xgboost):
    dfCVResults = pd.DataFrame(custom_xgboost.param_search.cv_results_)
    mask_raisonable = dfCVResults['mean_train_score'] >= -10
    dfCVResults = dfCVResults[mask_raisonable]

    fig = plt.figure(figsize=(10, 5))
    mean_train = abs(dfCVResults.mean_train_score)
    std_train = abs(dfCVResults.std_train_score)
    mean_test = abs(dfCVResults.mean_test_score)
    std_test = abs(dfCVResults.std_test_score)

    plt.plot(mean_train, label='train', color=color_xgb)
    plt.plot(mean_test, label='validation', color=color_tim)

    # add std
    plt.fill_between(dfCVResults.index,
                     mean_train - std_train,
                     mean_train + std_train,
                     alpha=0.2,
                     color=color_xgb)
    plt.fill_between(dfCVResults.index,
                     mean_test - std_test,
                     mean_test + std_test,
                     alpha=0.2,
                     color=color_tim)
    
    # Add a line at the minimum
    pos_min = dfCVResults.mean_test_score.abs().idxmin()
    plt.axvline(pos_min, color='red', linestyle='--', label='min validation')
    
    plt.xlabel('Iteration')
    plt.ylabel(f'{config.LOSS} {loss_units[config.LOSS]}')
    plt.title('Grid search score over iterations')
    plt.legend()

def visualiseValPreds(model, splits, train_set, feature_columns, all_columns):
    fig, axs = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(25, 8))
    a = 0
    for (train_index, val_index), ax in zip(splits, axs.flatten()):
        # Get the training and validation data
        X_train = train_set['df_X'][all_columns].iloc[train_index]
        X_val = train_set['df_X'][all_columns].iloc[val_index]
        y_train = train_set['y'][train_index]
        y_val = train_set['y'][val_index]
        
        # fit on training set
        model.fit(X_train, y_train)
        xgb = model.set_params(device='cpu')
        
        # Make predictions on validation set:
        features_val, metadata_val = xgb._create_features_metadata(
            X_val, config.META_DATA)
        y_pred = xgb.predict(features_val)
        y_pred_agg = xgb.aggrPredict(metadata_val, config.META_DATA, features_val)

        # Aggregate predictions to annual or winter:
        all_columns = feature_columns + config.META_DATA + config.NOT_METADATA_NOT_FEATURES
        df_pred = X_val[all_columns].copy()
        df_pred['target'] = y_val
        grouped_ids = df_pred.groupby('ID').agg({
            'target': 'mean',
            'YEAR': 'first',
            'POINT_ID': 'first'
        })
        grouped_ids['pred'] = y_pred_agg
        grouped_ids['PERIOD'] = X_val[all_columns].groupby('ID')['PERIOD'].first()
        grouped_ids['GLACIER'] = grouped_ids['POINT_ID'].apply(
            lambda x: x.split('_')[0])

        mse, rmse, mae, pearson_corr = xgb.evalMetrics(metadata_val, y_pred,
                                                y_val)
        scores = {'mse': mse, 'rmse': rmse, 'mae': mae, 'pearson_corr': pearson_corr}
        predVSTruth(ax, grouped_ids, scores, hue = None)

    plt.tight_layout()
    
    
def visualiseInputs(train_set, test_set, vois_climate):    
    f, ax = plt.subplots(2, 11, figsize=(16, 6), sharey='row', sharex='col')
    train_set['df_X']['POINT_BALANCE'].plot.hist(ax=ax[0, 0],
                                                color=color_xgb,
                                                alpha=0.6,
                                                density=False)
    ax[0, 0].set_title('PMB')
    ax[0, 0].set_ylabel('Frequency (train)')
    train_set['df_X']['ELEVATION_DIFFERENCE'].plot.hist(ax=ax[0, 1],
                                                color=color_xgb,
                                                alpha=0.6,
                                                density=False)
    ax[0, 1].set_title('ELV_DIFF')
    train_set['df_X']['YEAR'].plot.hist(ax=ax[0, 2],
                                        color=color_xgb,
                                        alpha=0.6,
                                        density=False)
    ax[0, 2].set_title('YEARS')


    for i, voi_clim in enumerate(vois_climate + ['pcsr']):
        ax[0, 3 + i].set_title(voi_clim)
        train_set['df_X'][voi_clim].plot.hist(ax=ax[0, 3 + i],
                                            color=color_xgb,
                                            alpha=0.6,
                                            density=False)

    test_set['df_X']['POINT_BALANCE'].plot.hist(ax=ax[1, 0],
                                                color=color_tim,
                                                alpha=0.6,
                                                density=False)
    ax[1, 0].set_ylabel('Frequency (test)')
    test_set['df_X']['ELEVATION_DIFFERENCE'].plot.hist(bins=50,
                                                ax=ax[1, 1],
                                                color=color_tim,
                                                alpha=0.6,
                                                density=False)
    test_set['df_X']['YEAR'].plot.hist(ax=ax[1, 2],
                                    color=color_tim,
                                    alpha=0.6,
                                    density=False)

    for i, voi_clim in enumerate(vois_climate + ['pcsr']):
        test_set['df_X'][voi_clim].plot.hist(ax=ax[1, 3 + i],
                                            color=color_tim,
                                            alpha=0.6,
                                            density=False)
    # rotate xticks
    for ax in ax.flatten():
        ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel('')

    plt.tight_layout()
    
    
def PlotPredictions(xgb, test_glaciers, metadata_test, test_set, y_pred, y_pred_agg, feature_columns, all_columns):
    # Aggregate predictions to annual or winter:
    df_pred = test_set['df_X'][all_columns].copy()
    df_pred['target'] = test_set['y']
    grouped_ids = df_pred.groupby('ID').agg({
        'target': 'mean',
        'YEAR': 'first',
        'POINT_ID': 'first'
    })
    grouped_ids['pred'] = y_pred_agg
    grouped_ids['PERIOD'] = test_set['df_X'][
        feature_columns + config.META_DATA +
        config.NOT_METADATA_NOT_FEATURES].groupby('ID')['PERIOD'].first()
    grouped_ids['GLACIER'] = grouped_ids['POINT_ID'].apply(
        lambda x: x.split('_')[0])

    # grouped_ids = grouped_ids[grouped_ids.YEAR <= 2021]

    fig = plt.figure(figsize=(15, 10))
    colors_glacier = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f']
    color_palette_glaciers = dict(zip(grouped_ids.GLACIER.unique(),
                                    colors_glacier))

    ax = plt.subplot(2, 2, 1)
    grouped_ids_annual = grouped_ids[grouped_ids.PERIOD == 'annual']
    mse_annual, rmse_annual, mae_annual, pearson_corr_annual = xgb.evalMetrics(metadata_test, y_pred,
                                                test_set['y'], period = 'annual')
    scores_annual = {'mse': mse_annual,
                        'rmse': rmse_annual,
                        'mae': mae_annual,
                        'pearson_corr': pearson_corr_annual}
    predVSTruth(ax,
                grouped_ids_annual,
                scores_annual,
                hue='GLACIER',
                palette=color_palette_glaciers)
    ax.set_title('Annual MB', fontsize=24)

    grouped_ids_annual.sort_values(by='YEAR', inplace=True)
    ax = plt.subplot(2, 2, 2)
    plotMeanPred(grouped_ids_annual, ax)

    if 'winter' in grouped_ids.PERIOD.unique():
        grouped_ids_winter = grouped_ids[grouped_ids.PERIOD == 'winter']
        ax = plt.subplot(2, 2, 3)
        mse_winter, rmse_winter, mae_winter, pearson_corr_winter = xgb.evalMetrics(metadata_test, y_pred,
                                                test_set['y'], period = 'winter')
        scores_winter = {'mse': mse_winter,
                        'rmse': rmse_winter,
                        'mae': mae_winter,
                        'pearson_corr': pearson_corr_winter}
        predVSTruth(ax,
                    grouped_ids_winter,
                    scores_winter,
                    hue='GLACIER',
                    palette=color_palette_glaciers)
        ax.set_title('Winter MB', fontsize=24)

        ax = plt.subplot(2, 2, 4)
        grouped_ids_winter.sort_values(by='YEAR', inplace=True)
        plotMeanPred(grouped_ids_winter, ax)

    # ax.set_title('Mean yearly target and prediction')
    plt.suptitle(f'XGBoost tested on {test_glaciers}', fontsize=20)
    plt.tight_layout()