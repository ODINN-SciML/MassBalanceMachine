import matplotlib.pyplot as plt
import seaborn as sns
from cmcrameri import cm
import matplotlib.colors as mcolors

import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import config
from matplotlib.patches import Rectangle

from scripts.helpers import *
from scripts.xgb_helpers import *

colors = get_cmap_hex(cm.batlow, 2)
color_xgb = colors[0]
# color_tim = colors[1]
color_tim = '#c51b7d'


def visualiseSplits(y_test, y_train, splits, colors=[color_xgb, color_tim]):
    # Visualise the cross validation splits
    fig, ax = plt.subplots(1, 6, figsize=(20, 5))
    ax[0].hist(y_train, color=colors[0], density=False, alpha=0.5)
    ax[0].set_title('Train & Test PMB')
    ax[0].hist(y_test, color=colors[1], density=False, alpha=0.5)
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


def predVSTruth(ax, grouped_ids, scores, hue='GLACIER', palette=None):
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
        hue=hue,
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
            fontsize=20,
            bbox=props)
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
    ax.legend()


def FIPlot(best_estimator, feature_columns, vois_climate):
    FI = best_estimator.feature_importances_
    cmap = cm.devon
    color_palette_glaciers = get_cmap_hex(cmap, len(FI) + 5)
    fig = plt.figure(figsize=(15, 10))
    ax = plt.subplot(1, 1, 1)
    feature_importdf = pd.DataFrame(data={
        "variables": feature_columns,
        "feat_imp": FI
    })

    feature_importdf['variables'] = feature_importdf['variables'].apply(
        lambda x: vois_climate_long_name[x] + f' ({x})'
        if x in vois_climate_long_name.keys() else x)

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
    fig, ax = plt.subplots(2, 5, figsize=(18, 10))
    ds.masked_slope.plot(ax=ax[0, 0], cmap=cmap)
    ax[0, 0].set_title('Slope')
    ds.masked_elev.plot(ax=ax[0, 1], cmap=cmap)
    ax[0, 1].set_title('Elevation')
    ds.masked_aspect.plot(ax=ax[0, 2], cmap=cmap)
    ax[0, 2].set_title('Aspect')
    ds.masked_dis.plot(ax=ax[0, 3], cmap=cmap)
    ax[0, 3].set_title('Dis from border')
    ds.masked_hug.plot(ax=ax[0, 4], cmap=cmap)
    ax[0, 4].set_title('Hugonnet')

    ds.masked_cit.plot(ax=ax[1, 0], cmap=cmap)
    ax[1, 0].set_title('Consensus ice thickness')
    ds.masked_mit.plot(ax=ax[1, 1], cmap=cmap)
    ax[1, 1].set_title('Millan ice thickness')
    ds.masked_miv.plot(ax=ax[1, 2], cmap=cmap)
    ax[1, 2].set_title('Millan v')
    ds.masked_mivx.plot(ax=ax[1, 3], cmap=cmap)
    ax[1, 3].set_title('Millan vx')
    ds.masked_mivy.plot(ax=ax[1, 4], cmap=cmap)
    ax[1, 4].set_title('Millan vy')

    plt.tight_layout()


def plotGlGrid(df_grid_annual, data_gl):
    # Plot glacier oggm grid and stakes
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    one_year = df_grid_annual.YEAR.unique()[0]
    df_grid_annual_one_year = df_grid_annual[df_grid_annual.YEAR == one_year]
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


def plotGridSearchParams(cv_results_, param_grid, N=None):
    dfCVResults = pd.DataFrame(cv_results_)
    best_params = dfCVResults.sort_values('mean_test_score',
                                          ascending=False).iloc[0].params
    mask_raisonable = dfCVResults['mean_train_score'] >= -10
    dfCVResults_ = dfCVResults[mask_raisonable]
    dfCVResults_.sort_values('mean_test_score', ascending=False, inplace=True)
    if N is not None:
        dfCVResults_ = dfCVResults_.iloc[:10]
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


def plotGridSearchScore(cv_results_):
    dfCVResults = pd.DataFrame(cv_results_)
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


def visualiseValPreds(model, splits, train_set, feature_columns):
    all_columns = feature_columns + config.META_DATA + config.NOT_METADATA_NOT_FEATURES
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
        model_cpu = model.set_params(device='cpu')

        # Make predictions on validation set:
        features_val, metadata_val = model_cpu._create_features_metadata(
            X_val, config.META_DATA)
        y_pred = model_cpu.predict(features_val)
        y_pred_agg = model_cpu.aggrPredict(metadata_val, config.META_DATA,
                                           features_val)

        # Aggregate predictions to annual or winter:
        df_pred = X_val[all_columns].copy()
        df_pred['target'] = y_val
        grouped_ids = df_pred.groupby('ID').agg({
            'target': 'mean',
            'YEAR': 'first',
            'POINT_ID': 'first'
        })
        grouped_ids['pred'] = y_pred_agg
        grouped_ids['PERIOD'] = X_val[all_columns].groupby(
            'ID')['PERIOD'].first()
        grouped_ids['GLACIER'] = grouped_ids['POINT_ID'].apply(
            lambda x: x.split('_')[0])

        mse, rmse, mae, pearson_corr = model_cpu.evalMetrics(
            metadata_val, y_pred, y_val)
        scores = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'pearson_corr': pearson_corr
        }
        predVSTruth(ax, grouped_ids, scores, hue=None)

    plt.tight_layout()


def visualiseInputs(train_set, test_set, vois_climate):
    f, ax = plt.subplots(2,
                         len(vois_climate) + 4,
                         figsize=(16, 6),
                         sharey='row',
                         sharex='col')
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
    test_set['df_X']['ELEVATION_DIFFERENCE'].plot.hist(ax=ax[1, 1],
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


def PlotPredictions(grouped_ids, y_pred, metadata_test, test_set, model):
    fig = plt.figure(figsize=(15, 10))
    colors_glacier = [
        '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
        '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'
    ]
    color_palette_glaciers = dict(
        zip(grouped_ids.GLACIER.unique(), colors_glacier))
    print(color_palette_glaciers)
    ax1 = plt.subplot(2, 2, 1)
    grouped_ids_annual = grouped_ids[grouped_ids.PERIOD == 'annual']
    mse_annual, rmse_annual, mae_annual, pearson_corr_annual = model.evalMetrics(
        metadata_test, y_pred, test_set['y'], period='annual')
    scores_annual = {
        'mse': mse_annual,
        'rmse': rmse_annual,
        'mae': mae_annual,
        'pearson_corr': pearson_corr_annual
    }
    predVSTruth(ax1,
                grouped_ids_annual,
                scores_annual,
                hue='GLACIER',
                palette=color_palette_glaciers)
    ax1.set_title('Annual PMB', fontsize=24)

    grouped_ids_annual.sort_values(by='YEAR', inplace=True)
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title('Mean annual PMB', fontsize=24)
    plotMeanPred(grouped_ids_annual, ax2)

    if 'winter' in grouped_ids.PERIOD.unique():
        grouped_ids_winter = grouped_ids[grouped_ids.PERIOD == 'winter']
        ax3 = plt.subplot(2, 2, 3)
        mse_winter, rmse_winter, mae_winter, pearson_corr_winter = model.evalMetrics(
            metadata_test, y_pred, test_set['y'], period='winter')
        scores_winter = {
            'mse': mse_winter,
            'rmse': rmse_winter,
            'mae': mae_winter,
            'pearson_corr': pearson_corr_winter
        }
        predVSTruth(ax3,
                    grouped_ids_winter,
                    scores_winter,
                    hue='GLACIER',
                    palette=color_palette_glaciers)
        ax3.set_title('Winter PMB', fontsize=24)

        ax4 = plt.subplot(2, 2, 4)
        ax4.set_title('Mean winter PMB', fontsize=24)
        grouped_ids_winter.sort_values(by='YEAR', inplace=True)
        plotMeanPred(grouped_ids_winter, ax4)


def PlotIndividualGlacierPred(grouped_ids, figsize=(15, 22)):
    fig, axs = plt.subplots(len(grouped_ids['GLACIER'].unique()),
                            2,
                            figsize=figsize)

    colors_glacier = [
        '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
        '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'
    ]
    color_palette_glaciers = dict(
        zip(grouped_ids.GLACIER.unique(), colors_glacier))
    color_palette_period = dict(
        zip(grouped_ids.PERIOD.unique(),
            colors_glacier[:len(grouped_ids.GLACIER.unique()):]))

    for i, test_gl in enumerate(grouped_ids['GLACIER'].unique()):
        df_gl = grouped_ids[grouped_ids.GLACIER == test_gl]

        ax1 = axs[i, 0]
        ax2 = axs[i, 1]

        scores = {
            'mse':
            mean_squared_error(df_gl['target'], df_gl['pred']),
            'rmse':
            mean_squared_error(df_gl['target'], df_gl['pred'], squared=False),
            'mae':
            mean_absolute_error(df_gl['target'], df_gl['pred']),
            'pearson_corr':
            np.corrcoef(df_gl['target'], df_gl['pred'])[0, 1]
        }
        predVSTruth(ax1,
                    df_gl,
                    scores,
                    hue='PERIOD',
                    palette=color_palette_period)
        ax1.set_title(f'{test_gl.capitalize()}', fontsize=24)

        plotMeanPred(df_gl, ax2)

    plt.tight_layout()


def plotHeatmap(test_glaciers, data_glamos, glDirect, period='annual'):
    # Heatmap of mean mass balance per glacier:
    # Get the mean mass balance per glacier
    data_with_pot = data_glamos[(data_glamos.GLACIER.isin(glDirect))
                                & (data_glamos.PERIOD == period)]
    mean_mb_per_glacier = data_with_pot.groupby(['GLACIER', 'YEAR'
                                                 ])['POINT_BALANCE'].mean()
    matrix = pd.DataFrame(mean_mb_per_glacier).reset_index().pivot(
        index='GLACIER', columns='YEAR',
        values='POINT_BALANCE').sort_values(by='GLACIER')
    gl_per_el = data_with_pot.groupby(['GLACIER'])['POINT_ELEVATION'].mean()
    matrix = matrix.loc[gl_per_el.sort_values(ascending=False).index]

    # make index categorical
    matrix.index = pd.Categorical(matrix.index,
                                  categories=matrix.index,
                                  ordered=True)
    fig = plt.figure(figsize=(20, 15))
    ax = plt.subplot(1, 1, 1)
    sns.heatmap(data=matrix,
                center=0,
                cmap=cm.vik_r,
                cbar_kws={'label': '[m w.e. $a^{-1}$]'},
                ax=ax)

    # add patches for test glaciers
    for test_gl in test_glaciers:
        if test_gl not in matrix.index:
            continue
        height = matrix.index.get_loc(test_gl)
        row = np.where(matrix.loc[test_gl].notna())[0]
        split_indices = np.where(np.diff(row) != 1)[0] + 1
        continuous_sequences = np.split(row, split_indices)
        for patch in continuous_sequences:
            ax.add_patch(
                Rectangle((patch.min(), height),
                          patch.max() - patch.min() + 1,
                          1,
                          fill=False,
                          edgecolor='blue',
                          lw=3))


def TwoDPlots(ds, gdir, glacierName, grouped_ids_annual, grouped_ids_winter,
              year, axs):
    for j in range(2):
        ax = axs[j]
        # First column is winter
        pred_y_winter = grouped_ids_winter[grouped_ids_winter.YEAR ==
                                           year].drop(['YEAR'], axis=1)

        # Second column is annual
        pred_y_annual = grouped_ids_annual[grouped_ids_annual.YEAR ==
                                           year].drop(['YEAR'], axis=1)

        ds_pred_annual, ds_pred_annual_xy = predXarray(ds, gdir, pred_y_annual)

        ds_pred_winter, ds_pred_winter_xy = predXarray(ds, gdir, pred_y_winter)

        # # Find min and max values for color bar
        # min_val = min(pred_y_winter['pred'].min(), pred_y_annual['pred'].min())
        # max_val = max(pred_y_winter['pred'].max(), pred_y_annual['pred'].max())

        # # Set up a common normalization for both plots
        # norm = plt.Normalize(vmin=min_val, vmax=max_val)

        # norm = mcolors.TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)

        if j == 0:
            # Plot glacier grid with pred value
            # scatter = sns.scatterplot(
            #     pred_y_winter,
            #     x='POINT_LON',
            #     y='POINT_LAT',
            #     hue='pred',
            #     palette='coolwarm_r',
            #     s=20,
            #     edgecolor='k',
            #     legend=None,
            #     ax=ax,
            #     hue_norm=norm)  # Use the common norm for both plots)

            vmin, vmax = ds_pred_winter.pred_masked.min(
            ).values, ds_pred_winter.pred_masked.max().values
            if vmax >= 0 and vmin < 0:
                norm = mcolors.TwoSlopeNorm(
                    vmin=ds_pred_winter.pred_masked.min().values,
                    vcenter=0,
                    vmax=vmax)
                ds_pred_winter.pred_masked.plot(
                    cmap='coolwarm_r',
                    norm=norm,
                    ax=ax,
                    cbar_kwargs={'label': "[m w.e.]"})
            if vmax >= 0 and vmin >= 0:
                ds_pred_winter.pred_masked.plot(
                    cmap='Blues', ax=ax, cbar_kwargs={'label': "[m w.e.]"})
            else:
                ds_pred_winter.pred_masked.plot(
                    cmap='Reds_r', ax=ax, cbar_kwargs={'label': "[m w.e.]"})

            ax.set_title(f'{glacierName.capitalize()}: winter MB')

        if j == 1:
            # Plot glacier grid with pred value
            vmin, vmax = ds_pred_annual.pred_masked.min(
            ).values, ds_pred_annual.pred_masked.max().values
            if vmax >= 0 and vmin < 0:
                norm = mcolors.TwoSlopeNorm(
                    vmin=ds_pred_annual.pred_masked.min().values,
                    vcenter=0,
                    vmax=vmax)
                ds_pred_annual.pred_masked.plot(
                    cmap='coolwarm_r',
                    norm=norm,
                    ax=ax,
                    cbar_kwargs={'label': "[m w.e.]"})
            else:
                ds_pred_annual.pred_masked.plot(
                    cmap='Reds_r', ax=ax, cbar_kwargs={'label': "[m w.e.]"})
            ax.set_title('Annual MB, Year: {}'.format(year))

        # Add labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    # plt.suptitle(f'Glacier: {glacierName.capitalize()}, {year}')
    plt.tight_layout()
