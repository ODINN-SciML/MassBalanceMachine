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


def predVSTruth(ax, grouped_ids, mae, rmse, pearson_corr, hue='YEAR'):
    legend_xgb = "\n".join(
        (r"$\mathrm{MAE_{xgb}}=%.3f, \mathrm{RMSE_{xgb}}=%.3f,$ " % (
            mae,
            rmse,
        ), (r"$\mathrm{\rho_{xgb}}=%.3f$" % (pearson_corr, ))))

    marker_xgb = 'o'
    # colors = get_cmap_hex(cm.devon, len(grouped_ids[hue].unique())+2)
    # palette = sns.color_palette(colors, as_cmap=True)
    # if hue == 'YEAR':
    #     palette = cm.devon_r
    sns.scatterplot(
        grouped_ids,
        x="target",
        y="pred",
        # palette=palette,
        # hue=hue,
        ax=ax,
        # alpha=0.8,
        color=color_xgb,
        marker=marker_xgb)

    ax.set_ylabel('Predicted PMB [m w.e.]', fontsize=20)
    ax.set_xlabel('Observed PMB [m w.e.]', fontsize=20)

    ax.text(0.03,
            0.98,
            legend_xgb,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=20)
    #ax.legend()
    ax.legend([], [], frameon=False)
    # diagonal line
    pt = (0, 0)
    ax.axline(pt, slope=1, color="grey", linestyle="-", linewidth=0.2)
    ax.axvline(0, color="grey", linestyle="-", linewidth=0.2)
    ax.axhline(0, color="grey", linestyle="-", linewidth=0.2)
    ax.grid()
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
    plt.xlabel('Iteration')
    plt.ylabel(f'{config.LOSS} {loss_units[config.LOSS]}')
    plt.title('Grid search score over iterations')
    plt.legend()
