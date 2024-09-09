import matplotlib.pyplot as plt
import seaborn as sns
from cmcrameri import cm
import pandas as pd

from scripts.helpers import *

colors = get_cmap_hex(cm.batlow, 2)
color_xgb = colors[0]
# color_tim = colors[1]
color_tim = '#c51b7d'


def visualiseSplits(y_test, y_train, splits, colors=[color_xgb, color_tim]):
    # Visualise the cross validation splits
    fig, ax = plt.subplots(1, 6, figsize=(20, 5))
    ax[0].hist(y_train, bins=20, color=colors[0])
    ax[0].set_title('Train & Test PMB')
    ax[0].hist(y_test, bins=20, color=colors[1])

    for i, (train_idx, val_idx) in enumerate(splits):
        # Check that there is no overlap between the training, val and test IDs
        # train_meas_id = df_X_train.iloc[train_idx]['ID'].unique()
        # val_meas_id = df_X_train.iloc[val_idx]['ID'].unique()
        # assert len(set(train_meas_id).intersection(set(val_meas_id))) == 0
        # assert(len(set(train_meas_id).intersection(set(test_meas_id))) == 0)
        # assert(len(set(val_meas_id).intersection(set(test_meas_id))) == 0)
        ax[i + 1].hist(y_train[train_idx], bins=20, color=colors[0])
        ax[i + 1].hist(y_train[val_idx], bins=20, color=colors[1])
        ax[i + 1].set_title('CV train Fold ' + str(i + 1))


def predVSTruth(ax, grouped_ids, mae, rmse, pearson_corr):
    legend_xgb = "\n".join(
        (r"$\mathrm{MAE_{xgb}}=%.3f, \mathrm{RMSE_{xgb}}=%.3f,$ " % (
            mae,
            rmse,
        ), (r"$\mathrm{\rho_{xgb}}=%.3f$" % (pearson_corr, ))))

    marker_xgb = 'o'
    #colors = get_cmap_hex(cm.glasgow, 2)
    palette = sns.color_palette("magma_r", as_cmap=True)
    sns.scatterplot(
        grouped_ids,
        x="target",
        y="pred",
        palette=palette,
        hue='YEAR',
        ax=ax,
        # alpha=0.8,
        marker=marker_xgb)

    ax.set_ylabel('Predicted PMB [m w.e.]', fontsize=20)
    ax.set_xlabel('Observed PMB [m w.e.]', fontsize=20)

    ax.text(0.03,
            0.98,
            legend_xgb,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=20)
    ax.legend()
    # ax.legend([], [], frameon=False)
    # diagonal line
    pt = (0, 0)
    ax.axline(pt, slope=1, color="grey", linestyle="-", linewidth=0.2)
    ax.axvline(0, color="grey", linestyle="-", linewidth=0.2)
    ax.axhline(0, color="grey", linestyle="-", linewidth=0.2)
    ax.grid()
    plt.tight_layout()


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
        lambda x: vois_climate_long_name[x] + f' ({x})'
        if x in vois_climate else x)

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
    num_gl_yr.plot(x='YEAR', kind='bar', stacked=True, ax=ax,
               colormap='flare_r')
    ax.set_ylabel('Number of measurements')
    ax.set_title(f'Number of measurements per year: {glacierName}',
                 fontsize=14)
    plt.tight_layout()
