import matplotlib.pyplot as plt
import seaborn as sns
from cmcrameri import cm
import matplotlib.colors as mcolors
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import massbalancemachine as mbm

from matplotlib.patches import Rectangle

from scripts.helpers import *
from scripts.xgb_helpers import *

colors = get_cmap_hex(cm.batlow, 2)
color_xgb = colors[0]
color_tim = '#c51b7d'

color_winter = '#a6cee3'
color_annual = '#1f78b4'





def plotGlGrid(df_grid_annual, data_gl):
    # Plot glacier oggm grid and stakes
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    one_year = df_grid_annual.YEAR.unique()[0]
    df_grid_annual_one_year = df_grid_annual[df_grid_annual.YEAR == one_year]
    ax.scatter(df_grid_annual_one_year.POINT_LON,
               df_grid_annual_one_year.POINT_LAT,
               s=1,
               color="grey")
    data_winter = data_gl[data_gl.PERIOD == 'winter']
    data_annual = data_gl[data_gl.PERIOD == 'annual']
    ax.scatter(data_winter.POINT_LON,
               data_winter.POINT_LAT,
               s=10,
               label='Winter',
               marker='x',
               color=color_winter,
               alpha=0.8)
    ax.scatter(data_annual.POINT_LON,
               data_annual.POINT_LAT,
               s=10,
               label='Annual',
               marker='x',
               color=color_annual,
               alpha=0.8)
    ax.legend(fontsize=18, markerscale=2)


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








def visualiseValPreds(model, splits, train_set, feature_columns, cfg: mbm.Config):
    all_columns = feature_columns + cfg.fieldsNotFeatures
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
        features_val, metadata_val = model_cpu._create_features_metadata(X_val)
        y_pred = model_cpu.predict(features_val)
        y_pred_agg = model_cpu.aggrPredict(metadata_val, features_val)

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

        geoData_annual = mbm.GeoData(pred_y_annual)
        geoData_annual.pred_to_xr(ds, gdir, pred_var='pred')

        geoData_winter = mbm.GeoData(pred_y_winter)
        geoData_winter.pred_to_xr(ds, gdir, pred_var='pred')

        if j == 0:
            vmin, vmax = geoData_winter.ds_latlon.pred_masked.min(
            ).values, geoData_winter.ds_latlon.pred_masked.max().values
            if vmax >= 0 and vmin < 0:
                # find the biggest of the two absolute values
                max_abs_value = max(abs(vmin), abs(vmax))
                norm = mcolors.TwoSlopeNorm(vmin=-max_abs_value,
                                            vcenter=0,
                                            vmax=max_abs_value)
                geoData_winter.ds_latlon.pred_masked.plot(
                    cmap='coolwarm_r',
                    norm=norm,
                    ax=ax,
                    cbar_kwargs={'label': "[m w.e.]"})

            elif vmax >= 0 and vmin >= 0:
                geoData_winter.ds_latlon.pred_masked.plot(
                    cmap='Blues', ax=ax, cbar_kwargs={'label': "[m w.e.]"})
            else:
                geoData_winter.ds_latlon.pred_masked.plot(
                    cmap='Reds_r', ax=ax, cbar_kwargs={'label': "[m w.e.]"})

            ax.set_title(f'{glacierName.capitalize()}: winter MB')

        if j == 1:
            # Plot glacier grid with pred value
            vmin, vmax = geoData_annual.ds_latlon.pred_masked.min(
            ).values, geoData_annual.ds_latlon.pred_masked.max().values
            if vmax >= 0 and vmin < 0:
                max_abs_value = max(abs(vmin), abs(vmax))
                norm = mcolors.TwoSlopeNorm(vmin=-max_abs_value,
                                            vcenter=0,
                                            vmax=max_abs_value)
                geoData_annual.ds_latlon.pred_masked.plot(
                    cmap='coolwarm_r',
                    norm=norm,
                    ax=ax,
                    cbar_kwargs={'label': "[m w.e.]"})
            else:
                geoData_annual.ds_latlon.pred_masked.plot(
                    cmap='Reds_r', ax=ax, cbar_kwargs={'label': "[m w.e.]"})
            ax.set_title('Annual MB, Year: {}'.format(year))

        # Add labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    # plt.suptitle(f'Glacier: {glacierName.capitalize()}, {year}')
    plt.tight_layout()


def Plot2DPred(fig, vmin, vmax, ds_pred, YEAR, month, ax, cfg:mbm.Config, savefig=False):
    monthNb = cfg.month_abbr_hydr[month]
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    pcm = ds_pred.pred_masked.plot(cmap='coolwarm_r',
                                   norm=norm,
                                   add_colorbar=False,
                                   ax=ax)
    cb = fig.colorbar(pcm)
    cb.ax.set_yscale('linear')
    cb.set_label("[m w.e.]")  # Set the label here

    ax.set_title(f'{month.capitalize()}, {YEAR}')
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.grid()
    if savefig:
        plt.savefig(f"results/gif/{monthNb}_{month}.png",
                    format="png",
                    dpi=300,
                    bbox_inches="tight")





