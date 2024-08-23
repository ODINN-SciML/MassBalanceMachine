import matplotlib.pyplot as plt
import seaborn as sns
from cmcrameri import cm

from scripts.helpers import *

def visualiseSplits(y_test, y_train, splits):
    # Visualise the cross validation splits
    fig, ax = plt.subplots(1, 6, figsize=(20, 5))
    ax[0].hist(y_train, bins=20)
    ax[0].set_title('Train & Test PMB')
    ax[0].hist(y_test, bins=20)

    for i, (train_idx, val_idx) in enumerate(splits):
        # Check that there is no overlap between the training, val and test IDs
        # train_meas_id = df_X_train.iloc[train_idx]['ID'].unique()
        # val_meas_id = df_X_train.iloc[val_idx]['ID'].unique()
        # assert len(set(train_meas_id).intersection(set(val_meas_id))) == 0
        # assert(len(set(train_meas_id).intersection(set(test_meas_id))) == 0)
        # assert(len(set(val_meas_id).intersection(set(test_meas_id))) == 0)
        ax[i+1].hist(y_train[train_idx], bins=20)
        ax[i+1].hist(y_train[val_idx], bins=20)
        ax[i+1].set_title('CV train Fold ' + str(i + 1))
        
        
def predVSTruth(grouped_ids, mae, rmse, title):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    legend_xgb = "\n".join(
        (r"$\mathrm{MAE_{xgb}}=%.3f, \mathrm{RMSE_{xgb}}=%.3f$ " % (
            mae,
            rmse,
        ), ))

    marker_xgb = 'o'
    colors = get_cmap_hex(cm.batlow, 2)
    color_xgb = colors[0]
    sns.scatterplot(
        grouped_ids,
        x="target",
        y="pred",
        # color=color_xgb,
        hue = 'PERIOD',
        ax=ax,
        alpha=0.5,
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
    ax.set_title(title, fontsize=20)
    plt.tight_layout()