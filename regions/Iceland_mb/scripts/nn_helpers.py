import matplotlib.pyplot as plt
import os 
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from skorch.helper import SliceDataset
from skorch.callbacks import Callback
import torch

from scripts.plots import *

class SaveBestAtEpochs(Callback):
    def __init__(self, epochs, prefix="nn_model_best_epoch"):
        self.epochs = set(epochs)
        self.prefix = prefix
        self.best_score = float('inf')
        self.best_state = None

    def on_epoch_end(self, net, **kwargs):
        epoch = net.history[-1]['epoch']
        valid_loss = net.history[-1]['valid_loss']
        if valid_loss < self.best_score:
            self.best_score = valid_loss
            self.best_state = {k: v.cpu().clone() for k, v in net.module_.state_dict().items()}
        if epoch in self.epochs and self.best_state is not None:
            filename = f"{self.prefix}_{epoch}.pt"
            torch.save(self.best_state, filename)
            print(f"Best model up to epoch {epoch} saved as {filename}")

def plot_training_history(custom_nn, skip_first_n=0):
    history = custom_nn.history

    # Skip first N entries if specified
    if skip_first_n > 0:
        history = history[skip_first_n:]

    epochs = [entry['epoch'] for entry in history]
    train_loss = [entry.get('train_loss') for entry in history]
    valid_loss = [entry.get('valid_loss') for entry in history if 'valid_loss' in entry]

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, train_loss, label='Training Loss', marker='o')

    if valid_loss:
        # Align epochs with valid_loss length
        valid_epochs = epochs[:len(valid_loss)]
        plt.plot(valid_epochs, valid_loss, label='Validation Loss', marker='x')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss (Skipped first {skip_first_n} epochs)" if skip_first_n > 0 else "Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # save the plot
    # Create a folder to save figures (optional)
    save_dir = "figures"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, "training_history.png"),
                dpi=300,
                bbox_inches='tight')
    plt.close()  # closes the plot to avoid display in notebooks/scripts


def predVSTruth_all(grouped_ids, mae, rmse, title):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    legend_nn = "\n".join(
        (r"$\mathrm{MAE_{nn}}=%.3f, \mathrm{RMSE_{nn}}=%.3f$ " % (
            mae,
            rmse,
        ), ))

    marker_nn = 'o'
    sns.scatterplot(grouped_ids,
                    x="target",
                    y="pred",
                    ax=ax,
                    alpha=0.5,
                    marker=marker_nn)

    ax.set_ylabel('Predicted PMB [m w.e.]', fontsize=20)
    ax.set_xlabel('Observed PMB [m w.e.]', fontsize=20)

    ax.text(0.03,
            0.98,
            legend_nn,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=20)
    ax.legend([], [], frameon=False)
    # diagonal line
    pt = (0, 0)
    ax.axline(pt, slope=1, color="grey", linestyle="-", linewidth=0.2)
    ax.axvline(0, color="grey", linestyle="-", linewidth=0.2)
    ax.axhline(0, color="grey", linestyle="-", linewidth=0.2)
    ax.grid()
    ax.set_title(title, fontsize=20)
    plt.tight_layout()
    
    
def PlotPredictions_NN(grouped_ids):
    fig = plt.figure(figsize=(15, 10))

    palette = sns.color_palette("husl", n_colors=len(grouped_ids.GLACIER.unique()))
    color_palette_glaciers = dict(zip(grouped_ids.GLACIER.unique(), palette))
    ax1 = plt.subplot(2, 2, 1)
    grouped_ids_annual = grouped_ids[grouped_ids.PERIOD == 'annual']

    y_true_mean = grouped_ids_annual['target']
    y_pred_agg  = grouped_ids_annual['pred']

    scores_annual = {
        'mse': mean_squared_error(y_true_mean, y_pred_agg),
        'rmse': root_mean_squared_error(y_true_mean, y_pred_agg),
        'mae': mean_absolute_error(y_true_mean, y_pred_agg),
        'pearson_corr': np.corrcoef(y_true_mean, y_pred_agg)[0, 1]
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

    grouped_ids_winter = grouped_ids[grouped_ids.PERIOD == 'winter']
    y_true_mean = grouped_ids_winter['target']
    y_pred_agg  = grouped_ids_winter['pred']

    ax3 = plt.subplot(2, 2, 3)
    scores_winter = {
        'mse': mean_squared_error(y_true_mean, y_pred_agg),
        'rmse': root_mean_squared_error(y_true_mean, y_pred_agg),
        'mae': mean_absolute_error(y_true_mean, y_pred_agg),
        'pearson_corr': np.corrcoef(y_true_mean, y_pred_agg)[0, 1]
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

    plt.tight_layout()

def evaluate_model_and_group_predictions(custom_NN_model, df_X_subset, y, cfg, mbm):
    # Create features and metadata
    features, metadata = custom_NN_model._create_features_metadata(df_X_subset)

    # Ensure features and targets are on CPU
    if hasattr(features, 'cpu'):
        features = features.cpu()
    if hasattr(y, 'cpu'):
        y = y.cpu()

    # Define the dataset for the NN
    dataset = mbm.data_processing.AggregatedDataset(cfg,
                                                    features=features,
                                                    metadata=metadata,
                                                    targets=y)
    dataset = [SliceDataset(dataset, idx=0), SliceDataset(dataset, idx=1)]

    # Make predictions
    y_pred = custom_NN_model.predict(dataset[0])
    y_pred_agg = custom_NN_model.aggrPredict(dataset[0])

    # Get true values
    batchIndex = np.arange(len(y_pred_agg))
    y_true = np.array([e for e in dataset[1][batchIndex]])

    # Compute scores
    score = custom_NN_model.score(dataset[0], dataset[1])
    mse, rmse, mae, pearson = custom_NN_model.evalMetrics(y_pred, y_true)
    scores = {
        'score': score,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'pearson': pearson
    }

    # Create grouped prediction DataFrame
    ids = dataset[0].dataset.indexToId(batchIndex)
    grouped_ids = pd.DataFrame({
        'target': [e[0] for e in dataset[1]],
        'ID': ids,
        'pred': y_pred_agg
    })

    # Add period
    periods_per_ids = df_X_subset.groupby('ID')['PERIOD'].first()
    grouped_ids = grouped_ids.merge(periods_per_ids, on='ID')

    # Add glacier name
    glacier_per_ids = df_X_subset.groupby('ID')['GLACIER'].first()
    grouped_ids = grouped_ids.merge(glacier_per_ids, on='ID')

    # Add YEAR
    years_per_ids = df_X_subset.groupby('ID')['YEAR'].first()
    grouped_ids = grouped_ids.merge(years_per_ids, on='ID')
    
    return grouped_ids, scores, ids, y_pred


def PlotPredictionsCombined_NN(grouped_ids, region_name="", include_summer=False):
    fig = plt.figure(figsize=(12, 10))
    period_colors = {'annual': '#e31a1c', 'winter': '#1f78b4', 'summer': '#33a02c'}

    # Compute metrics for each period
    metrics = {}
    for period in ['annual', 'winter', 'summer']:
        if period == 'summer' and not include_summer:
            continue
        subset = grouped_ids[grouped_ids.PERIOD == period]
        if len(subset) > 0:
            rmse = np.sqrt(mean_squared_error(subset.target, subset.pred))
            # Pearson correlation
            if len(subset) > 1:
                rho = np.corrcoef(subset.target, subset.pred)[0, 1]
            else:
                rho = np.nan
            metrics[period] = (rmse, rho)

    # Combined metrics
    rmse_all = np.sqrt(mean_squared_error(grouped_ids.target, grouped_ids.pred))
    if len(grouped_ids) > 1:
        rho_all = np.corrcoef(grouped_ids.target, grouped_ids.pred)[0, 1]
    else:
        rho_all = np.nan
    metrics['combined'] = (rmse_all, rho_all)

    ax = plt.subplot(1, 1, 1)
    for period in grouped_ids.PERIOD.unique():
        if period == 'summer' and not include_summer:
            continue
        subset = grouped_ids[grouped_ids.PERIOD == period]
        if len(subset) > 0:
            ax.scatter(subset.target, subset.pred,
                       color=period_colors.get(period, 'gray'),
                       alpha=0.7, s=80, label=f"{period}")

    min_val = min(grouped_ids.target.min(), grouped_ids.pred.min())
    max_val = max(grouped_ids.target.max(), grouped_ids.pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2)

    # Build metrics text
    metrics_text = f"Combined: RMSE: {metrics['combined'][0]:.2f} m w.e., ρ: {metrics['combined'][1]:.2f}\n"
    if 'annual' in metrics:
        metrics_text += f"Annual: RMSE: {metrics['annual'][0]:.2f} m w.e., ρ: {metrics['annual'][1]:.2f}\n"
    if 'winter' in metrics:
        metrics_text += f"Winter: RMSE: {metrics['winter'][0]:.2f} m w.e., ρ: {metrics['winter'][1]:.2f}\n"
    if include_summer and 'summer' in metrics:
        metrics_text += f"Summer: RMSE: {metrics['summer'][0]:.2f} m w.e., ρ: {metrics['summer'][1]:.2f}"

    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=20)

    ax.legend(fontsize=24, loc='lower right')
    ax.set_xlabel('Observed PMB [m w.e.]', fontsize=27)
    ax.set_ylabel('Predicted PMB [m w.e.]', fontsize=27)
    ax.set_title(f'PMB - Pred vs. Obs ({region_name})', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=21)
    plt.tight_layout()