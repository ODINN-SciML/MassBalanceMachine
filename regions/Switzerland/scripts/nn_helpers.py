import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from skorch.helper import SliceDataset
from datetime import datetime
import massbalancemachine as mbm
from tqdm.notebook import tqdm
import ast

from regions.Switzerland.scripts.plots import *

def plot_training_history(custom_nn, skip_first_n=0, save=True):
    history = custom_nn.history

    # Skip first N entries if specified
    if skip_first_n > 0:
        history = history[skip_first_n:]

    epochs = [entry['epoch'] for entry in history]
    train_loss = [entry.get('train_loss') for entry in history]
    valid_loss = [
        entry.get('valid_loss') for entry in history if 'valid_loss' in entry
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, train_loss, label='Training Loss', marker='o')

    if valid_loss:
        # Align epochs with valid_loss length
        valid_epochs = epochs[:len(valid_loss)]
        plt.plot(valid_epochs, valid_loss, label='Validation Loss', marker='x')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"Training and Validation Loss (Skipped first {skip_first_n} epochs)"
        if skip_first_n > 0 else "Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save:
        # save the plot
        # Create a folder to save figures (optional)
        save_dir = "figures"
        os.makedirs(save_dir, exist_ok=True)

        # Save the figure
        plt.savefig(os.path.join(save_dir, "training_history.png"),
                    dpi=300,
                    bbox_inches='tight')
        plt.close()  # closes the plot to avoid display in notebooks/scripts


def PlotPredictions_NN(grouped_ids):
    fig = plt.figure(figsize=(15, 10))
    colors_glacier = [
        '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
        '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'
    ]
    color_palette_glaciers = dict(
        zip(grouped_ids.GLACIER.unique(), colors_glacier))
    ax1 = plt.subplot(2, 2, 1)
    grouped_ids_annual = grouped_ids[grouped_ids.PERIOD == 'annual']

    y_true_mean = grouped_ids_annual['target']
    y_pred_agg = grouped_ids_annual['pred']

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
    y_pred_agg = grouped_ids_winter['pred']

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


def evaluate_model_and_group_predictions(
    custom_NN_model,
    df_X_subset,
    y,
    cfg,
    months_head_pad,
    months_tail_pad,
):
    # Create features and metadata
    features, metadata = mbm.data_processing.utils.create_features_metadata(cfg, df_X_subset)

    # Ensure features and targets are on CPU
    if hasattr(features, 'cpu'):
        features = features.cpu()
    if hasattr(y, 'cpu'):
        y = y.cpu()

    # Define the dataset for the NN
    dataset = mbm.data_processing.AggregatedDataset(
        cfg,
        features=features,
        metadata=metadata,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
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
    mse, rmse, mae, pearson, r2, bias = custom_NN_model.evalMetrics(y_pred, y_true)
    scores = {
        'score': score,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'pearson': pearson,
        'r2': r2,
        'bias': bias,
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


def process_glacier_grids(cfg, glacier_list, periods_per_glacier, all_columns,
                          loaded_model, path_glacier_grid_glamos,
                          path_save_glw, path_xr_grids):
    """
    Process distributed MB grids for a list of glaciers using pre-trained models.

    Parameters
    ----------
    cfg : object
        Configuration object with dataPath attribute.
    glacier_list : list of str
        List of glacier names to process.
    periods_per_glacier : dict
        Dictionary mapping glacier names to periods (years) for processing.
    all_columns : list of str
        List of required column names in glacier grid files.
    loaded_model : object
        Pre-trained model to use for prediction.
    path_glacier_grid_glamos : str
        Relative path to glacier grids within cfg.dataPath.
    emptyfolder : function
        Function to empty a folder.
    path_save_glw : str
        Path where results will be saved.
    path_xr_grids : str
        Path to xr_masked_grids.
    """
    # Ensure save path exists
    os.makedirs(path_save_glw, exist_ok=True)

    emptyfolder(path_save_glw)

    for glacier_name in glacier_list:
        glacier_path = os.path.join(cfg.dataPath, path_glacier_grid_glamos, glacier_name)

        if not os.path.exists(glacier_path):
            print(f"Folder not found for {glacier_name}, skipping...")
            continue

        glacier_files = sorted([f for f in os.listdir(glacier_path) if glacier_name in f])

        geodetic_range = range(np.min(periods_per_glacier[glacier_name]),
                               np.max(periods_per_glacier['aletsch']) + 1)

        years = [int(file_name.split('_')[2].split('.')[0]) for file_name in glacier_files]
        years = [y for y in years if y in geodetic_range]

        print(f"Processing {glacier_name} ({len(years)} files)")

        for year in tqdm(years, desc=f"Processing {glacier_name}", leave=False):
            file_name = f"{glacier_name}_grid_{year}.parquet"
            file_path = os.path.join(cfg.dataPath, path_glacier_grid_glamos, glacier_name, file_name)

            df_grid_monthly = pd.read_parquet(file_path)
            df_grid_monthly.drop_duplicates(inplace=True)

            # Keep only necessary columns
            df_grid_monthly = df_grid_monthly[[col for col in all_columns if col in df_grid_monthly.columns]]
            df_grid_monthly = df_grid_monthly.dropna()

            # Create geodata object
            geoData = mbm.geodata.GeoData(df_grid_monthly)

            # Compute and save gridded MB
            path_glacier_dem = os.path.join(path_xr_grids, f"{glacier_name}_{year}.zarr")

            geoData.gridded_MB_pred(
                df_grid_monthly,
                loaded_model,
                glacier_name,
                year,
                all_columns,
                path_glacier_dem,
                path_save_glw,
                save_monthly_pred=True,
                type_model='NN'
            )
            
            
def retrieve_best_params(path, sort_values = 'valid_loss'):
    # Open grid_search results
    gs_results = pd.read_csv(path).sort_values(by=sort_values, ascending=True)
    
    # Take best row
    best_params = gs_results.iloc[0].to_dict()

    # Clean it up into a proper dict
    params = {}

    for key, value in best_params.items():
        if key in ['valid_loss', 'train_loss', 'test_rmse', 'status', 'error']:
            continue  # skip these

        if isinstance(value, str):
            # Convert optimizer string to actual torch class
            if "torch.optim" in value:
                # e.g. "<class 'torch.optim.adamw.AdamW'>" → torch.optim.AdamW
                cls_name = value.split("'")[1].split('.')[-1]
                params[key] = getattr(torch.optim, cls_name)
            else:
                # Convert string representations of lists, bools, numbers
                try:
                    params[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    params[key] = value
        else:
            params[key] = value

    return params