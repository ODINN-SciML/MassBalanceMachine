"""
This script contains functions for plotting the results of the mode.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/06/2024
"""

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.inspection import permutation_importance

from model_methods import *


def plot_fold_distribution(splits, df_X_train, plotting):
    # Count number of annual and seasonal per fold

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)

    # Create a color map or list for the bars
    colors = ['#4e8cd9', '#4b9c8d']

    n_months_to_season = {5: 'summer', 7: 'winter', 12: 'annual'}

    for i, (train_index, val_index) in enumerate(splits):
        ax = axes[i]

        n_months_train = df_X_train.iloc[train_index]['n_months']
        n_months_val = df_X_train.iloc[val_index]['n_months']

        # Counts
        n_months_train_counts = dict(zip(*np.unique(n_months_train, return_counts=True)))
        n_months_val_counts = dict(zip(*np.unique(n_months_val, return_counts=True)))

        # n_months_values = sorted(set(n_months_train_counts.keys()).union(n_months_val_counts.keys()))
        n_months_values = sorted(set(n_months_train_counts.keys()).union(n_months_val_counts.keys()))
        season_names = [n_months_to_season[n_months] for n_months in n_months_values]

        train_positions = np.arange(len(n_months_values))
        val_positions = train_positions + 0.4

        train_counts = [n_months_train_counts.get(x, 0) for x in n_months_values]
        ax.bar(train_positions, train_counts, width=0.4, label='Train', color=colors[0], alpha=0.8)

        val_counts = [n_months_val_counts.get(x, 0) for x in n_months_values]
        ax.bar(val_positions, val_counts, width=0.4, label='Validation', color=colors[1], alpha=0.8)

        # Annotate each bar with the respective count
        for j in range(len(n_months_values)):
            train_count = n_months_train_counts.get(n_months_values[j], 0)
            val_count = n_months_val_counts.get(n_months_values[j], 0)
            train_pos = train_positions[j]
            val_pos = val_positions[j]

            ax.text(train_pos, train_count + max(train_count, val_count) * 0.01, str(train_count),
                    ha='center', va='bottom', fontsize=8, color='k')

            ax.text(val_pos, val_count + max(train_count, val_count) * 0.01, str(val_count),
                    ha='center', va='bottom', fontsize=8, color='k')

        ax.set_title(f'Fold {i + 1}')
        ax.set_ylabel('Count')
        ax.set_xticks(train_positions + 0.2)
        ax.set_xticklabels(season_names)
        ax.yaxis.grid(linestyle='--')
        ax.set_axisbelow(True)

        if i == 0:
            ax.legend()

    if plotting:
        plt.savefig(f'.././data/plots/model-training/distribution_folds.svg', dpi=600, format='svg',
                    bbox_inches='tight')

    plt.tight_layout()
    plt.show()


# Get true values (means) and predicted values (aggregates)

def get_ytrue_y_pred_agg(y_true, y_pred, X):
    # Extract the metadata
    metadata = X[:, -3:]  # Assuming last three columns are the metadata
    unique_ids = np.unique(metadata[:, 0])  # Assuming ID is the first column
    y_pred_agg_all = []
    y_true_mean_all = []

    # Loop over each unique ID to calculate MSE
    for uid in unique_ids:
        # Indexes for the current ID
        indexes = metadata[:, 0] == uid
        # Aggregate y_pred for the current ID
        y_pred_agg = np.sum(y_pred[indexes])
        y_pred_agg_all.append(y_pred_agg)
        # True value is the mean of true values for the group
        y_true_mean = np.mean(y_true[indexes])
        y_true_mean_all.append(y_true_mean)

    y_pred_agg_all_arr = np.array(y_pred_agg_all)
    y_true_mean_all_arr = np.array(y_true_mean_all)

    return y_true_mean_all_arr, y_pred_agg_all_arr


def plot_prediction_validation(X, y, model, idc_list, title, plotting, fname):
    """
    Plots model predictions and evaluation metrics for each fold of cross-validation.

    Parameters:
    X (array-like): Features.
    y (array-like): Target variable.
    model: Model object with 'fit' and 'predict' methods.
    idc_list (list): List of tuples containing train-test indices for each fold.
    title (str): Title for the plot.
    plotting (bool): Whether to plot the model predictions
    fname: Filename for the plot.

    Returns:
    None
    """
    # Determine the number of folds and the layout of subplots
    num_folds = len(idc_list)

    # Calculate the number of rows based on the value of num_folds
    num_rows = 1 if num_folds <= 5 else num_folds // 5 + 1

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_folds, sharey=True, figsize=(24, 4 * num_rows))

    plt.suptitle("Model Evaluation Validation, " + title, fontsize=15, y=1.05, x=0.45)

    im = None

    # Iterate over each fold
    for idx, (train_idx, val_idx) in enumerate(idc_list):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        if num_rows > 1:
            row, col = divmod(idx, 5)
            ax = axs[row, col]
        else:
            ax = axs[idx]

        y_val_agg, y_pred_agg = get_ytrue_y_pred_agg(y_val, y_pred, X_val)

        # Reshape the truth and predicted arrays
        y_truth = y_val_agg.ravel()
        y_pred = y_pred_agg.ravel()

        values = np.vstack([y_pred, y_truth])
        kernel = stats.gaussian_kde(values)(values)

        # Plot scatter and kernel density estimation of true vs predicted values
        im = sns.scatterplot(x=y_truth, y=y_pred, ax=ax, c=kernel, cmap="viridis", linewidth=0)
        # im = sns.kdeplot(x=y_truth, y=y_pred, ax=ax, fill=True, cmap="viridis", levels=200)

        ax.axhline(y=0, color='k', alpha=0.8, linestyle='-.')
        ax.axvline(x=0, color='k', alpha=0.8, linestyle='-.')
        ax.plot([-10, 7.5], [-10, 7.5], color='k', alpha=0.8)

        ax.set_xlim([-10, 7.5])
        ax.set_ylim([-10, 7.5])
        ax.set_title(f'Fold {idx + 1}')
        ax.set_xlabel('Ground truth SMB [m w.e.]', fontsize=11)
        if idx % 5 == 0:
            ax.set_ylabel('Predicted SMB [m w.e.]', fontsize=11)

        textstr = '\n'.join((
            r'$RMSE=%.2f$' % (root_mean_squared_error(y_truth, y_pred),),
            r'$MSE=%.2f$' % (mean_squared_error(y_truth, y_pred),),
            r'$MAE=%.2f$' % (mean_absolute_error(y_truth, y_pred),),
            r'$R^2=%.2f$' % (r2_score(y_truth, y_pred),)
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

    # Add a single colorbar to the right of the plots
    fig.colorbar(im.collections[0], ax=axs, location='right', pad=0.01)

    if plotting:
        plt.savefig(f'.././data/plots/model-training/{fname}_validation_folds_{num_folds}.svg', dpi=600, format='svg',
                    bbox_inches='tight')
    plt.show()


def plot_gsearch_results(grid, model_name, plotting):
    """
    Plots the results of a grid search for hyperparameter tuning, displaying the mean test and train scores
    with their standard deviations for each hyperparameter.

    Parameters:
    - grid: Fitted GridSearchCV object containing the results of the hyperparameter search.
    - model_name: Name of the model (string) to be included in the plot title.
    """

    # Extract results from the grid search
    results = grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    means_train = results['mean_train_score']
    stds_train = results['std_train_score']

    # Retrieve the best parameters and their names
    best_params = grid.best_params_
    masks_names = list(best_params.keys())

    # Create boolean masks to identify the best parameter configurations
    masks = [results[f'param_{p}'].data == v for p, v in best_params.items()]

    params = grid.param_grid

    # Determine plot width based on the number of parameters
    width = len(best_params) * 5

    # Create subplots for each hyperparameter
    fig, axs = plt.subplots(nrows=1, ncols=len(params), sharex='none', sharey='all', figsize=(width, 4))
    fig.suptitle(f'Score per Learning Parameter for {model_name}')

    # Plot results for each parameter
    for i, p in enumerate(masks_names):
        # Combine masks for other parameters to identify the best configurations for the current parameter
        other_masks = np.stack(masks[:i] + masks[i + 1:])
        best_parms_mask = other_masks.all(axis=0)
        best_index = np.where(best_parms_mask)[0]

        # Extract the parameter values and corresponding mean and std scores
        x = np.array(params[p])
        y_test = means_test[best_index]
        e_test = stds_test[best_index]
        y_train = means_train[best_index]
        e_train = stds_train[best_index]

        # Plot the mean test and train scores with error bars for standard deviation
        axs[i].plot(x, y_test, linestyle='--', marker='o', label='test')
        axs[i].fill_between(x, y_test - e_test, y_test + e_test, linestyle='--', alpha=0.2, label='std test')
        axs[i].errorbar(x, y_train, e_train, linestyle='-', marker='^', label='train')
        axs[i].set_xlabel(p.upper())
        axs[i].grid()

        # Add y-axis label to the first subplot
        if i == 0:
            axs[i].set_ylabel('Mean Score')

    plt.legend()
    if plotting:
        plt.savefig(f'.././data/plots/model-training/{model_name}_parameter_scores.svg', dpi=600, format='svg',
                    bbox_inches='tight')
    plt.show()


# Plot permutation importance
def plot_permutation_importance(df_train_X_s, X_train_s, y_train_s, splits_s, best_model, plotting, model_name,
                                max_features_plot=10):
    fig, ax = plt.subplots(1, 5, figsize=(30, 10))
    for idx, (train_index, test_index) in enumerate(splits_s):
        # Loops over n_splits iterations and gets train and test splits in each fold
        X_train, X_test = X_train_s[train_index], X_train_s[test_index]
        y_train, y_test = y_train_s[train_index], y_train_s[test_index]

        best_model.fit(X_train, y_train)

        result = permutation_importance(best_model, X_train, y_train, n_repeats=20, random_state=42, n_jobs=10)

        sorted_idx = result.importances_mean.argsort()
        labels = np.array(df_train_X_s.columns)[sorted_idx][-max_features_plot:]

        ax[idx].boxplot(result.importances[sorted_idx].T[:, -max_features_plot:], vert=False, labels=labels)
        ax[idx].set_title("Permutation Importance Fold " + str(idx + 1))

    if plotting:
        plt.savefig(f'.././data/plots/model-training/{model_name}_feature_importance.svg', dpi=600, format='svg',
                    bbox_inches='tight')
    plt.show()


def plot_prediction_per_season(plotting, name_model, X_data, y_data, model_splits, model):
    y_test_annual, y_pred_annual = get_aggregated_predictions(X_data, y_data, model_splits,
                                                                      model, 12)
    y_test_winter, y_pred_winter = get_aggregated_predictions(X_data, y_data, model_splits,
                                                                      model, 7)
    y_test_summer, y_pred_summer = get_aggregated_predictions(X_data, y_data, model_splits,
                                                                      model, 5)

    test_data = {
        'Annual': {
            'truth': y_test_annual,
            'pred': y_pred_annual,
            'months': 12,
        },
        'Winter': {
            'truth': y_test_winter,
            'pred': y_pred_winter,
            'months': 7,
        },
        'Summer': {
            'truth': y_test_summer,
            'pred': y_pred_summer,
            'months': 5,
        }
    }

    fig, axs = plt.subplots(nrows=1, ncols=3, sharey=False, figsize=(16, 4))

    plt.suptitle("Model Evaluation per Season and Annually", fontsize=15, y=1.05, x=0.45)

    im = None

    # Iterate over each fold
    for (ax, season) in zip(axs, test_data.keys()):
        # Reshape the truth and predicted arrays
        y_truth = test_data[season]['truth'].ravel()
        y_pred = test_data[season]['pred'].ravel()

        values = np.vstack([y_pred, y_truth])
        kernel = stats.gaussian_kde(values)(values)

        # Plot scatter and kernel density estimation of true vs predicted values
        im = sns.scatterplot(x=y_truth, y=y_pred, ax=ax, c=kernel, cmap="viridis", linewidth=0)

        ax.axhline(y=0, color='k', alpha=0.8, linestyle='-.')
        ax.axvline(x=0, color='k', alpha=0.8, linestyle='-.')
        ax.plot([values.min(), values.max()], [values.min(), values.max()], color='k', alpha=0.8)

        # ax.set_xlim([values.min(), values.max()])
        # ax.set_ylim([values.min(), values.max()])
        ax.set_title(f'{season}')
        ax.set_xlabel('Ground truth SMB [m w.e.]', fontsize=11)
        if season == 'Annual':
            ax.set_ylabel('Predicted SMB [m w.e.]', fontsize=11)

        textstr = '\n'.join((
            r'$RMSE=%.2f$' % (root_mean_squared_error(y_truth, y_pred),),
            r'$MSE=%.2f$' % (mean_squared_error(y_truth, y_pred),),
            r'$MAE=%.2f$' % (mean_absolute_error(y_truth, y_pred),),
            r'$R^2=%.2f$' % (r2_score(y_truth, y_pred),)
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

    # Add a single colorbar to the right of the plots
    fig.colorbar(im.collections[0], ax=axs, location='right', pad=0.01)

    if plotting:
        plt.savefig(f'.././data/plots/model-training/{name_model}_seasons_individual_prediction.svg', dpi=600,
                    format='svg',
                    bbox_inches='tight')
    plt.show()
