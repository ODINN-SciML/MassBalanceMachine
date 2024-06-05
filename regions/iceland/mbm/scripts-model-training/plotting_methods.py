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


# Function to plot histograms for both the training and test dataset
def plot_histograms(ax, df_train, df_test, column, x_label):
    hist_params = {
        'alpha': 0.7,
        'edgecolor': 'white',
        'linewidth': 1
    }

    if column is not None:
        df_train[column].plot.hist(ax=ax, label='Train', color='#4b9c8d', **hist_params)
        df_test[column].plot.hist(ax=ax, label='Test', color='#4e8cd9', **hist_params)
    else:
        df_train.plot.hist(ax=ax, label='Train', color='#4b9c8d', **hist_params)
        df_test.plot.hist(ax=ax, label='Test', color='#4e8cd9', **hist_params)

    ax.set_xlabel(x_label)
    ax.set_ylabel('Frequency')
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='white')


def make_hist_plot(title, smb_type, df_train, df_test, temp_columns, prec_columns, plotting):
    # Plot the frequencies for the SMB, elevation, years, mean t2m, cumulative total precipitation, for both the
    # training and test dataset
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(16, 4), sharex='col', sharey=True)

    plot_histograms(ax[0], df_train, df_test, smb_type, 'SMB [m w.e.]')
    plot_histograms(ax[1], df_train, df_test, 'elevation', 'Elevation [m]')
    plot_histograms(ax[2], df_train, df_test, 'yr', 'Years')
    plot_histograms(ax[3], df_train[temp_columns].mean(axis=1), df_test[temp_columns].mean(axis=1), None,
                    'Mean t2m [K]')
    plot_histograms(ax[4], df_train[prec_columns].sum(axis=1), df_test[prec_columns].sum(axis=1), None, 'Sum tp [m]')

    fig.suptitle(f'{title}')
    plt.tight_layout()
    plt.legend()
    if plotting:
        plt.savefig(f'.././data/plots/{title}_feature_dists.svg', dpi=600, format='svg')
    plt.show()


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
    for idx, (train_idx, test_idx) in enumerate(idc_list):
        X_train, X_truth = X[train_idx], X[test_idx]
        y_train, y_truth = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_truth)

        if num_rows > 1:
            row, col = divmod(idx, 5)
            ax = axs[row, col]
        else:
            ax = axs[idx]

        # Reshape the truth and predicted arrays
        y_truth = y_truth.ravel()
        y_pred = y_pred.ravel()

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
        plt.savefig(f'.././data/plots/{fname}_validation_folds_{num_folds}.svg', dpi=600, format='svg',
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
        plt.savefig(f'.././data/plots/{model_name}_parameter_scores.svg', dpi=600, format='svg', bbox_inches='tight')
    plt.show()


# Plot permutation importance
def plot_permutation_importance(df_train_X_s, X_train_s, y_train_s, splits_s, best_model, plotting, model_name, max_features_plot=10):
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
        plt.savefig(f'.././data/plots/{model_name}_feature_importance.svg', dpi=600, format='svg', bbox_inches='tight')
    plt.show()


def plot_prediction_per_season(plotting, name_model, X_train, y_train, model_splits, model):
    y_test_annual, y_pred_annual = get_prediction_per_season(X_train, y_train, model_splits, model, months=12)
    y_test_winter, y_pred_winter = get_prediction_per_season(X_train, y_train, model_splits, model, months=8)
    y_test_summer, y_pred_summer = get_prediction_per_season(X_train, y_train, model_splits, model, months=6)

    test_data = {
        'Annual': {
            'truth': y_test_annual,
            'pred': y_pred_annual,
        },
        'Winter': {
            'truth': y_test_winter,
            'pred': y_pred_winter,
        },
        'Summer': {
            'truth': y_test_summer,
            'pred': y_pred_summer,
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

        ax.set_xlim([values.min(), values.max()])
        ax.set_ylim([values.min(), values.max()])
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
        plt.savefig(f'.././data/plots/{name_model}_seasons_individual_prediction.svg', dpi=600, format='svg',
                    bbox_inches='tight')
    plt.show()