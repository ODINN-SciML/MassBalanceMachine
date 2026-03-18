from cmcrameri import cm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from regions.Switzerland.scripts.plotting.palettes import get_cmap_hex
from regions.Switzerland.scripts.config_CH import *


def plot_grid_search_score(cv_results_, lossType: str):
    """
    Plot train and validation scores across grid-search iterations.

    Parameters
    ----------
    cv_results_ : dict or pandas.DataFrame-like
        Cross-validation results (e.g., from sklearn GridSearchCV.cv_results_),
        expected to contain mean/std train/test scores.
    lossType : str
        Label for the y-axis (e.g., name of the loss/metric).

    Returns
    -------
    None
        Creates a matplotlib figure.
    """
    dfCVResults = pd.DataFrame(cv_results_)
    mask_raisonable = dfCVResults["mean_train_score"] >= -10
    dfCVResults = dfCVResults[mask_raisonable]

    fig = plt.figure(figsize=(10, 5))
    mean_train = abs(dfCVResults.mean_train_score)
    std_train = abs(dfCVResults.std_train_score)
    mean_test = abs(dfCVResults.mean_test_score)
    std_test = abs(dfCVResults.std_test_score)

    plt.plot(mean_train, label="train", color=COLOR_ANNUAL)
    plt.plot(mean_test, label="validation", color=COLOR_WINTER)

    # add std
    plt.fill_between(
        dfCVResults.index,
        mean_train - std_train,
        mean_train + std_train,
        alpha=0.2,
        color=COLOR_ANNUAL,
    )
    plt.fill_between(
        dfCVResults.index,
        mean_test - std_test,
        mean_test + std_test,
        alpha=0.2,
        color=COLOR_WINTER,
    )

    # Add a line at the minimum
    pos_min = dfCVResults.mean_test_score.abs().idxmin()
    plt.axvline(pos_min, color="red", linestyle="--", label="min validation")

    plt.xlabel("Iteration")
    plt.ylabel(f"{lossType}")
    plt.title("Grid search score over iterations")
    plt.legend()


def plot_grid_search_params(cv_results_, param_grid, lossType: str, N=10):
    """
    Visualize grid-search performance as a function of each hyperparameter.

    For each parameter in `param_grid`, plots mean ± std of train/validation
    scores (aggregated across CV splits) and marks the best parameter value.

    Parameters
    ----------
    cv_results_ : dict or pandas.DataFrame-like
        Cross-validation results (e.g., from sklearn GridSearchCV.cv_results_).
    param_grid : dict
        Parameter grid used in the search (keys are parameter names).
    lossType : str
        Label for the y-axis (e.g., name of the loss/metric).
    N : int or None, optional
        Number of top configurations (by mean_test_score) to retain for plotting.
        If None, uses all configurations after filtering.

    Returns
    -------
    None
        Creates a matplotlib figure.
    """
    dfCVResults = pd.DataFrame(cv_results_)
    best_params = (
        dfCVResults.sort_values("mean_test_score", ascending=False).iloc[0].params
    )
    mask_raisonable = dfCVResults["mean_train_score"] >= -10
    dfCVResults_ = dfCVResults[mask_raisonable]
    dfCVResults_.sort_values("mean_test_score", ascending=False, inplace=True)
    if N is not None:
        dfCVResults_ = dfCVResults_.iloc[:N]
    fig = plt.figure(figsize=(15, 5))
    for i, param in enumerate(param_grid.keys()):

        dfParam = dfCVResults_.groupby(f"param_{param}")[
            [
                "split0_test_score",
                "split1_test_score",
                "split2_test_score",
                "split3_test_score",
                "split4_test_score",
                "mean_test_score",
                "std_test_score",
                "rank_test_score",
                "split0_train_score",
                "split1_train_score",
                "split2_train_score",
                "split3_train_score",
                "split4_train_score",
                "mean_train_score",
                "std_train_score",
            ]
        ].mean()

        mean_test = abs(
            dfParam[[f"split{i}_test_score" for i in range(5)]].mean(axis=1)
        )
        std_test = abs(dfParam[[f"split{i}_test_score" for i in range(5)]].std(axis=1))

        mean_train = abs(
            dfParam[[f"split{i}_train_score" for i in range(5)]].mean(axis=1)
        )
        std_train = abs(
            dfParam[[f"split{i}_train_score" for i in range(5)]].std(axis=1)
        )

        # plot mean values with std
        ax = plt.subplot(1, len(param_grid.keys()), i + 1)
        ax.scatter(
            x=mean_test.index, y=mean_test.values, marker="x", color=COLOR_WINTER
        )
        ax.plot(mean_test.index, mean_test, color=COLOR_WINTER, label="validation")
        ax.fill_between(
            mean_test.index,
            mean_test - std_test,
            mean_test + std_test,
            alpha=0.2,
            color=COLOR_WINTER,
        )

        ax.scatter(
            x=mean_train.index, y=mean_train.values, marker="x", color=COLOR_ANNUAL
        )
        ax.plot(mean_train.index, mean_train, color=COLOR_ANNUAL, label="train")
        ax.fill_between(
            mean_train.index,
            mean_train - std_train,
            mean_train + std_train,
            alpha=0.2,
            color=COLOR_ANNUAL,
        )
        # add vertical line of best param
        ax.axvline(best_params[param], color="red", linestyle="--")

        ax.set_ylabel(f"{lossType}")
        ax.set_title(param)
        ax.legend()

    plt.suptitle("Grid search results")
    plt.tight_layout()


def FI_plot(best_estimator, feature_columns, vois_climate):
    """
    Plot feature importances of a fitted tree-based estimator.

    Parameters
    ----------
    best_estimator : object
        Fitted estimator exposing `feature_importances_`.
    feature_columns : list of str
        Feature names in the order used for training.
    vois_climate : list or dict-like
        Climate variable identifiers used for mapping to long names
        (via `vois_climate_long_name` if available).

    Returns
    -------
    None
        Creates a seaborn/matplotlib bar plot.
    """
    FI = best_estimator.feature_importances_
    cmap = cm.devon
    color_palette_glaciers = get_cmap_hex(cmap, len(FI) + 5)
    fig = plt.figure(figsize=(10, 15))
    ax = plt.subplot(1, 1, 1)
    feature_importdf = pd.DataFrame(data={"variables": feature_columns, "feat_imp": FI})

    feature_importdf["variables"] = feature_importdf["variables"].apply(
        lambda x: (
            vois_climate_long_name[x] + f" ({x})"
            if x in vois_climate_long_name.keys()
            else x
        )
    )

    feature_importdf.sort_values(by="feat_imp", ascending=True, inplace=True)
    sns.barplot(
        feature_importdf,
        x="feat_imp",
        y="variables",
        dodge=False,
        ax=ax,
        palette=color_palette_glaciers,
    )

    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Feature")
