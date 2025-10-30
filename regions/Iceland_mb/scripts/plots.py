import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from cmcrameri import cm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, r2_score
from matplotlib.ticker import MaxNLocator

from scripts.helpers import *
from scripts.config_ICE import *

colors_vik = get_cmap_hex(cm.vik, 10)
color_xgb = colors_vik[0]
color_tim = "#c51b7d"

color_dark_blue = "#00008B"
color_orange = "#FFA500"
color_pink = "#c51b7d"


def plotHeatmap(data_wgms, test_glaciers=None, period="annual", plot_elevation=False):
    # Heatmap of mean mass balance per glacier:
    # Get the mean mass balance per glacier
    data_with_pot = data_wgms[data_wgms.PERIOD == period]

    mean_mb_per_glacier = (
        data_with_pot.groupby(["GLACIER", "YEAR", "PERIOD"])["POINT_BALANCE"]
        .mean()
        .reset_index()
    )
    mean_mb_per_glacier = mean_mb_per_glacier[mean_mb_per_glacier["PERIOD"] == period]
    matrix = mean_mb_per_glacier.pivot(
        index="GLACIER", columns="YEAR", values="POINT_BALANCE"
    ).sort_values(by="GLACIER")

    # get elevation of glaciers:
    gl_per_el = data_with_pot.groupby(["GLACIER"])["POINT_ELEVATION"].mean()
    gl_per_el = gl_per_el.sort_values(ascending=False)

    matrix = matrix.loc[gl_per_el.index]
    # make index categorical
    matrix.index = pd.Categorical(matrix.index, categories=matrix.index, ordered=True)
    fig = plt.figure(figsize=(20, 15))
    ax = plt.subplot(1, 1, 1)
    sns.heatmap(
        data=matrix,
        center=0,
        cmap=cm.vik_r,
        cbar_kws={"label": "[m w.e. $a^{-1}$]"},
        ax=ax,
    )

    # add patches for test glaciers
    if test_glaciers is not None:
        for test_gl in test_glaciers:
            if test_gl not in matrix.index:
                continue
            height = matrix.index.get_loc(test_gl)
            row = np.where(matrix.loc[test_gl].notna())[0]
            split_indices = np.where(np.diff(row) != 1)[0] + 1
            continuous_sequences = np.split(row, split_indices)
            for patch in continuous_sequences:
                ax.add_patch(
                    Rectangle(
                        (patch.min(), height),
                        patch.max() - patch.min() + 1,
                        1,
                        fill=False,
                        edgecolor="black",
                        lw=3,
                    )
                )

    if plot_elevation:
        fig = plt.figure(figsize=(10, 3))
        ax = plt.subplot(1, 1, 1)
        sorted_elevations = gl_per_el.sort_values(ascending=True)

        sns.lineplot(sorted_elevations, ax=ax, color="gray", marker="v")

        ax.set_xticks(range(len(sorted_elevations)))
        ax.set_xticklabels(sorted_elevations.index, rotation=45, ha="right")
        ax.set_ylabel("Elevation [m]")
        plt.tight_layout()


def plot_feature_correlation(df, exclude_columns=None):
    """
    Generate and plot a correlation heatmap of features.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data to analyze
    exclude_columns : list, optional
        List of columns to exclude from the correlation analysis
    """
    # Default columns to exclude if not specified
    if exclude_columns is None:
        exclude_columns = [
            "GLACIER",
            "PERIOD",
            "YEAR",
            "POINT_LON",
            "POINT_LAT",
            "POINT_BALANCE",
            "ALTITUDE_CLIMATE",
            "POINT_ELEVATION",
            "RGIId",
            "POINT_ID",
            "ID",
            "N_MONTHS",
            "MONTHS",
            "GLACIER_ZONE",
        ]

    df_test = df.copy()

    # Define the columns to plot
    columns_to_keep = [col for col in df_test.columns if col not in exclude_columns]
    df_test = df_test[columns_to_keep]

    # Rename columns based on long names (if available in the global scope)
    try:
        df_test.rename(columns=vois_climate_long_name, inplace=True)
    except NameError:
        pass

    # Compute correlation matrix
    corr = df_test.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        vmax=1,
        vmin=-1,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    plt.title("Feature Intercorrelation Heatmap", fontsize=16)
    plt.tight_layout()
    plt.show()


def visualiseSplits(y_test, y_train, splits, colors=[color_xgb, color_tim]):
    # Visualise the cross validation splits
    fig, ax = plt.subplots(1, 6, figsize=(20, 5))
    ax[0].hist(y_train, color=colors[0], density=False, alpha=0.5)
    ax[0].set_title("Train & Test PMB")
    ax[0].hist(y_test, color=colors[1], density=False, alpha=0.5)
    ax[0].set_ylabel("Frequency")
    for i, (train_idx, val_idx) in enumerate(splits):
        # Check that there is no overlap between the training, val and test IDs
        ax[i + 1].hist(
            y_train[train_idx], bins=20, color=colors[0], density=False, alpha=0.5
        )
        ax[i + 1].hist(
            y_train[val_idx], bins=20, color=colors[1], density=False, alpha=0.5
        )
        ax[i + 1].set_title("CV train Fold " + str(i + 1))
        ax[i + 1].set_xlabel("[m w.e.]")
    plt.tight_layout()


def visualiseInputs(train_set, test_set, vois_climate):
    f, ax = plt.subplots(
        2, len(vois_climate) + 4, figsize=(16, 6), sharey="row", sharex="col"
    )
    train_set["df_X"]["POINT_BALANCE"].plot.hist(
        ax=ax[0, 0], color=color_xgb, alpha=0.6, density=False
    )
    ax[0, 0].set_title("PMB")
    ax[0, 0].set_ylabel("Frequency (train)")
    train_set["df_X"]["ELEVATION_DIFFERENCE"].plot.hist(
        ax=ax[0, 1], color=color_xgb, alpha=0.6, density=False
    )
    ax[0, 1].set_title("ELV_DIFF")
    train_set["df_X"]["YEAR"].plot.hist(
        ax=ax[0, 2], color=color_xgb, alpha=0.6, density=False
    )
    ax[0, 2].set_title("YEARS")

    for i, voi_clim in enumerate(vois_climate):
        ax[0, 3 + i].set_title(voi_clim)
        train_set["df_X"][voi_clim].plot.hist(
            ax=ax[0, 3 + i], color=color_xgb, alpha=0.6, density=False
        )

    test_set["df_X"]["POINT_BALANCE"].plot.hist(
        ax=ax[1, 0], color=color_tim, alpha=0.6, density=False
    )
    ax[1, 0].set_ylabel("Frequency (test)")
    test_set["df_X"]["ELEVATION_DIFFERENCE"].plot.hist(
        ax=ax[1, 1], color=color_tim, alpha=0.6, density=False
    )
    test_set["df_X"]["YEAR"].plot.hist(
        ax=ax[1, 2], color=color_tim, alpha=0.6, density=False
    )

    for i, voi_clim in enumerate(vois_climate):
        test_set["df_X"][voi_clim].plot.hist(
            ax=ax[1, 3 + i], color=color_tim, alpha=0.6, density=False
        )
    # rotate xticks
    for ax in ax.flatten():
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("")

    plt.tight_layout()


def plotGridSearchScore(cv_results_, lossType: str):
    dfCVResults = pd.DataFrame(cv_results_)
    mask_raisonable = dfCVResults["mean_train_score"] >= -10
    dfCVResults = dfCVResults[mask_raisonable]

    fig = plt.figure(figsize=(10, 5))
    mean_train = abs(dfCVResults.mean_train_score)
    std_train = abs(dfCVResults.std_train_score)
    mean_test = abs(dfCVResults.mean_test_score)
    std_test = abs(dfCVResults.std_test_score)

    plt.plot(mean_train, label="train", color=color_xgb)
    plt.plot(mean_test, label="validation", color=color_tim)

    # add std
    plt.fill_between(
        dfCVResults.index,
        mean_train - std_train,
        mean_train + std_train,
        alpha=0.2,
        color=color_xgb,
    )
    plt.fill_between(
        dfCVResults.index,
        mean_test - std_test,
        mean_test + std_test,
        alpha=0.2,
        color=color_tim,
    )

    # Add a line at the minimum
    pos_min = dfCVResults.mean_test_score.abs().idxmin()
    plt.axvline(pos_min, color="red", linestyle="--", label="min validation")

    plt.xlabel("Iteration")
    plt.ylabel(f"{lossType}")
    plt.title("Grid search score over iterations")
    plt.legend()


def plotGridSearchParams(cv_results_, param_grid, lossType: str, N=10):
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
        ax.scatter(x=mean_test.index, y=mean_test.values, marker="x", color=color_tim)
        ax.plot(mean_test.index, mean_test, color=color_tim, label="validation")
        ax.fill_between(
            mean_test.index,
            mean_test - std_test,
            mean_test + std_test,
            alpha=0.2,
            color=color_tim,
        )

        ax.scatter(x=mean_train.index, y=mean_train.values, marker="x", color=color_xgb)
        ax.plot(mean_train.index, mean_train, color=color_xgb, label="train")
        ax.fill_between(
            mean_train.index,
            mean_train - std_train,
            mean_train + std_train,
            alpha=0.2,
            color=color_xgb,
        )
        # add vertical line of best param
        ax.axvline(best_params[param], color="red", linestyle="--")

        ax.set_ylabel(f"{lossType}")
        ax.set_title(param)
        ax.legend()

    plt.suptitle("Grid search results")
    plt.tight_layout()


def print_top_n_models(cv_results_, n=10, lossType="rmse"):

    results = pd.DataFrame(cv_results_)

    # Sort by test score (taking absolute value since it's negative)
    results["abs_test_score"] = results["mean_test_score"].abs()
    results = results.sort_values("abs_test_score", ascending=True).head(n)

    # Extract parameters and scores
    table = []
    for i, row in enumerate(results.iterrows()):
        row = row[1]
        model_info = {
            "Model": i + 1,
            "learning_rate": row["param_learning_rate"],
            "max_depth": int(row["param_max_depth"]),
            "n_estimators": int(row["param_n_estimators"]),
            f"Validation {lossType}": abs(row["mean_test_score"]),
            f"Train {lossType}": abs(row["mean_train_score"]),
        }
        table.append(model_info)

    df = pd.DataFrame(table)
    df = df.set_index("Model")

    return display(df)


def FIPlot(best_estimator, feature_columns, vois_climate):
    FI = best_estimator.feature_importances_
    cmap = cm.devon
    color_palette_glaciers = get_cmap_hex(cmap, len(FI) + 5)
    fig = plt.figure(figsize=(15, 10))
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


def PlotPredictions(
    grouped_ids, y_pred, metadata_test, test_set, model, include_summer=False
):
    # Determine number of rows based on whether summer is included and present in the data
    rows = 3 if include_summer and "summer" in grouped_ids.PERIOD.unique() else 2
    fig = plt.figure(figsize=(20, 7.5 * rows))

    # Use seaborn's color palette for consistent glacier colors
    palette = sns.color_palette("husl", n_colors=len(grouped_ids.GLACIER.unique()))
    color_palette_glaciers = dict(zip(grouped_ids.GLACIER.unique(), palette))

    # Always plot annual data (first row)
    ax1 = plt.subplot(rows, 2, 1)
    grouped_ids_annual = grouped_ids[grouped_ids.PERIOD == "annual"]
    mse_annual, rmse_annual, mae_annual, pearson_corr_annual = model.evalMetrics(
        metadata_test, y_pred, test_set["y"], period="annual"
    )
    scores_annual = {
        "mse": mse_annual,
        "rmse": rmse_annual,
        "mae": mae_annual,
        "pearson_corr": pearson_corr_annual,
    }
    predVSTruth(
        ax1,
        grouped_ids_annual,
        scores_annual,
        hue="GLACIER",
        palette=color_palette_glaciers,
    )
    ax1.set_title("Annual PMB", fontsize=24)

    grouped_ids_annual.sort_values(by="YEAR", inplace=True)
    ax2 = plt.subplot(rows, 2, 2)
    ax2.set_title("Mean annual PMB", fontsize=24)
    plotMeanPred(grouped_ids_annual, ax2)

    # Always plot winter data (second row)
    grouped_ids_winter = grouped_ids[grouped_ids.PERIOD == "winter"]
    ax3 = plt.subplot(rows, 2, 3)
    mse_winter, rmse_winter, mae_winter, pearson_corr_winter = model.evalMetrics(
        metadata_test, y_pred, test_set["y"], period="winter"
    )
    scores_winter = {
        "mse": mse_winter,
        "rmse": rmse_winter,
        "mae": mae_winter,
        "pearson_corr": pearson_corr_winter,
    }
    predVSTruth(
        ax3,
        grouped_ids_winter,
        scores_winter,
        hue="GLACIER",
        palette=color_palette_glaciers,
    )
    ax3.set_title("Winter PMB", fontsize=24)

    ax4 = plt.subplot(rows, 2, 4)
    ax4.set_title("Mean winter PMB", fontsize=24)
    grouped_ids_winter.sort_values(by="YEAR", inplace=True)
    plotMeanPred(grouped_ids_winter, ax4)

    # Conditionally plot summer data (third row) if requested and available
    if include_summer and "summer" in grouped_ids.PERIOD.unique():
        grouped_ids_summer = grouped_ids[grouped_ids.PERIOD == "summer"]
        ax5 = plt.subplot(rows, 2, 5)
        mse_summer, rmse_summer, mae_summer, pearson_corr_summer = model.evalMetrics(
            metadata_test, y_pred, test_set["y"], period="summer"
        )
        scores_summer = {
            "mse": mse_summer,
            "rmse": rmse_summer,
            "mae": mae_summer,
            "pearson_corr": pearson_corr_summer,
        }
        predVSTruth(
            ax5,
            grouped_ids_summer,
            scores_summer,
            hue="GLACIER",
            palette=color_palette_glaciers,
        )
        ax5.set_title("Summer PMB", fontsize=24)

        ax6 = plt.subplot(rows, 2, 6)
        ax6.set_title("Mean summer PMB", fontsize=24)
        grouped_ids_summer.sort_values(by="YEAR", inplace=True)
        plotMeanPred(grouped_ids_summer, ax6)

    plt.tight_layout()


def PlotPredictionsCombined(
    grouped_ids,
    y_pred,
    metadata_test,
    test_set,
    model,
    region_name="",
    include_summer=False,
):
    fig = plt.figure(figsize=(12, 10))

    # Define colors for period (annual/winter/summer)
    period_colors = {"annual": "#e31a1c", "winter": "#1f78b4", "summer": "#33a02c"}

    # Calculate metrics for combined, annual and winter periods
    mse_all, rmse_all, mae_all, pearson_corr_all = model.evalMetrics(
        metadata_test, y_pred, test_set["y"]
    )

    mse_annual, rmse_annual, mae_annual, pearson_corr_annual = model.evalMetrics(
        metadata_test, y_pred, test_set["y"], period="annual"
    )

    mse_winter, rmse_winter, mae_winter, pearson_corr_winter = model.evalMetrics(
        metadata_test, y_pred, test_set["y"], period="winter"
    )

    # Calculate metrics for summer if requested
    if include_summer and "summer" in grouped_ids.PERIOD.unique():
        mse_summer, rmse_summer, mae_summer, pearson_corr_summer = model.evalMetrics(
            metadata_test, y_pred, test_set["y"], period="summer"
        )

    # Create a single plot for all data points
    ax = plt.subplot(1, 1, 1)

    # Plot points colored by period
    for period in grouped_ids.PERIOD.unique():
        # Skip summer if not requested
        if period == "summer" and not include_summer:
            continue

        subset = grouped_ids[grouped_ids.PERIOD == period]
        if len(subset) > 0:
            ax.scatter(
                subset.target,
                subset.pred,
                color=period_colors[period],
                alpha=0.7,
                s=80,
                label=f"{period}",
            )

    # Add identity line
    min_val = min(grouped_ids.target.min(), grouped_ids.pred.min())
    max_val = max(grouped_ids.target.max(), grouped_ids.pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, linewidth=2)

    # Add metrics text with separate statistics for each period
    metrics_text = (
        f"Combined: RMSE: {rmse_all:.2f} m w.e., ρ: {pearson_corr_all:.2f}\n"
        f"Annual: RMSE: {rmse_annual:.2f} m w.e., ρ: {pearson_corr_annual:.2f}\n"
        f"Winter: RMSE: {rmse_winter:.2f} m w.e., ρ: {pearson_corr_winter:.2f}"
    )

    # Add summer metrics if requested
    if include_summer and "summer" in grouped_ids.PERIOD.unique():
        metrics_text += (
            f"\nSummer: RMSE: {rmse_summer:.2f} m w.e., ρ: {pearson_corr_summer:.2f}"
        )

    ax.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=20,
    )

    ax.legend(fontsize=24, loc="lower right")

    ax.set_xlabel("Observed PMB [m w.e.]", fontsize=27)
    ax.set_ylabel("Predicted PMB [m w.e.]", fontsize=27)
    ax.set_title(f"PMB - Pred vs. Obs ({region_name})", fontsize=30)

    ax.tick_params(axis="both", which="major", labelsize=21)


def PlotPredictionsCombined_NN(
    grouped_ids, region_name="", include_summer=False, nticks=6
):
    fig = plt.figure(figsize=(9.7, 9.7))
    period_colors = {"annual": "#e31a1c", "winter": "#1f78b4", "summer": "#33a02c"}

    # Compute metrics for each period
    metrics = {}
    for period in ["annual", "winter", "summer"]:
        if period == "summer" and not include_summer:
            continue
        subset = grouped_ids[grouped_ids.PERIOD == period]
        if len(subset) > 0:
            rmse = np.sqrt(mean_squared_error(subset.target, subset.pred))
            r2 = r2_score(subset.target, subset.pred)
            # Pearson correlation
            if len(subset) > 1:
                rho = np.corrcoef(subset.target, subset.pred)[0, 1]
            else:
                rho = np.nan
            metrics[period] = (rmse, rho, r2)

    # Combined metrics
    rmse_all = np.sqrt(mean_squared_error(grouped_ids.target, grouped_ids.pred))
    r2_all = r2_score(grouped_ids.target, grouped_ids.pred)
    if len(grouped_ids) > 1:
        rho_all = np.corrcoef(grouped_ids.target, grouped_ids.pred)[0, 1]
    else:
        rho_all = np.nan
    metrics["combined"] = (rmse_all, rho_all, r2_all)

    ax = plt.subplot(1, 1, 1)
    for period in grouped_ids.PERIOD.unique():
        if period == "summer" and not include_summer:
            continue
        subset = grouped_ids[grouped_ids.PERIOD == period]
        if len(subset) > 0:
            ax.scatter(
                subset.target,
                subset.pred,
                color=period_colors.get(period, "gray"),
                alpha=0.7,
                s=80,
                label=f"{period}",
            )

    # Calculate common axis limits and ticks
    min_val = min(grouped_ids.target.min(), grouped_ids.pred.min())
    print(min_val)
    max_val = max(grouped_ids.target.max(), grouped_ids.pred.max())
    print(max_val)

    # Add some padding
    range_val = max_val - min_val
    padding = range_val * 0.05
    min_val -= padding
    max_val += padding

    # Set equal limits for both axes
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Force equal aspect ratio
    ax.set_aspect("equal", adjustable="box")

    ax.xaxis.set_major_locator(MaxNLocator(nbins=nticks))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nticks))

    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, linewidth=2)

    # Build metrics text for top left (RMSE values)
    rmse_text = f"RMSE$_{{\\mathbf{{C}}}}$: {metrics['combined'][0]:.2f}\n"
    if "annual" in metrics:
        rmse_text += f"RMSE$_{{\\mathbf{{A}}}}$: {metrics['annual'][0]:.2f}\n"
    if "winter" in metrics:
        rmse_text += f"RMSE$_{{\\mathbf{{W}}}}$: {metrics['winter'][0]:.2f}"
    if include_summer and "summer" in metrics:
        rmse_text += f"\nRMSE$_{{\\mathbf{{S}}}}$: {metrics['summer'][0]:.2f}"

    # Build metrics text for bottom right (rho and R² values)
    corr_text = f"ρ$_{{\\mathbf{{C}}}}$: {metrics['combined'][1]:.2f}, R²$_{{\\mathbf{{C}}}}$: {metrics['combined'][2]:.2f}\n"
    if "annual" in metrics:
        corr_text += f"ρ$_{{\\mathbf{{A}}}}$: {metrics['annual'][1]:.2f}, R²$_{{\\mathbf{{A}}}}$: {metrics['annual'][2]:.2f}\n"
    if "winter" in metrics:
        corr_text += f"ρ$_{{\\mathbf{{W}}}}$: {metrics['winter'][1]:.2f}, R²$_{{\\mathbf{{W}}}}$: {metrics['winter'][2]:.2f}"
    if include_summer and "summer" in metrics:
        corr_text += f"\nρ$_{{\\mathbf{{S}}}}$: {metrics['summer'][1]:.2f}, R²$_{{\\mathbf{{S}}}}$: {metrics['summer'][2]:.2f}"

    # Top left text box (RMSE)
    ax.text(
        0.02,
        0.98,
        rmse_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white", alpha=0.5, linewidth=0.5
        ),
        fontsize=32,
    )

    # Bottom right text box (rho and R²)
    ax.text(
        0.98,
        0.02,
        corr_text,
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white", alpha=0.5, linewidth=0.5
        ),
        fontsize=32,
    )

    # ax.legend(fontsize=32, loc='upper left', borderpad=0.2,labelspacing=0.2,handletextpad=0.1)
    ax.set_xlabel("Observed PMB [m w.e.]", fontsize=32)
    ax.set_ylabel("Predicted PMB [m w.e.]", fontsize=32)
    # ax.set_title(f'PMB - Pred vs. Obs ({region_name})', fontsize=32)
    ax.tick_params(axis="both", which="major", labelsize=32)
    plt.tight_layout()


def predVSTruth(ax, grouped_ids, scores, hue="GLACIER", palette=None):

    legend_xgb = "\n".join(
        (
            (r"$\mathrm{RMSE}=%.3f$," % (scores["rmse"],)),
            (r"$\mathrm{\rho}=%.3f$" % (scores["pearson_corr"],)),
        )
    )

    marker_xgb = "o"
    sns.scatterplot(
        grouped_ids,
        x="target",
        y="pred",
        palette=palette,
        hue=hue,
        ax=ax,
        # alpha=0.8,
        color=color_xgb,
        marker=marker_xgb,
    )

    ax.set_ylabel("Predicted PMB [m w.e.]", fontsize=20)
    ax.set_xlabel("Observed PMB [m w.e.]", fontsize=20)

    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    ax.text(
        0.03,
        0.98,
        legend_xgb,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=20,
        bbox=props,
    )
    if hue is not None:
        ax.legend(fontsize=5, loc="center left", bbox_to_anchor=(1, 0.5))
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

    # Set aspect ratio to equal (square plot)
    ax.set_aspect("equal")

    plt.tight_layout()


def plotMeanPred(grouped_ids, ax):
    mean = grouped_ids.groupby("YEAR")["target"].mean().values
    std = grouped_ids.groupby("YEAR")["target"].std().values
    years = grouped_ids.YEAR.unique()
    ax.fill_between(
        years,
        mean - std,
        mean + std,
        color="orange",
        alpha=0.3,
    )
    ax.plot(years, mean, color="orange", label="mean target")
    ax.scatter(years, mean, color="orange", marker="x")
    ax.plot(
        years,
        grouped_ids.groupby("YEAR")["pred"].mean().values,
        color=color_xgb,
        label="mean pred",
        linestyle="--",
    )
    ax.scatter(
        years,
        grouped_ids.groupby("YEAR")["pred"].mean().values,
        color=color_xgb,
        marker="x",
    )
    ax.fill_between(
        years,
        grouped_ids.groupby("YEAR")["pred"].mean().values
        - grouped_ids.groupby("YEAR")["pred"].std().values,
        grouped_ids.groupby("YEAR")["pred"].mean().values
        + grouped_ids.groupby("YEAR")["pred"].std().values,
        color=color_xgb,
        alpha=0.3,
    )
    # rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    mae, rmse, pearson_corr = (
        mean_absolute_error(grouped_ids.groupby("YEAR")["pred"].mean().values, mean),
        mean_squared_error(
            grouped_ids.groupby("YEAR")["pred"].mean().values, mean, squared=False
        ),
        np.corrcoef(grouped_ids.groupby("YEAR")["pred"].mean().values, mean)[0, 1],
    )
    legend_xgb = "\n".join(
        (
            r"$\mathrm{RMSE}=%.3f, \mathrm{\rho}=%.3f$ "
            % (
                rmse,
                pearson_corr,
            ),
        )
    )
    ax.text(
        0.03,
        0.98,
        legend_xgb,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=20,
    )
    ax.legend(fontsize=20, loc="lower right")


def PlotIndividualGlacierPredVsTruth(
    grouped_ids, base_figsize=(20, 15), height_per_row=5
):
    # Calculate number of rows needed based on number of glaciers
    n_glaciers = len(grouped_ids["GLACIER"].unique())
    n_rows = (n_glaciers + 2) // 3  # Ceiling division to get enough rows for 3 columns

    figsize = (base_figsize[0], n_rows * height_per_row)

    fig, axs = plt.subplots(n_rows, 3, figsize=figsize)

    color_palette_period = {
        "annual": "#e31a1c",
        "winter": "#1f78b4",
        "summer": "#33a02c",
    }

    for i, test_gl in enumerate(grouped_ids["GLACIER"].unique()):
        df_gl = grouped_ids[grouped_ids.GLACIER == test_gl]

        ax1 = axs.flatten()[i]

        scores = {
            "mse": mean_squared_error(df_gl["target"], df_gl["pred"]),
            "rmse": mean_squared_error(df_gl["target"], df_gl["pred"], squared=False),
            "mae": mean_absolute_error(df_gl["target"], df_gl["pred"]),
            "pearson_corr": np.corrcoef(df_gl["target"], df_gl["pred"])[0, 1],
        }
        predVSTruth(ax1, df_gl, scores, hue="PERIOD", palette=color_palette_period)
        ax1.set_title(f"{test_gl.capitalize()}", fontsize=28)

    # Hide empty subplots
    for j in range(i + 1, n_rows * 3):
        if j < len(axs.flatten()):
            axs.flatten()[j].set_visible(False)

    plt.tight_layout()


def plot_climate_glacier_elevations(
    test_glaciers, test_set, plots_per_row=7, base_figsize=(20, 5)
):

    num_rows = (len(test_glaciers) + plots_per_row - 1) // plots_per_row

    fig, ax = plt.subplots(
        num_rows,
        plots_per_row,
        figsize=(base_figsize[0], base_figsize[1] * num_rows),
        sharey="all",
        sharex="all",
    )

    ax = ax.flatten() if num_rows > 1 else ax

    # Plot each glacier's elevation histogram
    for i, test_gl in enumerate(test_glaciers):
        test_df_gl = test_set["df_X"][test_set["df_X"].GLACIER == test_gl]
        test_df_gl.POINT_ELEVATION.plot.hist(
            color=color_dark_blue, alpha=0.5, density=False, ax=ax[i]
        )
        # Add vertical line for altitude climate
        alt_climate = test_df_gl.ALTITUDE_CLIMATE.mean()
        ax[i].axvline(
            x=alt_climate, color="red", linestyle="--", label="Altitude climate"
        )
        ax[i].set_xlabel("Elevation [m]")
        ax[i].legend()
        ax[i].set_title(test_gl)

    # Hide any unused subplots
    for j in range(len(test_glaciers), len(ax)):
        ax[j].set_visible(False)

    plt.tight_layout()
    return


def plot_point_climate_variables(
    point_ids,
    data_monthly,
    vois_climate,
    vois_units=None,
    figsize=(18, 12),
    ncol=3,
    title="Climate Variables for specific POINT_IDs",
):
    if vois_units is None:
        vois_units = {}

    # Filter data for these specific point IDs
    point_data = data_monthly[data_monthly["POINT_ID"].isin(point_ids)]

    fig = plt.figure(figsize=figsize)

    month_order = [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]

    # Plot each climate variable
    for var_idx, var in enumerate(vois_climate):
        ax = plt.subplot(3, 3, var_idx + 1)

        # Group by point ID and month, taking mean if multiple values exist
        pivot_data = point_data.pivot_table(
            index="MONTHS", columns="POINT_ID", values=var, aggfunc="mean"
        )

        for point_id in point_ids:
            if point_id in pivot_data.columns:
                valid_months = [m for m in month_order if m in pivot_data.index]
                if valid_months:
                    x_pos = [month_order.index(m) for m in valid_months]
                    y_vals = [pivot_data.loc[m, point_id] for m in valid_months]

                    ax.plot(x_pos, y_vals, marker="o", linewidth=2, label=point_id)

        ax.set_title(f"{var}", fontsize=12)
        ax.set_ylabel(vois_units.get(var, ""))
        ax.set_xticks(range(len(month_order)))
        ax.set_xticklabels(month_order, rotation=45)
        ax.grid(True, linestyle="--", alpha=0.7)

    # Create a separate legend figure below the plots
    plt.subplots_adjust(bottom=0.25)
    handles, labels = ax.get_legend_handles_labels()
    plt.figlegend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.02),
    )

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    return
