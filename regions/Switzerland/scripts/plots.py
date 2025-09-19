import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from cmcrameri import cm
from matplotlib import gridspec

from regions.Switzerland.scripts.helpers import *
from regions.Switzerland.scripts.config_CH import *

# CONSTANT COLORS FOR PLOTS
colors = get_cmap_hex(cm.batlow, 10)
color_annual = colors[0]
color_winter = "#c51b7d"
color_obs = "#e08214"
color_pred = color_annual


def plotHeatmap(test_glaciers, data_glamos, glacierCap, period="annual"):
    # Heatmap of mean mass balance per glacier:
    # Get the mean mass balance per glacier
    data_with_pot = data_glamos[data_glamos.PERIOD == period]
    data_with_pot["GLACIER"] = data_glamos["GLACIER"].apply(lambda x: glacierCap[x])

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
    ax.set_xlabel("")
    # Update colorbar label fontsize
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(24)  # Adjust 14 to your desired fontsize

    # add patches for test glaciers
    test_glaciers = [glacierCap[gl] for gl in test_glaciers]
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
    ax.tick_params(axis="y", labelsize=20)  # Adjust 16 to your preferred size
    ax.tick_params(axis="x", labelsize=20)  # Adjust 16 to your preferred size
    plt.tight_layout()


def visualiseSplits(y_test, y_train, splits, colors=[color_annual, color_winter]):
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
    colors = get_cmap_hex(cm.vik, 10)
    color_annual = colors[0]
    color_winter = colors[2]
    f, ax = plt.subplots(
        2, len(vois_climate) + 3, figsize=(16, 6), sharey="row", sharex="col"
    )
    train_set["df_X"]["POINT_BALANCE"].plot.hist(
        ax=ax[0, 0], color=color_annual, alpha=0.6, density=False
    )
    ax[0, 0].set_title("PMB")
    ax[0, 0].set_ylabel("Frequency (train)")
    train_set["df_X"]["ELEVATION_DIFFERENCE"].plot.hist(
        ax=ax[0, 1], color=color_annual, alpha=0.6, density=False
    )
    ax[0, 1].set_title("ELV_DIFF")
    train_set["df_X"]["YEAR"].plot.hist(
        ax=ax[0, 2], color=color_annual, alpha=0.6, density=False
    )
    ax[0, 2].set_title("YEARS")

    for i, voi_clim in enumerate(vois_climate):
        ax[0, 3 + i].set_title(voi_clim)
        train_set["df_X"][voi_clim].plot.hist(
            ax=ax[0, 3 + i], color=color_annual, alpha=0.6, density=False
        )

    test_set["df_X"]["POINT_BALANCE"].plot.hist(
        ax=ax[1, 0], color=color_winter, alpha=0.6, density=False
    )
    ax[1, 0].set_ylabel("Frequency (test)")
    test_set["df_X"]["ELEVATION_DIFFERENCE"].plot.hist(
        ax=ax[1, 1], color=color_winter, alpha=0.6, density=False
    )
    test_set["df_X"]["YEAR"].plot.hist(
        ax=ax[1, 2], color=color_winter, alpha=0.6, density=False
    )

    for i, voi_clim in enumerate(vois_climate):
        test_set["df_X"][voi_clim].plot.hist(
            ax=ax[1, 3 + i], color=color_winter, alpha=0.6, density=False
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

    plt.plot(mean_train, label="train", color=color_annual)
    plt.plot(mean_test, label="validation", color=color_winter)

    # add std
    plt.fill_between(
        dfCVResults.index,
        mean_train - std_train,
        mean_train + std_train,
        alpha=0.2,
        color=color_annual,
    )
    plt.fill_between(
        dfCVResults.index,
        mean_test - std_test,
        mean_test + std_test,
        alpha=0.2,
        color=color_winter,
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
        ax.scatter(
            x=mean_test.index, y=mean_test.values, marker="x", color=color_winter
        )
        ax.plot(mean_test.index, mean_test, color=color_winter, label="validation")
        ax.fill_between(
            mean_test.index,
            mean_test - std_test,
            mean_test + std_test,
            alpha=0.2,
            color=color_winter,
        )

        ax.scatter(
            x=mean_train.index, y=mean_train.values, marker="x", color=color_annual
        )
        ax.plot(mean_train.index, mean_train, color=color_annual, label="train")
        ax.fill_between(
            mean_train.index,
            mean_train - std_train,
            mean_train + std_train,
            alpha=0.2,
            color=color_annual,
        )
        # add vertical line of best param
        ax.axvline(best_params[param], color="red", linestyle="--")

        ax.set_ylabel(f"{lossType}")
        ax.set_title(param)
        ax.legend()

    plt.suptitle("Grid search results")
    plt.tight_layout()


def FIPlot(best_estimator, feature_columns, vois_climate):
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


def PlotPredictions(grouped_ids, y_pred, metadata_test, test_set, model):
    fig = plt.figure(figsize=(15, 10))
    colors_glacier = [
        "#a6cee3",
        "#1f78b4",
        "#b2df8a",
        "#33a02c",
        "#fb9a99",
        "#e31a1c",
        "#fdbf6f",
        "#ff7f00",
        "#cab2d6",
        "#6a3d9a",
        "#ffff99",
        "#b15928",
    ]
    color_palette_glaciers = dict(zip(grouped_ids.GLACIER.unique(), colors_glacier))
    ax1 = plt.subplot(2, 2, 1)
    grouped_ids_annual = grouped_ids[grouped_ids.PERIOD == "annual"]
    mse_annual, rmse_annual, mae_annual, pearson_corr_annual, r2_annual, bias_annual = (
        model.evalMetrics(metadata_test, y_pred, test_set["y"], period="annual")
    )
    scores_annual = {
        "mse": mse_annual,
        "rmse": rmse_annual,
        "mae": mae_annual,
        "pearson_corr": pearson_corr_annual,
        "r2": r2_annual,
        "bias": bias_annual,
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
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title("Mean annual PMB", fontsize=24)
    plotMeanPred(grouped_ids_annual, ax2)

    if "winter" in grouped_ids.PERIOD.unique():
        grouped_ids_winter = grouped_ids[grouped_ids.PERIOD == "winter"]
        ax3 = plt.subplot(2, 2, 3)
        (
            mse_winter,
            rmse_winter,
            mae_winter,
            pearson_corr_winter,
            r2_winter,
            bias_winter,
        ) = model.evalMetrics(metadata_test, y_pred, test_set["y"], period="winter")
        scores_winter = {
            "mse": mse_winter,
            "rmse": rmse_winter,
            "mae": mae_winter,
            "pearson_corr": pearson_corr_winter,
            "r2": r2_winter,
            "bias": bias_winter,
        }
        predVSTruth(
            ax3,
            grouped_ids_winter,
            scores_winter,
            hue="GLACIER",
            palette=color_palette_glaciers,
        )
        ax3.set_title("Winter PMB", fontsize=24)

        ax4 = plt.subplot(2, 2, 4)
        ax4.set_title("Mean winter PMB", fontsize=24)
        grouped_ids_winter.sort_values(by="YEAR", inplace=True)
        plotMeanPred(grouped_ids_winter, ax4)


def predVSTruth(
    ax,
    grouped_ids,
    scores,
    hue="GLACIER",
    palette=None,
    color=color_annual,
    ax_xlim=(-8, 6),
    ax_ylim=(-8, 6),
):
    """
    Wrapper around mbm.plots.predVSTruth with custom parameters for Switzerland.
    """
    return mbm.plots.predVSTruth(
        grouped_ids,
        ax=ax,
        scores=scores,
        hue=hue,
        markers={"annual": "o", "winter": "o"},
        palette=palette,
        color=color,
        style="PERIOD",
        xlabel="Observed PMB [m w.e.]",
        ylabel="Modelled PMB [m w.e.]",
        ax_xlim=ax_xlim,
        ax_ylim=ax_ylim,
    )


def plotMeanPred(grouped_ids, ax, color_pred=color_annual, color_obs="orange"):
    mean = grouped_ids.groupby("YEAR")["target"].mean().values
    std = grouped_ids.groupby("YEAR")["target"].std().values
    years = grouped_ids.YEAR.unique()
    ax.fill_between(
        years,
        mean - std,
        mean + std,
        color=color_obs,
        alpha=0.3,
    )
    ax.plot(years, mean, color=color_obs, label="observed")
    ax.scatter(years, mean, color=color_obs, marker="x")
    ax.plot(
        years,
        grouped_ids.groupby("YEAR")["pred"].mean().values,
        color=color_pred,
        label="predicted",
        linestyle="--",
    )
    ax.scatter(
        years,
        grouped_ids.groupby("YEAR")["pred"].mean().values,
        color=color_pred,
        marker="x",
    )
    ax.fill_between(
        years,
        grouped_ids.groupby("YEAR")["pred"].mean().values
        - grouped_ids.groupby("YEAR")["pred"].std().values,
        grouped_ids.groupby("YEAR")["pred"].mean().values
        + grouped_ids.groupby("YEAR")["pred"].std().values,
        color=color_pred,
        alpha=0.3,
    )
    # rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    scores = mbm.metrics.scores(mean, grouped_ids.groupby("YEAR")["pred"].mean().values)
    mae = scores["mae"]
    rmse = scores["rmse"]
    pearson_corr = scores["pearson_corr"]
    legend_xgb = "\n".join((r"$\mathrm{RMSE}=%.3f$ " % (rmse,),))
    ax.text(
        0.03,
        0.96,
        legend_xgb,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=20,
    )
    ax.legend(fontsize=20, loc="lower right")


def PlotIndividualGlacierPredVsTruth(
    grouped_ids,
    color_annual,
    color_winter,
    axs,
    custom_order=None,
    ax_xlim=(-8, 6),
    ax_ylim=(-8, 6),
):
    """
    Wrapper around mbm.plots.predVSTruthPerGlacier with custom parameters for Switzerland.
    """

    scores = {}
    for i, test_gl in enumerate(grouped_ids["GLACIER"].unique()):
        df_gl = grouped_ids[grouped_ids.GLACIER == test_gl]

        df_gl_annual = df_gl[df_gl["PERIOD"] == "annual"]
        if not df_gl_annual.empty:
            scores_annual = mbm.metrics.scores(
                df_gl_annual["target"], df_gl_annual["pred"]
            )

        df_gl_winter = df_gl[df_gl["PERIOD"] == "winter"]
        if not df_gl_winter.empty:
            scores_winter = mbm.metrics.scores(
                df_gl_winter["target"], df_gl_winter["pred"]
            )
            scores[test_gl] = {
                "RMSE": {"a": scores_annual["rmse"], "w": scores_winter["rmse"]},
                "R2": {"a": scores_annual["r2"], "w": scores_winter["r2"]},
                "B": {"a": scores_annual["bias"], "w": scores_winter["bias"]},
            }
        elif not df_gl_annual.empty:
            scores[test_gl] = {
                "RMSE": {"a": scores_annual["rmse"]},
                "R2": {"a": scores_annual["r2"]},
                "B": {"a": scores_annual["bias"]},
            }

    mbm.plots.predVSTruthPerGlacier(
        grouped_ids,
        axs=axs,
        scores=scores,
        custom_order=custom_order,
        xlabel="Observed PMB [m w.e.]",
        ylabel="Modelled PMB [m w.e.]",
        markers={"annual": "o", "winter": "o"},
        style="PERIOD",  # markers
        hue="PERIOD",
        palette=[color_annual, color_winter],
        precLegend=2,
    )


def plotGlAttr(ds, cmap=cm.batlow):
    # Plot glacier attributes
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    ds.masked_slope.plot(ax=ax[0, 0], cmap=cmap)
    ax[0, 0].set_title("Slope")
    ds.masked_elev.plot(ax=ax[0, 1], cmap=cmap)
    ax[0, 1].set_title("Elevation")
    ds.masked_aspect.plot(ax=ax[0, 2], cmap=cmap)
    ax[0, 2].set_title("Aspect")
    ds.masked_hug.plot(ax=ax[1, 0], cmap=cmap)
    ax[1, 0].set_title("Hugonnet")
    ds.masked_cit.plot(ax=ax[1, 1], cmap=cmap)
    ax[1, 1].set_title("Consensus ice thickness")
    ds.masked_miv.plot(ax=ax[1, 2], cmap=cmap)
    ax[1, 2].set_title("Millan v")
    plt.tight_layout()


def plot_predictions_summary(
    grouped_ids,
    scores_annual,
    scores_winter,
    predVSTruth,
    plotMeanPred,
    ax_xlim=(-8, 6),
    ax_ylim=(-8, 6),
):
    """
    Plots a summary figure with NN predictions and PMB trends.

    Parameters:
    - grouped_ids_xgb: DataFrame with prediction results and 'PERIOD' and 'YEAR' columns.
    - scores_annual_xgb: dict with keys 'rmse' and 'R2' for annual scores.
    - scores_winter_xgb: dict with keys 'rmse' and 'R2' for winter scores.
    - predVSTruth: function to plot predicted vs true values.
    - plotMeanPred: function to plot mean prediction time series.
    - color_annual, color_winter: colors for the NN plot.
    """
    # Create figure
    fig = plt.figure(figsize=(20, 8))

    # Define grid: 2 rows x 2 columns
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2])

    # Left plot spans both rows
    ax1 = fig.add_subplot(gs[:, 0])

    # Right column has two plots
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    # Left panel: Predictions vs Truth
    ax1.set_title("", fontsize=24)
    predVSTruth(
        ax1,
        grouped_ids,
        scores_annual,
        hue="PERIOD",
        add_legend=False,
        palette=[color_annual, color_winter],
        ax_xlim=ax_xlim,
        ax_ylim=ax_ylim,
    )

    legend_NN = "\n".join(
        [
            r"$\mathrm{RMSE_a}=%.3f$, $\mathrm{RMSE_w}=%.3f$"
            % (scores_annual["rmse"], scores_winter["rmse"]),
            r"$\mathrm{R^2_a}=%.3f$, $\mathrm{R^2_w}=%.3f$"
            % (scores_annual["R2"], scores_winter["R2"]),
            r"$\mathrm{B_a}=%.3f$, $\mathrm{B_w}=%.3f$"
            % (scores_annual["Bias"], scores_winter["Bias"]),
        ]
    )
    ax1.text(
        0.03,
        0.98,
        legend_NN,
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=20,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    # Top-right: Mean annual PMB
    ax2.set_title("Mean yearly annual point mass balance", fontsize=24)
    grouped_ids_xgb_annual = grouped_ids[grouped_ids.PERIOD == "annual"].sort_values(
        by="YEAR"
    )
    plotMeanPred(
        grouped_ids_xgb_annual, ax2, color_pred=color_pred, color_obs=color_obs
    )
    ax2.set_ylabel("PMB [m w.e.]", fontsize=20)

    # Bottom-right: Mean winter PMB
    ax3.set_title("Mean yearly winter point mass balance", fontsize=24)
    grouped_ids_xgb_winter = grouped_ids[grouped_ids.PERIOD == "winter"].sort_values(
        by="YEAR"
    )
    plotMeanPred(
        grouped_ids_xgb_winter, ax3, color_pred=color_pred, color_obs=color_obs
    )
    ax3.set_ylabel("PMB [m w.e.]", fontsize=20)

    # Remove legend from ax3 if it exists
    if ax3.get_legend() is not None:
        ax3.get_legend().remove()

    plt.tight_layout()
    return fig  # return figure in case further customization or saving is needed


def compute_seasonal_scores(df, target_col="target", pred_col="pred"):
    """
    Computes regression scores separately for annual and winter data.

    Parameters:
    - df: DataFrame with at least 'PERIOD', target_col, and pred_col columns.
    - target_col: name of the column with ground truth values.
    - pred_col: name of the column with predicted values.

    Returns:
    - scores_annual: dict of metrics for annual data.
    - scores_winter: dict of metrics for winter data.
    """

    scores = {}
    for season in ["annual", "winter"]:
        df_season = df[df["PERIOD"] == season]
        y_true = df_season[target_col]
        y_pred = df_season[pred_col]
        seasonal_scores = mbm.metrics.scores(y_true, y_pred)
        seasonal_scores["R2"] = seasonal_scores.pop("r2")
        seasonal_scores["Bias"] = seasonal_scores.pop("bias")
        scores[season] = seasonal_scores
    return scores["annual"], scores["winter"]


def plot_scatter_geodetic_MB(df_all, hue, size, ax, y_col, rmse, corr):
    """Helper function to plot a scatter plot with annotations"""
    sns.scatterplot(
        data=df_all,
        x="Geodetic MB",
        y=y_col,
        hue=hue,
        size="Area" if size else None,
        sizes=(10, 1000),
        alpha=0.7,
        ax=ax,
    )

    # Identity line through the origin
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    lims = [min(xlims[0], ylims[0]), max(xlims[1], ylims[1])]
    ax.plot(lims, lims, "--", color="grey", linewidth=1)

    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Grid and axis labels
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.set_xlabel("Geodetic MB [m w.e.]")
    ax.set_ylabel(f"{y_col} [m w.e.]")

    # RMSE and correlation annotation
    legend_text = "\n".join(
        (r"$\mathrm{RMSE}=%.3f$" % rmse, r"$\mathrm{\rho}=%.3f$" % corr)
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    ax.text(
        0.03,
        0.94,
        legend_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=18,
        bbox=props,
    )
    ax.legend([], [], frameon=False)


def plot_permutation_importance(
    df_importance, top_n=None, figsize=(10, 6), title="Permutation Feature Importance"
):
    # Sort features by importance
    df_plot = df_importance.sort_values(by="mean_importance", ascending=True)
    if top_n:
        df_plot = df_plot.tail(top_n)

    # give long name to features
    df_plot["feature_long"] = df_plot["feature"].apply(
        lambda x: vois_climate_long_name.get(x, x)
    )

    plt.figure(figsize=figsize)
    plt.barh(
        df_plot["feature_long"],
        df_plot["mean_importance"],
        xerr=df_plot["std_importance"],
        align="center",
        alpha=0.7,
        ecolor="black",
        color=color_annual,
        capsize=5,
    )
    plt.xlabel("Increase in RMSE (mean ± std)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
