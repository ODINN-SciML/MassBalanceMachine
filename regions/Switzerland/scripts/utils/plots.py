import seaborn as sns
from cmcrameri import cm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
import pandas as pd
from matplotlib.patches import Patch
import joypy
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d

import massbalancemachine as mbm

from regions.Switzerland.scripts.utils import *
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


def plot_predictions_summary(
    grouped_ids,
    scores_annual,
    scores_winter,
    ax_xlim=(-8, 6),
    ax_ylim=(-8, 6),
    color_annual=COLOR_ANNUAL,
    color_winter=COLOR_WINTER,
):
    """
    Create a summary figure showing predicted vs observed PMB and mean time series.

    The figure includes:
      - Predicted vs observed scatter (annual + winter),
      - Mean annual PMB time series (predicted vs observed),
      - Mean winter PMB time series (predicted vs observed).

    Parameters
    ----------
    grouped_ids : pandas.DataFrame
        Table containing predictions and targets with at least columns
        ['target', 'pred', 'PERIOD', 'YEAR'].
    scores_annual : dict
        Annual performance metrics (e.g., keys like 'rmse', 'R2', 'Bias').
    scores_winter : dict
        Winter performance metrics (e.g., keys like 'rmse', 'R2', 'Bias').
    ax_xlim, ax_ylim : tuple, optional
        Axis limits for the predicted-vs-observed panel.
    color_annual, color_winter : str, optional
        Colors used for annual and winter points in the scatter plot.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure (useful for saving or further customization).
    """
    subplot_labels = ["(a)", "(b)", "(c)"]
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
    pred_vs_truth(
        ax1,
        grouped_ids,
        scores_annual,
        hue="PERIOD",
        add_legend=False,
        palette=[color_annual, color_winter],
        ax_xlim=ax_xlim,
        ax_ylim=ax_ylim,
    )
    ax1.text(
        0.02,
        0.98,
        subplot_labels[0],
        transform=ax1.transAxes,
        fontsize=24,
        verticalalignment="top",
        horizontalalignment="left",
    )

    legend_NN = "\n".join(
        [
            r"$\mathrm{RMSE_a}=%.2f$, $\mathrm{RMSE_w}=%.2f$"
            % (scores_annual["rmse"], scores_winter["rmse"]),
            r"$\mathrm{R^2_a}=%.2f$, $\mathrm{R^2_w}=%.2f$"
            % (scores_annual["R2"], scores_winter["R2"]),
            r"$\mathrm{Bias_a}=%.2f$, $\mathrm{Bias_w}=%.2f$"
            % (scores_annual["Bias"], scores_winter["Bias"]),
        ]
    )
    ax1.text(
        0.25,
        0.98,
        legend_NN,
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=20,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    # Top-right: Mean annual PMB
    color_pred = "#762a83"
    color_obs = "black"
    ax2.set_title("Mean yearly annual point mass balance", fontsize=24)
    grouped_ids_xgb_annual = grouped_ids[grouped_ids.PERIOD == "annual"].sort_values(
        by="YEAR"
    )
    plot_mean_pred(
        grouped_ids_xgb_annual,
        ax2,
        color_pred=color_pred,
        color_obs=color_obs,
        linestyle_pred="-",
        linestyle_obs="--",
    )
    ax2.set_ylabel("PMB [m w.e.]", fontsize=20)
    ax2.text(
        0.01,
        0.98,
        subplot_labels[1],
        transform=ax2.transAxes,
        fontsize=24,
        verticalalignment="top",
        horizontalalignment="left",
    )

    # Bottom-right: Mean winter PMB
    ax3.set_title("Mean yearly winter point mass balance", fontsize=24)
    grouped_ids_xgb_winter = grouped_ids[grouped_ids.PERIOD == "winter"].sort_values(
        by="YEAR"
    )
    plot_mean_pred(
        grouped_ids_xgb_winter,
        ax3,
        color_pred=color_pred,
        color_obs=color_obs,
        linestyle_pred="-",
        linestyle_obs="--",
    )
    ax3.set_ylabel("PMB [m w.e.]", fontsize=20)
    ax3.text(
        0.01,
        0.98,
        subplot_labels[2],
        transform=ax3.transAxes,
        fontsize=24,
        verticalalignment="top",
        horizontalalignment="left",
    )
    # Remove legend from ax3 if it exists
    if ax3.get_legend() is not None:
        ax3.get_legend().remove()

    plt.tight_layout()
    return fig  # return figure in case further customization or saving is needed


def pred_vs_truth(
    ax,
    grouped_ids,
    scores,
    hue="GLACIER",
    palette=None,
    color=COLOR_ANNUAL,
    add_legend=True,
    ax_xlim=(-8, 6),
    ax_ylim=(-8, 6),
):
    """
    Scatter plot of predicted vs observed point mass balance with 1:1 reference.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    grouped_ids : pandas.DataFrame
        DataFrame with at least ['target', 'pred', 'PERIOD'] and optionally
        the column specified by `hue`.
    scores : dict
        Metrics to display in the annotation (expects keys like 'rmse',
        'pearson_corr' if `add_legend=True`).
    hue : str or None, optional
        Column used for coloring groups (e.g., 'GLACIER' or 'PERIOD').
        If None, no categorical legend is shown.
    palette : list or dict, optional
        Seaborn palette used when `hue` is provided.
    color : str, optional
        Fallback color when no palette/hue is used.
    add_legend : bool, optional
        If True, adds a metrics text box.
    ax_xlim, ax_ylim : tuple, optional
        Axis limits.

    Returns
    -------
    None
        Modifies the provided axis in place.
    """
    sns.scatterplot(
        grouped_ids,
        x="target",
        y="pred",
        palette=palette,
        hue=hue,
        ax=ax,
        color=color,
        style="PERIOD",
        markers={"annual": "o", "winter": "o"},
    )  # optional custom marker map)

    ax.set_ylabel("Modeled PMB [m w.e.]", fontsize=20)
    ax.set_xlabel("Observed PMB [m w.e.]", fontsize=20)

    if add_legend:
        legend_xgb = "\n".join(
            (
                (r"$\mathrm{RMSE}=%.2f$," % (scores["rmse"],)),
                (r"$\mathrm{\rho}=%.2f$" % (scores["pearson_corr"],)),
            )
        )
        ax.text(
            0.03,
            0.98,
            legend_xgb,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=20,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )
    if hue is not None:
        ax.legend(fontsize=20, loc="lower right", ncol=2)
    else:
        ax.legend([], [], frameon=False)
    # diagonal line
    pt = (0, 0)
    ax.axline(pt, slope=1, color="grey", linestyle="-", linewidth=0.2)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.grid()
    leg = ax.get_legend()
    if leg is not None:
        for txt in leg.get_texts():
            t = txt.get_text().strip().lower()
            if t in ("annual", "winter"):
                txt.set_text(t.capitalize())
    # Set ylimits to be the same as xlimits
    ax.set_xlim(ax_xlim)
    ax.set_ylim(ax_ylim)
    plt.tight_layout()


def plot_mean_pred(
    grouped_ids,
    ax,
    color_pred=COLOR_ANNUAL,
    color_obs="orange",
    linestyle_pred="--",
    linestyle_obs="-",
):
    """
    Plot mean observed and predicted PMB time series with ±1 std envelopes.

    Parameters
    ----------
    grouped_ids : pandas.DataFrame
        DataFrame with columns ['YEAR', 'target', 'pred'].
    ax : matplotlib.axes.Axes
        Axis to draw on.
    color_pred, color_obs : str, optional
        Line/fill colors for predicted and observed series.
    linestyle_pred, linestyle_obs : str, optional
        Line styles for predicted and observed series.

    Returns
    -------
    None
        Modifies the provided axis in place.
    """
    # Aggregate once
    g = grouped_ids.groupby("YEAR")
    years = np.sort(g.size().index.values)

    obs_mean = g["target"].mean().reindex(years).values
    obs_std = g["target"].std().reindex(years).values

    pred_mean = g["pred"].mean().reindex(years).values
    pred_std = g["pred"].std().reindex(years).values

    # Observations
    ax.fill_between(
        years, obs_mean - obs_std, obs_mean + obs_std, color=color_obs, alpha=0.3
    )
    ax.plot(
        years,
        obs_mean,
        color=color_obs,
        label="observed",
        linestyle=linestyle_obs,
    )

    # Predictions
    ax.plot(
        years,
        pred_mean,
        color=color_pred,
        label="predicted",
        linestyle=linestyle_pred,
        marker="v",
    )
    ax.fill_between(
        years, pred_mean - pred_std, pred_mean + pred_std, color=color_pred, alpha=0.3
    )

    # Rotate x labels (safer than set_xticklabels)
    ax.tick_params(axis="x", rotation=45)

    # Metrics
    scores = mbm.metrics.scores(obs_mean, pred_mean)
    mae = scores["mae"]
    rmse = scores["rmse"]
    pearson_corr = scores["pearson_corr"]
    legend_text = "\n".join((rf"$\mathrm{{RMSE}}={rmse:.2f}$",))
    ax.text(0.055, 0.96, legend_text, transform=ax.transAxes, va="top", fontsize=20)

    ax.legend(fontsize=20, loc="lower right")


def plot_individual_glacier_pred(
    grouped_ids,
    color_annual,
    color_winter,
    axs,
    subplot_labels=None,  # <— now optional
    custom_order=None,
    add_text=True,
    ax_xlim=(-9, 6),
    ax_ylim=(-9, 6),
    gl_area={},
):
    """
    Plot predicted vs observed PMB for multiple glaciers in a panel grid.

    Creates one scatter subplot per glacier (annual + winter), adds 1:1 line,
    optional per-glacier metric text, and a title including area and mean elevation.

    Parameters
    ----------
    grouped_ids : pandas.DataFrame
        DataFrame containing at least ['GLACIER', 'PERIOD', 'target', 'pred', 'gl_elv'].
    color_annual, color_winter : str
        Colors for annual and winter points.
    axs : numpy.ndarray of matplotlib.axes.Axes
        Array of axes (e.g., from plt.subplots) to populate.
    subplot_labels : list of str, optional
        Labels like ['(a)', '(b)', ...]. If None, labels are auto-generated.
    custom_order : list of str, optional
        Glacier plotting order. If None, uses unique values from grouped_ids['GLACIER'].
    add_text : bool, optional
        If True, annotates each subplot with per-period metrics when available.
    ax_xlim, ax_ylim : tuple or None, optional
        Axis limits. If None, limits are set from the glacier-specific data.
    gl_area : dict, optional
        Mapping from glacier name (lowercase) to area in km^2 for subplot titles.

    Returns
    -------
    numpy.ndarray
        Flattened array of axes used for plotting.
    """
    color_palette_period = [color_annual, color_winter]

    if custom_order is None:
        custom_order = grouped_ids["GLACIER"].unique()

    ax_flat = axs.flatten()
    n_plots = min(len(custom_order), len(ax_flat))

    # Auto-generate labels if none provided (a), (b), ...
    if subplot_labels is None:
        subplot_labels = _alpha_labels(n_plots)
    else:
        # if provided shorter/longer, trim or extend deterministically
        if len(subplot_labels) < n_plots:
            subplot_labels = list(subplot_labels) + _alpha_labels(
                n_plots - len(subplot_labels)
            )
        else:
            subplot_labels = list(subplot_labels)[:n_plots]

    for i, test_gl in enumerate(custom_order[:n_plots]):
        gl_elv = int(
            np.round(grouped_ids[grouped_ids.GLACIER == test_gl]["gl_elv"].values[0], 0)
        )
        df_gl = grouped_ids[grouped_ids.GLACIER == test_gl]

        ax1 = ax_flat[i]

        sns.scatterplot(
            df_gl,
            x="target",
            y="pred",
            palette=color_palette_period,
            hue="PERIOD",
            style="PERIOD",
            markers={"annual": "o", "winter": "o"},
            ax=ax1,
            hue_order=["annual", "winter"],
        )

        # diagonal and axes zero lines
        ax1.axline((0, 0), slope=1, color="grey", linestyle="-", linewidth=0.2)
        ax1.axvline(0, color="grey", linestyle="--", linewidth=1)
        ax1.axhline(0, color="grey", linestyle="--", linewidth=1)

        # Set symmetric limits or provided limits
        if ax_xlim is None:
            ymin = math.floor(min(df_gl.pred.min(), df_gl.target.min()))
            ymax = math.ceil(max(df_gl.pred.max(), df_gl.target.max()))
            ax1.set_xlim(ymin, ymax)
            ax1.set_ylim(ymin, ymax)
        else:
            ax1.set_xlim(ax_xlim)
            ax1.set_ylim(ax_ylim)

        ax1.grid(alpha=0.2)
        ax1.tick_params(labelsize=18, pad=2)
        ax1.set_ylabel("")
        ax1.set_xlabel("")

        # remove legend (we’ll make a global one elsewhere if needed)
        leg = ax1.get_legend()
        if leg is not None:
            leg.remove()

        # Subplot label (auto or provided)
        ax1.text(
            0.02,
            0.98,
            subplot_labels[i],
            transform=ax1.transAxes,
            fontsize=24,
            va="top",
            ha="left",
        )

        # Metrics text
        legend_lines = []
        df_gl_annual = df_gl[df_gl["PERIOD"] == "annual"]
        if not df_gl_annual.empty:
            scores_annual = mbm.metrics.scores(
                df_gl_annual["target"], df_gl_annual["pred"]
            )
            legend_lines.append(
                rf"$\mathrm{{RMSE_a}}={scores_annual['rmse']:.2f},\ "
                rf"\mathrm{{R^2_a}}={scores_annual['r2']:.2f},\ "
                rf"\mathrm{{B_a}}={scores_annual['bias']:.2f}$"
            )

        df_gl_winter = df_gl[df_gl["PERIOD"] == "winter"]
        if not df_gl_winter.empty:
            scores_winter = mbm.metrics.scores(
                df_gl_winter["target"], df_gl_winter["pred"]
            )
            legend_lines.append(
                rf"$\mathrm{{RMSE_b}}={scores_winter['rmse']:.2f},\ "
                rf"\mathrm{{R^2_b}}={scores_winter['r2']:.2f},\ "
                rf"\mathrm{{B_b}}={scores_winter['bias']:.2f}$"
            )

        if add_text and legend_lines:
            ax1.text(
                0.98,
                0.02,
                "\n".join(legend_lines),
                transform=ax1.transAxes,
                fontsize=18,
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.0),
            )

        area = gl_area.get(test_gl.lower(), np.nan)
        if area < 0.1:
            area = np.round(area, 3)
        else:
            area = np.round(area, 1)

        ax1.set_title(
            f"{test_gl.capitalize()} ({area} km$^2$, {gl_elv} m a.s.l.)", fontsize=20
        )

    return ax_flat


def _alpha_labels(n: int):
    """
    Generate subplot labels '(a)', '(b)', ... '(z)', '(aa)', '(ab)', ... .

    Parameters
    ----------
    n : int
        Number of labels to generate.

    Returns
    -------
    list of str
        List of n labels formatted as '(...)'.
    """

    def to_label(k: int) -> str:
        # 0 -> a, 25 -> z, 26 -> aa, ...
        s = ""
        k += 1
        while k > 0:
            k, r = divmod(k - 1, 26)
            s = chr(97 + r) + s
        return f"({s})"

    return [to_label(i) for i in range(n)]


def plot_history_lstm(history):
    """
    Plot LSTM training/validation loss curves and optionally learning rate.

    Parameters
    ----------
    history : dict
        History dict with keys:
          - 'train_loss' : list-like
          - 'val_loss'   : list-like
          - 'lr'         : list-like, optional

    Returns
    -------
    None
        Creates and shows a matplotlib figure.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot losses
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="tab:blue")
    ax1.plot(epochs, history["val_loss"], label="Validation Loss", color="tab:orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # If LR is present, plot on secondary axis
    if "lr" in history:
        ax2 = ax1.twinx()
        ax2.plot(
            epochs,
            history["lr"],
            label="Learning Rate",
            color="tab:green",
            linestyle="--",
        )
        ax2.set_ylabel("Learning Rate")
        ax2.legend(loc="upper center")

    plt.show()


def plot_monthly_joyplot(
    df_long,
    month_order=None,
    color_lstm="tab:blue",
    color_nn="tab:orange",
    color_xgb="tab:green",
    color_glamos="gray",
    figsize_cm=(12, 14),
    x_range=(-2.2, 2.2),
    alpha=1,
):
    """
    Plot a ridge (joy) plot comparing monthly mass-balance distributions across models.

    Parameters
    ----------
    df_long : pandas.DataFrame
        Long-format table containing a 'Month' column and model columns
        ['mb_lstm', 'mb_nn', 'mb_xgb', 'mb_glamos'].
    month_order : list of str, optional
        Month ordering for the ridge plot (default: Oct–Sep hydrological order).
    color_lstm, color_nn, color_xgb, color_glamos : str, optional
        Colors for the respective distributions.
    figsize_cm : tuple, optional
        Figure size in centimeters (width, height).
    x_range : tuple, optional
        X-axis range for mass balance values.
    alpha : float, optional
        Transparency used in legend patches.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object returned by joypy (useful for saving).
    """

    if month_order is None:
        month_order = [
            "oct",
            "nov",
            "dec",
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
        ]

    cm = 1 / 2.54

    model_cols = ["mb_lstm", "mb_nn", "mb_xgb", "mb_glamos"]
    model_labels = ["LSTM", "NN", "XGB", "GLAMOS"]
    model_colors = [color_lstm, color_nn, color_xgb, color_glamos]

    fig, ax = joypy.joyplot(
        df_long,
        by="Month",
        column=model_cols,
        alpha=0.8,
        overlap=0,
        fill=False,
        linewidth=1.5,
        xlabelsize=8.5,
        ylabelsize=8.5,
        x_range=x_range,
        grid=False,
        color=model_colors,
        figsize=(figsize_cm[0] * cm, figsize_cm[1] * cm),
        ylim="own",
    )

    # Zero-line
    plt.axvline(x=0, color="grey", alpha=0.5, linewidth=1)

    # Axis labels & ticks
    plt.xlabel("Mass balance (m w.e.)", fontsize=8.5)
    plt.yticks(ticks=range(1, 13), labels=month_order, fontsize=8.5)
    plt.gca().set_yticklabels(month_order)

    # Legend
    legend_patches = [
        Patch(facecolor=color, label=label, alpha=alpha, edgecolor="k")
        for label, color in zip(model_labels, model_colors)
    ]

    plt.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.48, -0.1),
        ncol=4,
        fontsize=8.5,
        handletextpad=0.5,
        columnspacing=1,
    )

    plt.show()
    return fig


def plot_monthly_joyplot_single(
    df_long,
    variable,
    month_order=None,
    color_model="tab:blue",
    color_glamos="gray",
    figsize_cm=(12, 14),
    x_range=(-2.2, 2.2),
    alpha=1,
    show=True,
    model_name="lstm",
    y_offset=0.3,
):
    """
    Plot a monthly ridge (joy) plot for one model variable against GLAMOS.

    Creates a JoyPy ridge plot of the monthly distributions of `variable` and
    'mb_glamos' (reference). For each month, the plot is annotated with:
      - OVL: distribution overlap coefficient between model and GLAMOS
      - Δμ: mean bias (model minus GLAMOS)

    Parameters
    ----------
    df_long : pandas.DataFrame
        Long-format DataFrame with a 'Month' column and columns for `variable`
        and 'mb_glamos'.
    variable : str
        Column name in `df_long` for the model mass-balance values
        (e.g., 'mb_lstm', 'mb_nn', 'mb_xgb').
    month_order : list of str, optional
        Month ordering for the ridge plot (default: Oct–Sep hydrological order).
    color_model : str, optional
        Color used for the model distribution curves.
    color_glamos : str, optional
        Color used for the GLAMOS distribution curves.
    figsize_cm : tuple, optional
        Figure size in centimeters (width, height).
    x_range : tuple, optional
        X-axis range for mass balance values.
    alpha : float, optional
        Transparency used in legend patches.
    show : bool, optional
        If True, calls `plt.show()` to display the figure.
    model_name : str, optional
        Label used for the model in the legend (converted to uppercase).
    y_offset : float, optional
        Vertical offset (in axis coordinates of each ridge panel) for the
        annotation text.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object returned by JoyPy (useful for saving).
    """

    if month_order is None:
        month_order = [
            "oct",
            "nov",
            "dec",
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
        ]

    cm = 1 / 2.54  # centimeters to inches conversion

    # --- Ridge plot: model + GLAMOS ---
    fig, ax = joypy.joyplot(
        df_long,
        by="Month",
        column=[variable, "mb_glamos"],
        alpha=0.8,
        overlap=0,
        fill=False,
        linewidth=1.5,
        xlabelsize=10,
        ylabelsize=10,
        x_range=x_range,
        grid=False,
        color=[color_model, color_glamos],
        figsize=(figsize_cm[0] * cm, figsize_cm[1] * cm),
        ylim="own",
    )

    # --- Aesthetics ---
    plt.axvline(x=0, color="grey", alpha=0.5, linewidth=1)
    plt.xlabel("Mass balance (m w.e.)", fontsize=8.5)
    plt.yticks(ticks=range(1, 13), labels=month_order, fontsize=8.5)
    plt.gca().set_yticklabels(month_order)

    # --- Legend ---
    model_name = model_name.upper()
    legend_patches = [
        Patch(facecolor=color_model, label=model_name, alpha=alpha, edgecolor="k"),
        Patch(facecolor=color_glamos, label="GLAMOS", alpha=alpha, edgecolor="k"),
    ]
    plt.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.48, -0.1),
        ncol=2,
        fontsize=10,
        handletextpad=0.5,
        columnspacing=1,
    )

    # --- Compute monthly overlap coefficients ---
    overlap_by_month = monthly_overlap_coefficients(
        df_long,
        model_col=variable,
        x_range=x_range,
    )
    bias_by_month = monthly_mean_bias(
        df_long,
        model_col=variable,
    )

    # --- Annotate overlap on plot ---
    # x_text = x_range[1] * 0.96  # right-hand side
    x_text = x_range[0]  # + 0.01 * (x_range[1] - x_range[0])

    for i, month in enumerate(month_order):
        ov = overlap_by_month.get(month)

        if ov is None or not np.isfinite(ov):
            continue

        ax[i].text(
            x_text,
            y_offset,  # y=0 is the ridge baseline in each axis
            f"OVL={ov:.2f}, Δμ={bias_by_month[month]:+.2f}",
            ha="left",
            va="center",
            fontsize=9,
            color="black",
        )

    if show:
        plt.show()

    return fig


def monthly_overlap_coefficients(
    df_long,
    model_col,
    glamos_col="mb_glamos",
    month_col="Month",
    x_range=(-2.2, 2.2),
    n_grid=1000,
):
    """
    Compute monthly overlap coefficients between model and GLAMOS distributions.

    For each month, kernel density estimates (KDEs) are fitted to the model
    and GLAMOS mass-balance values, and the overlap coefficient is computed
    as the integral of the minimum of the two densities.

    Parameters
    ----------
    df_long : pandas.DataFrame
        Long-format DataFrame containing monthly mass-balance values.
    model_col : str
        Column name for the model mass-balance values.
    glamos_col : str, optional
        Column name for the GLAMOS reference values.
    month_col : str, optional
        Column used to group data by month.
    x_range : tuple, optional
        Range over which KDEs are evaluated.
    n_grid : int, optional
        Number of points used for KDE evaluation.

    Returns
    -------
    dict
        Dictionary mapping month -> overlap coefficient. Months with
        insufficient data return NaN.
    """
    overlaps = {}

    x = np.linspace(x_range[0], x_range[1], n_grid)

    for month, df_m in df_long.groupby(month_col):
        model_vals = df_m[model_col].dropna().values
        glamos_vals = df_m[glamos_col].dropna().values

        # skip months with too little data
        if len(model_vals) < 5 or len(glamos_vals) < 5:
            overlaps[month] = np.nan
            continue

        kde_model = gaussian_kde(model_vals)
        kde_glamos = gaussian_kde(glamos_vals)

        overlap = np.trapz(np.minimum(kde_model(x), kde_glamos(x)), x)

        overlaps[month] = overlap

    return overlaps


def monthly_mean_bias(
    df_long,
    model_col,
    glamos_col="mb_glamos",
    month_col="Month",
):
    """
    Compute monthly mean bias between model and GLAMOS mass balance.

    Bias is defined as the difference between the mean model value and
    the mean GLAMOS value for each month.

    Parameters
    ----------
    df_long : pandas.DataFrame
        Long-format DataFrame containing monthly mass-balance values.
    model_col : str
        Column name for the model mass-balance values.
    glamos_col : str, optional
        Column name for the GLAMOS reference values.
    month_col : str, optional
        Column used to group data by month.

    Returns
    -------
    dict
        Dictionary mapping month -> mean bias (model − GLAMOS). Months with
        insufficient data return NaN.
    """
    bias = {}

    for month, df_m in df_long.groupby(month_col):
        model_vals = df_m[model_col].dropna()
        glamos_vals = df_m[glamos_col].dropna()

        if len(model_vals) == 0 or len(glamos_vals) == 0:
            bias[month] = np.nan
            continue

        bias[month] = model_vals.mean() - glamos_vals.mean()

    return bias


def plot_pfi_annual(df):
    """
    Plot permutation feature importance for annual mass-balance predictions.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing permutation importance results with at least
        the columns:
        ['feature', 'mean_delta_annual', 'std_delta_annual', 'baseline_annual'].

    Returns
    -------
    None
        Creates and displays a horizontal bar plot.
    """
    d = df.sort_values("mean_delta_annual", ascending=False)

    plt.figure(figsize=(8, max(3, 0.35 * len(d))))
    plt.barh(d["feature"], d["mean_delta_annual"], xerr=d["std_delta_annual"])
    plt.gca().invert_yaxis()
    plt.title(
        f"Permutation Importance – Annual (baseline RMSE={d.baseline_annual.iloc[0]:.3f})"
    )
    plt.xlabel("Increase in RMSE_annual")
    plt.tight_layout()
    plt.show()


def plot_pfi_winter(df):
    """
    Plot permutation feature importance for winter mass-balance predictions.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing permutation importance results with at least
        the columns:
        ['feature', 'mean_delta_winter', 'std_delta_winter', 'baseline_winter'].

    Returns
    -------
    None
        Creates and displays a horizontal bar plot.
    """
    d = df.sort_values("mean_delta_winter", ascending=False)

    plt.figure(figsize=(8, max(3, 0.35 * len(d))))
    plt.barh(d["feature"], d["mean_delta_winter"], xerr=d["std_delta_winter"])
    plt.gca().invert_yaxis()
    plt.title(
        f"Permutation Importance – Winter (baseline RMSE={d.baseline_winter.iloc[0]:.3f})"
    )
    plt.xlabel("Increase in RMSE_winter")
    plt.tight_layout()
    plt.show()


def plot_monthly_pfi_ridges(
    pfi_monthly,
    MONTHLY_COLS,
    vois_climate_long_name,
    months_tail_pad,
    months_head_pad,
    metric="global",  # "winter", "annual", "global"
    drop_padded_months=True,
    fname=None,
    title=None,
):
    """
    Plot month-resolved permutation feature importance (PFI) as stacked ridge curves.

    The function filters the provided month-wise PFI table to monthly climate
    predictors, aggregates importance by (feature, month), optionally removes
    padded months, smooths the month-to-month signal, and visualizes each
    feature as a vertically offset ridge line. Values can be shown as either
    relative or absolute ΔRMSE for winter, annual, or a global metric.

    Parameters
    ----------
    pfi_monthly : pandas.DataFrame
        Month-wise PFI results with columns including:
        'feature', 'month', and the relevant importance columns
        (e.g., mean_delta_winter_rel / mean_delta_winter / mean_delta_global, etc.).
    MONTHLY_COLS : list of str
        List of feature names to include (monthly predictors only).
    vois_climate_long_name : dict
        Mapping from short feature names to long display names.
    months_tail_pad : array-like
        Month labels used as padding at the end of the hydrological year
        (excluded when `drop_padded_months=True`).
    months_head_pad : array-like
        Month labels used as padding at the start of the hydrological year
        (excluded when `drop_padded_months=True`).
    metric : {"winter", "annual", "global"}, optional
        Which metric to visualize (controls which PFI column is used).
    drop_padded_months : bool, optional
        If True, remove padded months from the plot and month order.
    fname : str or None, optional
        If provided, path where the figure is saved (dpi=300).
    title : str or None, optional
        Custom plot title. If None, a default title is generated.

    Returns
    -------
    None
        Creates and displays a matplotlib figure (and optionally saves it).
    """
    if metric == "winter":
        value_col = "mean_delta_winter"
        label = "ΔWinter RMSE"
    elif metric == "annual":
        value_col = "mean_delta_annual"
        label = "ΔAnnual RMSE"
    else:
        value_col = "mean_delta_global"
        label = "ΔGlobal RMSE"
    annot_fmt = "ΔRMSE={:.3f}"

    full_month_order = [
        "aug_",
        "sep_",
        "oct",
        "nov",
        "dec",
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct_",
    ]

    df = pfi_monthly.copy()
    df = df[df.feature.isin(MONTHLY_COLS)]
    df["feature_long"] = df["feature"].apply(lambda x: vois_climate_long_name.get(x, x))

    if drop_padded_months:
        padded = np.concatenate([months_tail_pad, months_head_pad])
        df = df[~df.month.isin(padded)]
        month_order = [m for m in full_month_order if m not in padded]
    else:
        month_order = [
            "sep_",
            "oct",
            "nov",
            "dec",
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct_",
        ]

    df = df.groupby(["feature_long", "month"], as_index=False).mean(numeric_only=True)

    all_idx = pd.MultiIndex.from_product(
        [df.feature_long.unique(), month_order], names=["feature_long", "month"]
    )

    df = (
        df.set_index(["feature_long", "month"])
        .reindex(all_idx)
        .fillna(0.0)
        .reset_index()
    )

    piv = df.pivot(index="feature_long", columns="month", values=value_col)[month_order]

    feat_order = piv.mean(axis=1).sort_values(ascending=True).index
    piv = piv.loc[feat_order]

    piv_smooth = pd.DataFrame(
        np.vstack([gaussian_filter1d(piv.loc[f], sigma=1) for f in feat_order]),
        index=feat_order,
        columns=piv.columns,
    )

    if metric == "winter":
        winter_months = [
            "aug_",
            "sep_",
            "oct",
            "nov",
            "dec",
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
        ]
        invalid = [m for m in piv_smooth.columns if m not in winter_months]
        piv_smooth[invalid] = 0.0

    fig, ax = plt.subplots(figsize=(10, 10))
    palette = sns.color_palette("magma", n_colors=len(feat_order))
    month_idx = np.arange(len(piv_smooth.columns))

    offset_step = np.nanmax(piv_smooth.values) * 0.7
    current_offset = 0.0
    max_importance = piv.max(axis=1)

    for feat, color in zip(feat_order, palette):
        y = piv_smooth.loc[feat].values

        ax.plot(month_idx, y + current_offset, color=color, lw=2)
        ax.fill_between(
            month_idx, current_offset, y + current_offset, color=color, alpha=0.4
        )

        ax.text(-0.6, current_offset, feat, va="center", ha="right", fontsize=13)

        max_idx = np.argmax(y)
        ax.text(
            month_idx[max_idx],
            y[max_idx] + current_offset + 0.05 * offset_step,
            annot_fmt.format(max_importance[feat]),
            ha="center",
            va="bottom",
            fontsize=11,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.2),
        )

        current_offset += offset_step

    ax.set_yticks([])
    ax.set_xlim(0, len(month_idx) - 1)
    ax.set_xticks(month_idx)
    ax.set_xticklabels(
        [m.strip("_").capitalize() for m in piv_smooth.columns], rotation=45, ha="right"
    )
    ax.set_xlabel("Month")
    ax.set_title(title or f"Monthly Permutation Feature Importance – {label}")

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()
