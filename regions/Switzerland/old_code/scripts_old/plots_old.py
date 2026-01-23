# --- Standard library ---
import math
from typing import Sequence, Optional, Tuple

# --- Third-party libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from cmcrameri import cm

# --- Project-specific modules ---
from regions.Switzerland.scripts.helpers import *
from regions.Switzerland.scripts.config_CH import *

# CONSTANT COLORS FOR PLOTS
colors = get_cmap_hex(cm.batlow, 10)
color_annual = colors[0]
color_winter = "#c51b7d"


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


def PlotIndividualGlacierPredVsTruth(
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
        (r"$\mathrm{RMSE}=%.2f$" % rmse, r"$\mathrm{\rho}=%.2f$" % corr)
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
