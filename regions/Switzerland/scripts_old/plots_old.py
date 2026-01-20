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


def plotHeatmap(
    test_glaciers,
    data_glamos,
    glacierCap,
    period="annual",
    var_to_plot="POINT_BALANCE",
    cbar_label="[m w.e. $a^{-1}$]",
):
    # Heatmap of mean mass balance per glacier:
    # Get the mean mass balance per glacier
    data_with_pot = data_glamos[data_glamos.PERIOD == period]
    data_with_pot["GLACIER"] = data_glamos["GLACIER"].apply(lambda x: glacierCap[x])

    mean_mb_per_glacier = (
        data_with_pot.groupby(["GLACIER", "YEAR", "PERIOD"])[var_to_plot]
        .mean()
        .reset_index()
    )
    mean_mb_per_glacier = mean_mb_per_glacier[mean_mb_per_glacier["PERIOD"] == period]
    matrix = mean_mb_per_glacier.pivot(
        index="GLACIER", columns="YEAR", values=var_to_plot
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
        cbar_kws={"label": cbar_label},
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
    return fig


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


# ---------- Stratified sampling helper ----------
def stratified_sample_by_group(
    df: pd.DataFrame,
    group_col: str = "GLACIER",
    n_per_group: int = 100,
    *,
    random_state: int = 42,
    exact: bool = False,
) -> pd.DataFrame:
    """
    Return a stratified sample with ~n_per_group rows per group (df[group_col]).
    - exact=False (default): downsample without replacement; groups smaller than n_per_group are kept entirely.
    - exact=True: always return exactly n_per_group per group; small groups are sampled with replacement.
    """
    rng = np.random.RandomState(random_state)
    parts = []
    for g, gdf in df.groupby(group_col, sort=False):
        if exact:
            replace = len(gdf) < n_per_group
            take = n_per_group
        else:
            replace = False
            take = min(len(gdf), n_per_group)
        parts.append(gdf.sample(n=take, replace=replace, random_state=rng))
    return pd.concat(parts, axis=0).reset_index(drop=True)


# ---------- Existing helpers from before ----------
def _safe_perplexity(n_samples: int, target: int = 30) -> int:
    if n_samples <= 10:
        return max(2, n_samples // 3)
    return int(np.clip(target, 5, max(5, (n_samples - 1) // 3)))


def _prepare_matrix(train_df, test_df, cols):
    X_train = train_df[cols].to_numpy()
    X_test = test_df[cols].to_numpy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    X_train_std = scaler.fit_transform(X_train_imp)
    X_test_std = scaler.transform(X_test_imp)

    X = np.vstack([X_train_std, X_test_std])
    mask_train = np.zeros(X.shape[0], dtype=bool)
    mask_train[: X_train_std.shape[0]] = True
    return X, mask_train


def _tsne_embed(
    X,
    n_components=2,
    perplexity: Optional[int] = None,
    random_state: int = 42,
    n_iter: int = 500,
    n_iter_without_progress: int = 100,
):
    n = X.shape[0]
    px = _safe_perplexity(n) if perplexity is None else min(perplexity, max(2, n - 2))
    kwargs = dict(
        n_components=n_components,
        perplexity=px,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
        metric="euclidean",
        n_iter_without_progress=n_iter_without_progress,
        verbose=0,
    )
    try:
        tsne = TSNE(max_iter=n_iter, **kwargs)  # sklearn >= 1.5
    except TypeError:
        tsne = TSNE(n_iter=n_iter, **kwargs)  # older sklearn
    return tsne.fit_transform(X), px


def plot_tsne_overlap(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    STATIC_COLS: Sequence[str],
    MONTHLY_COLS: Sequence[str],
    *,
    stratify_by: str = "GLACIER",
    n_per_group: int = 100,
    exact: bool = False,
    perplexity: Optional[int] = None,
    random_state: int = 20,
    n_iter: int = 500,
    n_iter_without_progress: int = 100,
    figsize=(18, 5),
    alpha_train=0.5,
    alpha_test=0.5,
    s_train=45,
    s_test=45,
    custom_palette: Optional[dict] = None,
    # (optional) subpanel label controls if you kept them earlier
    sublabels: Sequence[str] = ("a", "b", "c"),
    label_fmt: str = "({})",
    label_xy: Tuple[float, float] = (0.02, 0.98),
    label_fontsize: int = 14,
    label_bbox: Optional[dict] = None,
):
    # Default palette (your colors)
    if custom_palette is None:
        colors = get_cmap_hex(cm.batlow, 10)
        color_dark_blue = colors[0]
        custom_palette = {"Train": color_dark_blue, "Test": "#b2182b"}

    # 1) Stratified downsample per split
    train_s = stratified_sample_by_group(
        data_train,
        group_col=stratify_by,
        n_per_group=n_per_group,
        random_state=random_state,
        exact=exact,
    )
    test_s = stratified_sample_by_group(
        data_test,
        group_col=stratify_by,
        n_per_group=n_per_group,
        random_state=random_state + 1,
        exact=exact,
    )

    feature_sets = [
        ("All features", list(STATIC_COLS) + list(MONTHLY_COLS)),
        ("Dynamic", list(MONTHLY_COLS)),
        ("Static", list(STATIC_COLS)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    if label_bbox is None:
        label_bbox = dict(facecolor="white", alpha=0.7, pad=2, edgecolor="none")

    for i, (ax, (title, cols)) in enumerate(zip(axes, feature_sets)):
        if len(cols) == 0:
            ax.text(0.5, 0.5, f"No columns in: {title}", ha="center", va="center")
            ax.set_axis_off()
            continue

        # 2) Build matrix with train-fitted preprocessing
        X, mask_train = _prepare_matrix(train_s, test_s, cols)

        # 3) t-SNE on combined samples
        emb, px_used = _tsne_embed(
            X,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=n_iter,
            n_iter_without_progress=n_iter_without_progress,
        )

        # 4) Split back to train/test embeddings
        emb_train = emb[mask_train]
        emb_test = emb[~mask_train]

        # 5) Plot with your colors
        ax.scatter(
            emb_train[:, 0],
            emb_train[:, 1],
            marker="o",
            s=s_train,
            alpha=alpha_train,
            color=custom_palette["Train"],
            label="Train",
            linewidths=0,
        )
        ax.scatter(
            emb_test[:, 0],
            emb_test[:, 1],
            marker="^",
            s=s_test,
            alpha=alpha_test,
            color=custom_palette["Test"],
            label="Test",
            linewidths=0,
        )

        ax.set_title(title)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(frameon=True)

        # optional subpanel labels
        if sublabels and i < len(sublabels):
            ax.text(
                label_xy[0],
                label_xy[1],
                label_fmt.format(sublabels[i]),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=label_fontsize,
                bbox=label_bbox,
            )

    plt.tight_layout()
    plt.show()
    return fig


def plot_predictions_three_models_side_by_side(
    grouped_ids_lstm,
    grouped_ids_nn,
    grouped_ids_xgb,
    scores_annual_lstm,
    scores_winter_lstm,
    scores_annual_nn,
    scores_winter_nn,
    scores_annual_xgb,
    scores_winter_xgb,
    ax_xlim=(-8, 6),
    ax_ylim=(-8, 6),
    color_annual=COLOR_ANNUAL,
    color_winter=COLOR_WINTER,
):
    """
    Produces a 3-panel summary figure:
    LSTM ─ NN ─ XGBoost (each: predictions vs truth),
    with a single shared legend underneath.
    """

    model_labels = ["LSTM", "MLP", "XGBoost"]
    subplot_labels = ["(a)", "(b)", "(c)"]
    grouped_inputs = [grouped_ids_lstm, grouped_ids_nn, grouped_ids_xgb]
    scores_annuals = [scores_annual_lstm, scores_annual_nn, scores_annual_xgb]
    scores_winters = [scores_winter_lstm, scores_winter_nn, scores_winter_xgb]

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.20)

    for ax, grouped, label, sl, scores_annual, scores_winter in zip(
        axes,
        grouped_inputs,
        model_labels,
        subplot_labels,
        scores_annuals,
        scores_winters,
    ):
        # Predictions vs Truth panel
        predVSTruth(
            ax,
            grouped,
            scores_annual,
            hue="PERIOD",
            add_legend=False,
            palette=[color_annual, color_winter],
            ax_xlim=ax_xlim,
            ax_ylim=ax_ylim,
        )

        # Title
        ax.set_title(label, fontsize=20)

        # Subplot label (a, b, c)
        ax.text(
            0.03,
            0.97,
            sl,
            transform=ax.transAxes,
            fontsize=20,
            verticalalignment="top",
            horizontalalignment="left",
        )

        # Score box
        legend_str = "\n".join(
            [
                r"$\mathrm{RMSE_a}=%.2f$, $\mathrm{RMSE_w}=%.2f$"
                % (scores_annual["rmse"], scores_winter["rmse"]),
                r"$\mathrm{R^2_a}=%.2f$, $\mathrm{R^2_w}=%.2f$"
                % (scores_annual["R2"], scores_winter["R2"]),
                r"$\mathrm{B_a}=%.2f$, $\mathrm{B_w}=%.2f$"
                % (scores_annual["Bias"], scores_winter["Bias"]),
            ]
        )
        ax.text(
            0.98,  # near the right edge
            0.03,  # near the bottom
            legend_str,
            transform=ax.transAxes,
            fontsize=16,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.60),
        )

        # remove legend
        ax.legend().remove()

    # Common axis label
    axes[0].set_ylabel("Predicted PMB (m w.e.)", fontsize=18)

    # ---------- Shared legend BELOW subplots ----------
    from matplotlib.patches import Patch

    handles = [
        Patch(color=COLOR_ANNUAL, label="Annual"),
        Patch(color=COLOR_WINTER, label="Winter"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=16,
        frameon=True,
        bbox_to_anchor=(0.5, -0.04),  # move down if needed
    )
    # --------------------------------------------------

    fig.tight_layout(rect=(0, 0.05, 1, 1))  # leave room at bottom for legend
    return fig
