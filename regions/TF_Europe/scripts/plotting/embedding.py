import seaborn as sns
from cmcrameri import cm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
import numpy as np
import os
from matplotlib.patches import Rectangle
from typing import Sequence, Optional, Tuple
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr
import colormaps as cmaps

from regions.TF_Europe.scripts.config_TF_Europe import *
from regions.TF_Europe.scripts.plotting.palettes import get_cmap_hex


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
    """
    Visualize overlap between training and test datasets using t-SNE embeddings.

    The function performs a stratified subsampling of train and test data,
    computes t-SNE embeddings on the combined feature space, and displays
    train/test overlap for three feature configurations:
      - all features (static + monthly),
      - dynamic (monthly) features only,
      - static features only.

    Each configuration is shown in a separate subplot.

    Parameters
    ----------
    data_train : pandas.DataFrame
        Training dataset containing static and/or monthly features.
    data_test : pandas.DataFrame
        Test dataset with the same feature structure as `data_train`.
    STATIC_COLS : sequence of str
        Names of static feature columns.
    MONTHLY_COLS : sequence of str
        Names of monthly (dynamic) feature columns.
    stratify_by : str, optional
        Column name used for stratified subsampling (e.g. glacier ID).
    n_per_group : int, optional
        Number of samples per group drawn from train and test sets.
    exact : bool, optional
        If True, enforce exact group sizes during stratified sampling.
    perplexity : int or None, optional
        Perplexity parameter for t-SNE. If None, it is chosen automatically
        based on sample size.
    random_state : int, optional
        Random seed for reproducibility.
    n_iter : int, optional
        Maximum number of t-SNE iterations.
    n_iter_without_progress : int, optional
        Number of iterations without improvement before early stopping.
    figsize : tuple, optional
        Figure size passed to matplotlib.
    alpha_train, alpha_test : float, optional
        Marker transparency for train and test points.
    s_train, s_test : int, optional
        Marker sizes for train and test points.
    custom_palette : dict or None, optional
        Mapping defining colors for 'Train' and 'Test'. If None, defaults
        are used.
    sublabels : sequence of str, optional
        Labels used to annotate subplots (e.g. ('a', 'b', 'c')).
    label_fmt : str, optional
        Format string applied to subplot labels.
    label_xy : tuple, optional
        Relative (x, y) position of subplot labels in axes coordinates.
    label_fontsize : int, optional
        Font size of subplot labels.
    label_bbox : dict or None, optional
        Matplotlib bbox properties for subplot label backgrounds.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure containing the t-SNE overlap plots.
    """
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


def _tsne_embed(
    X,
    n_components=2,
    perplexity: Optional[int] = None,
    random_state: int = 42,
    n_iter: int = 500,
    n_iter_without_progress: int = 100,
):
    """
    Compute a t-SNE embedding for a feature matrix with a safe perplexity choice.

    Parameters
    ----------
    X : numpy.ndarray
        Input feature matrix of shape (n_samples, n_features).
    n_components : int, optional
        Dimension of the embedding (default: 2).
    perplexity : int or None, optional
        t-SNE perplexity. If None, it is chosen automatically based on
        `X.shape[0]`. If provided, it is clipped to a valid range.
    random_state : int, optional
        Random seed for reproducibility.
    n_iter : int, optional
        Maximum number of t-SNE iterations (compatible with different sklearn
        versions via `max_iter` or `n_iter`).
    n_iter_without_progress : int, optional
        Early stopping patience for t-SNE.

    Returns
    -------
    emb : numpy.ndarray
        Embedded coordinates of shape (n_samples, n_components).
    px : int
        Perplexity value actually used.
    """
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


def _prepare_matrix(train_df, test_df, cols):
    """
    Build a combined standardized feature matrix from train/test DataFrames.

    Missing values are imputed using the median (fit on train only), and
    features are standardized (fit on train only). The returned matrix stacks
    train above test and includes a boolean mask indicating train rows.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training DataFrame.
    test_df : pandas.DataFrame
        Test DataFrame.
    cols : sequence of str
        Feature column names to extract.

    Returns
    -------
    X : numpy.ndarray
        Combined feature matrix of shape (n_train + n_test, n_features).
    mask_train : numpy.ndarray of bool
        Boolean mask of shape (n_train + n_test,) where True indicates
        rows belonging to the training set.
    """
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


def _safe_perplexity(n_samples: int, target: int = 30) -> int:
    """
    Choose a t-SNE perplexity that is safe for the given sample size.

    Parameters
    ----------
    n_samples : int
        Number of samples to embed.
    target : int, optional
        Desired perplexity for large sample sizes (default: 30).

    Returns
    -------
    int
        Perplexity clipped to a valid range for the given `n_samples`.
    """
    if n_samples <= 10:
        return max(2, n_samples // 3)
    return int(np.clip(target, 5, max(5, (n_samples - 1) // 3)))


def stratified_sample_by_group(
    df: pd.DataFrame,
    group_col: str = "GLACIER",
    n_per_group: int = 100,
    *,
    random_state: int = 42,
    exact: bool = False,
) -> pd.DataFrame:
    """
    Draw a stratified sample from a DataFrame by grouping column.

    The function samples rows independently within each group defined by
    `group_col`, aiming to return approximately `n_per_group` samples per
    group.

    Sampling behavior:
      - exact=False (default): sample without replacement; groups smaller than
        `n_per_group` are kept entirely.
      - exact=True: always return exactly `n_per_group` samples per group;
        groups smaller than `n_per_group` are sampled with replacement.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to sample from.
    group_col : str, optional
        Column name defining the groups (default: 'GLACIER').
    n_per_group : int, optional
        Target number of samples per group.
    random_state : int, optional
        Random seed for reproducibility.
    exact : bool, optional
        If True, enforce exactly `n_per_group` samples per group using
        replacement when necessary.

    Returns
    -------
    pandas.DataFrame
        Stratified sample containing the selected rows from all groups,
        with the index reset.
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


def plot_feature_kde_overlap(
    df_train,
    df_test,
    features,
    palette=None,
    outfile=None,
    sublabels: Sequence[str] = (
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
    ),
    label_fmt: str = "({})",
    label_xy: Tuple[float, float] = (0.02, 0.98),
    label_fontsize: int = 14,
    label_bbox: Optional[dict] = None,
):
    """
    Plot kernel density estimates (KDEs) of feature distributions for train
    and test datasets.

    For each feature, the function overlays KDE curves for the training and
    test sets in a grid of subplots, enabling visual comparison of
    distributional differences. A single global legend is added below
    the figure.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training dataset containing the specified features.
    df_test : pandas.DataFrame
        Test dataset containing the same features as `df_train`.
    features : sequence of str
        List of feature names to plot.
    palette : dict or None, optional
        Mapping specifying colors for 'Train' and 'Test'. If None, a default
        palette is used.
    outfile : str or None, optional
        Path to save the figure. If None, the figure is not saved.
    sublabels : sequence of str, optional
        Labels used to annotate individual subplots (e.g. ('a', 'b', 'c', ...)).
    label_fmt : str, optional
        Format string applied to subplot labels.
    label_xy : tuple of float, optional
        Relative (x, y) position of subplot labels in axes coordinates.
    label_fontsize : int, optional
        Font size of subplot labels.
    label_bbox : dict or None, optional
        Matplotlib bbox properties for subplot label backgrounds.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure containing KDE overlap plots.
    """
    if palette is None:
        palette = {"Train": "steelblue", "Test": "darkred"}

    n = len(features)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]

        sns.kdeplot(
            df_train[feat].dropna(),
            ax=ax,
            color=palette["Train"],
            fill=True,
            alpha=0.4,
            linewidth=2,
            label="Train",
        )
        sns.kdeplot(
            df_test[feat].dropna(),
            ax=ax,
            color=palette["Test"],
            fill=True,
            alpha=0.4,
            linewidth=2,
            label="Test",
        )

        ax.set_title(vois_climate_long_name[feat])
        ax.set_xlabel(f"[{vois_units[feat]}]")

        # remove axis-level legend
        ax.legend_.remove() if ax.legend_ else None

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

    # remove unused axes
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j])

    # --- global legend below the plots ---
    handles = [
        plt.Line2D([0], [0], color=palette["Train"], lw=10, alpha=0.6, label="Train"),
        plt.Line2D([0], [0], color=palette["Test"], lw=10, alpha=0.6, label="Test"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
        fontsize=16,
    )

    fig.tight_layout()

    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")

    return fig


def plot_heatmap(
    test_glaciers,
    data_glamos,
    glacierCap,
    period="annual",
    var_to_plot="POINT_BALANCE",
    cbar_label="[m w.e. $a^{-1}$]",
):
    """
    Plot a heatmap of mean glacier mass balance by year and glacier.

    The function aggregates point mass-balance data by glacier and year,
    computes the mean value per glacier–year, and displays the result as a
    heatmap. Glaciers are ordered by decreasing mean elevation. Selected
    test glaciers are highlighted with rectangular outlines.

    Parameters
    ----------
    test_glaciers : list of str
        List of glacier identifiers to highlight in the heatmap.
    data_glamos : pandas.DataFrame
        GLAMOS point mass-balance dataset containing at least the columns
        ['GLACIER', 'YEAR', 'PERIOD', var_to_plot, 'POINT_ELEVATION'].
    glacierCap : dict
        Mapping from glacier identifiers to formatted glacier names used
        for plotting (e.g. capitalization or display names).
    period : {"annual", "winter"}, optional
        Mass-balance period to plot.
    var_to_plot : str, optional
        Column name in `data_glamos` containing the mass-balance variable
        to visualize.
    cbar_label : str, optional
        Label for the colorbar.

    Returns
    -------
    matplotlib.figure.Figure
        The generated heatmap figure.
    """
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


def plot_overlap_for_all_results(
    results_dict,
    cfg,
    STATIC_COLS,
    MONTHLY_COLS,
    n_iter=1000,
):
    """
    Generate t-SNE train/test feature-overlap plots for multiple regions.

    This function iterates over a dictionary of monthly-preparation results
    (e.g., output from `prepare_monthlies_for_all_regions`) and creates a
    t-SNE-based feature overlap visualization for each region/subregion.

    For each key:
      - Extracts `df_train` and `df_test`
      - Skips if either is missing or empty
      - Calls `plot_tsne_overlap`
      - Stores the resulting matplotlib Figure

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping region/subregion keys (e.g., "07_SJM", "11_CH")
        to result dictionaries containing at least:
            - "df_train"
            - "df_test"

    cfg : object
        Configuration object. Must contain a `seed` attribute used for
        reproducibility of the t-SNE random state.

    STATIC_COLS : list of str
        Names of static (time-invariant) feature columns used for plotting.

    MONTHLY_COLS : list of str
        Names of monthly (time-varying) feature columns used for plotting.

    n_iter : int, optional
        Number of t-SNE optimization iterations (default is 1000).

    Returns
    -------
    dict
        Dictionary mapping region/subregion keys to matplotlib Figure objects.

    Notes
    -----
    - Uses a consistent color palette across all regions
      (dark blue for Train, red for Test).
    - Skips regions with missing or empty datasets.
    - Figures are not automatically saved; the caller can save them
      individually if desired.
    """
    colors = get_cmap_hex(cm.batlow, 10)
    color_dark_blue = colors[0]
    custom_palette = {"Train": color_dark_blue, "Test": "#b2182b"}

    figs = {}

    for key, res in results_dict.items():
        if res is None:
            continue

        df_train = res.get("df_train")
        df_test = res.get("df_test")

        if (
            df_train is None
            or df_test is None
            or len(df_train) == 0
            or len(df_test) == 0
        ):
            print(f"[{key}] Missing/empty df_train or df_test, skipping.")
            continue

        print(
            f"Plotting t-SNE overlap for {key}: train={len(df_train)}, test={len(df_test)}"
        )

        fig = plot_tsne_overlap(
            df_train,
            df_test,
            STATIC_COLS,
            MONTHLY_COLS,
            sublabels=("a", "b", "c"),
            label_fmt="({})",
            label_xy=(0.02, 0.98),
            label_fontsize=14,
            n_iter=n_iter,
            random_state=cfg.seed,
            custom_palette=custom_palette,
        )

        figs[key] = fig

    return figs


def plot_feature_overlap_all_regions(
    results_dict,
    STATIC_COLS,
    MONTHLY_COLS,
    output_dir="figures",
    include_target=True,
):
    """
    Plot KDE-based feature distribution overlap (Train vs Test) for all regions.

    This function iterates over a dictionary of region/subregion results
    (e.g., from `prepare_monthlies_for_all_regions`) and generates kernel
    density estimate (KDE) overlap plots comparing train and test feature
    distributions.

    For each key:
      - Extracts `df_train` and `df_test`
      - Skips missing or empty datasets
      - Plots feature KDE overlap using `plot_feature_kde_overlap`
      - Stores the resulting matplotlib Figure

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping region/subregion keys (e.g., "07_SJM", "11_CH")
        to result dictionaries containing at least:
            - "df_train"
            - "df_test"

    STATIC_COLS : list of str
        Names of static (time-invariant) feature columns.

    MONTHLY_COLS : list of str
        Names of monthly (time-varying) feature columns.

    output_dir : str, optional
        Directory where figures could be saved (default: "figures").
        The directory is created if it does not exist.
        Note: Figures are not automatically saved unless handled inside
        `plot_feature_kde_overlap`.

    include_target : bool, optional
        If True, includes the target variable ("POINT_BALANCE") in the
        KDE overlap plots (default: True).

    Returns
    -------
    dict
        Dictionary mapping region/subregion keys to matplotlib Figure objects.

    Notes
    -----
    - Uses a consistent color palette across all regions
      (dark blue for Train, red for Test).
    - Skips regions with missing or empty train/test datasets.
    - Figures are returned for optional further processing (saving, styling).
    """
    os.makedirs(output_dir, exist_ok=True)

    colors = get_cmap_hex(cm.batlow, 10)
    color_dark_blue = colors[0]
    palette = {"Train": color_dark_blue, "Test": "#b2182b"}

    features = STATIC_COLS + MONTHLY_COLS
    if include_target:
        features = features + ["POINT_BALANCE"]

    figs = {}

    for key, res in results_dict.items():
        if res is None:
            continue

        df_train = res.get("df_train")
        df_test = res.get("df_test")

        if df_train is None or df_test is None:
            print(f"[{key}] Missing df_train/df_test, skipping.")
            continue

        if len(df_train) == 0 or len(df_test) == 0:
            print(f"[{key}] Empty train/test, skipping.")
            continue

        print(f"Plotting KDE overlap for {key}")

        fig = plot_feature_kde_overlap(
            df_train, df_test, features, palette, outfile=None
        )

        figs[key] = fig

    return figs
