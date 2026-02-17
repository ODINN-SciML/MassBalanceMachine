import os
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joypy

from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
import seaborn as sns

from regions.Switzerland.scripts.utils import *
from regions.Switzerland.scripts.config_CH import *


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


def month_keep_idx(month_keys, drop_padded_months: bool):
    """
    Return indices of months to keep, optionally dropping padded months.

    Parameters
    ----------
    month_keys : sequence of str
        Month labels (e.g. ['oct', 'nov', ..., 'sep'] or with padding like 'sep_').
    drop_padded_months : bool
        If True, exclude month labels ending with '_' (used for padding).

    Returns
    -------
    numpy.ndarray
        Integer indices selecting the months to keep.
    """
    if not drop_padded_months:
        return np.arange(len(month_keys))
    return np.array([i for i, k in enumerate(month_keys) if not k.endswith("_")])


def plot_sensitivity_elev_band(
    sens_bands: list,
    plot_var: str,
    text_var: str,
    id_elev_bands,
    month_labels,
    monthly_cols,
    bands,
    ax=None,
    ylim=None,
    drop_padded_months: bool = False,
):
    """
    Plot mean ± std sensitivity across months for selected elevation bands.

    Parameters
    ----------
    sens_bands : list
        List-like container of sensitivity arrays/tensors for each elevation band.
        Each element is expected to support indexing like band[:, :, f_idx] and
        reductions with .mean(dim=0) / .std(dim=0) (e.g. torch tensor / xarray DataArray).
    plot_var : str
        Variable name to plot (must be present in `monthly_cols`).
    text_var : str
        Label used for the subplot title (typically a month key like 'feb', 'mar', ...).
    id_elev_bands : sequence of int
        Indices of the elevation bands to plot (commonly two: lowest and highest).
    month_labels : sequence of str
        Month labels corresponding to the month dimension of the sensitivity data.
    monthly_cols : list of str
        Ordered list of monthly feature names (used to locate `plot_var` index).
    bands : sequence of float or int
        Elevation band edges (length nbands+1). Used for labeling, e.g.
        label shows bands[id]-bands[id+1] in meters.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If None, a new figure and axis are created.
    ylim : tuple or None, optional
        y-axis limits (ymin, ymax). If None, matplotlib autoscaling is used.
    drop_padded_months : bool, optional
        If True, padded months ending with '_' are removed from the x-axis.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the plot.
    """
    f_idx = monthly_cols.index(plot_var)

    keep_idx = month_keep_idx(month_labels, drop_padded_months)
    xval = np.arange(len(keep_idx))
    xlabels = [month_labels[i] for i in keep_idx]

    if ax is None:
        _, ax = plt.subplots()

    colors = ["tab:red", "tab:blue"]

    for e, id_band in enumerate(id_elev_bands):
        band = sens_bands[id_band]
        mean = band[:, :, f_idx].mean(dim=0)[keep_idx]
        std = band[:, :, f_idx].std(dim=0)[keep_idx]

        label_ = "lowest band" if id_band == 0 else "highest band"
        label = f"{label_} ({int(bands[id_band])}-{int(bands[id_band + 1])} m)"

        ax.plot(xval, mean, label=label, color=colors[e])
        ax.fill_between(
            xval,
            mean - std,
            mean + std,
            color=colors[e],
            alpha=0.25,
        )

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xticks(xval)
    ax.set_xticklabels(xlabels, rotation=90)
    ax.set_ylabel("Sensitivity")

    whole_months = {
        "jan": "January",
        "feb": "February",
        "mar": "March",
        "apr": "April",
        "may": "May",
        "jun": "June",
        "jul": "July",
    }
    ax.set_title(whole_months.get(text_var, f"{text_var}"))
    ax.legend(fontsize=12)

    return ax


def plot_monthly_sensitivity_elevbands(
    sens_bands,
    plot_var,
    glacier_name,
    months_keys,
    vois_climate_long_name,
    monthly_cols,
    bands,
    selected_months=("feb", "mar", "apr", "may", "jun", "jul"),
    id_elev_bands=(0, 6),
    outdir="figures/paper",
    add_panel_labels=True,
    drop_padded_months: bool = False,
):
    """
    Plot a multi-panel figure of monthly sensitivities by elevation band and save as PDF.

    For each month in `selected_months`, a panel is created showing mean ± std
    sensitivity across months for selected elevation bands (e.g., lowest vs highest).
    A global y-limit is computed across all selected months for comparability.

    Parameters
    ----------
    sens_bands : dict-like
        Mapping month_key -> list of sensitivity arrays/tensors per elevation band.
        Example: sens_bands['feb'] is a list where each element corresponds to an
        elevation band and has shape compatible with band[:, :, f_idx].
    plot_var : str
        Variable name to plot (must be present in `monthly_cols`).
    glacier_name : str
        Glacier name/identifier used in the figure title and output filename.
    months_keys : sequence of str
        Month labels for the x-axis (may include padded months ending with '_').
    vois_climate_long_name : dict
        Mapping from variable short names to descriptive long names.
    monthly_cols : list of str
        Ordered list of monthly feature names.
    bands : sequence of float or int
        Elevation band edges (length nbands+1), used for labels.
    selected_months : tuple of str, optional
        Month keys to plot as separate panels (must exist in `sens_bands`).
    id_elev_bands : tuple of int, optional
        Indices of the elevation bands to compare (default: (0, 6)).
    outdir : str, optional
        Output directory where the figure PDF will be saved.
    add_panel_labels : bool, optional
        If True, add panel labels (a), (b), ... to subplots.
    drop_padded_months : bool, optional
        If True, exclude padded months (ending with '_') from the x-axis.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure (also saved to disk).
    """
    f_idx = monthly_cols.index(plot_var)

    keep_idx = month_keep_idx(months_keys, drop_padded_months)

    # Compute global y-limits across the same months that will be plotted
    vals = []
    for m in selected_months:
        for band in sens_bands[m]:
            mean = band[:, :, f_idx].mean(dim=0)[keep_idx]
            std = band[:, :, f_idx].std(dim=0)[keep_idx]
            vals.append((mean - std).min().item())
            vals.append((mean + std).max().item())

    ylim = (1.1 * min(vals), 1.1 * max(vals))
    print("Global ylim:", ylim)

    fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    panel_labels = iter(string.ascii_lowercase)

    for idx, (ax, m) in enumerate(zip(axs.ravel(), selected_months)):
        plot_sensitivity_elev_band(
            sens_bands=sens_bands[m],
            plot_var=plot_var,
            text_var=m,
            id_elev_bands=id_elev_bands,
            month_labels=months_keys,
            monthly_cols=monthly_cols,
            bands=bands,
            ax=ax,
            ylim=ylim,
            drop_padded_months=drop_padded_months,
        )

        if idx % 2 == 1:
            ax.set_ylabel("")
        if idx != 0 and ax.get_legend() is not None:
            ax.get_legend().remove()

        if add_panel_labels:
            ax.text(
                0.02,
                0.97,
                f"({next(panel_labels)})",
                transform=ax.transAxes,
                fontsize=14,
                ha="left",
                va="top",
            )

    plt.suptitle(
        f"Sensitivity of monthly mass balance to "
        f"{vois_climate_long_name[plot_var].lower()} on {glacier_name.capitalize()}gletscher"
    )
    plt.tight_layout()
    plt.show()

    os.makedirs(outdir, exist_ok=True)
    output_figure_path = os.path.join(
        outdir,
        f"appendix/{glacier_name}_sensitivity_{plot_var}_elev_bands.pdf",
    )
    fig.savefig(output_figure_path, dpi=300, bbox_inches="tight")

    return fig
