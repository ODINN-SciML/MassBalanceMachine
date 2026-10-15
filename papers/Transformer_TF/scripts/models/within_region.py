"""Evaluation and figures of the within-region experiment (notebook `2.2 Within_region`).

- `count_trainable_params`: number of trainable parameters of a model.
- `plot_within_region_grid`: LSTM vs Transformer, predicted vs observed PMB per region (Fig. A2).
- `plot_within_region_grid_transformer_only`: same for the Transformer only (Fig. 3).
- `plot_region_pred_vs_truth_by_glacier`: predicted vs observed PMB of one region,
  coloured by glacier (Fig. A3a).
- `build_split_comparison` / `plot_split_comparison`: Sinkhorn-informed vs random
  glacier-level splits (Fig. A4).
"""

import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

import massbalancemachine as mbm
from papers.Transformer_TF.scripts.config_TF_Europe import REGION_LABELS
from papers.Transformer_TF.scripts.plotting.style import (
    NATURE_COLORS,
    NATURE_SPECS,
    apply_nature_style,
    nature_figsize,
)
from papers.Transformer_TF.scripts.models.predictions import (
    evaluate_one_model,
    make_test_loader_for_key,
)


def count_trainable_params(model, trainable_only=True):
    """Number of (trainable) parameters of a torch model."""
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )


def _axis_limits_from_test_targets(lstm_assets, region_to_key, pad_frac=0.3):
    """Per-region axis limits: range of the observed test PMB, padded by pad_frac."""
    lims = {}
    for region, key in region_to_key.items():
        vals = lstm_assets[key]["ds_test"].y.cpu().numpy().flatten()
        vals = vals[~np.isnan(vals)]
        pad = (vals.max() - vals.min()) * pad_frac
        lims[region] = (float(vals.min() - pad), float(vals.max() + pad))
    return lims


def _season_legend_handles():
    return [
        mpatches.Patch(color=mbm.plots.COLOR_ANNUAL, label="Annual"),
        mpatches.Patch(color=mbm.plots.COLOR_WINTER, label="Winter"),
    ]


def plot_within_region_grid(
    cfg, device, region_to_key, models_lstm, models_tf, lstm_assets, out_path
):
    """
    Predicted vs observed PMB on the test set of each region, for the LSTM (top row
    of each block) and the Transformer (bottom row), at most 4 regions per row (Fig. A2).

    region_to_key maps a region label (e.g. 'CH', 'CA_3') to its key in the model and
    dataset dicts (e.g. 'CH', 'CA_3_GL').
    """
    regions = list(region_to_key.keys())
    max_cols = 4
    n_regions = len(regions)
    n_cols = min(max_cols, n_regions)
    n_region_rows = math.ceil(n_regions / n_cols)

    region_lims = _axis_limits_from_test_targets(lstm_assets, region_to_key)

    # One block (LSTM row + Transformer row) per row of regions
    fig = plt.figure(figsize=(4 * n_cols, 3.8 * 2 * n_region_rows))
    outer_gs = GridSpec(
        n_region_rows,
        1,
        figure=fig,
        hspace=0.2,
        top=0.96,
        bottom=0.08,
        left=0.08,
        right=0.98,
    )
    inner_gs_list = []
    for block_i in range(n_region_rows):
        inner_gs = GridSpecFromSubplotSpec(
            2,
            n_cols,
            subplot_spec=outer_gs[block_i],
            hspace=0.1,
            wspace=0.32,
        )
        inner_gs_list.append(inner_gs)

    axes_grid = []
    for block_i in range(n_region_rows):
        block_axes = []
        for model_i in range(2):
            row_axes = [
                fig.add_subplot(inner_gs_list[block_i][model_i, col])
                for col in range(n_cols)
            ]
            block_axes.append(row_axes)
        axes_grid.append(block_axes)

    for reg_idx, region in enumerate(regions):
        key = region_to_key[region]
        block_i = reg_idx // n_cols
        col = reg_idx % n_cols
        lim = region_lims[region]

        for model_i, (model_label, models_by_key) in enumerate(
            [
                ("LSTM", models_lstm),
                ("Transformer", models_tf),
            ]
        ):
            ax = axes_grid[block_i][model_i][col]
            model = models_by_key[key]

            _, _, _, ax = evaluate_one_model(
                cfg=cfg,
                model=model,
                device=device,
                lstm_assets_for_key=lstm_assets[key],
                ax=ax,
                ax_xlim=lim,
                ax_ylim=lim,
                legend_fontsize=NATURE_SPECS["font_min_pt"],
            )
            apply_nature_style(ax, fontsize=NATURE_SPECS["font_max_pt"], box=True)
            leg = ax.get_legend()
            if leg:
                leg.remove()

            # Panel labels run row by row over all LSTM and Transformer rows
            global_row = block_i * 2 + model_i
            panel_index = global_row * n_cols + col
            panel_label = chr(ord("a") + panel_index)
            ax.text(
                0.02,
                0.98,
                f"({panel_label})",
                transform=ax.transAxes,
                fontsize=NATURE_SPECS["font_max_pt"],
                va="top",
                ha="left",
            )

            if model_i == 0:
                ax.set_title(
                    "{}".format(REGION_LABELS.get(region, region)),
                    fontsize=NATURE_SPECS["font_max_pt"] + 1,
                )
            if col == 0:
                ax.set_ylabel(
                    f"{model_label}\nModeled PMB (m w.e.)",
                    fontsize=NATURE_SPECS["font_max_pt"],
                )
            else:
                ax.set_ylabel("")
            if model_i == 0:
                ax.set_xlabel("")

    # Hide unused panels in the last row
    for empty_idx in range(n_regions, n_region_rows * n_cols):
        block_i = empty_idx // n_cols
        col = empty_idx % n_cols
        for model_i in range(2):
            axes_grid[block_i][model_i][col].set_visible(False)

    fig.legend(
        handles=_season_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncols=2,
        frameon=False,
        fontsize=NATURE_SPECS["font_max_pt"] + 2,
    )
    fig.suptitle(
        "Within-region point mass balance modelling — Transformer vs LSTM model",
        fontsize=NATURE_SPECS["font_max_pt"] + 2,
        y=1.0,
    )
    fig.savefig(out_path, dpi=NATURE_SPECS["dpi"], bbox_inches="tight")
    plt.show()


def plot_within_region_grid_transformer_only(
    cfg, device, region_to_key, models_tf, lstm_assets, out_path
):
    """
    Predicted vs observed PMB on the test set of each region for the Transformer,
    at most 4 regions per row (Fig. 3). See `plot_within_region_grid` for region_to_key.
    """
    regions = list(region_to_key.keys())
    max_cols = 4
    n_regions = len(regions)
    n_cols = min(max_cols, n_regions)
    n_region_rows = math.ceil(n_regions / n_cols)

    region_lims = _axis_limits_from_test_targets(lstm_assets, region_to_key)

    fig = plt.figure(figsize=(4 * n_cols, 3.8 * n_region_rows))
    outer_gs = GridSpec(
        n_region_rows,
        n_cols,
        figure=fig,
        hspace=0.35,
        wspace=0.32,
        top=0.94,
        bottom=0.15,
        left=0.08,
        right=0.98,
    )
    axes_grid = [
        [fig.add_subplot(outer_gs[row, col]) for col in range(n_cols)]
        for row in range(n_region_rows)
    ]

    for reg_idx, region in enumerate(regions):
        key = region_to_key[region]
        row = reg_idx // n_cols
        col = reg_idx % n_cols
        lim = region_lims[region]
        ax = axes_grid[row][col]
        model = models_tf[key]

        _, _, _, ax = evaluate_one_model(
            cfg=cfg,
            model=model,
            device=device,
            lstm_assets_for_key=lstm_assets[key],
            ax=ax,
            ax_xlim=lim,
            ax_ylim=lim,
            legend_fontsize=NATURE_SPECS["font_min_pt"],
        )
        apply_nature_style(ax, fontsize=NATURE_SPECS["font_max_pt"], box=True)
        leg = ax.get_legend()
        if leg:
            leg.remove()

        panel_label = chr(ord("a") + reg_idx)
        ax.text(
            0.02,
            0.98,
            f"({panel_label})",
            transform=ax.transAxes,
            fontsize=NATURE_SPECS["font_max_pt"],
            va="top",
            ha="left",
        )

        ax.set_title(
            "{}".format(REGION_LABELS.get(region, region)),
            fontsize=NATURE_SPECS["font_max_pt"] + 1,
        )
        if col == 0:
            ax.set_ylabel("Modeled PMB (m w.e.)", fontsize=NATURE_SPECS["font_max_pt"])
        else:
            ax.set_ylabel("")

    # Hide unused panels in the last row
    for empty_idx in range(n_regions, n_region_rows * n_cols):
        row = empty_idx // n_cols
        col = empty_idx % n_cols
        axes_grid[row][col].set_visible(False)

    fig.legend(
        handles=_season_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncols=2,
        frameon=False,
        fontsize=NATURE_SPECS["font_max_pt"],
    )
    fig.suptitle(
        "Within-region point mass balance modelling by the Transformer",
        fontsize=NATURE_SPECS["font_max_pt"] + 2,
        y=1.01,
    )
    fig.savefig(out_path, dpi=NATURE_SPECS["dpi"], bbox_inches="tight")
    plt.show()


def plot_region_pred_vs_truth_by_glacier(
    cfg,
    model,
    device,
    lstm_assets_for_key,
    ax=None,
    ax_xlim=(-16, 9),
    ax_ylim=(-16, 9),
    legend_fontsize=NATURE_SPECS["font_min_pt"],
    s=60,
    color_by_glacier=None,
    legend_separate=False,
    legend_figsize=(3, 4),
    legend_title_fontsize=12,
):
    """
    Predicted vs observed PMB on the test set of one region, coloured by glacier (Fig. A3a).

    color_by_glacier : {glacier: colour}, optional
        Shared colours (e.g. to match a map); defaults to NATURE_COLORS in
        alphabetical glacier order. Glaciers missing from it are drawn in grey.
    legend_separate : bool
        If True, the glacier legend is drawn in its own figure (returned as legend_fig).

    Returns
    -------
    (test_df_preds, created_fig, ax, color_by_glacier, legend_fig)
    """
    _, ds_test_copy, test_dl = make_test_loader_for_key(cfg, lstm_assets_for_key)

    _, test_df_preds = model.evaluate_with_preds(device, test_dl, ds_test_copy)
    test_df_preds = test_df_preds.dropna(subset=["target", "pred"]).copy()

    if "GLACIER" not in test_df_preds.columns and "POINT_ID" in test_df_preds.columns:
        test_df_preds["GLACIER"] = test_df_preds["POINT_ID"].apply(
            lambda x: x.split("_")[0]
        )

    glaciers = sorted(test_df_preds["GLACIER"].unique())

    if color_by_glacier is None:
        color_by_glacier = {g: NATURE_COLORS[i] for i, g in enumerate(glaciers)}
    else:
        missing = [g for g in glaciers if g not in color_by_glacier]
        if missing:
            print(
                f"  ⚠ {len(missing)} glacier(s) in test set not in provided "
                f"color_by_glacier, falling back to grey: {missing}"
            )
            for g in missing:
                color_by_glacier[g] = "#999999"

    created_fig = None
    if ax is None:
        created_fig = plt.figure(figsize=nature_figsize(cols=2, height_mm=120))
        ax = plt.subplot(1, 1, 1)

    fig = ax.figure

    for glacier in glaciers:
        sub = test_df_preds[test_df_preds["GLACIER"] == glacier]
        if sub.empty:
            continue
        ax.scatter(
            sub["target"],
            sub["pred"],
            color=color_by_glacier[glacier],
            marker="o",
            s=s,
            alpha=0.8,
            edgecolors="none",
            zorder=2,
        )

    # 1:1 line and zero lines
    ax.axline((0, 0), slope=1, color="grey", linestyle="-", linewidth=0.4)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.grid(alpha=0.3)
    ax.set_xlim(ax_xlim)
    ax.set_ylim(ax_ylim)
    ax.set_xlabel("Observed PMB [m w.e.]", fontsize=legend_fontsize)
    ax.set_ylabel("Modeled PMB [m w.e.]", fontsize=legend_fontsize)

    glacier_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=color_by_glacier[g],
            label=g,
            markersize=7,
        )
        for g in glaciers
    ]

    legend_fig = None
    if legend_separate:
        legend_fig = plt.figure(figsize=legend_figsize)
        legend_fig.legend(
            handles=glacier_handles,
            title="Glacier",
            fontsize=legend_fontsize - 2,
            title_fontsize=legend_title_fontsize,
            loc="center",
            frameon=True,
            borderpad=0.8,
            labelspacing=0.8,
        )
        legend_fig.gca().set_axis_off() if legend_fig.axes else None
    else:
        if created_fig is not None:
            fig.subplots_adjust(right=0.72)
        fig.legend(
            handles=glacier_handles,
            title="Glacier",
            fontsize=legend_fontsize - 2,
            title_fontsize=legend_title_fontsize,
            loc="center left",
            bbox_to_anchor=(0.74, 0.5),
        )

    apply_nature_style(ax, fontsize=NATURE_SPECS["font_min_pt"], box=True)

    return test_df_preds, created_fig, ax, color_by_glacier, legend_fig


# ── Sinkhorn-informed vs random glacier-level splits (Fig. A4) ──────────────

# Regions with only a measurement-level (ID) split: no Sinkhorn/random GL comparison
SPLIT_CMP_EXCLUDE_CODES = {"CA_4", "SJM"}

_RAND_RE = re.compile(r"^(?P<base>.+)_RAND(?P<idx>\d+)$", re.IGNORECASE)
_GL_RE = re.compile(r"^(?P<base>.+)_GL$", re.IGNORECASE)
_ID_RE = re.compile(r"^(?P<base>.+)_ID$", re.IGNORECASE)

# Columns of df_metrics that are sample counts, not metrics
_NON_METRIC_COLS = {"n_preds", "n_annual", "n_winter"}


def classify_keys(index):
    """
    Group df_metrics keys by base region code.
    Returns {base_code: {"smart": key_or_None, "random": [keys...]}}, where "smart"
    is the Sinkhorn-informed GL split ('<code>_GL' or the plain code) and "random"
    the random splits ('<code>_rand<i>'). ID-split keys are ignored.
    """
    groups = {}
    for key in index:
        m = _RAND_RE.match(key)
        if m:
            base = m.group("base")
            groups.setdefault(base, {"smart": None, "random": []})
            groups[base]["random"].append(key)
            continue
        m = _GL_RE.match(key)
        if m:
            base = m.group("base")
            groups.setdefault(base, {"smart": None, "random": []})
            groups[base]["smart"] = key
            continue
        if _ID_RE.match(key):
            continue  # ID-level result, not the "smart" split — ignore here
        # bare key, no suffix -> region where only a GL split was ever run
        groups.setdefault(key, {"smart": None, "random": []})
        if groups[key]["smart"] is None:
            groups[key]["smart"] = key
    return groups


def build_split_comparison(
    df_metrics, model_label, exclude_codes=SPLIT_CMP_EXCLUDE_CODES
):
    """
    Tidy table comparing, per region and metric, the Sinkhorn-informed split with the
    mean (and std) over the random splits. 'delta' is positive when the Sinkhorn
    split is better (lower error, or higher R²/correlation).
    """
    metric_cols = [
        c
        for c in df_metrics.columns
        if pd.api.types.is_numeric_dtype(df_metrics[c]) and c not in _NON_METRIC_COLS
    ]
    groups = classify_keys(df_metrics.index)

    rows = []
    for base, info in sorted(groups.items()):
        if base.upper() in exclude_codes:
            continue
        smart_key, rand_keys = info["smart"], info["random"]
        if smart_key is None or not rand_keys:
            print(
                f"  ⚠ Skipping {base} ({model_label}): "
                f"smart_key={smart_key}, n_random_found={len(rand_keys)}"
            )
            continue

        for metric in metric_cols:
            smart_val = float(df_metrics.loc[smart_key, metric])
            rand_vals = df_metrics.loc[rand_keys, metric].astype(float)

            higher_better = any(k in metric for k in ("R2", "r2", "pearson", "corr"))
            delta = (
                (smart_val - rand_vals.mean())
                if higher_better
                else (rand_vals.mean() - smart_val)
            )

            rows.append(
                {
                    "model": model_label,
                    "region": base,
                    "metric": metric,
                    "smart_key": smart_key,
                    "smart_val": smart_val,
                    "random_mean": rand_vals.mean(),
                    "random_std": rand_vals.std(ddof=1) if len(rand_vals) > 1 else 0.0,
                    "n_random": len(rand_vals),
                    "higher_is_better": higher_better,
                    "delta": delta,
                    "smart_better": delta > 0,
                }
            )

    return pd.DataFrame(rows)


SPLIT_CMP_METRIC_LABELS = {
    "RMSE_annual": "RMSE annual\n(m w.e.)",
    "RMSE_winter": "RMSE winter\n(m w.e.)",
    "RMSE_summer": "RMSE summer\n(m w.e.)",
    "MAE_annual": "MAE annual\n(m w.e.)",
    "R2_annual": "R² annual",
    "R2_winter": "R² winter",
    "pearson_annual": "Pearson r annual",
}


def plot_split_comparison(
    df_split_cmp,
    out_path,
    model="Transformer",
    metrics=("RMSE_annual", "RMSE_winter"),
    region_order=("CH", "FR", "IT", "AT", "NOR", "ISL", "ALA", "CA_3"),
    smart_color="#3182bd",
    random_color="#aaaaaa",
):
    """
    Bar chart of Δ = random − Sinkhorn split (one panel per metric) for one model;
    positive bars mean the Sinkhorn split is better (Fig. A4). Error bars show the
    std over the random splits.
    """
    sub_model = df_split_cmp[df_split_cmp["model"] == model]
    metrics = list(metrics)

    missing = [r for r in region_order if r not in sub_model["region"].unique()]
    if missing:
        print(f"  ⚠ Not found in df_split_cmp, dropping from plot: {missing}")

    regions = [r for r in region_order if r in sub_model["region"].unique()]
    tick_labels = [REGION_LABELS.get(r, r) for r in regions]
    x = np.arange(len(regions))

    n_metrics = len(metrics)
    mm2in = 1 / 25.4
    fig = plt.figure(figsize=(110 * mm2in, n_metrics * 1.55 + 0.9))

    outer_gs = GridSpec(
        n_metrics,
        1,
        figure=fig,
        hspace=0.08,
        left=0.20,
        right=0.97,
        top=0.90,
        bottom=0.25,
    )

    for row_i, metric in enumerate(metrics):
        is_bottom = row_i == n_metrics - 1
        ax = fig.add_subplot(outer_gs[row_i, 0])

        ax.text(
            0.02,
            0.95,
            f"({chr(ord('a') + row_i)})",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            zorder=5,
        )

        # Alternating background bands per region
        ax.set_facecolor("#f5f5f5")
        for xi in range(0, len(regions), 2):
            ax.axvspan(xi - 0.5, xi + 0.5, color="white", alpha=1.0, zorder=0)

        sub = sub_model[sub_model["metric"] == metric].set_index("region")
        bar_vals = [sub.loc[r, "delta"] if r in sub.index else np.nan for r in regions]
        err_vals = [
            sub.loc[r, "random_std"] if r in sub.index else 0.0 for r in regions
        ]

        # Symmetric y-limits around 0 that fit the bars and error bars
        extents = [abs(v) + e for v, e in zip(bar_vals, err_vals) if not np.isnan(v)]
        abs_max = max(max(extents), 0.05)
        ypad = abs_max * 0.1

        colors = [smart_color if v > 0 else random_color for v in bar_vals]

        ax.bar(x, bar_vals, width=0.6, color=colors, zorder=3, linewidth=0)
        ax.errorbar(
            x,
            bar_vals,
            yerr=err_vals,
            fmt="none",
            ecolor="black",
            elinewidth=0.6,
            capsize=2,
            zorder=4,
        )
        ax.axhline(0, color="black", linewidth=0.6, zorder=4)

        ax.set_xlim(-0.5, len(regions) - 0.5)
        ax.set_ylim(-(abs_max + ypad), abs_max + ypad)
        ax.set_xticks(x)

        if is_bottom:
            ax.set_xticklabels(tick_labels, fontsize=6, rotation=90, ha="center")
        else:
            ax.set_xticklabels([])

        ax.tick_params(axis="y", labelsize=6, width=0.5, length=2)
        ax.tick_params(axis="x", length=0)

        ax.set_ylabel(
            f"Δ {SPLIT_CMP_METRIC_LABELS.get(metric, metric)}", fontsize=6.5, labelpad=4
        )

    fig.suptitle(f"{model} — random vs. sinkhorn glacier split", fontsize=8, y=0.94)

    fig.legend(
        handles=[
            mpatches.Patch(
                color=smart_color, alpha=0.88, label="Sinkhorn split better"
            ),
            mpatches.Patch(color=random_color, alpha=0.88, label="Random split better"),
        ],
        loc="lower center",
        ncol=2,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.58, 0.0),
        handlelength=1.2,
        handletextpad=0.4,
    )

    fig.savefig(out_path, dpi=NATURE_SPECS["dpi"], bbox_inches="tight")
    plt.show()
