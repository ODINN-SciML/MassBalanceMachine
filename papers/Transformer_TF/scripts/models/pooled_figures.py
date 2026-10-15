"""Figures of the multi-region model (notebook `3.2 Pooled_model`).

- `plot_zs_pooled_vs_single_barplot`: zero-shot metric of the multi-region and
  single-region models per target region (Fig. 5 annual, Fig. A6 winter).
- `plot_zs_pooled_vs_single_scatter`: modeled vs observed PMB per target and model (Fig. A7).
- `plot_ft_gain_barplot`: zero-shot vs head-only and full fine-tuning (Fig. 6).
"""

import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import massbalancemachine as mbm

from papers.Transformer_TF.scripts.config_TF_Europe import COLORS, REGION_LABELS
from papers.Transformer_TF.scripts.plotting.style import (
    NATURE_SPECS,
    apply_nature_style,
)
from papers.Transformer_TF.scripts.plotting.pmb_performance import pred_vs_truth_density
from papers.Transformer_TF.scripts.models.predictions import compute_seasonal_scores
from papers.Transformer_TF.scripts.models.ft_sensitivity import FT_COLORS, FT_LABELS


def _target_title(tgt_label, glacier_counts):
    title_str = "Target: {}".format(REGION_LABELS.get(tgt_label, tgt_label))
    n_glaciers = glacier_counts.get(tgt_label)
    if n_glaciers is not None:
        title_str += f" ({n_glaciers} glaciers)"
    return title_str


def plot_zs_pooled_vs_single_barplot(
    df_metrics_comparison,
    metric_key,
    metric_label,
    source_regions,
    glacier_counts,
    out_path,
    suptitle,
    exclude_regions=("CENTRALASIA",),
    max_cols=3,
    global_ylim=False,
):
    """
    One bar panel per target region with the zero-shot metric of the multi-region
    model (black outline) and of each single-region model.

    global_ylim=False (Fig. 5): panels share the y axis (sharey=True), the limit
        follows the last panel drawn, and every panel shows y tick labels.
    global_ylim=True (Fig. A6): one y range over all panels; y tick labels only in
        the first column; NaN values are not annotated.
    """
    model_order = ["pooled_zs"] + list(source_regions)
    labels = {src: f"{src} only" for src in source_regions}
    labels["pooled_zs"] = "Multi-region"

    tgt_labels = [t for t in df_metrics_comparison.keys() if t not in exclude_regions]
    n_tgts = len(tgt_labels)
    n_cols = min(max_cols, n_tgts)
    n_rows = math.ceil(n_tgts / n_cols)

    shared_ylim = None
    if global_ylim:
        # one y range across all panels
        all_vals = []
        for tgt_label in tgt_labels:
            results = df_metrics_comparison[tgt_label]
            models = [m for m in model_order if m in results]
            all_vals += [
                results[m][metric_key]
                for m in models
                if not np.isnan(results[m][metric_key])
            ]
        if all_vals:
            shared_ylim = (0, max(all_vals) * 1.25)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 3.8 * n_rows),
        sharey=not global_ylim,
    )
    axes = np.array(axes).reshape(n_rows, n_cols)

    label_fontsize = NATURE_SPECS["font_max_pt"] - 2
    for tgt_idx, tgt_label in enumerate(tgt_labels):
        row, col = tgt_idx // n_cols, tgt_idx % n_cols
        ax = axes[row, col]

        ax.text(
            0.03,
            0.97,
            f"({chr(ord('a') + tgt_idx)})",
            transform=ax.transAxes,
            fontsize=NATURE_SPECS["font_max_pt"],
            va="top",
            ha="left",
            zorder=5,
        )

        results = df_metrics_comparison[tgt_label]
        models = [m for m in model_order if m in results]
        vals = [results[m][metric_key] for m in models]
        colors = [COLORS[m] for m in models]

        bars = ax.bar(
            range(len(models)),
            vals,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.5,
            zorder=3,
        )

        apply_nature_style(ax, fontsize=NATURE_SPECS["font_max_pt"], box=True)

        # highlight the multi-region model
        for i, mod in enumerate(models):
            if mod.startswith("pooled_zs"):
                bars[i].set_edgecolor("black")
                bars[i].set_linewidth(1.5)
                bars[i].set_alpha(1.0)

        # value labels on top of the bars
        for bar, val in zip(bars, vals):
            if global_ylim and np.isnan(val):
                continue
            y_pos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.08
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(
            [labels[m] for m in models],
            rotation=90,
            ha="right",
            fontsize=label_fontsize,
        )

        if col == 0:
            ax.set_ylabel(metric_label, fontsize=label_fontsize)
        ax.tick_params(axis="y", labelsize=NATURE_SPECS["font_min_pt"] - 2)

        if global_ylim:
            if shared_ylim is not None:
                ax.set_ylim(shared_ylim)
            # hide redundant y tick labels on inner columns
            if col != 0:
                ax.tick_params(labelleft=False)
        else:
            valid_vals = [v for v in vals if not np.isnan(v)]
            ax.set_ylim(0, max(valid_vals) * 1.25)

        ax.set_title(
            _target_title(tgt_label, glacier_counts),
            fontsize=NATURE_SPECS["font_max_pt"] - 2,
            pad=6,
        )

    # Hide any unused axes in the last row
    for empty_idx in range(n_tgts, n_rows * n_cols):
        axes[empty_idx // n_cols, empty_idx % n_cols].set_visible(False)

    fig.suptitle(suptitle, fontsize=NATURE_SPECS["font_max_pt"] + 2)
    fig.tight_layout(pad=1.8)
    fig.subplots_adjust(top=0.9)
    fig.savefig(out_path, dpi=NATURE_SPECS["dpi"], bbox_inches="tight")
    plt.show()
    return fig


def _eval_and_plot(
    cfg,
    device,
    model,
    ds_test,
    scaler,
    ax,
    lim,
    legend_fontsize=NATURE_SPECS["font_min_pt"],
):
    """Density scatter of modeled vs observed PMB with RMSE, R² and bias annotations."""
    ds_copy = mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
        ds_test
    )
    test_dl = mbm.data_processing.MBSequenceDataset.make_test_loader(
        ds_copy, scaler, batch_size=128, seed=cfg.seed
    )
    metrics, test_df_preds = model.evaluate_with_preds(device, test_dl, ds_copy)
    test_df_preds = test_df_preds.dropna(subset=["target", "pred"])
    scores_annual, scores_winter = compute_seasonal_scores(
        test_df_preds, target_col="target", pred_col="pred"
    )

    pred_vs_truth_density(
        ax,
        test_df_preds,
        scores_annual,
        add_legend=False,
        palette=[mbm.plots.COLOR_ANNUAL, mbm.plots.COLOR_WINTER],
        ax_xlim=lim,
        ax_ylim=lim,
        s=50,
    )

    def _fmt(x):
        return (
            "NA"
            if (x is None or (isinstance(x, float) and np.isnan(x)))
            else f"{x:.2f}"
        )

    legend_NN = "\n".join(
        [
            rf"$\mathrm{{RMSE_a}}={_fmt(scores_annual['rmse'])},\ \mathrm{{RMSE_w}}={_fmt(scores_winter['rmse'])}$",
            rf"$\mathrm{{R^2_a}}={_fmt(scores_annual['R2'])},\ \mathrm{{R^2_w}}={_fmt(scores_winter['R2'])}$",
            rf"$\mathrm{{Bias_a}}={_fmt(scores_annual['Bias'])},\ \mathrm{{Bias_w}}={_fmt(scores_winter['Bias'])}$",
        ]
    )
    ax.text(
        0.98,
        0.02,
        legend_NN,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=legend_fontsize,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )
    return metrics


def plot_zs_pooled_vs_single_scatter(
    cfg,
    device,
    df_metrics_comparison,
    eval_configs,
    source_regions,
    model_pooled,
    ds_pooled_scaler,
    single_source_models,
    single_source_scalers,
    out_path,
    exclude_regions=("CENTRALASIA",),
):
    """
    Fig. A7: modeled vs observed PMB, one row per target region and one column per
    model (multi-region, then each single-region model).
    """
    regions_to_plot = [
        t for t in df_metrics_comparison.keys() if t not in exclude_regions
    ]

    plot_model_order = ["pooled_zs"] + list(source_regions)
    col_titles = {
        "pooled_zs": "Multi-region",
        **{
            src: f"Single-region: {REGION_LABELS.get(src, src)}"
            for src in source_regions
        },
    }
    model_and_scaler = {
        "pooled_zs": (model_pooled, ds_pooled_scaler),
        **{
            src: (single_source_models[src], single_source_scalers[src])
            for src in source_regions
        },
    }

    ncols = len(plot_model_order)
    nrows = len(regions_to_plot)
    eval_configs_dict = dict(eval_configs)

    # axis limits per target: range of the observed PMB, padded by 50%
    region_lims_zs = {}
    for region in regions_to_plot:
        vals = eval_configs_dict[region]["ds_test"].y.cpu().numpy().flatten()
        vals = vals[~np.isnan(vals)]
        pad = (vals.max() - vals.min()) * 0.5
        region_lims_zs[region] = (float(vals.min() - pad), float(vals.max() + pad))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        sharex="row",
        sharey="row",
    )
    axes = np.array(axes).reshape(nrows, ncols)

    for row_i, region in enumerate(regions_to_plot):
        lim = region_lims_zs[region]
        ds_test = eval_configs_dict[region]["ds_test"]

        for col_i, model_key in enumerate(plot_model_order):
            ax = axes[row_i, col_i]
            model, scaler = model_and_scaler[model_key]

            _eval_and_plot(
                cfg,
                device,
                model,
                ds_test,
                scaler,
                ax,
                lim,
                legend_fontsize=NATURE_SPECS["font_min_pt"],
            )
            apply_nature_style(ax, fontsize=NATURE_SPECS["font_max_pt"] + 2, box=True)
            leg = ax.get_legend()
            if leg:
                leg.remove()

            panel_label = f"{chr(ord('a') + row_i)}{col_i + 1}"
            ax.text(
                0.02,
                0.98,
                f"({panel_label})",
                transform=ax.transAxes,
                fontsize=NATURE_SPECS["font_max_pt"],
                va="top",
                ha="left",
            )

            if row_i == 0:
                ax.set_title(
                    col_titles.get(model_key, model_key),
                    fontsize=NATURE_SPECS["font_max_pt"] + 5,
                )
            if col_i == 0:
                ax.set_ylabel(
                    f"Target: {REGION_LABELS.get(region, region)}\nModeled PMB (m w.e.)",
                    fontsize=NATURE_SPECS["font_max_pt"] + 2,
                )
            else:
                ax.set_ylabel("")
            if row_i < nrows - 1:
                ax.set_xlabel("")

    fig.legend(
        handles=[
            mpatches.Patch(color=mbm.plots.COLOR_ANNUAL, label="Annual"),
            mpatches.Patch(color=mbm.plots.COLOR_WINTER, label="Winter"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncols=2,
        frameon=False,
        fontsize=NATURE_SPECS["font_max_pt"] + 5,
    )
    fig.suptitle(
        "Multi-region vs single-region zero-shot transfer",
        fontsize=NATURE_SPECS["font_max_pt"] + 6,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.subplots_adjust(top=0.94)
    fig.savefig(out_path, dpi=NATURE_SPECS["dpi"], bbox_inches="tight")
    plt.show()
    return fig


def plot_ft_gain_barplot(
    df_ft_metrics_id,
    glacier_counts,
    out_path,
    metric_key="RMSE_annual",
    metric_label="RMSE annual (m w.e.)",
    exclude_regions=("CENTRALASIA",),
    max_cols=3,
    bar_alpha=0.7,
):
    """
    Fig. 6: metric of the zero-shot model (black outline) and of head-only and full
    fine-tuning per target region, mean ± std over the repeats; one y range for all
    panels.
    """
    model_order = ["no_ft", "head_only", "full"]

    tgt_labels = [t for t in df_ft_metrics_id.keys() if t not in exclude_regions]
    n_tgts = len(tgt_labels)
    n_cols = min(max_cols, n_tgts)
    n_col_groups = math.ceil(n_tgts / n_cols)

    # one plot row per group of targets, with a spacer row in between
    plot_h, spacer_h = 3, 1.3
    height_ratios = []
    for g in range(n_col_groups):
        height_ratios.append(plot_h)
        if g < n_col_groups - 1:
            height_ratios.append(spacer_h)

    fig = plt.figure(figsize=(5 * n_cols, sum(height_ratios) + 0.8))
    gs = GridSpec(
        len(height_ratios),
        n_cols,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.24,
        wspace=0.15,  # tighter, since inner y tick labels are hidden
    )

    # one y range across all panels (including the error bars)
    std_key = f"{metric_key}_std"
    shared_ylim = None
    for tgt_label in tgt_labels:
        results = df_ft_metrics_id[tgt_label]
        models = [m for m in model_order if m in results]
        vals = [results[m][metric_key] for m in models]
        stds = [results[m].get(std_key, 0.0) for m in models]
        valid_pairs = [(v, s) for v, s in zip(vals, stds) if not np.isnan(v)]
        if not valid_pairs:
            continue
        ymax = max(v + s for v, s in valid_pairs) * 1.25
        shared_ylim = (
            (0, ymax)
            if shared_ylim is None
            else (min(shared_ylim[0], 0), max(shared_ylim[1], ymax))
        )

    for tgt_idx, tgt_label in enumerate(tgt_labels):
        col = tgt_idx % n_cols
        group_row = tgt_idx // n_cols

        results = df_ft_metrics_id[tgt_label]
        models = [m for m in model_order if m in results]
        colors = [FT_COLORS[m] for m in models]

        ax = fig.add_subplot(gs[group_row * 2, col])

        panel_idx = group_row * n_cols + col
        ax.text(
            0.03,
            0.97,
            f"({chr(ord('a') + panel_idx)})",
            transform=ax.transAxes,
            fontsize=NATURE_SPECS["font_max_pt"],
            va="top",
            ha="left",
            zorder=5,
        )

        vals = [results[m][metric_key] for m in models]
        stds = [results[m].get(std_key, 0.0) for m in models]

        bars = ax.bar(
            range(len(models)),
            vals,
            yerr=stds,
            capsize=3,
            error_kw={"linewidth": 1.0, "ecolor": "black", "zorder": 4},
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            alpha=bar_alpha,
            zorder=3,
        )

        # highlight the zero-shot model
        for i, mod in enumerate(models):
            if mod == "no_ft":
                bars[i].set_edgecolor("black")
                bars[i].set_linewidth(1.5)
                bars[i].set_alpha(1.0)

        for bar, val, std in zip(bars, vals, stds):
            if np.isnan(val):
                continue
            y_pos = (
                bar.get_height() + std + 0.02
                if val >= 0
                else bar.get_height() - std - 0.08
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(
            [FT_LABELS[m] for m in models],
            rotation=90,
            ha="right",
            fontsize=NATURE_SPECS["font_max_pt"],
        )

        if col == 0:
            ax.set_ylabel(metric_label, fontsize=NATURE_SPECS["font_max_pt"])

        ax.set_title(
            _target_title(tgt_label, glacier_counts),
            fontsize=NATURE_SPECS["font_max_pt"] - 4,
            pad=6,
        )

        if shared_ylim is not None:
            ax.set_ylim(shared_ylim)

        apply_nature_style(ax, fontsize=NATURE_SPECS["font_max_pt"], box=False)

        # hide redundant y tick labels (after styling so they stay hidden)
        if col != 0:
            ax.tick_params(labelleft=False)

    fig.suptitle(
        "Effect of fine-tuning compared to zero-shot transfer performance (10% local data)",
        fontsize=NATURE_SPECS["font_max_pt"] + 2,
        y=1.0,
    )
    fig.savefig(out_path, dpi=NATURE_SPECS["dpi"], bbox_inches="tight")
    plt.show()
    return fig
