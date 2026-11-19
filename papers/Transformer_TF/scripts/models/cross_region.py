"""Single-region zero-shot transfer (notebook `2.3 Cross_region`).

- `report_nan_in_xreg_inputs` / `report_nan_in_xreg_datasets`: sanity checks of the
  monthly data and sequence datasets.
- `train_or_load_source_models`: one model per source region.
- `evaluate_xreg_metrics`: metrics of each source model on every target region.
- `build_xreg_metric_tables`: long table of both models and Δ (LSTM − Transformer).
- `plot_zs_absolute_rmse`: absolute RMSE per source and target (Fig. A5).
- `plot_zeroshot_combined`: Δ RMSE bars and example scatters (Fig. 4), plus its legend.
"""

import re

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import massbalancemachine as mbm
from papers.Transformer_TF.scripts.config_TF_Europe import NATURE_PALETTE, REGION_LABELS
from papers.Transformer_TF.scripts.plotting.style import (
    NATURE_SPECS,
    apply_nature_style,
)
from papers.Transformer_TF.scripts.models.lstm_helpers import (
    train_or_load_one_source_model,
)
from papers.Transformer_TF.scripts.models.predictions import (
    evaluate_one_model,
    evaluate_all_models_metrics_only,
)

# Colours of the two models in Fig. 4 and A5
TF_COLOR = NATURE_PALETTE["sky_blue"]
LSTM_COLOR = "#aaaaaa"

XREG_METRIC_LABELS = {
    "RMSE_annual": "RMSE annual\n(m w.e.)",
    "RMSE_winter": "RMSE winter\n(m w.e.)",
    "RMSE_summer": "RMSE summer\n(m w.e.)",
    "MAE_annual": "MAE annual\n(m w.e.)",
    "R2_annual": "R² annual",
    "R2_winter": "R² winter",
    "pearson_annual": "Pearson r annual",
}


# ── Sanity checks ────────────────────────────────────────────────────────────


def report_nan_in_xreg_inputs(res_xreg_by_source, monthly_cols, static_cols):
    """Number of NaNs in the input features and target of the augmented monthly data."""
    print("--- NaN check on augmented dataframes ---")
    for src_region, res_xreg in res_xreg_by_source.items():
        for split_name, df in [
            ("df_train_aug", res_xreg["df_train_aug"]),
            ("df_test_aug", res_xreg["df_test_aug"]),
        ]:
            feat_cols = [c for c in monthly_cols + static_cols if c in df.columns]
            n_nan_feat = df[feat_cols].isna().sum().sum()
            n_nan_tgt = (
                df["POINT_BALANCE"].isna().sum()
                if "POINT_BALANCE" in df.columns
                else "N/A"
            )
            print(
                f"  {src_region} {split_name}: feature NaNs={n_nan_feat}, target NaNs={n_nan_tgt}"
            )


def report_nan_in_xreg_datasets(src_region, outputs_xreg):
    """NaN/Inf check of the input and target tensors of every source→target dataset."""
    print(f"--- NaN/Inf check for {src_region} ---")
    for exp_key, assets in outputs_xreg.items():
        for ds_name in ["ds_train", "ds_test"]:
            ds = assets.get(ds_name)
            if ds is None:
                continue
            checks = {"x_m": ds.Xm, "x_s": ds.Xs, "y": ds.y}
            any_issue = False
            for name, t in checks.items():
                n_nan = torch.isnan(t).sum().item()
                n_inf = torch.isinf(t).sum().item()
                if n_nan > 0 or n_inf > 0:
                    print(f"  {exp_key} {ds_name} {name}: NaNs={n_nan}, Infs={n_inf}")
                    any_issue = True
            if not any_issue:
                print(f"  {exp_key} {ds_name}: OK")


# ── Models and metrics ──────────────────────────────────────────────────────


def train_or_load_source_models(
    cfg,
    device,
    source_regions,
    outputs_xreg_by_source,
    params,
    models_dir,
    prefix,
    model_type,
    epochs,
    date,
    train_flag=False,
    force_retrain=False,
):
    """
    Train (train_flag=True) or load one model per source region, trained on all of
    that region's data. The training data are the same for every target key, so the
    first key is used. Models are stored in models_dir / <source region>.

    Returns (models_by_source, infos_by_source).
    """
    models_by_source, infos_by_source = {}, {}
    for src_region in source_regions:
        any_key = next(iter(outputs_xreg_by_source[src_region]))
        train_assets = outputs_xreg_by_source[src_region][any_key]

        model, model_path, info = train_or_load_one_source_model(
            cfg=cfg,
            key=src_region,
            lstm_assets=train_assets,
            best_params=params,
            device=device,
            models_dir=models_dir / src_region,
            prefix=prefix,
            train_flag=train_flag,
            force_retrain=force_retrain,
            epochs=epochs,
            date=date,
            model_type=model_type,
        )

        models_by_source[src_region] = model
        infos_by_source[src_region] = {"model_path": model_path, **(info or {})}
        print(f"{src_region} -> model trained/loaded from {model_path}")
    return models_by_source, infos_by_source


def evaluate_xreg_metrics(
    cfg,
    device,
    models_by_source,
    outputs_xreg_by_source,
    source_regions,
    exclude_targets,
    display_order,
):
    """
    Metrics of each source model on all its targets (except exclude_targets), with the
    targets in display_order first. Returns {source: DataFrame indexed by key
    'XREG_<source>_TO_<target>'}.
    """
    df_metrics_by_source = {}
    for src_region in source_regions:
        print(f"\nEvaluating source region: {src_region}")
        target_keys = [
            k
            for k in outputs_xreg_by_source[src_region].keys()
            if k.split("_TO_")[-1] not in exclude_targets
        ]
        models_by_key = {key: models_by_source[src_region] for key in target_keys}

        available_targets = [k.split("_TO_")[-1] for k in target_keys]
        custom_order = [t for t in display_order if t in available_targets]
        custom_order += [t for t in available_targets if t not in custom_order]

        df_metrics_by_source[src_region] = evaluate_all_models_metrics_only(
            cfg=cfg,
            models_by_key=models_by_key,
            lstm_assets_by_key=outputs_xreg_by_source[src_region],
            device=device,
            complement_key=f"XREG_{src_region}_TO_",
            custom_order=custom_order,
        )
    return df_metrics_by_source


def _parse_key(key):
    m = re.match(r"XREG_(.+)_TO_(.+)", key)
    return (m.group(1), m.group(2)) if m else (key, key)


def _sort_key(val, order):
    try:
        return order.index(val)
    except ValueError:
        return len(order)


def build_xreg_metric_tables(
    df_metrics_by_source_lstm,
    df_metrics_by_source_tf,
    metrics,
    source_order,
    target_order,
    exclude_targets=(),
):
    """
    Long table of the metrics of both models per (source, target), and the difference
    LSTM − Transformer (positive = Transformer better for RMSE).

    Returns
    -------
    (df_all, df_delta, sources, targets_per_source), with sources and the targets of
    each source sorted by source_order / target_order.
    """
    rows = []
    for model_label, metrics_dict in [
        ("LSTM", df_metrics_by_source_lstm),
        ("Transformer", df_metrics_by_source_tf),
    ]:
        for src, df in metrics_dict.items():
            for key, row in df.iterrows():
                _, tgt = _parse_key(key)
                if tgt in exclude_targets:
                    continue
                entry = {"source": src, "target": tgt, "model": model_label}
                entry.update({m: row[m] for m in metrics if m in row.index})
                rows.append(entry)

    df_all = pd.DataFrame(rows)

    # delta: LSTM − Transformer (positive = Transformer has lower RMSE)
    df_lstm = df_all[df_all["model"] == "LSTM"].set_index(["source", "target"])
    df_tf = df_all[df_all["model"] == "Transformer"].set_index(["source", "target"])
    df_delta = (df_lstm[metrics] - df_tf[metrics]).reset_index()
    df_delta.columns = ["source", "target"] + metrics

    sources = sorted(
        df_delta["source"].unique(), key=lambda s: _sort_key(s, source_order)
    )
    targets_per_source = {
        src: sorted(
            df_delta[df_delta["source"] == src]["target"].unique(),
            key=lambda t: _sort_key(t, target_order),
        )
        for src in sources
    }
    return df_all, df_delta, sources, targets_per_source


# ── Figures ─────────────────────────────────────────────────────────────────


def plot_zs_absolute_rmse(df_all, metrics, sources, targets_per_source, out_path):
    """
    Absolute metric per target for the LSTM (grey) and Transformer (blue), one column
    per source region and one row per metric (Fig. A5).
    """
    n_metrics, n_sources = len(metrics), len(sources)
    mm2in = 1 / 25.4

    fig = plt.figure(figsize=(200 * mm2in, n_metrics * 1.55 + 0.9))

    outer_gs = GridSpec(
        1,
        n_sources,
        figure=fig,
        hspace=0.0,
        wspace=0.38,
        left=0.11,
        right=0.98,
        top=0.88,
        bottom=0.28,
    )

    inner_gs_list = [
        GridSpecFromSubplotSpec(
            n_metrics, 1, subplot_spec=outer_gs[0, col_i], hspace=0.08
        )
        for col_i in range(n_sources)
    ]

    # y limits per metric (across all sources, both models)
    ylims_abs = {}
    for metric in metrics:
        vals = df_all[metric].dropna()
        ypad = (vals.max() - vals.min()) * 0.12
        ylims_abs[metric] = (max(0, vals.min() - ypad), vals.max() + ypad)

    for col_i, src in enumerate(sources):
        tgts = targets_per_source[src]
        x = np.arange(len(tgts))
        tick_labels = [REGION_LABELS.get(t, t) for t in tgts]

        for row_i, metric in enumerate(metrics):
            is_bottom = row_i == n_metrics - 1

            ax = fig.add_subplot(inner_gs_list[col_i][row_i, 0])

            # Panels are lettered row by row: (a)-(d) first metric, (e)-(h) second
            panel_idx = row_i * n_sources + col_i
            ax.text(
                0.03,
                0.97,
                f"({chr(ord('a') + panel_idx)})",
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                ha="left",
                zorder=5,
            )

            # alternating column backgrounds
            ax.set_facecolor("#f5f5f5")
            for xi in range(0, len(tgts), 2):
                ax.axvspan(xi - 0.5, xi + 0.5, color="white", alpha=1.0, zorder=0)

            for model_label, color, offset in [
                ("LSTM", LSTM_COLOR, -0.15),
                ("Transformer", TF_COLOR, 0.15),
            ]:
                sub = df_all[
                    (df_all["source"] == src) & (df_all["model"] == model_label)
                ]
                dot_vals = [
                    (
                        float(sub.loc[sub["target"] == t, metric].iloc[0])
                        if not sub[sub["target"] == t].empty
                        else np.nan
                    )
                    for t in tgts
                ]
                ax.scatter(
                    x + offset,
                    dot_vals,
                    color=color,
                    s=30,
                    zorder=4,
                    alpha=0.88,
                    linewidths=0,
                )

            ax.set_xlim(-0.5, len(tgts) - 0.5)
            ax.set_ylim(ylims_abs[metric])
            ax.set_xticks(x)

            if is_bottom:
                ax.set_xticklabels(tick_labels, fontsize=6, rotation=90, ha="center")
            else:
                ax.set_xticklabels([])

            ax.tick_params(axis="y", labelsize=6, width=0.5, length=2)
            ax.tick_params(axis="x", length=0)

            if row_i == 0:
                ax.set_title(
                    f"Source: {REGION_LABELS.get(src, src)}", fontsize=9, pad=4
                )
            if col_i == 0:
                ax.set_ylabel(
                    XREG_METRIC_LABELS.get(metric, metric), fontsize=6.5, labelpad=4
                )

    fig.legend(
        handles=[
            mpatches.Patch(color=LSTM_COLOR, alpha=0.88, label="LSTM"),
            mpatches.Patch(color=TF_COLOR, alpha=0.88, label="Transformer"),
        ],
        loc="lower center",
        ncol=2,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.54, 0.02),
        handlelength=1.2,
        handletextpad=0.4,
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


def plot_zeroshot_combined(
    cfg,
    device,
    df_delta,
    metrics,
    sources,
    targets_per_source,
    pairs,
    models_by_source_lstm,
    models_by_source_tf,
    outputs_xreg_by_source,
    out_path,
    legend_out_path,
):
    """
    Fig. 4: top, Δ metric (LSTM − Transformer) per target, one column per source and
    one row per metric; bottom, modeled vs observed PMB for the source→target pairs,
    for the LSTM (first row) and the Transformer (second row). Panels are lettered
    bars first, then scatters. The legend is saved as a separate figure.
    """
    mm2in = 1 / 25.4

    n_pairs = len(pairs)
    n_sources = len(sources)
    n_metrics = len(metrics)
    n_cols = max(n_pairs, n_sources)

    # height ratios (inches): bar rows, spacer, two scatter rows
    scatter_row_h = 2.8
    spacer_h = 1.5
    bar_row_h = 2.5

    height_ratios = [bar_row_h] * n_metrics + [spacer_h] + [scatter_row_h] * 2
    fig_h = sum(height_ratios) + 1.0
    fig_w = max(200 * mm2in, 4.5 * n_cols)

    subplot_title_fontsize = NATURE_SPECS["font_max_pt"] + 4

    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(
        2 + 1 + n_metrics,
        n_cols,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.2,
        wspace=0.35,
        left=0.08,
        right=0.98,
        top=0.97,
        bottom=0.06,
    )

    # ── barplot section ──────────────────────────────────────────────────────
    # symmetric y limits per metric (across all sources)
    ylims = {}
    for metric in metrics:
        vals = df_delta[metric].dropna()
        abs_max = max(abs(vals.min()), abs(vals.max()), 0.05)
        ypad = abs_max * 0.18
        ylims[metric] = (-(abs_max + ypad), abs_max + ypad)

    for col_i, src in enumerate(sources):
        tgts = targets_per_source[src]
        x = np.arange(len(tgts))
        tick_labels = [REGION_LABELS.get(t, t) for t in tgts]

        for row_i, metric in enumerate(metrics):
            is_r2 = any(k in metric for k in ("R2", "r2", "pearson"))
            is_bottom = row_i == n_metrics - 1

            ax = fig.add_subplot(gs[row_i, col_i])

            panel_idx = row_i * n_sources + col_i
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

            ax.set_facecolor("#f5f5f5")
            for xi in range(0, len(tgts), 2):
                ax.axvspan(xi - 0.5, xi + 0.5, color="white", alpha=1.0, zorder=0)

            sub = df_delta[df_delta["source"] == src]
            bar_vals = [
                (
                    float(sub.loc[sub["target"] == t, metric].iloc[0])
                    if not sub[sub["target"] == t].empty
                    else np.nan
                )
                for t in tgts
            ]
            # Transformer better: lower error (positive Δ) or higher R² (negative Δ)
            tf_better = [(v > 0) if not is_r2 else (v < 0) for v in bar_vals]
            bar_colors = [TF_COLOR if b else LSTM_COLOR for b in tf_better]

            ax.bar(x, bar_vals, width=0.6, color=bar_colors, zorder=3, linewidth=0)
            ax.axhline(0, color="black", linewidth=0.6, zorder=4)
            ax.set_xlim(-0.5, len(tgts) - 0.5)
            ax.set_ylim(ylims[metric])
            ax.set_xticks(x)

            if is_bottom:
                ax.set_xticklabels(
                    tick_labels,
                    fontsize=NATURE_SPECS["font_min_pt"],
                    rotation=90,
                    ha="center",
                )
            else:
                ax.set_xticklabels([])

            ax.tick_params(
                axis="y", labelsize=NATURE_SPECS["font_min_pt"], width=0.5, length=2
            )
            ax.tick_params(axis="x", length=0)

            apply_nature_style(ax, fontsize=NATURE_SPECS["font_max_pt"], box=True)
            if row_i == 0:
                ax.set_title(
                    f"Source: {REGION_LABELS.get(src, src)}",
                    fontsize=subplot_title_fontsize,
                    pad=4,
                )
            if col_i == 0:
                ax.set_ylabel(
                    f"Δ {XREG_METRIC_LABELS.get(metric, metric)}\n(LSTM − TF)",
                    fontsize=NATURE_SPECS["font_max_pt"],
                    labelpad=4,
                )
            ax.set_facecolor("#f5f5f5")  # reapply after nature style reset

    # ── scatter section ──────────────────────────────────────────────────────
    # axis limits per pair: range of the observed test PMB, padded by 50%
    pair_lims = {}
    for src, tgt in pairs:
        key = f"XREG_{src}_TO_{tgt}"
        vals = outputs_xreg_by_source[src][key]["ds_test"].y.cpu().numpy().flatten()
        vals = vals[~np.isnan(vals)]
        pad = (vals.max() - vals.min()) * 0.5
        pair_lims[(src, tgt)] = (float(vals.min() - pad), float(vals.max() + pad))

    scatter_letter_start = n_metrics * n_sources
    scatter_row_offset = n_metrics + 1  # after barplots + spacer

    for row_i, (model_label, models_by_source) in enumerate(
        [
            ("LSTM", models_by_source_lstm),
            ("Transformer", models_by_source_tf),
        ]
    ):
        for col_i, (src, tgt) in enumerate(pairs):
            ax = fig.add_subplot(gs[scatter_row_offset + row_i, col_i])
            key = f"XREG_{src}_TO_{tgt}"
            lim = pair_lims[(src, tgt)]

            evaluate_one_model(
                cfg=cfg,
                model=models_by_source[src],
                device=device,
                lstm_assets_for_key=outputs_xreg_by_source[src][key],
                ax=ax,
                ax_xlim=lim,
                ax_ylim=lim,
                legend_fontsize=NATURE_SPECS["font_min_pt"],
            )
            apply_nature_style(ax, fontsize=NATURE_SPECS["font_max_pt"], box=True)
            leg = ax.get_legend()
            if leg:
                leg.remove()

            panel_idx = scatter_letter_start + row_i * n_pairs + col_i
            ax.text(
                0.02,
                0.98,
                f"({chr(ord('a') + panel_idx)})",
                transform=ax.transAxes,
                fontsize=NATURE_SPECS["font_max_pt"],
                va="top",
                ha="left",
            )

            if row_i == 0:
                ax.set_title(
                    f"{REGION_LABELS.get(src, src)} → {REGION_LABELS.get(tgt, tgt)}",
                    fontsize=subplot_title_fontsize,
                )
            if col_i == 0:
                ax.set_ylabel(
                    f"{model_label}\nModeled PMB (m w.e.)",
                    fontsize=NATURE_SPECS["font_max_pt"],
                )
            else:
                ax.set_ylabel("")
            # suppress x label — no room between sections
            ax.set_xlabel("")

    fig.savefig(out_path, dpi=NATURE_SPECS["dpi"], bbox_inches="tight")
    plt.show()

    # ── legend as a separate figure ──────────────────────────────────────────
    fig_legend, ax_legend = plt.subplots(figsize=(6, 0.5))
    ax_legend.axis("off")
    fig_legend.legend(
        handles=[
            mpatches.Patch(color=mbm.plots.COLOR_ANNUAL, label="Annual PMB"),
            mpatches.Patch(color=mbm.plots.COLOR_WINTER, label="Winter PMB"),
            mpatches.Patch(color=TF_COLOR, alpha=0.88, label="Transformer better"),
            mpatches.Patch(color=LSTM_COLOR, alpha=0.88, label="LSTM better"),
        ],
        loc="center",
        ncol=4,
        fontsize=NATURE_SPECS["font_max_pt"],
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
    )
    fig_legend.savefig(legend_out_path, dpi=NATURE_SPECS["dpi"], bbox_inches="tight")
    plt.close(fig_legend)
    return fig
