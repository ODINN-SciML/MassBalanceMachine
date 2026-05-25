import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.ticker as mtick

import massbalancemachine as mbm
from regions.TF_Europe.scripts.dataset import build_combined_LSTM_dataset
from regions.TF_Europe.scripts.config_TF_Europe import *
from regions.TF_Europe.scripts.plotting import *
from regions.TF_Europe.scripts.models import *


def build_glacier_subsets(df_ranked, fracs, n_random_seeds, seed=42):
    """
    For each fraction build:
      - closest_first: top-k glaciers by distance
      - random: N seeds
    All matched to same cumulative measurement count.
    """
    total_meas = df_ranked["n_meas"].sum()
    subsets = {}

    for frac in fracs:
        pct = int(frac * 100)
        target_meas = int(round(frac * total_meas))
        subsets[pct] = {}

        # --- closest first ---
        cumsum, glaciers_close = 0, []
        for _, row in df_ranked.iterrows():
            glaciers_close.append(row["glacier"])
            cumsum += row["n_meas"]
            if cumsum >= target_meas:
                break
        subsets[pct]["closest"] = glaciers_close

        # --- random ---
        rng = np.random.default_rng(seed)
        subsets[pct]["random"] = []
        all_glaciers = df_ranked["glacier"].tolist()
        for seed_idx in range(n_random_seeds):
            s = int(rng.integers(0, 2**31))
            rng_local = np.random.default_rng(s)
            shuffled = list(rng_local.permutation(all_glaciers))
            cumsum, glaciers_rnd = 0, []
            for g in shuffled:
                glaciers_rnd.append(g)
                cumsum += int(
                    df_ranked.loc[df_ranked["glacier"] == g, "n_meas"].iloc[0]
                )
                if cumsum >= target_meas:
                    break
            subsets[pct]["random"].append(
                {
                    "seed_idx": seed_idx,
                    "seed": s,
                    "glaciers": glaciers_rnd,
                    "n_meas": cumsum,
                }
            )

        n_close = len(glaciers_close)
        n_rnd = np.mean([len(s["glaciers"]) for s in subsets[pct]["random"]])
        print(
            f"  {pct:3d}%  target_meas={target_meas}  "
            f"close={n_close}gl  rnd~{n_rnd:.1f}gl"
        )

    return subsets


def build_assets_from_glacier_list(
    glaciers: list,
    df_ranked: pd.DataFrame,
    res_xreg_by_source: dict,
    monthly_cols: list,
    static_cols: list,
    cfg,
    months_head_pad,
    months_tail_pad,
    cache_path=None,
    force_recompute=False,
    src_regions=None,
):
    """
    Build MBSequenceDataset from a specific list of glaciers
    (drawn from multiple regions).
    """

    if cache_path and not force_recompute and os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        cached = joblib.load(cache_path)
        return cached["assets"]

    # map glacier -> region
    gl_to_region = df_ranked.set_index("glacier")["region"].to_dict()

    # collect df_loss and df_full per region
    train_dfs_loss, train_dfs_full = [], []
    for region in src_regions:
        region_glaciers = [g for g in glaciers if gl_to_region.get(g) == region]
        if not region_glaciers:
            continue

        df_loss = res_xreg_by_source[region]["data_monthly"]
        df_full = res_xreg_by_source[region]["data_monthly_aug"]

        train_dfs_loss.append(df_loss[df_loss["GLACIER"].isin(region_glaciers)])
        train_dfs_full.append(df_full[df_full["GLACIER"].isin(region_glaciers)])

    df_pooled_loss = pd.concat(train_dfs_loss, ignore_index=True)
    df_pooled_full = pd.concat(train_dfs_full, ignore_index=True)

    mbm.utils.seed_all(cfg.seed)

    ds = build_combined_LSTM_dataset(
        df_loss=df_pooled_loss,
        df_full=df_pooled_full,
        monthly_cols=monthly_cols,
        static_cols=static_cols,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
        normalize_target=True,
        expect_target=True,
        show_progress=False,
    )

    train_idx, val_idx = mbm.data_processing.MBSequenceDataset.split_indices(
        len(ds), val_ratio=0.2, seed=cfg.seed
    )

    assets = {"ds_train": ds, "train_idx": train_idx, "val_idx": val_idx}

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        joblib.dump({"assets": assets}, cache_path, compress=3)
        print(f"Saved to cache: {cache_path}")

    return assets


def plot_ranking_results_extended(
    df_results,
    ranking_label,
    test_region,
    source_regions,
    n_rand_seeds,
    model_label="transformer",  # <-- new: "transformer" or "lstm"
    metrics=None,
    save_path=None,
):
    if metrics is None:
        metrics = ["RMSE_annual", "RMSE_winter", "Bias_annual", "Bias_winter"]

    df_sub = df_results[
        (df_results["ranking_target"] == ranking_label)
        & (df_results["model"] == model_label)
    ].copy()
    PCTS_PLOT = sorted(df_sub[df_sub["pct"] != 100]["pct"].unique())
    x = np.array(PCTS_PLOT)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        is_r2 = metric.startswith("R2")

        # --- random ---
        df_rnd = df_sub[df_sub["strategy"] == "random"]
        agg = df_rnd.groupby("pct")[metric].agg(["mean", "std"]).reindex(PCTS_PLOT)

        ax.plot(
            PCTS_PLOT,
            agg["mean"],
            marker="o",
            color="steelblue",
            label="Random (mean)",
            zorder=4,
        )
        ax.fill_between(
            PCTS_PLOT,
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            alpha=0.2,
            color="steelblue",
            label="Random ±1 std",
        )
        for _, row in df_rnd.iterrows():
            ax.scatter(
                row["pct"], row[metric], color="steelblue", alpha=0.25, s=15, zorder=3
            )

        mean_rnd = agg["mean"].values

        # --- closest ---
        df_close = df_sub[df_sub["strategy"] == "closest"]
        close_vals = df_close.set_index("pct")[metric].reindex(PCTS_PLOT).values

        ax.plot(
            PCTS_PLOT,
            close_vals,
            marker="D",
            linestyle="--",
            color="darkorange",
            label="Closest (Sinkhorn)",
            zorder=5,
            markersize=8,
        )
        better = close_vals < mean_rnd if not is_r2 else close_vals > mean_rnd
        ax.fill_between(
            x, mean_rnd, close_vals, where=better, alpha=0.12, color="green"
        )
        ax.fill_between(x, mean_rnd, close_vals, where=~better, alpha=0.12, color="red")

        # --- full baseline for this model type ---
        full_label = f"{model_label}_full"
        row = df_results[
            (df_results["ranking_target"] == ranking_label)
            & (df_results["strategy"] == full_label)
        ]
        if not row.empty:
            ax.axhline(
                float(row[metric].iloc[0]),
                color="darkorange",
                linestyle=":",
                linewidth=1.5,
                label=f"{full_label} (100%)",
                zorder=2,
            )

        ax.set_title(metric, fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        ax.set_xlabel("Training data fraction (%)", fontsize=14)
        ax.set_xticks(PCTS_PLOT)
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%d%%"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, framealpha=0.9)
        apply_nature_style(ax, fontsize=NATURE_SPECS["font_min_pt"] + 2, box=True)

    mode = ranking_label.split("_")[-1]
    fig.suptitle(
        f"{model_label.upper()}: closest vs random ({mode}) → {test_region} (zero-shot)\n"
        f"Training pool: {' + '.join(source_regions)}  |  {n_rand_seeds} random seeds",
        fontsize=13,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path) + ".pdf", bbox_inches="tight")
        fig.savefig(str(save_path) + ".png", dpi=150, bbox_inches="tight")

    plt.show()
    return fig


def plot_pred_vs_truth_grid(
    plot_configs: list,  # list of (label, model, region_assets)
    ds_test,  # pristine test dataset
    device,
    cfg,
    ncols: int = 3,
    ax_xlim=(-16, 10),
    ax_ylim=(-16, 10),
    title: str = "",
    save_path=None,
    figsize=None,
):
    nrows = int(np.ceil(len(plot_configs) / ncols))
    if figsize is not None:
        figsize = (ncols * 5, nrows * 5)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)

    def _fmt(x):
        return (
            "NA"
            if (x is None or (isinstance(x, float) and np.isnan(x)))
            else f"{x:.2f}"
        )

    for idx, (label, model, region_assets) in enumerate(plot_configs):
        ax = axes[idx]

        ds_scaler_copy = (
            mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
                region_assets["ds_train"]
            )
        )
        ds_scaler_copy.make_loaders(
            train_idx=region_assets["train_idx"],
            val_idx=region_assets["val_idx"],
            fit_and_transform=True,
            seed=cfg.seed,
            verbose=False,
        )

        ds_test_copy = (
            mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(ds_test)
        )
        test_dl = mbm.data_processing.MBSequenceDataset.make_test_loader(
            ds_test_copy,
            ds_scaler_copy,
            batch_size=128,
            seed=cfg.seed,
        )

        _, df_preds = model.evaluate_with_preds(device, test_dl, ds_test_copy)
        df_preds = df_preds.dropna(subset=["target", "pred"])

        scores_annual, scores_winter = compute_seasonal_scores(
            df_preds, target_col="target", pred_col="pred"
        )

        pred_vs_truth_density(
            ax,
            df_preds,
            scores_annual,
            add_legend=False,
            palette=[mbm.plots.COLOR_ANNUAL, mbm.plots.COLOR_WINTER],
            ax_xlim=ax_xlim,
            ax_ylim=ax_ylim,
            s=100,
        )

        legend_txt = "\n".join(
            [
                rf"$\mathrm{{RMSE_a}}={_fmt(scores_annual['rmse'])},\ \mathrm{{RMSE_w}}={_fmt(scores_winter['rmse'])}$",
                rf"$\mathrm{{R^2_a}}={_fmt(scores_annual['R2'])},\ \mathrm{{R^2_w}}={_fmt(scores_winter['R2'])}$",
                rf"$\mathrm{{Bias_a}}={_fmt(scores_annual['Bias'])},\ \mathrm{{Bias_w}}={_fmt(scores_winter['Bias'])}$",
            ]
        )
        ax.text(
            0.02,
            0.98,
            legend_txt,
            transform=ax.transAxes,
            va="top",
            fontsize=14,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )
        ax.set_title(label, fontsize=14)

        col = idx % ncols
        row = idx // ncols
        ax.set_xlabel(
            "Observed PMB (m w.e.)" if row == nrows - 1 else "",
            fontsize=NATURE_SPECS["font_max_pt"],
        )
        ax.set_ylabel(
            "Modeled PMB (m w.e.)" if col == 0 else "",
            fontsize=NATURE_SPECS["font_max_pt"],
        )

        apply_nature_style(ax, fontsize=14, box=True)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    for j in range(len(plot_configs), len(axes)):
        axes[j].axis("off")

    legend_handles = [
        mpatches.Patch(color=mbm.plots.COLOR_ANNUAL, label="Annual"),
        mpatches.Patch(color=mbm.plots.COLOR_WINTER, label="Winter"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncols=2,
        frameon=False,
        fontsize=14,
    )
    fig.suptitle(title, fontsize=18, y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path) + ".pdf", bbox_inches="tight")
        fig.savefig(str(save_path) + ".png", dpi=150, bbox_inches="tight")

    plt.show()
    return fig
