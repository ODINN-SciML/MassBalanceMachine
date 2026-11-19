"""Sensitivity of fine-tuning to the fine-tuning fraction (notebook `3.2 Pooled_model`,
Fig. A8 and A9).

- `finetune_with_history`: fine-tuning as `finetune_transformer_on_target`, also
  returning the per-epoch train/validation loss.
- `build_ft_fraction_splits`: nested FT subsets per fraction and one fixed test set.
- `run_ft_fraction_sensitivity`: fine-tune and evaluate for every region, fraction,
  strategy and repeat.
- `summarize_ft_fraction_sensitivity`: summary tables.
- `plot_ft_fraction_sensitivity` (Fig. A8) / `plot_loss_grid` (Fig. A9).
"""

import os
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython.display import display
import massbalancemachine as mbm

from papers.Transformer_TF.scripts.config_TF_Europe import REGION_LABELS
from papers.Transformer_TF.scripts.plotting.style import (
    NATURE_SPECS,
    apply_nature_style,
)
from papers.Transformer_TF.scripts.dataset.xreg_lstm_dataset import (
    build_or_load_lstm_dataset_only,
)
from papers.Transformer_TF.scripts.models.pooled_model import add_nrmse, evaluate_on

# Colours and labels of zero-shot and fine-tuning strategies (same as Fig. 6)
FT_COLORS = {"no_ft": "#aaaaaa", "head_only": "#EE854A", "full": "#956CB4"}
FT_LABELS = {"no_ft": "Zero-shot", "head_only": "Head only FT", "full": "Full FT"}


def finetune_with_history(
    cfg,
    model_pooled,
    ds_ft,
    ds_pooled_scaler,
    device,
    best_params,
    model_filename,
    strategy="full",
    epochs=50,
    lr_factor=0.1,
    split_seed=None,
    force_retrain=False,
    verbose=False,
):
    """
    Same fine-tuning procedure as `finetune_transformer_on_target` (same split,
    sampler, optimiser, scheduler, early stopping, best-val checkpoint), but it
    also returns and caches the per-epoch train/val loss history.

    The history includes epoch 0 (pooled model before any fine-tuning step),
    evaluated in eval mode on the same train/val loaders.
    Cached next to the checkpoint as `<model_filename>.history.json`.
    """
    assert strategy in ("head_only", "full")
    model_filename = str(model_filename)
    hist_path = Path(model_filename).with_suffix(".history.json")

    model_ft = copy.deepcopy(model_pooled)

    if (not force_retrain) and os.path.exists(model_filename) and hist_path.exists():
        model_ft.load_state_dict(torch.load(model_filename, map_location=device))
        with open(hist_path, "r") as f:
            history = json.load(f)
        return model_ft, history

    split_seed = cfg.seed if split_seed is None else split_seed
    train_idx, val_idx = mbm.data_processing.MBSequenceDataset.split_indices(
        len(ds_ft), val_ratio=0.2, seed=split_seed
    )

    ds_ft_copy = mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
        ds_ft
    )
    ds_ft_copy.set_scalers_from(ds_pooled_scaler)
    ds_ft_copy.transform_inplace()
    train_dl, val_dl = ds_ft_copy.make_loaders(
        train_idx=train_idx,
        val_idx=val_idx,
        fit_and_transform=False,
        seed=cfg.seed,
        use_weighted_sampler=True,
        verbose=False,
    )

    # --- freeze / unfreeze (identical to finetune_transformer_on_target) ---
    if strategy == "head_only":
        for p in model_ft.parameters():
            p.requires_grad = False
        head_modules = (
            [model_ft.head_w, model_ft.head_a]
            if model_ft.two_heads
            else [model_ft.head]
        )
        for m in head_modules:
            for p in m.parameters():
                p.requires_grad = True

    n_trainable = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model_ft.parameters())

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_ft.parameters()),
        lr=best_params["lr"] * lr_factor,
        weight_decay=best_params["weight_decay"],
    )
    loss_fn = mbm.models.Transformer_MB.resolve_loss_fn(best_params)

    # --- epoch 0: loss of the pooled model before fine-tuning ---
    tr0 = model_ft.run_epoch(device, None, train_dl, loss_fn=loss_fn, train=False)
    va0 = model_ft.run_epoch(device, None, val_dl, loss_fn=loss_fn, train=False)

    hist, best_val, _ = model_ft.train_loop(
        device=device,
        train_dl=train_dl,
        val_dl=val_dl,
        epochs=epochs,
        optimizer=optimizer,
        loss_fn=loss_fn,
        es_patience=10,
        es_min_delta=1e-4,
        sched_factor=0.5,
        sched_patience=4,
        sched_min_lr=1e-7,
        log_every=5,
        verbose=verbose,
        save_best_path=model_filename,
    )

    val_curve = hist["val_loss"]
    # epoch index (1-based, epoch 0 = before FT) of the restored checkpoint
    best_epoch = int(np.argmin(np.abs(np.array(val_curve) - best_val))) + 1

    history = {
        "train_loss": [float(tr0)] + [float(x) for x in hist["train_loss"]],
        "val_loss": [float(va0)] + [float(x) for x in val_curve],
        "lr": [float("nan")] + [float(x) for x in hist["lr"]],
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_trainable_params": int(n_trainable),
        "n_total_params": int(n_total),
        "strategy": strategy,
    }
    with open(hist_path, "w") as f:
        json.dump(history, f)

    # train_loop already restored the best state; reload from disk to be explicit
    model_ft.load_state_dict(torch.load(model_filename, map_location=device))
    return model_ft, history


def _frac_tag(f):
    return f"{int(round(f * 100)):02d}pct"


def _n_ft(n_ids, frac):
    # identical rule to the main experiment
    return max(1, int(n_ids * frac)) if n_ids else 0


def build_ft_fraction_splits(
    cfg,
    monthly_cache,
    regions,
    fracs,
    n_repeats,
    monthly_cols,
    static_cols,
    cache_dir,
    ft_assets_id=None,
    force_recompute=False,
):
    """
    For each region and repeat, shuffle the measurement IDs as in the main experiment
    (seed cfg.seed + rep, winter and annual separately), so the 10% subset equals the
    main FT set. FT subsets are nested across fractions; the test set is fixed to the
    IDs left after the largest fraction.

    Returns (sens_assets, sens_test, df_sens_sizes):
    sens_assets[region][frac] = [{"rep", "seed", "ds_ft", ...}], sens_test[region] =
    [{"rep", "seed", "ds_test", "test_ids"}], and a table of FT/test set sizes.
    """
    frac_max = max(fracs)
    sens_assets = {r: {f: [] for f in fracs} for r in regions}
    sens_test = {r: [] for r in regions}
    sens_sizes = []

    for region in regions:
        print(f"\n{'='*50}\nFT-fraction sensitivity splits: {region}")
        df_monthly = monthly_cache[region]["data_monthly"]
        df_monthly_aug = monthly_cache[region]["data_monthly_aug"]
        head_pad = monthly_cache[region]["months_head_pad"]
        tail_pad = monthly_cache[region]["months_tail_pad"]
        common = dict(
            months_head_pad=head_pad,
            months_tail_pad=tail_pad,
            MONTHLY_COLS=monthly_cols,
            STATIC_COLS=static_cols,
            cache_dir=str(cache_dir),
            force_recompute=force_recompute,
            show_progress=False,
        )

        winter_ids_full = (
            df_monthly[df_monthly["PERIOD"] == "winter"]["ID"].unique().tolist()
        )
        annual_ids_full = (
            df_monthly[df_monthly["PERIOD"] == "annual"]["ID"].unique().tolist()
        )

        for rep in range(n_repeats):
            # identical shuffle to the main experiment -> 10% subset == main FT set
            winter_ids = list(winter_ids_full)
            annual_ids = list(annual_ids_full)
            rep_seed = cfg.seed + rep
            rng = np.random.default_rng(rep_seed)
            rng.shuffle(winter_ids)
            rng.shuffle(annual_ids)

            # fixed test set: everything after the largest FT subset
            n_w_max = _n_ft(len(winter_ids), frac_max)
            n_a_max = _n_ft(len(annual_ids), frac_max)
            test_ids = winter_ids[n_w_max:] + annual_ids[n_a_max:]

            ds_test = build_or_load_lstm_dataset_only(
                cfg=cfg,
                key=f"FD_to_{region}_IDsplit_rep{rep}_FTSENS_TEST",
                df_loss=df_monthly[df_monthly["ID"].isin(test_ids)],
                df_full=df_monthly_aug[df_monthly_aug["ID"].isin(test_ids)],
                kind="test",
                **common,
            )
            sens_test[region].append(
                {
                    "rep": rep,
                    "seed": rep_seed,
                    "ds_test": ds_test,
                    "test_ids": set(test_ids),
                }
            )

            prev_ids = set()
            for frac in fracs:
                n_w = _n_ft(len(winter_ids), frac)
                n_a = _n_ft(len(annual_ids), frac)
                ft_ids = winter_ids[:n_w] + annual_ids[:n_a]

                # sanity: nested and disjoint from test
                assert not (
                    set(ft_ids) & set(test_ids)
                ), f"FT/test overlap {region} rep{rep} {frac}"
                assert prev_ids <= set(
                    ft_ids
                ), f"FT subsets not nested {region} rep{rep} {frac}"
                prev_ids = set(ft_ids)

                ds_ft = build_or_load_lstm_dataset_only(
                    cfg=cfg,
                    key=f"FD_to_{region}_IDsplit_rep{rep}_FTSENS_{_frac_tag(frac)}_FT",
                    df_loss=df_monthly[df_monthly["ID"].isin(ft_ids)],
                    df_full=df_monthly_aug[df_monthly_aug["ID"].isin(ft_ids)],
                    kind="ft",
                    **common,
                )
                sens_assets[region][frac].append(
                    {
                        "rep": rep,
                        "seed": rep_seed,
                        "ds_ft": ds_ft,
                        "n_ft_ids": len(ft_ids),
                        "n_ft_w": n_w,
                        "n_ft_a": n_a,
                    }
                )
                sens_sizes.append(
                    {
                        "region": region,
                        "rep": rep,
                        "frac": frac,
                        "n_ft_ids": len(ft_ids),
                        "n_ft_winter": n_w,
                        "n_ft_annual": n_a,
                        "n_ft_seqs": len(ds_ft),
                        "n_test_seqs": len(ds_test),
                        "n_ft_glaciers": df_monthly[df_monthly["ID"].isin(ft_ids)][
                            "GLACIER"
                        ].nunique(),
                    }
                )

            # check that the 10% subset equals the main-experiment FT set
            if (
                ft_assets_id is not None
                and 0.10 in fracs
                and region in ft_assets_id
                and len(ft_assets_id[region]) > rep
            ):
                n_main = len(ft_assets_id[region][rep]["ds_ft"])
                n_sens = len(sens_assets[region][0.10][rep]["ds_ft"])
                if n_main != n_sens:
                    print(
                        f"  WARNING [{region} rep{rep}]: 10% FT set size differs from main "
                        f"experiment ({n_sens} vs {n_main})"
                    )

    df_sens_sizes = pd.DataFrame(sens_sizes)
    print("\nFT set sizes (mean over repeats):")
    display(
        df_sens_sizes.groupby(["region", "frac"])[
            ["n_ft_seqs", "n_ft_glaciers", "n_test_seqs"]
        ]
        .mean()
        .round(1)
        .unstack("frac")
    )
    return sens_assets, sens_test, df_sens_sizes


def _scalar_metrics(m):
    return {k: float(v) for k, v in m.items() if np.isscalar(v)}


def run_ft_fraction_sensitivity(
    cfg,
    device,
    model_pooled,
    ds_pooled_scaler,
    sens_assets,
    sens_test,
    regions,
    fracs,
    strategies,
    n_repeats,
    models_dir,
    params,
    ft_params,
    date,
    min_ft_seqs=5,
    force_retrain=False,
    results_csv=None,
):
    """
    Fine-tune (or load) one model per region, fraction, strategy and repeat and
    evaluate it, and the zero-shot model, on the fixed test set. Runs with fewer than
    min_ft_seqs FT sequences are skipped. Saves the per-run table to results_csv.

    Returns (df_sens, sens_histories) with loss histories keyed by
    (region, frac, strategy, rep).
    """
    os.makedirs(models_dir, exist_ok=True)

    def _eval_on_test(model, ds_test):
        metrics, preds_df = evaluate_on(cfg, device, model, ds_test, ds_pooled_scaler)
        return add_nrmse(metrics, preds_df)

    sens_rows = []
    sens_histories = {}

    for region in regions:
        print(f"\n{'='*60}\nFT-fraction sensitivity: {region}")
        for rep in range(n_repeats):
            ds_test = sens_test[region][rep]["ds_test"]

            # zero-shot on the same fixed test set (reference line)
            m0 = _scalar_metrics(_eval_on_test(model_pooled, ds_test))
            sens_rows.append(
                {
                    "region": region,
                    "rep": rep,
                    "frac": 0.0,
                    "strategy": "no_ft",
                    "n_ft_seqs": 0,
                    **m0,
                }
            )

            for frac in fracs:
                a = sens_assets[region][frac][rep]
                n_ft = len(a["ds_ft"])
                if n_ft < min_ft_seqs:
                    print(
                        f"  [{region} rep{rep} {_frac_tag(frac)}] skipped: "
                        f"only {n_ft} FT sequences (< {min_ft_seqs})"
                    )
                    continue

                for strategy in strategies:
                    fname = (
                        models_dir
                        / f"transformer_ft_{strategy}_{region}_{_frac_tag(frac)}"
                        f"_rep{rep}_{date}.pt"
                    )
                    model_ft, hist = finetune_with_history(
                        cfg=cfg,
                        model_pooled=model_pooled,
                        ds_ft=a["ds_ft"],
                        ds_pooled_scaler=ds_pooled_scaler,
                        device=device,
                        best_params=params,
                        model_filename=fname,
                        strategy=strategy,
                        epochs=ft_params["epochs"],
                        lr_factor=ft_params["lr_factor"],
                        force_retrain=force_retrain,
                        verbose=False,
                    )
                    sens_histories[(region, frac, strategy, rep)] = hist

                    m = _scalar_metrics(_eval_on_test(model_ft, ds_test))
                    be = hist["best_epoch"]
                    sens_rows.append(
                        {
                            "region": region,
                            "rep": rep,
                            "frac": frac,
                            "strategy": strategy,
                            "n_ft_seqs": n_ft,
                            "n_ft_train": hist["n_train"],
                            "n_ft_val": hist["n_val"],
                            "n_trainable_params": hist["n_trainable_params"],
                            "best_epoch": be,
                            "n_epochs_run": len(hist["train_loss"]) - 1,
                            "train_loss_ep0": hist["train_loss"][0],
                            "val_loss_ep0": hist["val_loss"][0],
                            "train_loss_best": hist["train_loss"][be],
                            "val_loss_best": hist["val_loss"][be],
                            "val_loss_final": hist["val_loss"][-1],
                            "train_loss_final": hist["train_loss"][-1],
                            **m,
                        }
                    )
                    del model_ft
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                print(f"  [{region} rep{rep} {_frac_tag(frac)}] n_ft={n_ft} done")

    df_sens = pd.DataFrame(sens_rows)
    # overfitting diagnostics
    df_sens["gen_gap_best"] = df_sens["val_loss_best"] - df_sens["train_loss_best"]
    df_sens["gen_ratio_best"] = df_sens["val_loss_best"] / df_sens["train_loss_best"]
    df_sens["val_rise_after_best"] = (
        df_sens["val_loss_final"] - df_sens["val_loss_best"]
    )

    if results_csv is not None:
        df_sens.to_csv(results_csv, index=False)
        print(f"\nSaved per-run results to: {results_csv}")
    return df_sens, sens_histories


def summarize_ft_fraction_sensitivity(df_sens, n_repeats):
    """
    Display: RMSE per fraction (mean ± std over repeats), change in RMSE versus
    zero-shot, full − head-only FT (paired per repeat), and overfitting diagnostics.
    """
    summary_metrics = [
        "RMSE_annual",
        "RMSE_winter",
        "NRMSE_annual",
        "NRMSE_winter",
        "R2_annual",
        "R2_winter",
    ]
    agg = df_sens.groupby(["region", "strategy", "frac"])[summary_metrics].agg(
        ["mean", "std"]
    )

    def _fmt_pm(df, metric):
        return (
            df[(metric, "mean")].map("{:.3f}".format)
            + " ± "
            + df[(metric, "std")].map("{:.3f}".format)
        )

    for metric in ["RMSE_annual", "RMSE_winter"]:
        print(f"\n{metric} (mean ± std over {n_repeats} repeats, fixed test set)")
        display(_fmt_pm(agg, metric).unstack("frac"))

    # Gain over zero-shot, per repeat (paired, same test set)
    zs = df_sens[df_sens.strategy == "no_ft"].set_index(["region", "rep"])[
        ["RMSE_annual", "RMSE_winter"]
    ]
    df_ft_only = df_sens[df_sens.strategy != "no_ft"].copy()
    for m in ["RMSE_annual", "RMSE_winter"]:
        df_ft_only[f"d{m}_vs_zs"] = (
            df_ft_only[m].values
            - zs.loc[list(zip(df_ft_only.region, df_ft_only.rep)), m].values
        )

    print("\nChange in RMSE vs zero-shot (negative = improvement), mean over repeats")
    display(
        df_ft_only.groupby(["region", "strategy", "frac"])[
            ["dRMSE_annual_vs_zs", "dRMSE_winter_vs_zs"]
        ]
        .mean()
        .round(3)
        .unstack("frac")
    )

    # Full minus head-only, paired per repeat: is full FT ever worse at small fractions?
    piv = df_ft_only.pivot_table(
        index=["region", "frac", "rep"],
        columns="strategy",
        values=["RMSE_annual", "RMSE_winter"],
    )
    full_minus_head = pd.DataFrame(
        {
            "RMSE_annual": piv[("RMSE_annual", "full")]
            - piv[("RMSE_annual", "head_only")],
            "RMSE_winter": piv[("RMSE_winter", "full")]
            - piv[("RMSE_winter", "head_only")],
        }
    )
    print(
        "\nFull FT − head-only FT RMSE (negative = full better); mean [frac. of repeats where full is better]"
    )
    fmh = (
        full_minus_head.groupby(["region", "frac"])
        .agg(
            dRMSE_a=("RMSE_annual", "mean"),
            full_better_a=("RMSE_annual", lambda x: (x < 0).mean()),
            dRMSE_w=("RMSE_winter", "mean"),
            full_better_w=("RMSE_winter", lambda x: (x < 0).mean()),
        )
        .round(3)
    )
    display(fmh.unstack("frac"))

    # Overfitting diagnostics from the loss curves
    print(
        "\nOverfitting diagnostics (mean over repeats): best epoch, epochs run, "
        "val−train loss gap at the restored checkpoint, val-loss rise after best epoch"
    )
    display(
        df_ft_only.groupby(["region", "strategy", "frac"])[
            [
                "n_ft_train",
                "n_ft_val",
                "best_epoch",
                "n_epochs_run",
                "gen_gap_best",
                "gen_ratio_best",
                "val_rise_after_best",
            ]
        ]
        .mean()
        .round(3)
    )


def plot_ft_fraction_sensitivity(
    df,
    metric,
    ylabel,
    regions,
    fracs,
    strategies,
    main_frac,
    exclude_regions=(),
    max_cols=3,
    xscale="linear",
    xtick_rotation=45,
    fname=None,
):
    """
    Fig. A8: metric versus fine-tuning fraction per region (mean ± std over repeats)
    for each strategy, with the zero-shot mean ± std as a grey band and the fraction
    used in the paper (main_frac) as a dashed line.
    """
    regions = [r for r in regions if r not in exclude_regions]
    n_cols = min(max_cols, len(regions))
    n_rows = math.ceil(len(regions) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.6 * n_cols, 3.0 * n_rows), squeeze=False, sharex=True
    )
    xs_pct = [100 * f for f in fracs]
    fs = NATURE_SPECS["font_max_pt"]

    for i, region in enumerate(regions):
        ax = axes[i // n_cols, i % n_cols]
        d = df[df.region == region]

        # zero-shot reference band
        z = d[d.strategy == "no_ft"][metric]
        if len(z):
            ax.axhspan(
                z.mean() - z.std(),
                z.mean() + z.std(),
                color=FT_COLORS["no_ft"],
                alpha=0.25,
                lw=0,
                zorder=1,
            )
            ax.axhline(
                z.mean(),
                color=FT_COLORS["no_ft"],
                ls="-",
                lw=1.2,
                label=FT_LABELS["no_ft"],
                zorder=2,
            )

        for strategy in strategies:
            g = (
                d[d.strategy == strategy]
                .groupby("frac")[metric]
                .agg(["mean", "std"])
                .reindex(fracs)
            )
            ax.errorbar(
                xs_pct,
                g["mean"],
                yerr=g["std"],
                marker="o",
                ms=4,
                lw=1.4,
                capsize=2.5,
                color=FT_COLORS[strategy],
                label=FT_LABELS[strategy],
                zorder=3,
            )

        ax.axvline(100 * main_frac, color="k", ls="--", lw=0.8, zorder=0)
        ax.set_xscale(xscale)
        ax.set_xticks(xs_pct)
        ax.set_xticklabels([f"{x:g}" for x in xs_pct])
        ax.minorticks_off()

        # panel letter in the title -> never collides with data or axis labels
        ax.set_title(
            f"({chr(ord('a') + i)}) {REGION_LABELS.get(region, region)}",
            fontsize=fs,
            loc="left",
        )
        apply_nature_style(ax, fontsize=fs, box=True)

        is_bottom = (i // n_cols == n_rows - 1) or (i + n_cols >= len(regions))
        ax.tick_params(axis="x", labelbottom=is_bottom)
        if is_bottom:
            plt.setp(
                ax.get_xticklabels(),
                rotation=xtick_rotation,
                ha="right" if xtick_rotation else "center",
                rotation_mode="anchor",
            )
            ax.set_xlabel("Fine-tuning fraction (%)", fontsize=fs)
        if i % n_cols == 0:
            ax.set_ylabel(ylabel, fontsize=fs)

    for j in range(len(regions), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=fs,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    if fname:
        fig.savefig(fname, dpi=NATURE_SPECS["dpi"], bbox_inches="tight")
    plt.show()
    return fig


def plot_loss_grid(
    row_specs,
    sens_histories,
    df_sens_sizes,
    strategies,
    n_repeats,
    min_ft_seqs=5,
    logy=False,
    fname=None,
    row_label_fn=None,
):
    """
    Fig. A9: training (dashed) and validation (solid) loss per epoch during
    fine-tuning. row_specs is a list of (region, frac) tuples, one row each; one
    column per strategy; one colour per repeat; dot = restored best-val checkpoint.
    Epoch 0 = pooled model before fine-tuning.
    """
    n_rows, n_cols = len(row_specs), len(strategies)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.8 * n_cols, 2.4 * n_rows), squeeze=False, sharex=True
    )
    cmap = plt.get_cmap("viridis")

    for r_i, (region, frac) in enumerate(row_specs):
        for c_i, strategy in enumerate(strategies):
            ax = axes[r_i, c_i]
            n_plotted = 0
            for rep in range(n_repeats):
                h = sens_histories.get((region, frac, strategy, rep))
                if h is None:
                    continue
                col = cmap(rep / max(1, n_repeats - 1))
                ep = np.arange(len(h["train_loss"]))
                ax.plot(ep, h["train_loss"], ls="--", lw=0.9, color=col, alpha=0.8)
                ax.plot(ep, h["val_loss"], ls="-", lw=1.2, color=col, alpha=0.9)
                be = h["best_epoch"]
                ax.plot(be, h["val_loss"][be], "o", ms=4, color=col, mec="k", mew=0.5)
                n_plotted += 1

            if n_plotted == 0:
                ax.text(
                    0.5,
                    0.5,
                    f"skipped (< {min_ft_seqs} FT seqs)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=NATURE_SPECS["font_min_pt"],
                )
                ax.set_yticks([])
            if logy:
                ax.set_yscale("log")
            apply_nature_style(ax, fontsize=NATURE_SPECS["font_max_pt"], box=True)

            if r_i == 0:
                ax.set_title(
                    FT_LABELS[strategy], fontsize=NATURE_SPECS["font_max_pt"] + 1
                )
            if c_i == 0:
                lab = (
                    row_label_fn(region, frac)
                    if row_label_fn
                    else f"{REGION_LABELS.get(region, region)}, {100*frac:g}%"
                )
                ax.set_ylabel(f"{lab}\nLoss", fontsize=NATURE_SPECS["font_max_pt"] - 1)
            if r_i == n_rows - 1:
                ax.set_xlabel("Epoch", fontsize=NATURE_SPECS["font_max_pt"])

            n_seq = df_sens_sizes[
                (df_sens_sizes.region == region) & (df_sens_sizes.frac == frac)
            ]["n_ft_seqs"].mean()
            ax.text(
                0.98,
                0.97,
                f"n$_{{FT}}$={n_seq:.0f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=NATURE_SPECS["font_min_pt"] - 1,
            )

    handles = [
        Line2D([], [], color="k", ls="--", lw=0.9, label="Training"),
        Line2D([], [], color="k", ls="-", lw=1.2, label="Validation"),
        Line2D(
            [],
            [],
            color="w",
            marker="o",
            mfc="grey",
            mec="k",
            ms=5,
            label="Restored checkpoint (min. val.)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=NATURE_SPECS["font_max_pt"],
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    if fname:
        fig.savefig(fname, dpi=NATURE_SPECS["dpi"], bbox_inches="tight")
    plt.show()
    return fig
