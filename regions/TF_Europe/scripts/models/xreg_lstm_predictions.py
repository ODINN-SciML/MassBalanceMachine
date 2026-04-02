import logging
import os
import matplotlib.pyplot as plt
import numpy as np

from regions.TF_Europe.scripts.dataset import make_test_loader_for_key_TL
from regions.TF_Europe.scripts.plotting import *
from regions.TF_Europe.scripts.models import compute_seasonal_scores


def _pick_tl_exp_key_for_region(tl_assets_by_key, region, split_name="5pct"):
    k = f"TL_CH_to_{region}_{split_name}"
    if k not in tl_assets_by_key:
        raise KeyError(f"Missing TL assets for {k}")
    return k


def evaluate_transfer_learning_grid(
    cfg,
    regions,
    models_xreg_by_region: dict,
    models_tl_by_key: dict,
    tl_assets_by_key: dict,
    device,
    *,
    split_name="5pct",
    strategies=None,  # : which columns to plot
    strategy_labels=None,  # : pretty names for column headers
    include_region_in_titles=True,  # : convenience
    save_dir=None,
    fig_size_per_cell=(5.2, 5.2),
    ax_xlim=None,
    ax_ylim=None,
    legend_fontsize=11,
    batch_size_eval=128,
    domain_vocab=None,
):
    """
    Flexible TL grid evaluator.

    Parameters
    ----------
    strategies : list[str] or None
        Strategies to plot as columns. Examples:
          ["no_ft", "safe", "two_stage"]
          ["no_ft", "safe", "l2sp_safe", "disc_full", "disc_l2sp_full"]
        If None, defaults to ["no_ft","safe","full","two_stage"].

    strategy_labels : dict[str,str] or None
        Mapping strategy -> column label. If None, uses defaults and falls back
        to the strategy string.

    Notes
    -----
    - Uses assets_row["ds_test"] as holdout set for that region.
    - Uses CH scalers via assets_row["ds_pretrain_scalers"] through evaluate_one_model_TL.
    """
    if strategies is None:
        strategies = ["no_ft", "safe", "full", "two_stage"]
    strategies = list(strategies)

    default_labels = {
        "no_ft": "No fine-tuning (xreg CH)",
        "safe": "Heads-only FT",
        "full": "Full FT",
        "two_stage": "Two-stage FT",
        "disc_full": "Disc-LR FT",
        "l2sp_safe": "L2SP + Heads-only",
        "l2sp_full": "L2SP + Full",
        "disc_l2sp_full": "Disc-LR + L2SP",
        "adapter": "Adapter FT",
        "dan": "Domain-Adversarial Network",
    }
    if strategy_labels is None:
        strategy_labels = {}
    col_labels = {**default_labels, **strategy_labels}

    nrows = len(regions)
    ncols = len(strategies)

    if save_dir:
        save_abs = os.path.join(save_dir)
        os.makedirs(save_abs, exist_ok=True)
    else:
        save_abs = None

    figsize = (fig_size_per_cell[0] * ncols, fig_size_per_cell[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=False, sharey=False)
    axes = np.array(axes)
    if nrows == 1:
        axes = axes.reshape(1, -1)
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    rows = []
    preds = {}

    for r, region in enumerate(regions):
        exp_key = _pick_tl_exp_key_for_region(
            tl_assets_by_key, region, split_name=split_name
        )
        assets_row = tl_assets_by_key.get(exp_key, None)

        # validate assets
        if assets_row is None or assets_row.get("ds_test", None) is None:
            logging.warning(f"Skipping region {region}: no ds_test in {exp_key}")
            for c in range(ncols):
                axes[r, c].axis("off")
            continue

        if assets_row.get("ds_pretrain_scalers", None) is None:
            logging.warning(
                f"Skipping region {region}: missing ds_pretrain_scalers in {exp_key}"
            )
            for c in range(ncols):
                axes[r, c].axis("off")
            continue

        for c, strat in enumerate(strategies):
            ax = axes[r, c]

            # pick model
            if strat == "no_ft":
                model = models_xreg_by_region.get(region, None)
            else:
                model_key = f"{exp_key}__{strat}"
                model = models_tl_by_key.get(model_key, None)

            if model is None:
                ax.axis("off")
                logging.warning(f"Missing model for region={region}, strategy={strat}")
                continue

            # title inside each cell (optional; you also set titles later)
            cell_title = f"{region}\n{strat}" if include_region_in_titles else strat

            dv = (
                domain_vocab
                if domain_vocab is not None
                else assets_row.get("domain_vocab", None)
            )

            metrics, df_preds, _fig_ind, _ = evaluate_one_model_TL(
                cfg=cfg,
                model=model,
                device=device,
                tl_assets_for_key=assets_row,
                ax=ax,
                ax_xlim=ax_xlim,
                ax_ylim=ax_ylim,
                title=cell_title if include_region_in_titles else None,
                legend_fontsize=legend_fontsize,
                batch_size=batch_size_eval,
                domain_vocab=dv,
            )

            metrics.update(
                {
                    "region": region,
                    "strategy": strat,
                    "exp_key": exp_key,
                    "split_name": split_name,
                }
            )
            rows.append(metrics)
            preds[(region, strat)] = df_preds

            # remove legend if present
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

    # ---- nicer axis titles: region as row + strategy label as column ----
    for rr in range(nrows):
        for cc in range(ncols):
            strat = strategies[cc]
            col_name = col_labels.get(strat, strat)
            axes[rr, cc].set_title(f"{regions[rr]} - {col_name}", fontsize=14)

            if cc == 0:
                axes[rr, cc].set_ylabel("Modeled PMB [m w.e.]", fontsize=12)
            else:
                axes[rr, cc].set_ylabel("")

            if rr == nrows - 1:
                axes[rr, cc].set_xlabel("Observed PMB [m w.e.]", fontsize=12)
            else:
                axes[rr, cc].set_xlabel("")

    fig.suptitle(
        f"Transfer learning evaluation (holdout test) — split={split_name}",
        fontsize=18,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if save_abs:
        # include strategies in filename to avoid overwriting
        tag = "_".join(strategies)
        out_png = os.path.join(save_abs, f"TL_grid_{split_name}_{tag}.png")
        fig.savefig(out_png, dpi=200, bbox_inches="tight")

    df_metrics = pd.DataFrame(rows)
    if len(df_metrics) > 0:
        df_metrics = df_metrics.set_index(["region", "strategy"]).sort_index()

    return df_metrics, preds, fig


def evaluate_one_model_TL(
    cfg,
    model,
    device,
    tl_assets_for_key,
    ax=None,
    ax_xlim=None,
    ax_ylim=None,
    title=None,
    legend_fontsize=16,
    batch_size=128,
    domain_vocab=None,  # optional: {"CH":0,"NOR":1,...}
    show_plot=True,
):
    """
    TL-only evaluator.

    - Builds a TL-wrapped test loader with CH scalers
    - Runs model.evaluate_with_preds(...)
    - Adds SOURCE_CODE (and domain_id if vocab provided) to df_preds
    - Plots pred-vs-truth density
    """
    # ---- normalize domain_vocab to dict[str,int] if user passed list/tuple ----
    if domain_vocab is not None and not isinstance(domain_vocab, dict):
        domain_vocab = {sc: i for i, sc in enumerate(list(domain_vocab))}

    _ds_scalers, ds_test_tl, test_dl, test_source_codes = make_test_loader_for_key_TL(
        cfg, tl_assets_for_key, batch_size=batch_size, domain_vocab=domain_vocab
    )

    # evaluate_with_preds expects the *base* dataset (with keys + scalers)
    test_metrics, test_df_preds = model.evaluate_with_preds(
        device, test_dl, ds_test_tl.base
    )

    # ---- attach SOURCE_CODE/domain_id aligned by sequence keys ----
    key_to_sc = {k: sc for k, sc in zip(ds_test_tl.base.keys, test_source_codes)}

    def _row_key(r):
        return (
            r["GLACIER"],
            int(r["YEAR"]),
            int(r["ID"]),
            str(r["PERIOD"]).strip().lower(),
        )

    test_df_preds["SOURCE_CODE"] = test_df_preds.apply(
        lambda r: key_to_sc.get(_row_key(r), None), axis=1
    )

    if domain_vocab is not None:
        test_df_preds["domain_id"] = test_df_preds["SOURCE_CODE"].map(domain_vocab)

    # seasonal scores
    scores_annual, scores_winter = compute_seasonal_scores(
        test_df_preds, target_col="target", pred_col="pred"
    )

    out = {
        "RMSE_annual": float(test_metrics.get("RMSE_annual", scores_annual["rmse"])),
        "RMSE_winter": float(test_metrics.get("RMSE_winter", scores_winter["rmse"])),
        "R2_annual": float(scores_annual["R2"]),
        "R2_winter": float(scores_winter["R2"]),
        "Bias_annual": float(scores_annual["Bias"]),
        "Bias_winter": float(scores_winter["Bias"]),
        "n_preds": int(len(test_df_preds)),
        "n_annual": (
            int(scores_annual.get("n", np.nan))
            if isinstance(scores_annual, dict)
            else np.nan
        ),
        "n_winter": (
            int(scores_winter.get("n", np.nan))
            if isinstance(scores_winter, dict)
            else np.nan
        ),
    }
    created_fig = None
    if show_plot:
        # Plot
        if ax is None:
            created_fig = plt.figure(figsize=(15, 10))
            ax = plt.subplot(1, 1, 1)

        # auto-lims if not provided
        if ax_xlim is None or ax_ylim is None:
            lo = float(np.min(test_df_preds[["target", "pred"]].min())) - 1
            hi = float(np.max(test_df_preds[["target", "pred"]].max())) + 1
            if ax_xlim is None:
                ax_xlim = (lo, hi)
            if ax_ylim is None:
                ax_ylim = (lo, hi)

        pred_vs_truth_density(
            ax,
            test_df_preds,
            scores_annual,
            add_legend=False,
            palette=[mbm.plots.COLOR_ANNUAL, mbm.plots.COLOR_WINTER],
            ax_xlim=ax_xlim,
            ax_ylim=ax_ylim,
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
            0.02,
            0.98,
            legend_NN,
            transform=ax.transAxes,
            va="top",
            fontsize=legend_fontsize,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

        if title:
            ax.set_title(title, fontsize=20)

    return out, test_df_preds, created_fig, ax


def load_one_xreg_model(
    cfg,
    region,
    best_params,
    device,
    models_dir="models",
    prefix="lstm_xreg_CH_to",
    date=None,  # if None → auto-detect latest
):
    """
    Loads one cross-regional CH→region model.
    """

    if date is None:
        # find latest file matching pattern
        pattern = f"{prefix}_{region}_"
        candidates = [
            f
            for f in os.listdir(models_dir)
            if f.startswith(pattern) and f.endswith(".pt")
        ]
        if len(candidates) == 0:
            raise FileNotFoundError(f"No checkpoint found for region {region}")

        candidates = sorted(candidates)  # last = latest by name
        filename = candidates[-1]
    else:
        filename = f"{prefix}_{region}_{date}.pt"

    path = os.path.join(models_dir, filename)

    # rebuild model
    model = mbm.models.LSTM_MB.build_model_from_params(cfg, best_params, device)

    state = torch.load(path, map_location=device)
    model.load_state_dict(state)

    return model, path


def load_xreg_models_all(
    cfg,
    regions,
    best_params,
    device,
    models_dir="models",
    prefix="lstm_xreg_CH_to",
    date=None,
):
    models = {}
    paths = {}

    for region in regions:
        try:
            model, path = load_one_xreg_model(
                cfg=cfg,
                region=region,
                best_params=best_params,
                device=device,
                models_dir=models_dir,
                prefix=prefix,
                date=date,
            )
            models[region] = model
            paths[region] = path
            print(f"Loaded CH→{region} from {path}")

        except FileNotFoundError as e:
            print(f"Skipping {region}: {e}")
            models[region] = None
            paths[region] = None

    return models, paths
