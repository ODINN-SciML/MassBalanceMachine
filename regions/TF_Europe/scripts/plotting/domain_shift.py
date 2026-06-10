from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from cmcrameri import cm
from matplotlib.ticker import FuncFormatter
import scipy.stats as scipy_stats

import pandas as pd

from regions.TF_Europe.scripts.config_TF_Europe import *
from regions.TF_Europe.scripts.plotting.palettes import *

from regions.TF_Europe.scripts.plotting.style import *


def plot_domain_shift(
    shift: dict,
    monthly_cols: list[str],
    static_cols: list[str],
    src="Iceland",
    tgt="Switzerland",
):

    var_keys = {f"D_mmd2_{col}": col for col in monthly_cols + static_cols}
    records_mmd2 = [
        (label, shift[key]) for key, label in var_keys.items() if key in shift
    ]
    records_mmd2.sort(key=lambda x: x[1], reverse=True)

    # Energy distance per variable
    var_keys_e = {f"D_energy_{col}": col for col in monthly_cols + static_cols}
    records_energy = {
        label: shift[key] for key, label in var_keys_e.items() if key in shift
    }

    labels, values_mmd2 = zip(*records_mmd2)
    values_energy = [records_energy.get(l, np.nan) for l in labels]

    radiation_vars = {"sshf", "slhf", "ssrd", "str", "fal"}
    topo_vars = {"slope", "svf", "aspect", "ELEVATION_DIFFERENCE"}

    def _color(label):
        if label in radiation_vars:
            return "#D85A30"
        if label in topo_vars:
            return "#534AB7"
        return "#888780"

    colors = [_color(l) for l in labels]

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 2.2, 2.2]}
    )

    # ── Left: summary bars ────────────────────────────────────────────────────
    ax_left = axes[0]
    summary_labels = ["joint", "climate", "topo"]
    summary_mmd2 = [
        shift["D_mmd2_joint"],
        shift["D_mmd2_climate"],
        shift["D_mmd2_topo"],
    ]
    summary_energy = [
        shift["D_energy_joint"],
        shift["D_energy_climate"],
        shift["D_energy_topo"],
    ]
    summary_colors = ["#3d3d3a", "#D85A30", "#534AB7"]

    x = np.arange(len(summary_labels))
    w = 0.35
    for i, (mv, ev, c) in enumerate(zip(summary_mmd2, summary_energy, summary_colors)):
        ax_left.barh(
            x[i] + w / 2, mv, height=w, color=c, label="MMD²" if i == 0 else ""
        )
        ax_left.barh(
            x[i] - w / 2,
            ev,
            height=w,
            color=c,
            alpha=0.45,
            label="Energy" if i == 0 else "",
        )
        ax_left.text(mv + 0.005, x[i] + w / 2, f"{mv:.3f}", va="center", fontsize=9)
        ax_left.text(ev + 0.005, x[i] - w / 2, f"{ev:.3f}", va="center", fontsize=9)

    ax_left.set_yticks(x)
    ax_left.set_yticklabels(summary_labels)
    ax_left.set_xlabel("Distance", fontsize=11)
    ax_left.set_title("Summary", fontsize=12, pad=10)
    ax_left.legend(frameon=False, fontsize=9)
    ax_left.spines[["top", "right"]].set_visible(False)
    ax_left.grid(axis="x", color="#e0e0e0", linewidth=0.6)
    ax_left.set_axisbelow(True)

    # ── Middle: per-variable MMD² ─────────────────────────────────────────────
    ax_mid = axes[1]
    y = np.arange(len(labels))
    bars = ax_mid.barh(y, values_mmd2, color=colors, height=0.6)
    ax_mid.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax_mid.set_yticks(y)
    ax_mid.set_yticklabels(labels, fontsize=10)
    ax_mid.set_xlim(0, max(values_mmd2) * 1.25)
    ax_mid.set_xlabel("MMD²", fontsize=11)
    ax_mid.set_title(f"Per-variable MMD²  ({src} → {tgt})", fontsize=12, pad=10)
    ax_mid.invert_yaxis()
    ax_mid.spines[["top", "right"]].set_visible(False)
    ax_mid.grid(axis="x", color="#e0e0e0", linewidth=0.6)
    ax_mid.set_axisbelow(True)

    # ── Right: per-variable Energy ────────────────────────────────────────────
    ax_right = axes[2]
    bars2 = ax_right.barh(y, values_energy, color=colors, height=0.6, alpha=0.7)
    ax_right.bar_label(bars2, fmt="%.3f", padding=4, fontsize=9)
    ax_right.set_yticks(y)
    ax_right.set_yticklabels(labels, fontsize=10)
    ax_right.set_xlim(0, max(values_energy) * 1.25)
    ax_right.set_xlabel("Energy distance", fontsize=11)
    ax_right.set_title(
        f"Per-variable Energy dist.  ({src} → {tgt})", fontsize=12, pad=10
    )
    ax_right.invert_yaxis()
    ax_right.spines[["top", "right"]].set_visible(False)
    ax_right.grid(axis="x", color="#e0e0e0", linewidth=0.6)
    ax_right.set_axisbelow(True)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="#D85A30", label="radiation / energy flux"),
        mpatches.Patch(color="#534AB7", label="topographic"),
        mpatches.Patch(color="#888780", label="temperature / precip"),
    ]
    ax_right.legend(
        handles=legend_handles, loc="lower right", frameon=False, fontsize=9
    )

    fig.tight_layout(w_pad=3)
    return fig


def plot_domain_shift_across_regions(all_shifts: dict, src_region: str):
    regions, mmd2_joint, mmd2_climate, mmd2_topo = [], [], [], []
    en_joint, en_climate, en_topo = [], [], []
    sk_joint, sk_climate, sk_topo = [], [], []

    for key, shift in all_shifts.items():
        region = key.split("_TO_")[-1]
        regions.append(region)
        mmd2_joint.append(shift["D_mmd2_joint"])
        mmd2_climate.append(shift["D_mmd2_climate"])
        mmd2_topo.append(shift["D_mmd2_topo"])
        en_joint.append(shift["D_energy_joint"])
        en_climate.append(shift["D_energy_climate"])
        en_topo.append(shift["D_energy_topo"])
        sk_joint.append(shift["D_sinkhorn_joint"])
        sk_climate.append(shift["D_sinkhorn_climate"])
        sk_topo.append(shift["D_sinkhorn_topo"])

    order = np.argsort(sk_joint)[::-1]
    regions = [regions[i] for i in order]
    mmd2_joint = [mmd2_joint[i] for i in order]
    mmd2_climate = [mmd2_climate[i] for i in order]
    mmd2_topo = [mmd2_topo[i] for i in order]
    en_joint = [en_joint[i] for i in order]
    en_climate = [en_climate[i] for i in order]
    en_topo = [en_topo[i] for i in order]
    sk_joint = [sk_joint[i] for i in order]
    sk_climate = [sk_climate[i] for i in order]
    sk_topo = [sk_topo[i] for i in order]

    y = np.arange(len(regions))
    h = 0.25

    colors = {
        "joint": NATURE_PALETTE["black"],
        "climate": NATURE_PALETTE["reddish_purple"],
        "topo": NATURE_PALETTE["blue"],
    }

    n_rows = max(4, len(regions))
    fig, axes = plt.subplots(
        1,
        3,
        figsize=nature_figsize(cols=2, height_mm=n_rows * 12),
        sharey=True,
    )

    def _draw_bars(ax, joint, climate, topo, xlabel, title):
        ax.barh(y + h, joint, height=h, label="Joint", color=colors["joint"])
        ax.barh(y, climate, height=h, label="Climate", color=colors["climate"])
        ax.barh(y - h, topo, height=h, label="Topo", color=colors["topo"])

        x_pad = max(joint + climate + topo) * 0.01
        fs = NATURE_SPECS["font_min_pt"]
        for i in range(len(regions)):
            ax.text(
                joint[i] + x_pad, y[i] + h, f"{joint[i]:.3f}", va="center", fontsize=fs
            )
            ax.text(
                climate[i] + x_pad, y[i], f"{climate[i]:.3f}", va="center", fontsize=fs
            )
            ax.text(
                topo[i] + x_pad, y[i] - h, f"{topo[i]:.3f}", va="center", fontsize=fs
            )

        ax.set_yticks(y)
        ax.set_yticklabels(regions)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=NATURE_SPECS["font_min_pt"])
        apply_nature_style(ax)

    _draw_bars(
        axes[0],
        sk_joint,
        sk_climate,
        sk_topo,
        xlabel="Sinkhorn distance",
        title="Sinkhorn",
    )
    _draw_bars(
        axes[1],
        mmd2_joint,
        mmd2_climate,
        mmd2_topo,
        xlabel="MMD² distance",
        title="MMD²",
    )
    _draw_bars(
        axes[2],
        en_joint,
        en_climate,
        en_topo,
        xlabel="Energy distance",
        title="Energy distance",
    )

    fig.suptitle(
        f"Domain shift: {src_region} → other regions",
        fontsize=NATURE_SPECS["font_max_pt"] + 1,
        y=1.01,
    )
    plt.tight_layout()
    return fig


def plot_region_shift_vs_performance_single_d(
    df_metrics: pd.DataFrame,
    all_shifts: dict,
    complement_key: str = "",
    performance_cols: list[str] | None = None,
    distance_variant: str = "mmd2",
    exclude_targets: list[str] | None = None,
    exclude_sources: list[str] | None = None,
    blur_m: float | None = None,
    blur_s: float | None = None,
    blur_joint: float | None = None,
    joint_variant: str = "averaged",
    distance_cols_override: list[str] | None = None,  # e.g. ["D_sinkhorn_joint"]
    ax_list: list | None = None,  # pass existing axes to skip fig creation
    color_palette: dict | None = None,
    panel_titles: dict | None = None,
    suptitle: str = "Region-level domain shift vs transfer performance",
):
    from scipy import stats

    if joint_variant not in {"averaged", "true"}:
        raise ValueError("joint_variant must be 'averaged' or 'true'.")

    exclude_targets = {t.upper() for t in (exclude_targets or [])}
    exclude_sources = {s.upper() for s in (exclude_sources or [])}

    if performance_cols is None:
        performance_cols = [
            c
            for c in df_metrics.columns
            if any(kw in c.lower() for kw in ["rmse", "bias"])
        ]
        if not performance_cols:
            raise ValueError(
                f"No performance columns auto-detected in df_metrics.\n"
                f"Available columns: {list(df_metrics.columns)}\n"
                f"Pass performance_cols= explicitly."
            )
        print(f"Auto-detected performance columns: {performance_cols}")

    if joint_variant == "true" and distance_variant == "sinkhorn":
        joint_col = "D_sinkhorn_joint"
        joint_label = "sinkhorn joint (true)"
    else:
        if joint_variant == "true":
            print(
                f"Warning: joint_variant='true' only available for sinkhorn, "
                f"falling back to 'averaged' for {distance_variant}."
            )
        joint_col = f"D_{distance_variant}_joint"
        joint_label = f"{distance_variant} joint (averaged)"

    all_distance_cols = [
        joint_col,
        f"D_{distance_variant}_climate",
        f"D_{distance_variant}_topo",
    ]
    all_xlabel_map = {
        joint_col: joint_label,
        f"D_{distance_variant}_climate": f"{distance_variant} climate",
        f"D_{distance_variant}_topo": f"{distance_variant} topo",
    }

    # restrict to requested distance cols if provided
    distance_cols = (
        distance_cols_override
        if distance_cols_override is not None
        else all_distance_cols
    )
    xlabel_map = {k: v for k, v in all_xlabel_map.items() if k in distance_cols}

    blur_map = {}
    if distance_variant == "sinkhorn":
        if blur_m is not None and blur_s is not None:
            blur_map["D_sinkhorn_joint"] = 0.5 * (blur_m + blur_s)
            blur_map["D_sinkhorn_climate"] = blur_m
            blur_map["D_sinkhorn_topo"] = blur_s
        if blur_joint is not None:
            blur_map["D_sinkhorn_joint"] = blur_joint

    # --- build flat region DataFrame ---
    records = []
    for full_key in df_metrics.index:
        shift_key = f"{complement_key}{full_key}" if complement_key else full_key
        if shift_key not in all_shifts:
            print(f"Warning: '{shift_key}' not in all_shifts, skipping.")
            continue

        parts = shift_key.split("_TO_")
        src = parts[0].replace("XREG_", "")
        tgt = parts[1] if len(parts) > 1 else shift_key

        if tgt.upper() in exclude_targets:
            continue
        if src.upper() in exclude_sources:
            continue

        shift = all_shifts[shift_key]
        row = {
            "full_key": full_key,
            "region": f"{src}→{tgt}",
            "src": src,
            "tgt": tgt,
        }
        for pc in performance_cols:
            row[pc] = df_metrics.loc[full_key, pc]
        for dc in all_distance_cols:  # always fetch all so override works
            row[dc] = shift.get(dc, float("nan"))
        records.append(row)

    if not records:
        raise ValueError(
            "No matching regions between df_metrics and all_shifts.\n"
            f"df_metrics index: {list(df_metrics.index)}\n"
            f"all_shifts keys:  {list(all_shifts.keys())}"
        )

    df_region = pd.DataFrame(records)
    print(f"Plotting {len(df_region)} source→target pairs: {list(df_region['region'])}")

    unique_sources = df_region["src"].unique()
    if color_palette is not None:
        src_colors = {
            s: color_palette.get(s, NATURE_COLORS[i % len(NATURE_COLORS)])
            for i, s in enumerate(unique_sources)
        }
    else:
        src_colors = {s: c for s, c in zip(unique_sources, NATURE_COLORS)}
    nrows = len(performance_cols)
    ncols = len(distance_cols)

    if ax_list is not None:
        axes = np.array(ax_list).reshape(nrows, ncols)
        fig = axes.flat[0].get_figure()
    else:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=nature_figsize(cols=2, height_mm=80 * nrows),
            squeeze=False,
            sharey="row",
        )

    for r, pc in enumerate(performance_cols):
        for c, dc in enumerate(distance_cols):
            ax = axes[r][c]

            x = df_region[dc].values.astype(float)
            y = df_region[pc].values.astype(float)
            mask = np.isfinite(x) & np.isfinite(y)

            for _, row in df_region.iterrows():
                xi = float(row[dc])
                yi = float(row[pc])
                if not (np.isfinite(xi) and np.isfinite(yi)):
                    continue
                ax.scatter(
                    xi,
                    yi,
                    color=src_colors[row["src"]],
                    s=80,  # slightly smaller, more Nature-appropriate
                    zorder=3,
                    edgecolors="white",
                    linewidths=0.4,  # was 0.6
                )
                x_range = x[mask].max() - x[mask].min() if mask.sum() > 1 else 1
                x_frac = (xi - x[mask].min()) / x_range if x_range > 0 else 0
                xytext, ha = ((-6, 4), "right") if x_frac > 0.7 else ((6, 4), "left")
                ax.annotate(
                    row["region"],
                    (xi, yi),
                    xytext=xytext,
                    textcoords="offset points",
                    fontsize=NATURE_SPECS["font_min_pt"],  # was 9
                    fontweight="bold",
                    color=src_colors[row["src"]],
                    ha=ha,
                )

            n_valid = mask.sum()
            if n_valid >= 3:
                rho, pval = stats.spearmanr(x[mask], y[mask])
                if dc in blur_map:
                    corr_txt = (
                        f"rho = {rho:.2f}\nblur = {blur_map[dc]:.3f}\nn = {n_valid}"
                    )
                else:
                    corr_txt = f"rho = {rho:.2f}\np = {pval:.2f}\nn = {n_valid}"
                slope, intercept, *_ = stats.linregress(x[mask], y[mask])
                x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                ax.plot(
                    x_line,
                    slope * x_line + intercept,
                    color="black",
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.3,
                    zorder=2,
                )
            else:
                corr_txt = f"n = {n_valid}\n(too few for rho)"

            ax.text(
                0.04,
                0.97,
                corr_txt,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=NATURE_SPECS["font_min_pt"] + 2,  # ← bigger
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8),
            )

            # ax.grid(color="#e0e0e0", linewidth=0.5)
            # ax.spines[["top", "right"]].set_visible(False)
            # ax.set_axisbelow(True)
            apply_nature_style(ax, fontsize=NATURE_SPECS["font_min_pt"], box=True)

            # Optional per-panel title
            if panel_titles is not None and c < len(panel_titles):
                ax.set_title(panel_titles[c], fontsize=8, fontweight="bold", pad=3)
            XLABEL_MAP = {
                "sinkhorn joint (true)": "Sinkhorn Distance (Joint)",
                "sinkhorn climate": "Sinkhorn Distance (Climate)",
                "sinkhorn topo": "Sinkhorn Distance (Topo)",
                "mmd2 joint (averaged)": "MMD² Distance (Joint)",
                "mmd2 climate": "MMD² Distance (Climate)",
                "mmd2 topo": "MMD² Distance (Topo)",
                "energy joint (averaged)": "Energy Distance (Joint)",
                "energy climate": "Energy Distance (Climate)",
                "energy topo": "Energy Distance (Topo)",
            }
            ax.set_xlabel(
                XLABEL_MAP.get(xlabel_map.get(dc, dc), xlabel_map.get(dc, dc)),
                fontsize=NATURE_SPECS["font_min_pt"],
            )
            YLABEL_MAP = {
                "RMSE_annual": "Annual RMSE (m w.e.)",
                "RMSE_winter": "Winter RMSE (m w.e.)",
            }
            ax.set_ylabel(YLABEL_MAP.get(pc, pc), fontsize=NATURE_SPECS["font_min_pt"])

    legend_handles = [
        plt.scatter([], [], color=src_colors[s], s=80, label=s) for s in unique_sources
    ]
    fig.legend(
        handles=legend_handles,
        title="Source region",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncols=len(unique_sources),
        frameon=False,
    )

    joint_desc = "true joint" if joint_variant == "true" else "averaged joint"
    fig.suptitle(
        suptitle,
        fontsize=NATURE_SPECS["font_max_pt"] + 1,  # was 13
        y=1.01,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    return fig, df_region


def format_axis_ticks(ax, label_size=8):
    """Format tick labels to avoid huge 1e6/1e7 offset labels."""
    # check if scientific notation offset is being used
    ax.xaxis.get_major_formatter().set_useOffset(False)
    try:
        # scale large numbers to readable units
        xmax = abs(ax.get_xlim()[1])
        if xmax > 1e6:
            scale = 1e6
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/scale:.1f}"))
            ax.set_xlabel(f"(×10⁶)", label_size=label_size, labelpad=1)
        elif xmax > 1e4:
            scale = 1e3
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/scale:.0f}"))
            ax.set_xlabel(f"(×10³)", label_size=label_size, labelpad=1)
    except Exception:
        pass


def plot_kde_pair(glaciers_to_plot, selected_cols, save_prefix):
    """KDE panels for a pair of glaciers."""
    ncols = 3
    nrows = int(np.ceil(len(selected_cols) / ncols))
    w, h = nature_figsize(cols=1, height_mm=160)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * 2, h), squeeze=False)

    legend_handles = []

    for idx, col in enumerate(selected_cols):
        ax = axes[idx // ncols][idx % ncols]

        all_vals = pd.concat(
            [cfg_gl["df"][col].dropna() for cfg_gl in glaciers_to_plot.values()]
        )
        x_grid = np.linspace(float(all_vals.min()), float(all_vals.max()), 500)

        for label, cfg_gl in glaciers_to_plot.items():
            vals = cfg_gl["df"][col].dropna().values
            if len(vals) < 10:
                continue
            kde = scipy_stats.gaussian_kde(vals, bw_method=0.3)
            y = kde(x_grid)
            y = y / y.max()
            (line,) = ax.plot(
                x_grid, y, color=cfg_gl["color"], linewidth=0.8, label=label
            )
            ax.fill_between(x_grid, y, alpha=0.08, color=cfg_gl["color"])

            # collect handles only from first panel to avoid duplicates
            if idx == 0:
                legend_handles.append(line)

        ax.set_title(col, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel("")
        ax.tick_params(labelsize=8, width=0.4, length=2, direction="in")
        ax.spines[["top", "right", "left", "bottom"]].set_visible(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
        ax.grid(axis="x", color="#e0e0e0", linewidth=0.3)
        ax.set_axisbelow(True)
        format_axis_ticks(ax, label_size=6)

    # ── place legend in first empty axis if one exists, else above figure ──
    empty_axes = [
        axes[idx // ncols][idx % ncols]
        for idx in range(len(selected_cols), nrows * ncols)
    ]

    if empty_axes:
        leg_ax = empty_axes[0]
        leg_ax.axis("off")
        leg_ax.legend(
            handles=legend_handles,
            loc="center",
            fontsize=8,
            frameon=False,
        )
        # turn off remaining empty axes
        for ax in empty_axes[1:]:
            ax.axis("off")
    else:
        # no empty panels — place legend above the figure
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(glaciers_to_plot),
            fontsize=8,
            frameon=False,
        )

    plt.tight_layout(h_pad=3.0)
    plt.savefig(f"figures/paperTF/{save_prefix}_kde.pdf", bbox_inches="tight")
    plt.savefig(f"figures/paperTF/{save_prefix}_kde.png", dpi=300, bbox_inches="tight")
    plt.show()
