from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


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


def plot_domain_shift_across_regions(all_shifts: dict):
    """
    Plot domain shift across regions as two side-by-side horizontal bar charts:
      - Subplot 1: MMD² (joint, climate, topo), ordered by joint MMD²
      - Subplot 2: Energy distance (joint, climate, topo), same region order

    Parameters
    ----------
    all_shifts : dict
        Keys like "XREG_CH_TO_ISL", values are shift dicts from
        compute_domain_shift (must contain D_mmd2_* and D_energy_* keys).

    Returns
    -------
    matplotlib Figure
    """
    regions, mmd2_joint, mmd2_climate, mmd2_topo = [], [], [], []
    en_joint, en_climate, en_topo = [], [], []

    for key, shift in all_shifts.items():
        region = key.split("_TO_")[-1]
        regions.append(region)
        mmd2_joint.append(shift["D_mmd2_joint"])
        mmd2_climate.append(shift["D_mmd2_climate"])
        mmd2_topo.append(shift["D_mmd2_topo"])
        en_joint.append(shift["D_energy_joint"])
        en_climate.append(shift["D_energy_climate"])
        en_topo.append(shift["D_energy_topo"])

    # --- sort by MMD² joint (most shifted first), apply same order to energy ---
    order = np.argsort(mmd2_joint)[::-1]
    regions = [regions[i] for i in order]
    mmd2_joint = [mmd2_joint[i] for i in order]
    mmd2_climate = [mmd2_climate[i] for i in order]
    mmd2_topo = [mmd2_topo[i] for i in order]
    en_joint = [en_joint[i] for i in order]
    en_climate = [en_climate[i] for i in order]
    en_topo = [en_topo[i] for i in order]

    y = np.arange(len(regions))
    h = 0.25

    colors = {
        "joint": "#3d3d3a",
        "climate": "#D85A30",
        "topo": "#534AB7",
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, max(4, len(regions) * 0.8)),
        sharey=True,
    )

    def _draw_bars(ax, joint, climate, topo, xlabel, title):
        ax.barh(y + h, joint, height=h, label="Joint", color=colors["joint"])
        ax.barh(y, climate, height=h, label="Climate", color=colors["climate"])
        ax.barh(y - h, topo, height=h, label="Topo", color=colors["topo"])

        # value annotations
        x_pad = max(joint + climate + topo) * 0.01
        for i in range(len(regions)):
            ax.text(
                joint[i] + x_pad, y[i] + h, f"{joint[i]:.3f}", va="center", fontsize=8
            )
            ax.text(
                climate[i] + x_pad, y[i], f"{climate[i]:.3f}", va="center", fontsize=8
            )
            ax.text(
                topo[i] + x_pad, y[i] - h, f"{topo[i]:.3f}", va="center", fontsize=8
            )

        ax.set_yticks(y)
        ax.set_yticklabels(regions)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.grid(axis="x", color="#e0e0e0", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    _draw_bars(
        axes[0],
        mmd2_joint,
        mmd2_climate,
        mmd2_topo,
        xlabel="MMD² distance",
        title="MMD²",
    )
    _draw_bars(
        axes[1],
        en_joint,
        en_climate,
        en_topo,
        xlabel="Energy distance",
        title="Energy distance",
    )

    fig.suptitle(
        "Domain shift: Switzerland → other regions",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    return fig
