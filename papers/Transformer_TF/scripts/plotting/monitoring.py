import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_sweep_Y_by_G_colormap(
    df,
    E_ZERO_a,
    E_FULL_a,
    E_ZERO_w,
    E_FULL_w,
    E_ZERO,
    E_FULL,
    metric="annual",
    band=(0.25, 0.75),
    title="Monitoring performance vs years monitored",
    ax=None,
    show_legend=True,
):
    if metric == "annual":
        rmse_col = "RMSE_annual"
        E0, EF = E_ZERO_a, E_FULL_a
        ylab = "RMSE_annual"
    elif metric == "winter":
        rmse_col = "RMSE_winter"
        E0, EF = E_ZERO_w, E_FULL_w
        ylab = "RMSE_winter"
    elif metric == "mean":
        rmse_col = "RMSE_TL"
        E0, EF = E_ZERO, E_FULL
        ylab = "RMSE (mean annual+winter)"
    else:
        raise ValueError

    qlo, qhi = band

    g = (
        df.groupby(["G", "Y"])[rmse_col]
        .agg(
            med="median",
            qlo=lambda s: np.quantile(s, qlo),
            qhi=lambda s: np.quantile(s, qhi),
            n=("size"),
        )
        .reset_index()
    )

    G_vals = np.array(sorted(g["G"].unique()))
    norm = mpl.colors.Normalize(vmin=G_vals.min(), vmax=G_vals.max())
    cmap = mpl.cm.viridis_r

    if ax is None:
        fig = plt.figure(figsize=(8.2, 5.8))
        ax = plt.subplot(1, 1, 1)
    else:
        fig = ax.figure

    for G_val in G_vals:
        sub = g[g["G"] == G_val].sort_values("Y")
        color = cmap(norm(G_val))

        ax.plot(
            sub["Y"],
            sub["med"],
            marker="o",
            linewidth=2,
            color=color,
            label=f"G={G_val}",
        )
        ax.fill_between(
            sub["Y"],
            sub["qlo"],
            sub["qhi"],
            alpha=0.15,
            color=color,
        )

    # baselines
    ax.axhline(
        E0, linestyle="--", color="black", linewidth=1, label="E_ZERO (no transfer)"
    )
    ax.axhline(
        EF, linestyle=":", color="black", linewidth=1.5, label="E_FULL (max monitoring)"
    )

    ax.set_xlabel("Y (years monitored)")
    ax.set_ylabel(f"{ylab} (median + {int(qlo*100)}–{int(qhi*100)}% band)")
    ax.set_title(title)

    # legend (multi-column to reduce clutter)
    if show_legend:
        ax.legend(
            loc="upper right",
            ncol=2,
            fontsize=9,
            frameon=False,
        )

    # optional colorbar (kept for gradient meaning)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("G (number of glaciers)")

    return fig, ax
