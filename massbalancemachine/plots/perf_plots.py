import matplotlib.pyplot as plt
import seaborn as sns


def predVSTruth_all(grouped_ids, mae, rmse, title, modelName="nn"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    legend_nn = "\n".join(
        (
            r"$\mathrm{MAE_{%s}}=%.3f, \mathrm{RMSE_{%s}}=%.3f$ "
            % (
                modelName,
                mae,
                modelName,
                rmse,
            ),
        )
    )

    marker_nn = "o"
    sns.scatterplot(
        grouped_ids, x="target", y="pred", ax=ax, alpha=0.5, marker=marker_nn
    )

    ax.set_ylabel("Predicted PMB [m w.e.]", fontsize=20)
    ax.set_xlabel("Observed PMB [m w.e.]", fontsize=20)

    ax.text(
        0.03,
        0.98,
        legend_nn,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=20,
    )
    ax.legend([], [], frameon=False)
    # diagonal line
    pt = (0, 0)
    ax.axline(pt, slope=1, color="grey", linestyle="-", linewidth=0.2)
    ax.axvline(0, color="grey", linestyle="-", linewidth=0.2)
    ax.axhline(0, color="grey", linestyle="-", linewidth=0.2)
    ax.grid()
    ax.set_title(title, fontsize=20)
    plt.tight_layout()

    return fig  # To log figure during training
