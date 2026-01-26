import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def profilePerGlacier(
    df_gridded,
    df_stakes=None,
    axs=None,
    titles={},
    custom_order=None,
    bin_width=100,
    band_alpha=0.25,
    lw=1.2,
    mean_linestyle="-",
    title="",
):
    assert "POINT_ELEVATION" in df_gridded.columns
    if df_stakes is not None:
        assert False, "Not supported yet"
        assert "POINT_ELEVATION" in df_stakes.columns

    order_key = "GLACIER" if "GLACIER" in df_gridded.keys() else "RGIId"
    custom_order = custom_order or sorted(df_gridded[order_key].unique())

    if axs is None:
        N = len(custom_order)
        n = np.sqrt(N / 2.0)
        nRows = int(np.ceil(n))  # Scales as 2n
        nCols = int(np.floor(N / nRows))  # Scales as n
        if nCols * nRows < N:
            nCols += 1
        fig, axs = plt.subplots(
            nRows, nCols, figsize=(20 * nCols / 3, 30 * nRows / 8), sharex=False
        )
    else:
        fig = None

    for i, test_gl in enumerate(custom_order):
        df_gl = df_gridded[df_gridded[order_key] == test_gl]
        if df_stakes is not None:
            df_gl_stakes = df_stakes[df_stakes[order_key] == test_gl]
        else:
            df_gl_stakes = None

        ax = axs.flatten()[i]

        nbins = int(
            np.ceil(
                (df_gl["POINT_ELEVATION"].max() - df_gl["POINT_ELEVATION"].min())
                / bin_width
            )
        )
        center = (df_gl["POINT_ELEVATION"].max() + df_gl["POINT_ELEVATION"].min()) / 2
        start = center - bin_width * nbins / 2
        stop = center + bin_width * nbins / 2
        bins = np.linspace(start, stop, nbins + 1)

        df_gl["altitude_interval"] = pd.cut(df_gl["POINT_ELEVATION"], bins=bins)
        centers = {
            iv: round((iv.left + iv.right) / 2)
            for iv in df_gl["altitude_interval"].cat.categories
        }
        df_gl["altitude_interval"] = df_gl["altitude_interval"].map(centers)

        altitude_interval = (
            df_gl.groupby(["altitude_interval"])["altitude_interval"].first().values
        )
        mean_per_bin = df_gl.groupby(["altitude_interval"])["pred"].mean().values
        min_per_bin = df_gl.groupby(["altitude_interval"])["pred"].min().values
        max_per_bin = df_gl.groupby(["altitude_interval"])["pred"].max().values
        ax.fill_betweenx(
            altitude_interval,
            min_per_bin,
            max_per_bin,
            # color=color_annual,
            alpha=band_alpha,
            # label=f"{label_prefix} band (annual)",
        )
        ax.plot(
            mean_per_bin,
            altitude_interval,
            # color=color_annual,
            linestyle=mean_linestyle,
            linewidth=lw,
            # label=f"{label_prefix} mean (annual)",
        )

        ax.grid()

        glacier_title = titles.get(test_gl) if titles is not None else None
        ax.set_title(glacier_title or test_gl.capitalize(), fontsize=20)

    # # Set axes limits
    # if ax_xlim is not None:
    #     ax.set_xlim(ax_xlim)
    # if ax_ylim is not None:
    #     ax.set_ylim(ax_ylim)

    plt.tight_layout()

    return fig
