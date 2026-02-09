import numpy as np
import matplotlib.pyplot as plt
from calendar import month_abbr
import time

from data_processing.utils.data_preprocessing import get_hash


def cumulatedMassChange(
    df_gridded,
    geo=None,
    axs=None,
    titles={},
    custom_order=None,
    xlabel="Time",
    ylabel="Cumulated MB [m w.e.]",
    ax_xlim=None,
    ax_ylim=None,
    color_pred="blue",
    color_obs="black",
):

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
        df_gl = df_gridded[df_gridded[order_key] == test_gl].copy()

        if isinstance(axs, list):
            ax = axs[i]
        else:
            ax = axs.flatten()[i]

        month_to_id = {month_abbr[i].lower(): i for i in range(1, 13)}
        # df_gl["MONTH_ID"] = df_gl.apply(
        #     lambda x: get_hash(f"{x.RGIId}_{x.YEAR}_{x.MONTHS}"),
        #     axis=1,
        # ).astype(str)
        df_gl["MONTH_ID"] = df_gl.YEAR * 12 + df_gl["MONTHS"].map(
            month_to_id
        )  # Computing a unique ID per month this way is much faster than using apply and get_hash
        monthly_df = df_gl.groupby("MONTH_ID").agg(
            {
                "RGIId": "first",
                "YEAR": "first",
                "MONTHS": "first",
                "pred": "mean",
            }
        )
        month_id = monthly_df["MONTHS"].map(month_to_id)
        monthly_df["time"] = month_id / 12 + monthly_df["YEAR"]
        monthly_df = monthly_df.sort_values(by="time")
        t = monthly_df.time.values
        y = monthly_df.pred.values
        first_year = np.sort(monthly_df.YEAR.unique())[0]
        t = np.concatenate([[first_year], t])
        y = np.concatenate([[0.0], y])
        ax.plot(t, np.cumsum(y), color=color_pred)

        nyear = monthly_df.YEAR.nunique()
        if geo is not None and test_gl in geo:
            tgt = geo[test_gl]["mean"]
            err = geo[test_gl]["err"]
            years = [first_year, first_year + nyear]
            ax.plot(years, [0, tgt * nyear], color=color_obs)
            ax.fill_between(
                years,
                [0, (tgt - 2 * err) * nyear],
                [0, (tgt + 2 * err) * nyear],
                color=color_obs,
                alpha=0.3,
            )
            # ax.errorbar(
            #     first_year+nyear,
            #     tgt*nyear,
            #     yerr=err*nyear,
            #     fmt="o",
            # )

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
