import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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
    linear_fit_breaks=None,
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

        month_to_id = {
            month_abbr[i].lower() + ("_" if i > 9 else ""): i for i in range(1, 13)
        }  # Since we are working with calendar years, the format is jan,..,sep,oct_,..dec_

        # df_gl["MONTH_ID"] = df_gl.apply(
        #     lambda x: get_hash(f"{x.RGIId}_{x.YEAR}_{x.MONTHS}"),
        #     axis=1,
        # ).astype(str)
        df_gl["MONTH_ID"] = df_gl.YEAR * 12 + df_gl["MONTHS"].map(
            month_to_id
        )  # Computing a unique ID per month this way is much faster than using apply and get_hash
        assert not df_gl[
            "MONTH_ID"
        ].hasnans, "The resulting MONTH_ID column contains NaNs. Check the month convention, especially that the gridded products are generated with calendar years."
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

        if geo is not None and test_gl in geo:
            # Filter monthly_df to keep only predictions inside the geodetic time window
            start_year = geo[test_gl]["start"]
            end_year = geo[test_gl]["end"]
            if isinstance(start_year, np.datetime64):
                start_year = (pd.Timestamp(start_year).month - 1) / 12 + pd.Timestamp(
                    start_year
                ).year
            if isinstance(end_year, np.datetime64):
                end_year = (pd.Timestamp(end_year).month - 1) / 12 + pd.Timestamp(
                    end_year
                ).year
            monthly_df = monthly_df[
                (monthly_df.time >= start_year) & (monthly_df.time <= end_year)
            ]

        monthly_df = monthly_df.sort_values(by="time")
        t = monthly_df.time.values
        y = monthly_df.pred.values
        begin_t = monthly_df.time.min() - 1 / 12
        end_t = monthly_df.time.max()
        window_width = end_t - begin_t
        t = np.concatenate([[begin_t], t])
        y = np.concatenate([[0.0], y])
        (line,) = ax.plot(t, np.cumsum(y), color=color_pred)

        if linear_fit_breaks is not None:
            bounds = [t[0]] + linear_fit_breaks + [t[-1]]
            std = np.std(y)
            c = np.cumsum(y)
            for i in range(len(bounds) - 1):
                ind_start = np.argwhere(t >= bounds[i])[0, 0]
                ind_end = np.argwhere(t <= bounds[i + 1])[-1, 0]
                ti = t[ind_start : ind_end + 1]
                ci = c[ind_start : ind_end + 1]
                coef = np.polyfit(ti, ci, 1)
                poly1d_fn = np.poly1d(coef)
                x = [ti[0], ti[-1]]
                (line,) = ax.plot(x, poly1d_fn(x), linestyle="--", color=color_pred)

                # Add slope label above the midpoint of the segment
                x_mid = (ti[0] + ti[-1]) / 2
                y_mid = poly1d_fn(x_mid)
                slope = coef[0]
                ax.text(
                    x_mid,
                    y_mid + 5 * std,
                    f"{slope:.2f}",
                    color=line.get_color(),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        nyear = monthly_df.YEAR.nunique()
        if geo is not None and test_gl in geo and "mean" in geo[test_gl]:
            tgt = geo[test_gl]["mean"]
            err = geo[test_gl]["err"]
            years = [begin_t, end_t]
            ax.plot(years, [0, tgt * window_width], color=color_obs)
            ax.fill_between(
                years,
                [0, (tgt - 2 * err) * window_width],
                [0, (tgt + 2 * err) * window_width],
                color=color_obs,
                alpha=0.3,
            )

        ax.grid()

        glacier_title = titles.get(test_gl) if titles is not None else None
        ax.set_title(glacier_title or test_gl.capitalize(), fontsize=20)

        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        step_years_xticks = nyear // 10 if nyear >= 10 else 1
        ax.set_xticks(
            np.arange(
                int(start_year),
                int(start_year) + nyear + step_years_xticks,
                step_years_xticks,
            )
        )
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    # Remove unused axes
    for i in range(len(custom_order), len(axs)):
        if isinstance(axs, list):
            ax = axs[i]
        else:
            ax = axs.flatten()[i]
        ax.set_visible(False)

    # # Set axes limits
    # if ax_xlim is not None:
    #     ax.set_xlim(ax_xlim)
    # if ax_ylim is not None:
    #     ax.set_ylim(ax_ylim)

    plt.tight_layout()

    return fig, line
