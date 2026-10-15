import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import time

from data_processing.utils.data_preprocessing import get_hash
from data_processing.gridded_utils import MONTH_TO_ID, _window_month_bounds


def window_cumulated_pred(monthly_df, lo, hi):
    """Cumulated glacier-wide MB over the months [lo, hi[ only, starting from zero.

    Args:
        monthly_df (pd.DataFrame): glacier-wide monthly predictions of one glacier,
            with the columns MONTH_INDEX (months since year 0) and pred.
        lo, hi: first month and one past the last month, as returned by
            `data_processing.gridded_utils._window_month_bounds`.

    Returns the times in decimal years and the cumulated MB. The first point is (lo /
    12, 0): the start of the window. Every other point is the end of a month, so that
    the last one of a window is at hi / 12, where the observed cumulated MB of the
    window is compared.
    """
    sel = monthly_df[
        (monthly_df.MONTH_INDEX >= lo) & (monthly_df.MONTH_INDEX < hi)
    ].sort_values("MONTH_INDEX")
    t = np.concatenate([[lo / 12], (sel.MONTH_INDEX.to_numpy() + 1) / 12])
    c = np.concatenate([[0.0], np.cumsum(sel.pred.to_numpy())])
    return t, c


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

        # Computing a unique ID per month this way is much faster than using apply and
        # get_hash. Since we are working with calendar years, the format is
        # jan,..,sep,oct_,..dec_
        df_gl["MONTH_INDEX"] = df_gl.YEAR * 12 + df_gl["MONTHS"].map(MONTH_TO_ID) - 1
        assert not df_gl[
            "MONTH_INDEX"
        ].hasnans, "The resulting MONTH_INDEX column contains NaNs. Check the month convention, especially since the gridded products are generated with calendar years."
        monthly_df = df_gl.groupby("MONTH_INDEX", as_index=False).agg(
            {
                "RGIId": "first",
                "YEAR": "first",
                "MONTHS": "first",
                "pred": "mean",
            }
        )

        has_geo = geo is not None and test_gl in geo
        if has_geo:
            # A glacier can have several geodetic windows: "start", "end", "mean" and
            # "err" are then sequences with one entry per window, and a scalar
            # otherwise
            starts = np.atleast_1d(geo[test_gl]["start"])
            ends = np.atleast_1d(geo[test_gl]["end"])
            bounds = [_window_month_bounds(w) for w in zip(starts, ends)]
            if "mean" in geo[test_gl]:
                targets = list(
                    zip(
                        np.atleast_1d(geo[test_gl]["mean"]),
                        np.atleast_1d(geo[test_gl]["err"]),
                    )
                )
            else:
                targets = [None] * len(bounds)
            # Every window is compared on its own: the cumulated MB, predicted and
            # observed, starts back from zero at the beginning of each window. This
            # is what the geodetic loss sees, since the rate of a window is computed
            # from its own months only (see
            # `data_processing.gridded_utils.geodetic_window_weights`)
            windows = sorted(zip(bounds, targets), key=lambda w: w[0])
        else:
            # Without geodetic windows the whole period is cumulated
            windows = [
                (
                    (
                        monthly_df.MONTH_INDEX.min(),
                        monthly_df.MONTH_INDEX.max() + 1,
                    ),
                    None,
                )
            ]
        in_a_window = np.zeros(len(monthly_df), dtype=bool)
        for (lo, hi), _ in windows:
            in_a_window |= (monthly_df.MONTH_INDEX >= lo) & (
                monthly_df.MONTH_INDEX < hi
            )
        monthly_df = monthly_df[in_a_window]
        std = np.std(monthly_df.pred.to_numpy())

        for (lo, hi), target in windows:
            t, c = window_cumulated_pred(monthly_df, lo, hi)
            (line,) = ax.plot(t, c, color=color_pred)

            if linear_fit_breaks is not None:
                # The slopes are fitted inside a window only, since the cumulated MB
                # starts back from zero in the next one
                breaks = [t[0]] + [b for b in linear_fit_breaks if t[0] < b < t[-1]]
                breaks += [t[-1]]
                for b_start, b_end in zip(breaks[:-1], breaks[1:]):
                    ind_start = np.argwhere(t >= b_start)[0, 0]
                    ind_end = np.argwhere(t <= b_end)[-1, 0]
                    ti = t[ind_start : ind_end + 1]
                    ci = c[ind_start : ind_end + 1]
                    if len(ti) < 2:
                        continue
                    coef = np.polyfit(ti, ci, 1)
                    poly1d_fn = np.poly1d(coef)
                    x = [ti[0], ti[-1]]
                    (fit_line,) = ax.plot(
                        x, poly1d_fn(x), linestyle="--", color=color_pred
                    )

                    # Add slope label above the midpoint of the segment
                    x_mid = (ti[0] + ti[-1]) / 2
                    y_mid = poly1d_fn(x_mid)
                    slope = coef[0]
                    ax.text(
                        x_mid,
                        y_mid + 5 * std,
                        f"{slope:.2f}",
                        color=fit_line.get_color(),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

            if target is not None:
                tgt, err = target
                years = [lo / 12, hi / 12]
                width = (hi - lo) / 12
                ax.plot(years, [0.0, tgt * width], color=color_obs)
                # The band reflects the 2 sigma uncertainty of this window
                ax.fill_between(
                    years,
                    [0.0, (tgt - 2 * err) * width],
                    [0.0, (tgt + 2 * err) * width],
                    color=color_obs,
                    alpha=0.3,
                )

        nyear = monthly_df.YEAR.nunique()
        begin_t = min(lo for (lo, _), _ in windows) / 12

        ax.grid()

        glacier_title = titles.get(test_gl) if titles is not None else None
        ax.set_title(glacier_title or test_gl.capitalize(), fontsize=20)

        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        step_years_xticks = nyear // 10 if nyear >= 10 else 1
        ax.set_xticks(
            np.arange(
                int(begin_t),
                int(begin_t) + nyear + step_years_xticks,
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
