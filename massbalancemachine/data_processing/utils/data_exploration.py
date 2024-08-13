"""
These functions are specifically for the data exploration phase; for users to gain more insights in the stake measurements
and their recordings. In addition, one can plot the cumulative point surface mass balance for the region of interest.
More kinds of figures are to come in the future that explore and visualize the available data.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 21/07/2024
"""

import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_stake_timeseries(df: pd.DataFrame, save_img=None, stakeIDs=None) -> None:
    """
    Plot the timeseries of each individual stake available in the dataset. Also, the mean and standard deviation of
    the stakes are plotted. The user can highlight a selection of stakes if desired.

    Args:
        df(pandas.Dataframe): All available stakes in the dataset for the region of interest with a monthly time resolution.
        save_img(list, optional): A list of details for saving the figure accordingly
        [image format (.svg/.png/.jpg/...), directory of images to save the figure]
        stakeIDs(list, optional): A list of a single or multiple stakes that the user want to be highlighted in the figure if desired.

    Returns:
        -

    """
    # Get the stake and stats timeseries data
    stake_timeseries = _get_stake_timeseries(df)
    stats_timeseries = _get_stats_stake_timeseries(df)

    # Create the figure and axis for plotting
    _, ax = plt.subplots(figsize=(16, 8))

    # Plot a zero line for reference
    years_range = np.arange(
        np.min(stats_timeseries.YEAR), np.max(stats_timeseries.YEAR) + 1
    )
    ax.plot(years_range, np.zeros_like(years_range), color="#2e2e2e", linewidth=1)

    # Set up markers for different stakes
    markers = itertools.cycle(
        ("o", ".", "v", "^", "<", ">", "8", "s", "p", "P", "d", "X", "+", "*")
    )

    # Define line styles based on conditions
    def get_plot_params(key, stakeIDs):
        if stakeIDs is not None:
            return (0.75, 1.1) if key in stakeIDs else (0.25, 0.9)
        return 0.75, 0.9

    # Plot each stake's data
    for _, row in stake_timeseries.iterrows():
        key = row["POINT_ID"]
        value = row["data"]
        alpha, linewidth = get_plot_params(key, stakeIDs)
        plt.plot(
            value["YEARS"],
            value["POINT_BALANCES"],
            marker=next(markers),
            markevery=1,
            markersize=4,
            alpha=alpha,
            linewidth=linewidth,
        )

    # Plot the mean annual point balance
    plt.plot(
        stats_timeseries.YEAR,
        stats_timeseries.MEAN_POINT_BALANCE,
        linestyle="--",
        color="#ff7f0e",
        label="Mean Annual Point SMB",
        linewidth=2,
    )

    # Fill the area between mean - std and mean + std
    ax.fill_between(
        stats_timeseries.YEAR,
        stats_timeseries.MEAN_POINT_BALANCE - stats_timeseries.STD_POINT_BALANCE,
        stats_timeseries.MEAN_POINT_BALANCE + stats_timeseries.STD_POINT_BALANCE,
        color="orange",
        alpha=0.3,
        label="Std Annual Point SMB",
    )

    # Add a dummy plot for the legend
    ax.plot(
        np.nan,
        np.nan,
        "-",
        marker="o",
        markevery=1,
        markersize=4,
        alpha=0.75,
        color="#4e8cd9",
        linewidth=1,
        label="Annual SMB per Stake",
    )

    # Set the x-axis limits
    ax.set_xlim(left=np.min(stats_timeseries.YEAR), right=np.max(stats_timeseries.YEAR))

    # Set tick parameters
    ax.xaxis.set_tick_params(labelsize=13)
    ax.yaxis.set_tick_params(labelsize=13)

    # Set the plot title
    plt.title(
        f"Average Annual SMB for all Stakes, from {np.min(stats_timeseries.YEAR)} to {np.max(stats_timeseries.YEAR)}",
        fontsize=15,
    )

    # Set axis labels and legend
    ax.set_xlabel("Years", fontsize=15)
    ax.set_ylabel("Point Surface Mass Balance [m w.e.]", fontsize=15)
    ax.legend()

    # Add gridlines
    ax.grid(which="major", color="lightgray", linestyle="--", linewidth=0.5)

    # Save the plot if desired
    if save_img is not None:
        plt.savefig(
            save_img[1] + f"timeseries_stakes.{save_img[0]}",
            dpi=600,
            format=f"{save_img[0]}",
            bbox_inches="tight",
        )

    plt.show()


def plot_cumulative_smb(df: pd.DataFrame, save_img=None) -> None:
    """
    Plots the cumulative annual SMB (Surface Mass Balance) of all stakes in the entire region of interest.

    Args:
        df (DataFrame): DataFrame containing the SMB data.
        save_img (list, optional): save_img(list, optional): A list of details for saving the figure accordingly
        [image format (.svg/.png/.jpg/...), directory of images to save the figure]

    Returns:
        None
    """

    # Get the aggregated point surface mass balance for all stakes in a single
    # year, region-wide
    aggregated_smb = _get_agg_smb_timeseries(df)

    # Plot the aggregated SMB data
    aggregated_smb.plot(
        x="YEAR",
        y="SUM_POINT_BALANCE",
        marker="X",
        markevery=1,
        markersize=6,
        linewidth=1.5,
        color="#4b9c8d",
        label="Cumulative Annual SMB",
    )

    # Set the x-, y-label, and the title
    plt.xlabel("Year", fontsize=11)
    plt.ylabel("Region-wide SMB [m.w.e.]", fontsize=11)
    plt.title("Cumulative Annual SMB of All Stakes", fontsize=12)
    # Set and specify the details of the gridlines
    plt.grid(which="major", color="lightgray", linestyle="--", linewidth=0.5)
    # Show the legend
    plt.legend()

    # Save the plot if desired
    if save_img is not None:
        plt.savefig(
            save_img[1] + f"timeseries_cumsum_smb.{save_img[0]}",
            dpi=600,
            format=f"{save_img[0]}",
            bbox_inches="tight",
        )

    plt.show()


def _get_stats_stake_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the mean and standard deviation of the total point surface mass balance per year, over all stakes

    Args:
        df (pandas.DataFrame):  Dataframe containing all stakes, with a monthly resolution per stake

    Returns:
        results (pandas.DataFrame): Dataframe with for each year a mean and standard deviation of the point surface mass balance
        for all available stakes in the dataset.

    """
    results = (
        df.groupby("YEAR")
        .apply(
            lambda x: pd.Series(
                {
                    "MEAN_POINT_BALANCE": x.drop_duplicates(subset=["POINT_BALANCE"])[
                        "POINT_BALANCE"
                    ].mean(),
                    "STD_POINT_BALANCE": x.drop_duplicates(subset=["POINT_BALANCE"])[
                        "POINT_BALANCE"
                    ].std(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    return results


def _get_stake_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by 'POINT_ID' and for each unique stake, make a dict with the unique years and point balances of the measurements.
    From this, one can make a timeseries of the point surface mass balance measurements over time.

    Args:
        df (pandas.DataFrame): Dataframe containing all stakes, with a monthly resolution per stake

    Returns:
        results (pandas.DataFrame): Dataframe containing all unique stakes, with a list for all the years a point surface
        mass balance recording, and the list of the point surface mass balance recordings.

    """
    results = (
        df.groupby("POINT_ID")
        .apply(
            lambda x: {
                "YEARS": x["YEAR"].unique().tolist(),
                "POINT_BALANCES": x.drop_duplicates("YEAR")["POINT_BALANCE"].tolist(),
            },
            include_groups=False,
        )
        .reset_index()
        .rename(columns={0: "data"})
    )

    return results


def _get_agg_smb_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    # Group by YEAR and apply function to calculate mean and std after
    # removing duplicates
    results = (
        df.groupby("YEAR")
        .apply(
            lambda x: pd.Series(
                {
                    "SUM_POINT_BALANCE": x.drop_duplicates(subset=["POINT_BALANCE"])[
                        "POINT_BALANCE"
                    ].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    results["SUM_POINT_BALANCE"] = results["SUM_POINT_BALANCE"].cumsum()

    return results
