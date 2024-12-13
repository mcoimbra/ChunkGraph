import gzip
import logging
import math
import os
import pickle
import sys
import time
from typing import List, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn
from scipy.sparse import coo_matrix, dok_matrix

# Logging import must come before other local project modules.
import util.logging as log_conf
logger: logging.Logger = log_conf.Logger.get_logger(__name__)
import util.functions as util_functions
import plot.blktrace as blktrace_plotting

import plot.util as plot_functions

def format_tick_label(x: int):
    B_SZ: int = 1024
    K_SZ: int = B_SZ * B_SZ
    M_SZ: int = K_SZ * B_SZ
    if x < B_SZ:
        return f"{int(x)}B"
    elif x < K_SZ:
        return f"{int(x / B_SZ)}KB"
    elif x < M_SZ:
        return f"{int(x / K_SZ)}MB"
    else:
        return f"{int(x / M_SZ)}MB"

# def plot_lba_count_heat_map(df: pd.DataFrame, plot_basename: str, output_dir: str,
#     title: str = "placeholder_title", logical_block_sz: int = 4096, chunk_sz: int = 512*1024) -> None:
#     """
#     Plots a heat map of LBAs against block counts.

#     Args:
#         df (pd.DataFrame): DataFrame containing parsed blktrace data.
#         plot_basename (str): Base name for the output plot file.
#         output_dir (str): Directory to save the plot.
#         title (str, optional): Title for the plot.
#         logical_block_sz (int, optional): Logical block size in bytes (default: 4096).
#         chunk_sz (int, optional): RAID chunk size in bytes (default: 512 * 1024).
#     """
#     # Filter the DataFrame for valid LBA and block count data.
#     df = df.dropna(subset=["lba", "blocks"])
#     df = df[df["blocks"] > 0]

#     # Group by LBA and block count, and calculate the number of occurrences.
#     heatmap_data = (
#         df.groupby(["lba", "blocks"])
#         .size()
#         .reset_index(name="count")
#         .pivot(index="lba", columns="blocks", values="count")
#     )

#     # Plot the heatmap using seaborn.
#     plt.figure()
#     seaborn.heatmap(
#         heatmap_data,
#         cmap="YlGnBu",
#         cbar=True,
#         linewidths=0.5,
#         linecolor="gray"
#     )

#     # Add labels and title.
#     plt.title(title)
#     plt.xlabel(f"Block Count (logical block size = {logical_block_sz} bytes)")
#     plt.ylabel("Logical Block Address (LBA)")
#     plt.tight_layout()

#     # Save the plot.
#     plot_functions.save_figure(output_dir, f"{plot_basename}_heatmap.png", ["pdf", "png"])

def plot_lba_count_heat_map(
    df: pd.DataFrame,
    plot_basename: str,
    output_dir: str,
    title: str = "LBA vs Block Count Heatmap",
    logical_block_sz: int = 4096,
    chunk_sz: int = 512 * 1024,
    lba_bins: int = 100,
    block_bins: int = 50,
    use_log_binning: bool = True,
    block_threshold_ratios: List[float] = [0.05],
    top_lbas: List[int] = [10],
    circle_scaling_factor: float = 1.0
) -> None:
    """
    Plots a heat map of LBAs against block counts with binning.

    Args:
        df (pd.DataFrame): DataFrame containing parsed blktrace data.
        plot_basename (str): Base name for the output plot file.
        output_dir (str): Directory to save the plot.
        title (str, optional): Title for the plot.
        logical_block_sz (int, optional): Logical block size in bytes (default: 4096).
        chunk_sz (int, optional): RAID chunk size in bytes (default: 512 * 1024).
        lba_bins (int, optional): Number of bins for LBAs (default: 100).
        block_bins (int, optional): Number of bins for block counts (default: 50).
        use_log_binning (bool, optional): Whether to use logarithmic binning (default: True).
    """

    
    # Filter and validate the data.
    filtered_df = df.dropna(subset=["lba", "blocks"]).copy()
    filtered_df = filtered_df[filtered_df["blocks"] > 0]

    # Calculate LBA and block count ranges.
    lba_range = (filtered_df["lba"].min(), filtered_df["lba"].max())
    block_range = (filtered_df["blocks"].min(), filtered_df["blocks"].max())

    # Determine bin edges (logarithmic or linear).
    if use_log_binning:
        lba_bins_edges = np.logspace(np.log10(lba_range[0] + 1), np.log10(lba_range[1]), lba_bins)
        block_bins_edges = np.logspace(np.log10(block_range[0] + 1), np.log10(block_range[1]), block_bins)
    else:
        lba_bins_edges = np.linspace(*lba_range, lba_bins)
        block_bins_edges = np.linspace(*block_range, block_bins)
        
    # Bin the data.
    filtered_df["lba_binned"] = pd.cut(filtered_df["lba"], bins=lba_bins_edges, labels=False)
    filtered_df["blocks_binned"] = pd.cut(filtered_df["blocks"], bins=block_bins_edges, labels=False)

    # Group by binned values and aggregate counts.
    heatmap_data = (
        filtered_df.groupby(["lba_binned", "blocks_binned"])
        .size()
        .reset_index(name="count")
        .pivot(index="lba_binned", columns="blocks_binned", values="count")
    )

    # Fill missing values with zeros for visualization.
    heatmap_data = heatmap_data.fillna(0)

    total_blocks: int = filtered_df["blocks"].sum()

    # Count the number of distinct LBAs.
    distinct_lbas = filtered_df["lba"].nunique()
    logger.info(f"Number of distinct LBAs: {distinct_lbas}")

    # PLOTS: heat map of LBAs with at least `block_threshold_ratio` blocks.
    for block_threshold_ratio in block_threshold_ratios:

        logger.info(f"Heat map: block size threshold: {block_threshold_ratio}")

        # Determine if we need to filter LBAs.
        include_all_block_sizes: bool = math.isclose(0, block_threshold_ratio)

        # Apply block threshold filtering if needed.
        if not include_all_block_sizes:
            
            # Calculate total block counts to filter LBAs.
            #total_blocks: int = df["blocks"].sum()
            lba_threshold: float = total_blocks * block_threshold_ratio

            # Filter LBAs based on the threshold.
            # lba_counts = df.groupby("lba")["blocks"].sum()
            # valid_lbas = lba_counts[lba_counts >= lba_threshold].index
            # df = df[df["lba"].isin(valid_lbas)]

            # Filter bins where the total count across all blocks is below the threshold.
            valid_lba_bins = heatmap_data.sum(axis=1)[heatmap_data.sum(axis=1) >= lba_threshold].index
            heatmap_data = heatmap_data.loc[valid_lba_bins]

        if heatmap_data.empty:
            logger.info(f"No data to plot after applying block threshold filter {block_threshold_ratio*100}% ({int(block_threshold_ratio*total_blocks)}/{total_blocks}).")
            continue
        
        lba_labels = [
            f"{int(lba_bins_edges[int(i)])}-{int(lba_bins_edges[int(i) + 1])}"
            for i in heatmap_data.index
        ]

        # Plot the heat map using seaborn.
        plt.figure(figsize=(12, 8))
        seaborn.heatmap(
            heatmap_data,
            cmap="YlGnBu",
            cbar=True,
            linewidths=0.5,
            linecolor="gray",
            yticklabels=lba_labels  # Use full range labels for LBAs
        )

        # Add labels and title.
        plt.title(title)
        plt.xlabel("Block Count (binned)")
        plt.ylabel("LBA (binned)")
        plt.tight_layout()

        # Save the plot.
        plot_functions.save_figure(output_dir, f"{plot_basename}_{block_threshold_ratio*100}pct_blocks", ["pdf", "png"])

    # PLOTS: heat map for the top `top_lba_count` LBAs by block count.
    for top_lba_count in top_lbas:

        # Filter and validate the data.
        filtered_df = df.dropna(subset=["lba", "blocks"]).copy()
        filtered_df = filtered_df[filtered_df["blocks"] > 0]

        # Calculate LBA and block count ranges.
        lba_range = (filtered_df["lba"].min(), filtered_df["lba"].max())
        block_range = (filtered_df["blocks"].min(), filtered_df["blocks"].max())
        
        # Print the top `top_lba_count`.
        # Calculate the top 10 LBAs by total block count
        top_lbas_values = (
            filtered_df.groupby("lba")["blocks"].sum()
            .nlargest(top_lba_count)
            .index
        )
        logger.info(f"Top {top_lba_count} LBAs: {list(top_lbas_values)}")

        # Filter DataFrame to include only these LBAs.
        filtered_df = filtered_df[filtered_df["lba"].isin(top_lbas_values)]

        if filtered_df.empty:
            logger.info("No data to plot after filtering for top LBAs.")
            continue

        # Calculate block count range.
        block_range = (filtered_df["blocks"].min(), filtered_df["blocks"].max())

        # Determine bin edges for block counts (logarithmic or linear).
        if use_log_binning:
            block_bins_edges = np.logspace(np.log10(block_range[0] + 1), np.log10(block_range[1]), block_bins)
        else:
            block_bins_edges = np.linspace(*block_range, block_bins)

        # Bin the data
        filtered_df["blocks_binned"] = pd.cut(filtered_df["blocks"], bins=block_bins_edges, labels=False)

        # Group by LBA and binned blocks, then aggregate counts.
        heatmap_data = (
            filtered_df.groupby(["lba", "blocks_binned"])
            .size()
            .reset_index(name="count")
            .pivot(index="lba", columns="blocks_binned", values="count")
        )

        # Fill missing values with zeros for visualization.
        heatmap_data = heatmap_data.fillna(0)

        # Generate labels for LBAs (show exact LBA values for top LBAs).
        lba_labels = heatmap_data.index.to_list()

        # Plot the heat map.
        plt.figure(figsize=(12, 8))
        seaborn.heatmap(
            heatmap_data,
            cmap="YlGnBu",
            cbar=True,
            linewidths=0.5,
            linecolor="gray",
            yticklabels=lba_labels  # Show top LBA labels directly
        )

        # Add labels and title.
        plt.title(title)
        plt.xlabel("Block Count (binned)")
        plt.ylabel(f"Logical Block Address (Top {top_lba_count} LBAs)")
        plt.tight_layout()

        # Save the plot.
        plot_functions.save_figure(output_dir, f"{plot_basename}_top_{top_lba_count}_lbas_blk_counts", ["pdf", "png"])


    # PLOTS: scatter plot for the average LBA block size.
    # Filter and validate the data.
    filtered_df = df.dropna(subset=["lba", "blocks"]).copy()
    filtered_df = filtered_df[filtered_df["blocks"] > 0]
    lba_stats = (
        filtered_df.groupby("lba")["blocks"]
        .agg(["mean", "sum"])
        .rename(columns={"mean": "avg_block_size", "sum": "total_blocks"})
    )
    for top_lba_count in top_lbas:

        logger.info(f"Scatter plot: top {top_lba_count} LBAs by average block size")

        # Get the top N LBAs by average block size.
        top_lbas_values = lba_stats.nlargest(top_lba_count, "avg_block_size")

        # Determine the range of avg_block_size for scaling.
        min_block_size = top_lbas_values["avg_block_size"].min()
        max_block_size = top_lbas_values["avg_block_size"].max()

        

        # Generate labels for LBAs (show exact LBA values for top LBAs).
        lba_labels: List = top_lbas_values.index.to_list()

        # Calculate average block size and total block count for each LBA.
        # Create the scatter plot.
        plt.figure()

        ax = plt.gca()  # Get the current axis.
        # Explicitly set a white background.
        ax.set_facecolor("white")

        # Set up circle scaling
        xlim = (filtered_df["blocks"].min(), filtered_df["blocks"].max())
        ylim = (0, len(top_lbas_values))  # The y-axis corresponds to the indices (0 to N-1)
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        # Choose a reasonable max radius (in data units, relative to the plot area)
        #max_radius: float = 0.05 * (plt.gca().get_xlim()[1] - plt.gca().get_xlim()[0])  # 5% of the x-axis range
        # Normalize avg_block_size to [0, 1] and scale to max_radius.
        #circle_scaling_factor: float = max_radius / (max_block_size - min_block_size)

        # Scale the radius based on the x and y ranges
        max_radius = 0.05 * x_range  # 5% of the x-axis range
        circle_scaling_factor = max_radius / (max_block_size - min_block_size)


        # This ensures that the circles retain their shape regardless of axis scaling.
        # Without this, some circles had a rectangle around them going from the bottom 
        # edge to the top edge of the figure.
        #plt.gca().set_aspect("equal", adjustable="datalim")

        #for lba, row in top_lbas_values.iterrows():
        for idx, (lba, row) in enumerate(top_lbas_values.iterrows()):
            avg_block_size = row["avg_block_size"]
            total_blocks = row["total_blocks"]

            # Plot the scatter point
            # plt.scatter(
            #     total_blocks,
            #     lba,
            #     color="blue",
            #     alpha=0.7,
            #     label=None,
            #     yticklabels=lba_labels
            # )
            plt.scatter(
                total_blocks,
                idx,
                color="blue",
                alpha=0.7,
                label=None,
                zorder=2
            )

        # Circles should be added after the plot's axes limits are finalized 
        # (i.e., after all scatter points are added), to ensure their positions 
        # and sizes align with the data.
        # This is why we do another loop just to add circls.
        #for idx, (lba, row) in enumerate(top_lbas_values.iterrows()):
            # Calculate radius.
            radius: float = (avg_block_size - min_block_size) * circle_scaling_factor

            # Adjust for the aspect ratio difference.
            radius = radius / (x_range / y_range)  

            # Overlay a transparent circle proportional to `avg_block_size``.
            # plt.gca().add_artist(plt.Circle(
            #     #(total_blocks, lba),
            #     xy=(total_blocks, idx),
            #     radius=avg_block_size * circle_scaling_factor,
            #     color="blue",
            #     alpha=0.2,
            #     zorder=1
            # ))
            plt.gca().add_artist(plt.Circle(
                #(total_blocks, lba),
                xy=(total_blocks, idx),
                radius=radius,
                color="blue",
                alpha=0.2,
                zorder=1
            ))

        # Customize y-axis to show full LBA addresses.
        lba_labels: List = top_lbas_values.index.tolist()  # Full LBA addresses
        plt.yticks(range(len(lba_labels)), lba_labels)

        # Customize the plot.
        plt.xlabel("Total Blocks")
        plt.ylabel("Logical Block Address (LBA)")
        plt.title(f"{title} (Top {top_lba_count} LBAs)")
        plt.grid(True, linestyle="--", alpha=plot_functions.PLOT_ALPHA)

        # Set aspect ratio to ensure circles appear as circles
        #ax.set_aspect("equal", adjustable="datalim")

        # Adjust layout to prevent label cutoff.
        plt.tight_layout()

        # Adjust aspect ratio for equal scaling
        # xlim = ax.get_xlim()
        # ylim = ax.get_ylim()
        # ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[1] - ylim[0])), adjustable="datalim")
        
        ax.set_aspect((x_range / y_range), adjustable="datalim")

        # Save the plot.
        plot_functions.save_figure(output_dir, f"{plot_basename}_top_{top_lba_count}_avg_block_sz", ["pdf", "png"])

def write_lba_stat_xlsx(df: pd.DataFrame, output_dir: str, logical_block_sz: int = 4096, engine: str = "xlsxwriter", max_rows: int = 1048576, overwrite=True) -> None:

    stats_csv_path: str = os.path.join(output_dir, "LBA_block_counts-stats.csv")
    specs_csv_path: str = os.path.join(output_dir, "LBA_block_counts-specs.csv")
    excel_out_path: str = os.path.join(output_dir, "LBA_block_counts.xlsx")

    if not overwrite:
        if util_functions.check_file_validity(stats_csv_path) and util_functions.check_file_validity(specs_csv_path) and util_functions.check_file_validity(excel_out_path):
            logger.info(f"Not overwritting existing files:\n\t{stats_csv_path}\n\t{specs_csv_path}\n\t{excel_out_path}")
            return

    # Filter valid data.
    filtered_df: pd.DataFrame = df.dropna(subset=["lba", "blocks"]).copy()
    filtered_df = filtered_df[filtered_df["blocks"] > 0]

    # Calculate LBA statistics.
    lba_stats: pd.DataFrame = (
        filtered_df.groupby("lba")["blocks"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"count": "block_count", "sum": "block_size_sum", "mean": "avg_block_size"})
    ).reset_index()

    # Convert block_size_sum and avg_block_size to bytes (if needed).
    lba_stats["block_size_sum_bytes"] = lba_stats["block_size_sum"] * logical_block_sz
    lba_stats["avg_block_size_bytes"] = lba_stats["avg_block_size"] * logical_block_sz

    # Create a DataFrame for the specifications sheet.
    specs_df: pd.DataFrame = pd.DataFrame(
        {"Parameter": ["Logical Block Size"], "Value": [logical_block_sz]}
    )

    # Write data to the Excel file
    with pd.ExcelWriter(excel_out_path, engine=engine) as writer:
        # Write specifications sheet
        specs_df.to_excel(writer, index=False, sheet_name="Specifications")

        # Split lba_stats into chunks and write each chunk to a new sheet
        for i in range(0, len(lba_stats), max_rows):
            chunk = lba_stats.iloc[i:i + max_rows]
            sheet_name = f"LBA Stats {i // max_rows + 1}"
            chunk.to_excel(writer, index=False, sheet_name=sheet_name)
    logger.info(f"Wrote LBA stats to Excel file:\n\t{excel_out_path}")

    # Write stats data to a CSV file.
    lba_stats.to_csv(stats_csv_path, index=False)
    logger.info(f"Wrote LBA stats to CSV file:\n\t{stats_csv_path}")

    # Write information on logical block value size to a CSV file.
    specs_df.to_csv(specs_csv_path, index=False)
    logger.info(f"Wrote LBA I/O specs to CSV file:\n\t{specs_csv_path}")

def plot_top_n_lba_counts_histogram(df: pd.DataFrame, plot_basename: str, output_dir: str,
    title: str = "placeholder_title", top_n: int = 20) -> None:
    # Filter valid data
    filtered_df = df.dropna(subset=["lba"]).copy()

    # Count occurrences of each LBA
    lba_counts = filtered_df["lba"].value_counts()

    # Select the top_n LBAs
    top_lbas = lba_counts.head(top_n)

    # Prepare the plot
    #plt.figure(figsize=(10, 8))
    plt.figure()
    y_positions = range(len(top_lbas))  # Y-axis positions for each LBA

    # Plot the horizontal bars
    plt.barh(y_positions, top_lbas.values, color="blue", alpha=plot_functions.PLOT_ALPHA, edgecolor="black")

    # Add LBA labels to the Y-axis
    plt.yticks(y_positions, top_lbas.index)

    # Customize the plot
    plt.xlabel("Number of Occurrences")
    plt.ylabel("Logical Block Address (LBA)")
    plt.title(title)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()  # Adjust layout to prevent label cutoff

    plot_functions.save_figure(output_dir, f"{plot_basename}_top_{top_n}", ["pdf", "png"])

    # Sort the LBAs by decreasing count
    sorted_lbas = top_lbas.sort_values(ascending=True)

    # Prepare the plot
    plt.figure()
    y_positions = range(len(sorted_lbas))  # Y-axis positions for each LBA

    # Plot the horizontal bars
    plt.barh(
        y_positions,
        sorted_lbas.values,
        color="blue",
        alpha=plot_functions.PLOT_ALPHA,
        edgecolor="black"
    )

    # Add LBA labels to the Y-axis
    plt.yticks(y_positions, sorted_lbas.index)

    # Customize the plot
    plt.xlabel("Number of Occurrences")
    plt.ylabel("Logical Block Address (LBA)")
    plt.title(title)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()  # Adjust layout to prevent label cutoff

    # Save the plot
    plot_functions.save_figure(output_dir, f"{plot_basename}_top_{top_n}_sorted", ["pdf", "png"])


# Assuming parse_blkparse_tsv_output is defined as per your provided code.
# These parameters assume: 
# - Logical block size of 4096 bytes.
# - Chunk size of 512 KB * 1024 = 524288 bytes.
def plot_block_size_histogram(df: pd.DataFrame, plot_basename: str, output_dir: str,
    title: str = "placeholder_title", logical_block_sz: int = 4096, chunk_sz: int = 512*1024) -> None:

    # Filter valid block sizes.
    block_sizes = df["blocks"].dropna()  # Remove NaN values
    block_sizes = block_sizes[block_sizes > 0]  # Keep only positive block sizes

    # Prepare power-of-two bins.
    min_exp = int(np.floor(np.log2(block_sizes.min())))
    max_exp = int(np.ceil(np.log2(block_sizes.max())))
    pow_of_two_bins = 2 ** np.arange(min_exp, max_exp + 1)

    # Plot histogram.
    plt.figure()
    plt.hist(
        block_sizes,
        bins = pow_of_two_bins,
        #bins=np.logspace(np.log10(block_sizes.min()), np.log10(block_sizes.max()), 50),  # Log-scaled bins
        edgecolor="black",
        alpha=plot_functions.PLOT_ALPHA
    )
    #plt.xscale("log")  # Log scale for x-axis
    plt.xscale("log", base=2)  # Set x-axis to log scale with base 2

    # Set x-axis ticks to powers of 2 and rotate labels
    x_ticks = pow_of_two_bins

    # This commented line shows 2^value for each tick.
    x_tick_labels = [f"$2^{{{int(np.log2(x))}}}$" for x in x_ticks]

    plt.xticks(x_ticks, x_tick_labels, rotation=45)
    plt.xlabel("Number of logical blocks per operation")
    plt.ylabel("#Occurrences")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    
    plot_functions.save_figure(output_dir, f"{plot_basename}_logical_blk_count_exp_notation", ["pdf", "png"])

    # Print the block size explicitly without exponent notation.
    x_tick_labels = [f"{x}" for x in x_ticks]
    plt.xticks(x_ticks, x_tick_labels, rotation=45)
    plt.xlabel("Number of logical blocks per operation")
    plt.ylabel("#Occurrences")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()  # Adjust layout to prevent label cutoff

    plot_functions.save_figure(output_dir, f"{plot_basename}_logical_blk_count", ["pdf", "png"])

    # Print with the logical block size multiplied by the `chunk_sz` argument.
    
    # Plot histogram.
    block_sizes = block_sizes * logical_block_sz
    min_exp = int(np.floor(np.log2(block_sizes.min())))
    max_exp = int(np.ceil(np.log2(block_sizes.max())))
    pow_of_two_bins = 2 ** np.arange(min_exp, max_exp + 1)

    plt.figure()
    plt.xscale("log", base=2)  # Set x-axis to log scale with base 2
    plt.hist(
        block_sizes,
        bins = pow_of_two_bins,
        #weights=weights,
        #bins=np.logspace(np.log10(block_sizes.min()), np.log10(block_sizes.max()), 50),  # Log-scaled bins
        edgecolor="black",
        alpha=plot_functions.PLOT_ALPHA
    )

    # Add a vertical line at chunk size.
    plt.axvline(chunk_sz, color="red", linestyle="--", linewidth=2, label=f"Chunk Size = {format_tick_label(chunk_sz)}")

    # Set x-axis ticks to powers of 2 and rotate labels.
    x_ticks = pow_of_two_bins

    # Create formatted tick labels.
    x_tick_labels = [format_tick_label(x) for x in x_ticks]

    plt.xticks(x_ticks, x_tick_labels, rotation=45)
    plt.xlabel(f"Physical (logical block size = {logical_block_sz}B) size sum of the logical blocks per operation")
    plt.ylabel("#Occurrences")
    plt.title(title)
    plt.legend()  # Add a legend for the vertical line
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()  # Adjust layout to prevent label cutoff

    plot_functions.save_figure(output_dir, f"{plot_basename}_logical_blk_sz", ["pdf", "png"])

def plot_throughput_bar_ts_bin(
    df: pd.DataFrame,
    plot_basename: str,
    output_dir: str,
    title: str = "placeholder_title"
) -> None:
    """
    Plots throughput using timestamp-based binning and saves the plot.

    Args:
        df (pd.DataFrame): Input data.
        plot_basename (str): Basename for the output files.
        output_dir (str): Directory to save the plots.
        title (str): Title for the plot.
    """
    # HISTOGRAM: using (binning on unique) raw timestamps (original method).
    throughput_timestamp = df.groupby("timestamp").size()

    # Check if there's data to plot
    if throughput_timestamp.empty:
        print(f"No data to plot for timestamp-based throughput. Skipping...")
        return

    # Determine Y-axis limit
    max_y_value = math.ceil(max(throughput_timestamp.values))  # Ceiling of the maximum value.

    # Create PlotConfig instance
    config = plot_functions.PlotConfig(
        title=title,
        xlabel="Time (s)",
        ylabel="Throughput (IOPS)",
        edgecolor="red",
        label="Throughput (IOPS)",
        figsize=(10, 6),
        alpha=0.7,
        width=0.01,  # Adjust bar width for better appearance
        xlim=(0, max(throughput_timestamp.index)),
        ylim=(0, max_y_value + 1),
        yaxis_major_locator_function=ticker.MultipleLocator(1),
        save_extensions=["png", "pdf"],
        plot_basename=plot_basename,
        output_dir=output_dir,
        function_type="bar"
    )

    # Call apply_plot_config
    plot_functions.apply_plot_config(
        x_values=throughput_timestamp.index.tolist(),
        y_values=throughput_timestamp.values.tolist(),
        config=config
    )

    print("Throughput (timestamp-based):")
    print(throughput_timestamp.head())

def plot_throughput_bar_1s_bin(
    df: pd.DataFrame,
    plot_basename: str,
    output_dir: str,
    title: str = "placeholder_title"
) -> None:
    """
    Plots throughput using 1-second binning and saves the plot.

    Args:
        df (pd.DataFrame): Input data.
        plot_basename (str): Basename for the output files.
        output_dir (str): Directory to save the plots.
        title (str): Title for the plot.
    """
    # HISTOGRAM: binning by 1 second.
    min_timestamp = max(df["timestamp"].min(), 0)

    # Subtracting this ensures that all timestamps are relative to the start of the trace.
    # // (integer division) divides the adjusted timestamp by 1, effectively truncating the 
    # fractional part of the time value.
    # This operation bins the timestamp into 1-second intervals. For example:
    # 0.5 // 1 = 0
    # 1.7 // 1 = 1
    # 2.3 // 1 = 2
    df["time_bin_1s"] = ((df["timestamp"] - min_timestamp) // 1).astype(int)

    # Calculate the number of events per 1-second bin, measuring the 
    # throughput (event density) over time.
    # Result: throughput_1s is a pandas.Series where:
    # The index represents the 1-second bins (e.g., 0, 1, 2, ...).
    # The values represent the number of events that occurred in each bin.
    throughput_1s = df.groupby("time_bin_1s").size()

    # Converts the bin indices (time_bin_1s) to actual time values in seconds.
    # * 1: Scales the indices to seconds (no effect here since the bin size is 
    # 1 second, but it would be important for other bin sizes like 0.5 or 10 seconds).
    x_values = throughput_1s.index * 1

    if throughput_1s.empty:
        print(f"No data to plot for 1-second binning. Skipping...")
        return

    # Determine X-axis range
    max_x = x_values.max() + 1

    # Create PlotConfig instance
    config = plot_functions.PlotConfig(
        title=title,
        xlabel="Time (s)",
        ylabel="Throughput (IOPS)",
        #edgecolor="red",
        label="Throughput (IOPS)",
        figsize=(10, 6),
        alpha=0.7,
        width=1,  # Adjust bar width for better appearance
        xlim=(0, max_x),
        ylim=(0, throughput_1s.max() + 1),
        yaxis_major_locator_function=None,  # Optional: Set to ticker.MultipleLocator(1) if desired
        save_extensions=["png", "pdf"],
        plot_basename=plot_basename,
        output_dir=output_dir,
        function_type="bar",
        align="edge"
    )

    # Call apply_plot
    plot_functions.apply_plot_config(
        x_values=x_values.tolist(),
        y_values=throughput_1s.values.tolist(),
        config=config
    )

def plot_throughput_bar_dynamic_bin(
    df: pd.DataFrame,
    plot_basename: str,
    output_dir: str,
    title: str = "placeholder_title",
    default_bin_count: int = 50
) -> None:
    """
    Plots throughput using dynamic binning and saves the plot.
    It computes 'default_bin_count' bins dynamically from the data.

    Args:
        df (pd.DataFrame): Input data.
        plot_basename (str): Basename for the output files.
        output_dir (str): Directory to save the plots.
        title (str): Title for the plot.
    """
    # HISTOGRAM: Dynamic binning logic.
    min_timestamp = df["timestamp"].min()
    max_timestamp = df["timestamp"].max()

    if pd.isna(min_timestamp) or pd.isna(max_timestamp):
        print(f"No data available for dynamic binning. Skipping...")
        return

    time_range = max_timestamp - min_timestamp

    if time_range == 0:
        print(f"No variation in timestamps (all data at {min_timestamp} s). Skipping dynamic binning...")
        return

    # Determine dynamic bin size.
    num_bins = default_bin_count  # Default number of bins.
    bin_size = time_range / num_bins
    print(f"Dynamic bin size: {bin_size:.6f} seconds")

    # Create time bins and calculate throughput.
    df["time_bin_dynamic"] = ((df["timestamp"] - min_timestamp) / bin_size).astype(int)
    throughput_dynamic = df.groupby("time_bin_dynamic").size()
    x_values_dynamic = throughput_dynamic.index * bin_size + min_timestamp

    if throughput_dynamic.empty:
        print(f"No data to plot for dynamically binned throughput. Skipping...")
        return
    
    # Determine Y-axis tick interval
    max_y_value = throughput_dynamic.max()
    tick_interval = max(1, max_y_value // 10)  # Aim for ~10 ticks, but at least 1 apart

    # Update the title to include bin size
    updated_title = f"{title} (Bin Size: {bin_size:.6f}s)"

    # Create PlotConfig instance
    config = plot_functions.PlotConfig(
        title=updated_title, #title,
        xlabel="Time (s)",
        ylabel="Throughput (IOPS)",
        edgecolor="blue",
        label="Throughput (IOPS)",
        figsize=(10, 6),
        alpha=0.7,
        width=bin_size,  # Bar width matches the dynamic bin size
        xlim=(min_timestamp, max_timestamp),
        ylim=(0, throughput_dynamic.max() + 1),
        yaxis_major_locator_function=ticker.MultipleLocator(tick_interval),
        save_extensions=["png", "pdf"],
        plot_basename=plot_basename,
        output_dir=output_dir,
        function_type="bar"
    )

    # Call apply_plot_config
    plot_functions.apply_plot_config(
        x_values=x_values_dynamic.tolist(),
        y_values=throughput_dynamic.values.tolist(),
        config=config
    )

    print("Throughput (dynamic binning):")
    print(throughput_dynamic.head())

def plot_latency_bar_static_50_bin_uniform(
    s: pd.Series,
    plot_basename: str,
    output_dir: str,
    title: str = "placeholder_title",
    default_bin_count: int = 50
) -> None:
    """
    Plots latency distribution with 'default_bin_count' bins and saves the plot.
    It computes 'default_bin_count' bins dynamically from the data.

    Args:
        df (pd.DataFrame): Input data containing latency values.
        plot_basename (str): Basename for the output files.
        output_dir (str): Directory to save the plots.
        title (str): Title for the plot.
    """
    if s.empty:
        print(f"No data available for latency histogram. Skipping...")
        return
    if not pd.api.types.is_numeric_dtype(s):
        print(f"Data is not numeric. Skipping...")
        return
    
    # Calculate dynamic bin range
    min_value = s.min()
    max_value = s.max()
    
    hist, bin_edges = np.histogram(s, bins=50, range=(min_value, max_value))
    print(f"Histogram counts: {hist}")
    print(f"Bin edges: {bin_edges}")
    assert sum(hist) == len(s), "Bar heights do not match the total data count."

    # Compute bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]  # Width of each bin

    # Set appropriate tick intervals for Y-axis to reduce clutter
    max_y_value = max(hist)
    tick_interval = max(1, max_y_value // 10)  # Set at most 10 major ticks
        
    # Create PlotConfig instance
    config = plot_functions.PlotConfig(
        title=title,
        xlabel="Latency (s)",
        ylabel="Count",
        bins=default_bin_count,
        edgecolor="black",
        label=f"Latency Distribution",
        figsize=(10, 6),
        do_grid=True,
        save_extensions=["png", "pdf"],
        plot_basename=f"{plot_basename}",
        output_dir=output_dir,
        function_type="bar",
        ylim=(0, max(hist) + 1),  # Extend ylim slightly above the highest bar
        xlim=(min_value, max_value),
        width=bin_width * 0.9,  # Adjust bar width to avoid overlap
        yaxis_major_locator_function=ticker.MultipleLocator(tick_interval)
    )

    # Call apply_plot_config
    plot_functions.apply_plot_config(
        x_values=bin_centers.tolist(),
        y_values=hist.tolist(),  # Not required for histograms
        config=config
    )

# THIS WORKS
# def plot_latency_logarithmic_bins(s: pd.Series, plot_basename: str, output_dir: str, title="placeholder_title") -> None:
#     bins = np.logspace(np.log10(s.min()), np.log10(s.max()), 50)

#     plt.figure(figsize=(10, 6))
#     plt.hist(s, bins=bins, edgecolor="black", label="Latency Distribution")
#     plt.xscale("log")
#     plt.title("Overall Latency Distribution (Logarithmic Binning)")
#     plt.xlabel("Latency (s, log scale)")
#     plt.ylabel("Count")
#     plt.legend()
#     plt.grid()
#     png_out_path = os.path.join(output_dir, f"{plot_basename}.png")
#     plt.savefig(png_out_path, dpi=300)
#     pdf_out_path = os.path.join(output_dir, f"{plot_basename}.pdf")
#     plt.savefig(pdf_out_path)
#     plt.close()

def plot_latency_time_series(
    latencies: pd.DataFrame,
    plot_basename: str,
    output_dir: str,
    title: str = "placeholder_title"
) -> None:
    config = plot_functions.PlotConfig(
        title=title,
        xlabel="Latency time series (s)",
        ylabel="Count",
        edgecolor="black",
        label="Latency",
        figsize=(10, 6),
        do_grid=True,
        do_legend=True,
        save_extensions=["png", "pdf"],
        plot_basename=f"{plot_basename}",
        output_dir=output_dir,
        function_type="scatter",
        xscale="linear"
    )

    

    # Call apply_plot_config
    # plot_functions.apply_plot_config(
    #     x_values=latencies.loc[latencies["latency"] >= 0, "timestamp_c"], 
    #     y_values=latencies.loc[latencies["latency"] >= 0, "latency"],
    #     config=config
    # )

    # Using valid latencies with their corresponding completion timestamps.
    plt.xscale('linear') 
    plt.scatter(latencies.loc[latencies["latency"] >= 0, "timestamp_c"], 
                latencies.loc[latencies["latency"] >= 0, "latency"], 
                s=1)
    plt.xlabel("Timestamp (seconds)")
    plt.ylabel("Latency (seconds)")
    plt.title(title)

    for ext in config.save_extensions:
        fig_output_path = os.path.join(output_dir, f"{plot_basename}.{ext}")
        plt.savefig(fig_output_path, dpi=config.dpi)


def plot_latency_logarithmic_bins(
    latencies: pd.Series,
    bins,
    plot_basename: str,
    output_dir: str,
    title: str = "placeholder_title"
) -> None:
    """
    Plots latency distribution with logarithmic binning and saves the plot.

    Args:
        df (pd.Series): Input data containing latency values.
        bins (int): Number of bins for the histogram.
        plot_basename (str): Basename for the output files.
        output_dir (str): Directory to save the plots.
        title (str): Title for the plot.
    """
    
    #bins = np.logspace(np.log10(latencies.min()), np.log10(latencies.max()), 50)

    # Define bin range
    min_value = latencies.min()
    max_value = latencies.max()

    if min_value <= 0 or pd.isna(min_value) or pd.isna(max_value) or min_value == max_value:
        print(f"Invalid range for latency histogram (min={min_value}, max={max_value}). Skipping...")
        return
    
    # plt.hist(s, bins=bins, edgecolor="black", label="Latency Distribution")


    # png_out_path = os.path.join(output_dir, f"{plot_basename}.png")
    # plt.savefig(png_out_path, dpi=300)
    # pdf_out_path = os.path.join(output_dir, f"{plot_basename}.pdf")
    # plt.savefig(pdf_out_path)
    # plt.close()

    # Create PlotConfig instance
    config = plot_functions.PlotConfig(
        title=title,
        xlabel="Latency (s, log scale)",
        ylabel="Count",
        bins=bins,
        edgecolor="black",
        label="Latency Distribution",
        figsize=(10, 6),
        do_grid=True,
        do_legend=True,
        save_extensions=["png", "pdf"],
        plot_basename=f"{plot_basename}",
        output_dir=output_dir,
        function_type="hist",
        xlim=(min_value, max_value),
        xscale="log"
    )

    # Call apply_plot_config
    # Plot histogram
    plt.hist(
        x=latencies,
        bins=bins,
        edgecolor=config.edgecolor,
        alpha=config.alpha,
        label=config.label,
    )
    plt.xscale("log")  # Ensure x-axis is logarithmic
    plt.xlabel("Latency (seconds)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Latencies")
    plt.gca().xaxis.set_major_formatter(ticker.LogFormatter())  # Use readable log formatting
    plt.legend()
    for ext in config.save_extensions:
        fig_output_path = os.path.join(output_dir, f"{plot_basename}.{ext}")
        plt.savefig(fig_output_path, dpi=config.dpi)

    # config.plot_basename = plot_basename + "_FUNC"
    # plot_functions.apply_plot_config(
    #     x_values=latencies,#.tolist(),
    #     #x_values=s[s > 0].tolist(),  # Exclude non-positive values for log scale
    #     y_values=None,  # Not required for histograms
    #     config=config
    # )

    # Set logarithmic scale on the X-axis
    # plt.xscale("log")

    # # Re-save the plot after modifying the scale
    # for ext in config.save_extensions:
    #     fig_output_path = os.path.join(output_dir, f"{plot_basename}_log.{ext}")
    #     plt.savefig(fig_output_path, dpi=config.dpi)

def plot_queue_depth(queue_depth_timestamp: pd.DataFrame, plot_basename: str, title: str, output_dir: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(queue_depth_timestamp.index, queue_depth_timestamp.values, label=title)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Queue Depth")
    plt.legend()
    plt.grid()
    plt.xlim(left=0)
    png_out_path = os.path.join(output_dir, f"{plot_basename}.png")
    plt.savefig(png_out_path, dpi=300)
    pdf_out_path = os.path.join(output_dir, f"{plot_basename}.pdf")
    plt.savefig(pdf_out_path)
    plt.close()

def plot_queue_depth_dynamic(queue_depth_timestamp: pd.DataFrame, plot_basename: str, output_dir: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(queue_depth_timestamp.index, queue_depth_timestamp.values, label="Queue Depth Over Time (Dynamic Binning)")
    plt.title("Queue Depth Over Time (Dynamic Binning)")
    plt.xlabel("Time (s)")
    plt.ylabel("Queue Depth")
    plt.legend()
    plt.grid()
    plt.xlim(left=0)
    png_out_path = os.path.join(output_dir, f"{plot_basename}.png")
    plt.savefig(png_out_path, dpi=300)
    pdf_out_path = os.path.join(output_dir, f"{plot_basename}.pdf")
    plt.savefig(pdf_out_path)
    plt.close()

# Memory-naive version
# def visualize_lba_blocks_over_time(df: pd.DataFrame, plot_basename: str, output_dir: str, granularity: float = 0.05
# ) -> None:
#     """
#     Visualizes LBA accesses over time as a heatmap.

#     Args:
#         df (pd.DataFrame): Input DataFrame with 'timestamp', 'lba', and 'blocks' columns.
#         plot_basename (str): Base name for the output plot file.
#         output_dir (str): Directory to save the plot.
#         granularity (float): Fraction of execution time for each time interval (default: 0.05).
#     """

#     # Compute execution time range
#     min_timestamp = df["timestamp"].min()
#     max_timestamp = df["timestamp"].max()
#     execution_time = max_timestamp - min_timestamp

#     # Define time bins based on granularity
#     num_bins = int(1 / granularity)
#     time_bins = np.linspace(min_timestamp, max_timestamp, num_bins + 1)
#     time_labels = (time_bins[:-1] + time_bins[1:]) / 2  # Midpoints of the bins

#     # Expand LBAs based on the "blocks" column
#     expanded_lbas = []
#     expanded_timestamps = []

#     for _, row in df.iterrows():
#         if pd.notna(row["lba"]) and pd.notna(row["blocks"]):
#             lba_range = range(int(row["lba"]), int(row["lba"] + row["blocks"]))
#             expanded_lbas.extend(lba_range)
#             expanded_timestamps.extend([row["timestamp"]] * len(lba_range))

#     expanded_df = pd.DataFrame({"timestamp": expanded_timestamps, "lba": expanded_lbas})

#     # Bin the timestamps
#     expanded_df["time_bin"] = pd.cut(expanded_df["timestamp"], bins=time_bins, labels=time_labels)

#     # Count occurrences of LBAs in each time bin
#     heatmap_data = expanded_df.groupby(["time_bin", "lba"]).size().unstack(fill_value=0)

#     # Normalize colors by the maximum count
#     max_count = heatmap_data.values.max()

#     # Plot heatmap
#     plt.figure(figsize=(12, 8))
#     seaborn.heatmap(
#         heatmap_data.T,  # Transpose to make LBAs on Y-axis
#         cmap="YlGnBu",
#         cbar=True,
#         linewidths=0.5,
#         linecolor="gray",
#         vmin=0,
#         vmax=max_count,
#     )

#     # Add labels and title
#     plt.title(f"LBA Access Over Time ({granularity * 100:.1f}% Time Intervals)")
#     plt.xlabel("Time (% Execution)")
#     plt.ylabel("Logical Block Address (LBA)")

#     # Adjust x-axis ticks
#     plt.xticks(
#         ticks=np.arange(len(time_labels)),
#         labels=[f"{round(label, 2)}" for label in time_labels],
#         rotation=45
#     )

#     # Save the plot
#     #plot_filename = os.path.join(output_dir, f"{plot_basename}_heatmap.png")
#     # plt.tight_layout()
#     # plt.savefig(plot_filename)
#     # plt.close()

#     # print(f"Heatmap saved to {plot_filename}")

#     plot_functions.save_figure(output_dir, f"{plot_basename}_{granularity*100}pct", ["pdf", "png"])

# Memory-efficient version:
# def visualize_lba_blocks_over_time(
#     df: pd.DataFrame,
#     plot_basename: str,
#     output_dir: str,
#     granularity: float = 0.05
# ) -> None:
#     """
#     Visualizes LBA accesses over time as a heatmap in a memory-efficient manner.

#     Args:
#         df (pd.DataFrame): Input DataFrame with 'timestamp', 'lba', and 'blocks' columns.
#         plot_basename (str): Base name for the output plot file.
#         output_dir (str): Directory to save the plot.
#         granularity (float): Fraction of execution time for each time interval (default: 0.05).
#     """

#     # Compute execution time range
#     min_timestamp = df["timestamp"].min()
#     max_timestamp = df["timestamp"].max()
#     execution_time = max_timestamp - min_timestamp

#     # Define time bins based on granularity
#     num_bins = int(1 / granularity)
#     time_bins = np.linspace(min_timestamp, max_timestamp, num_bins + 1)
#     time_labels = (time_bins[:-1] + time_bins[1:]) / 2  # Midpoints of the bins

#     # Initialize an empty sparse matrix for heatmap data
#     from scipy.sparse import dok_matrix

#     unique_lbas = df["lba"].dropna().unique()
#     lba_indices = {lba: idx for idx, lba in enumerate(sorted(unique_lbas))}
#     heatmap_matrix = dok_matrix((len(lba_indices), num_bins), dtype=int)

#     # Incrementally populate the heatmap matrix
#     for _, row in df.iterrows():
#         if pd.notna(row["lba"]) and pd.notna(row["blocks"]):
#             start_lba = int(row["lba"])
#             for offset in range(int(row["blocks"])):
#                 lba = start_lba + offset
#                 if lba in lba_indices:
#                     time_bin = np.digitize(row["timestamp"], time_bins) - 1
#                     if 0 <= time_bin < num_bins:
#                         heatmap_matrix[lba_indices[lba], time_bin] += 1

#     # Convert sparse matrix to dense for visualization
#     heatmap_data = pd.DataFrame.sparse.from_spmatrix(
#         heatmap_matrix, index=sorted(lba_indices.keys()), columns=time_labels
#     )

#     # Normalize colors by the maximum count
#     max_count = heatmap_data.max().max()

#     # Plot heatmap
#     plt.figure(figsize=(12, 8))
#     seaborn.heatmap(
#         heatmap_data,
#         cmap="YlGnBu",
#         cbar=True,
#         linewidths=0.5,
#         linecolor="gray",
#         vmin=0,
#         vmax=max_count,
#     )

#     # Add labels and title
#     plt.title(f"LBA Access Over Time ({granularity * 100:.1f}% Time Intervals)")
#     plt.xlabel("Time (% Execution)")
#     plt.ylabel("Logical Block Address (LBA)")

#     # Adjust x-axis ticks
#     plt.xticks(
#         ticks=np.arange(len(time_labels)),
#         labels=[f"{round(label, 2)}" for label in time_labels],
#         rotation=45
#     )

#     # Save the plot
#     plot_functions.save_figure(output_dir, f"{plot_basename}_{granularity*100}pct", ["pdf", "png"])


# Memory-efficient version, hopefully faster:
# def visualize_lba_blocks_over_time(
#     df: pd.DataFrame,
#     plot_basename: str,
#     output_dir: str,
#     granularity: float = 0.05
# ) -> None:
#     """
#     Visualizes LBA accesses over time as a heatmap in a memory-efficient manner.

#     Args:
#         df (pd.DataFrame): Input DataFrame with 'timestamp', 'lba', and 'blocks' columns.
#         plot_basename (str): Base name for the output plot file.
#         output_dir (str): Directory to save the plot.
#         granularity (float): Fraction of execution time for each time interval (default: 0.05).
#     """

#     # Compute execution time range
#     start_time: float = time.perf_counter()
#     min_timestamp = df["timestamp"].min()
#     max_timestamp = df["timestamp"].max()
#     execution_time = max_timestamp - min_timestamp
#     end_time: float = time.perf_counter()
#     logger.info(f"Computed execution time range: (took {end_time-start_time})\n\t[{min_timestamp} - {max_timestamp}] ({execution_time}).")
    

#     # Define time bins based on granularity
#     start_time: float = time.perf_counter()
#     num_bins = int(1 / granularity)
#     time_bins = np.linspace(min_timestamp, max_timestamp, num_bins + 1)
#     time_bin_indices = np.digitize(df["timestamp"], time_bins) - 1
#     end_time: float = time.perf_counter()
#     logger.info(f"Defined time bins based on granularity (took {end_time-start_time}).")

#     # Filter invalid rows.
#     start_time: float = time.perf_counter()
#     valid_rows = df.dropna(subset=["lba", "blocks"])
#     valid_rows = valid_rows[(time_bin_indices >= 0) & (time_bin_indices < num_bins)]
#     valid_rows["time_bin"] = time_bin_indices
#     end_time: float = time.perf_counter()
#     logger.info(f"Filtered invalid rows (took {end_time-start_time}).")

#     # Expand LBAs and Blocks.
#     start_time: float = time.perf_counter()
#     expanded_lbas = valid_rows.apply(
#         lambda row: [(int(row["lba"]) + offset, row["time_bin"]) for offset in range(int(row["blocks"]))],
#         axis=1
#     )
#     expanded_lbas = [item for sublist in expanded_lbas for item in sublist]
#     end_time: float = time.perf_counter()
#     logger.info(f"Expanded LBAs and blocks (took {end_time-start_time}).")

#     # Aggregate counts.
#     start_time: float = time.perf_counter()
#     lbas, time_bins = zip(*expanded_lbas)
#     lba_indices, unique_lbas = pd.factorize(lbas)
#     heatmap_coo = coo_matrix(
#         (np.ones_like(lba_indices), (lba_indices, time_bins)),
#         shape=(len(unique_lbas), num_bins)
#     )
#     heatmap_dense = heatmap_coo.toarray()
#     end_time: float = time.perf_counter()
#     logger.info(f"Aggregated counts (took {end_time-start_time}).")

#     # Normalize for plotting
#     start_time: float = time.perf_counter()
#     heatmap_df = pd.DataFrame(
#         heatmap_dense,
#         index=unique_lbas,
#         columns=np.linspace(0, 100, num_bins)  # Percent execution
#     )
#     end_time: float = time.perf_counter()
#     logger.info(f"Normalized for plotting (took {end_time-start_time}).")

#     # Plot heatmap
#     plt.figure(figsize=(12, 8))
#     seaborn.heatmap(
#         heatmap_df,
#         cmap="YlGnBu",
#         cbar=True,
#         linewidths=0.5,
#         linecolor="gray",
#         vmin=0
#     )

#     # Add labels and title
#     plt.title(f"LBA Access Over Time ({granularity * 100:.1f}% Time Intervals)")
#     plt.xlabel("Time (% Execution)")
#     plt.ylabel("Logical Block Address (LBA)")

#     # Save the plot
#     plot_functions.save_figure(output_dir, f"{plot_basename}_{granularity*100}pct", ["pdf", "png"])

# def visualize_lba_blocks_over_time(
#     df: pd.DataFrame,
#     plot_basename: str,
#     output_dir: str,
#     granularity: float = 0.05
# ) -> None:
#     """
#     Visualizes LBA accesses over time as a heatmap in a memory-efficient manner.

#     Args:
#         df (pd.DataFrame): Input DataFrame with 'timestamp', 'lba', and 'blocks' columns.
#         plot_basename (str): Base name for the output plot file.
#         output_dir (str): Directory to save the plot.
#         granularity (float): Fraction of execution time for each time interval (default: 0.05).
#     """

#     # Compute execution time range
#     start_time = time.perf_counter()
#     min_timestamp = df["timestamp"].min()
#     max_timestamp = df["timestamp"].max()
#     execution_time = max_timestamp - min_timestamp
#     end_time = time.perf_counter()
#     logger.info(f"Computed execution time range: (took {end_time-start_time})\n\t[{min_timestamp} - {max_timestamp}] ({execution_time}).")
    
#     # Define time bins based on granularity
#     start_time = time.perf_counter()
#     num_bins = int(1 / granularity)
#     time_bins = np.linspace(min_timestamp, max_timestamp, num_bins + 1)
#     end_time = time.perf_counter()
#     logger.info(f"Defined time bins based on granularity (took {end_time-start_time}).")

#     # Filter valid rows
#     start_time = time.perf_counter()
#     valid_rows = df.dropna(subset=["lba", "blocks"]).copy()
#     time_bin_indices = np.digitize(valid_rows["timestamp"], time_bins) - 1
#     valid_rows = valid_rows[(time_bin_indices >= 0) & (time_bin_indices < num_bins)]
#     valid_rows["time_bin"] = time_bin_indices
#     end_time = time.perf_counter()
#     logger.info(f"Filtered valid rows (took {end_time-start_time}).")

#     # Expand LBAs and Blocks
#     start_time = time.perf_counter()
#     expanded_lbas = valid_rows.apply(
#         lambda row: [(int(row["lba"]) + offset, row["time_bin"]) for offset in range(int(row["blocks"]))],
#         axis=1
#     )
#     expanded_lbas = [item for sublist in expanded_lbas for item in sublist]
#     end_time = time.perf_counter()
#     logger.info(f"Expanded LBAs and blocks (took {end_time-start_time}).")

#     # Aggregate counts
#     start_time = time.perf_counter()
#     lbas, time_bins = zip(*expanded_lbas)
#     lba_indices, unique_lbas = pd.factorize(lbas)
#     heatmap_coo = coo_matrix(
#         (np.ones_like(lba_indices), (lba_indices, time_bins)),
#         shape=(len(unique_lbas), num_bins)
#     )
#     heatmap_dense = heatmap_coo.toarray()
#     end_time = time.perf_counter()
#     logger.info(f"Aggregated counts (took {end_time-start_time}).")

#     # Normalize for plotting
#     start_time = time.perf_counter()
#     heatmap_df = pd.DataFrame(
#         heatmap_dense,
#         index=unique_lbas,
#         columns=np.linspace(0, 100, num_bins)  # Percent execution
#     )
#     end_time = time.perf_counter()
#     logger.info(f"Normalized for plotting (took {end_time-start_time}).")

#     # Plot heatmap
#     plt.figure(figsize=(12, 8))
#     seaborn.heatmap(
#         heatmap_df,
#         cmap="YlGnBu",
#         cbar=True,
#         linewidths=0.5,
#         linecolor="gray",
#         vmin=0
#     )

#     # Add labels and title
#     plt.title(f"LBA Access Over Time ({granularity * 100:.1f}% Time Intervals)")
#     plt.xlabel("Time (% Execution)")
#     plt.ylabel("Logical Block Address (LBA)")

#     # Save the plot
#     plot_functions.save_figure(output_dir, f"{plot_basename}_{granularity*100}pct", ["pdf", "png"])

# def process_in_chunks(df: pd.DataFrame, num_bins: int) -> Tuple[coo_matrix, List[int]]:
#     """
#     Process DataFrame in chunks to populate a sparse matrix.

#     Args:
#         df (pd.DataFrame): The input DataFrame.
#         num_bins (int): Number of time bins.
#         logger: Logger for logging progress.

#     Returns:
#         coo_matrix: Sparse matrix of aggregated counts.
#     """
#     # Extract unique LBAs and map them to indices.
#     unique_lbas: List[int] = sorted(df["lba"].unique())
#     lba_to_index = {lba: i for i, lba in enumerate(unique_lbas)}
#     num_lbas: int = len(unique_lbas)

#     # Initialize sparse matrix.
#     heatmap_coo: dok_matrix = dok_matrix((num_lbas, num_bins), dtype=np.int32)

#     # Process the DataFrame in chunks.
#     chunk_size: int = 1_000_000  # Adjust as needed.
#     num_chunks: int = (len(df) + chunk_size - 1) // chunk_size
#     logger.info(f"Processing DataFrame in {num_chunks} chunks...")

#     for chunk_start in range(0, len(df), chunk_size):

#         start_time: float = time.perf_counter()

#         chunk_end: int = min(chunk_start + chunk_size, len(df))
#         chunk = df.iloc[chunk_start:chunk_end]

#         logger.info(f"Processing rows {chunk_start} to {chunk_end}...")

#         # Process each row in the chunk.
#         # TODO: this part may benefit from memory improvements avoiding iterrows()
#         # as a we are only processing 1M-sized chunks, an approach that uses more memory
#         # may become possible since we are doing chunking.
#         for _, row in chunk.iterrows():
#             base_lba: int = int(row["lba"])
#             time_bin = row["time_bin"]
#             blocks: int = int(row["blocks"])

#             for offset in range(blocks):
#                 lba: int = base_lba + offset
#                 if lba in lba_to_index:
#                     heatmap_coo[lba_to_index[lba], time_bin] += 1

#         end_time: float = time.perf_counter()
#         logger.info(f"Rows {chunk_start}-{chunk_end} ({chunk_end-chunk_start}) took {end_time-start_time} s).")

#     # Convert to COO format.
#     return heatmap_coo.tocoo(), unique_lbas

def process_in_chunks(df: pd.DataFrame, num_bins: int, checkpoints_dir: str, compression: bool = True) -> Tuple[coo_matrix, List[int]]:
    """
    Process DataFrame in chunks to populate a sparse matrix.

    Args:
        df (pd.DataFrame): The input DataFrame.
        num_bins (int): Number of time bins.

    Returns:
        Tuple[coo_matrix, List[int]]: Sparse matrix of aggregated counts and unique LBAs.
    """
    # Extract unique LBAs and map them to indices.
    unique_lbas: List[int] = sorted(df["lba"].unique())
    lba_to_index = {lba: i for i, lba in enumerate(unique_lbas)}
    num_lbas: int = len(unique_lbas)

    # Compute intermediate checkpoint file names.
    checkpoint_file: str = os.path.join(checkpoints_dir, f"progress.pkl")
    completed_chunks: Set = set()

    # Check if there is a checkpoint to resume from.
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            checkpoint_data = pickle.load(f)
        completed_chunks = checkpoint_data["completed_chunks"]
        logger.info(f"Resuming from checkpoint. {len(completed_chunks)} chunks already processed.")
    else:
        # Initialize a fresh sparse matrix if no checkpoint
        logger.info("Starting new computation, no checkpoint found.")
        

    # Process the DataFrame in chunks.
    # TODO: chunk_size should be estimated from running system's total memory and from the size of `df` row...
    chunk_size: int = 1_000_000  # Adjust as needed.
    num_chunks: int = (len(df) + chunk_size - 1) // chunk_size
    logger.info(f"Processing DataFrame in {num_chunks} chunks...")

    for chunk_start in range(0, len(df), chunk_size):
        
        chunk_end: int = min(chunk_start + chunk_size, len(df))
        chunk_id = (chunk_start, chunk_end)

        # Skip already processed chunks.
        if chunk_id in completed_chunks:
            logger.info(f"Skipping already processed chunk: {chunk_id}.")
            continue

        start_time: float = time.perf_counter()

        chunk = df.iloc[chunk_start:chunk_end]

        logger.info(f"Processing rows {chunk_start} to {chunk_end}...")

        # Vectorized processing of the chunk.
        base_lbas = chunk["lba"].values
        time_bins = chunk["time_bin"].values
        block_counts = chunk["blocks"].values

        # Expand LBAs and replicate time bins for block counts.
        lba_offsets = np.concatenate([np.arange(block_count) for block_count in block_counts])
        expanded_lbas = np.repeat(base_lbas, block_counts) + lba_offsets
        expanded_time_bins = np.repeat(time_bins, block_counts)

        # Map expanded LBAs to indices.
        mapped_lba_indices = np.array([lba_to_index.get(lba, -1) for lba in expanded_lbas])

        # Filter valid indices.
        valid_indices = mapped_lba_indices >= 0
        valid_lbas = mapped_lba_indices[valid_indices]
        valid_time_bins = expanded_time_bins[valid_indices]

        # Accumulate updates for the chunk.
        chunk_coo = coo_matrix(
            (np.ones_like(valid_lbas), (valid_lbas, valid_time_bins)),
            shape=(num_lbas, num_bins)
        )

        # Save the current chunk's sparse matrix to disk.
        if compression:
            chunk_file: str = os.path.join(checkpoints_dir, f"chunk_{chunk_start}_{chunk_end}.pkl.gz")
            with gzip.open(chunk_file, "wb") as f:
                pickle.dump(chunk_coo, f)
        else:
            chunk_file: str = os.path.join(checkpoints_dir, f"chunk_{chunk_start}_{chunk_end}.npz")
            with open(chunk_file, "wb") as f:
                pickle.dump(chunk_coo, f)

        # Update the progress file.
        completed_chunks.add(chunk_id)
        with open(checkpoint_file, "wb") as f:
            pickle.dump({"completed_chunks": completed_chunks}, f)
        logger.info(f"Chunk {chunk_id} saved to {chunk_file}.")
        end_time: float = time.perf_counter()
        logger.info(f"Rows {chunk_start}-{chunk_end} ({chunk_end-chunk_start}) took {end_time-start_time} s).")


    # Combine all chunk matrices into a single sparse matrix
    start_time: float = time.perf_counter()
    final_matrix = coo_matrix((num_lbas, num_bins), dtype=np.int32)
    for chunk_start in range(0, len(df), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(df))
        if compression:
            chunk_file = os.path.join(checkpoints_dir, f"chunk_{chunk_start}_{chunk_end}.pkl.gz")
            if os.path.exists(chunk_file):
                with gzip.open(chunk_file, "rb") as f:
                    chunk_coo = pickle.load(f)
                final_matrix += chunk_coo
            else:
                logger.error("Exiting because I did not find expectd chunk file:\n\t{chunk_file}")
                sys.exit(1)
        else:
            chunk_file = os.path.join(checkpoints_dir, f"chunk_{chunk_start}_{chunk_end}.npz")
            if os.path.exists(chunk_file):
                with open(chunk_file, "rb") as f:
                    chunk_coo = pickle.load(f)
                final_matrix += chunk_coo
            else:
                logger.error("Exiting because I did not find expectd chunk file:\n\t{chunk_file}")
                sys.exit(1)
    
    end_time: float = time.perf_counter()
    logger.info(f"Combining all chunks into the final sparse matrix (took {end_time-start_time} s).")

    # Convert to COO format.
    return final_matrix, unique_lbas
    #return heatmap_coo.tocoo(), unique_lbas

def visualize_lba_blocks_over_time( 
    df: pd.DataFrame,
    plot_basename: str,
    output_dir: str,
    time_granularities: List[float] = [0.05],
    lba_aggregations: List[int] = [2, 4, 16, 64, 256, 1024, 4096]
) -> None:
    """
    Visualizes LBA accesses over time as a heatmap in a memory-efficient manner.

    Args:
        df (pd.DataFrame): Input DataFrame with 'timestamp', 'lba', and 'blocks' columns.
        plot_basename (str): Base name for the output plot file.
        output_dir (str): Directory to save the plot.
        granularity (float): Fraction of execution time for each time interval (default: 0.05).
    """

    # Compute execution time range.
    start_time: float = time.perf_counter()
    min_timestamp: float = df["timestamp"].min()
    max_timestamp: float = df["timestamp"].max()
    execution_time: float = max_timestamp - min_timestamp
    end_time: float = time.perf_counter()
    logger.info(f"Computed execution time range: (took {end_time-start_time})\n\t[{min_timestamp} - {max_timestamp}] ({execution_time} s).")

    # Filter valid rows. We call `copy()` here so that `df` is not modified outside this function.
    start_time: float = time.perf_counter()
    valid_rows_orig: pd.DataFrame = df.dropna(subset=["lba", "blocks"]).copy()
    end_time: float = time.perf_counter()
    logger.info(f"Filtered valid rows (took {end_time-start_time} s).")

    for granularity in time_granularities:

        logger.info(f"Generating LBAs over time in {int(granularity*100)}% steps of total execution...")

        # Define time bins based on granularity.
        start_time: float = time.perf_counter()
        num_bins: int = int(1 / granularity)
        time_bins: np.ndarray = np.linspace(min_timestamp, max_timestamp, num_bins + 1)
        end_time: float = time.perf_counter()
        logger.info(f"Defined time bins based on granularity (took {end_time-start_time} s).")

        # Compute time bin indices only for valid rows.
        valid_rows: pd.DataFrame = valid_rows_orig.copy()
        start_time: float = time.perf_counter()
        valid_rows["time_bin"] = np.digitize(valid_rows["timestamp"], time_bins) - 1
        end_time: float = time.perf_counter()
        logger.info(f"Computing time bin indices for valid rows (took {end_time-start_time} s).")

        # Remove rows with invalid time bins.
        start_time: float = time.perf_counter()
        valid_rows = valid_rows[(valid_rows["time_bin"] >= 0) & (valid_rows["time_bin"] < num_bins)]
        end_time: float = time.perf_counter()
        logger.info(f"Removed rows with invalid time bins (took {end_time-start_time} s).")

        # Process the `pd.DataFrame` in chunks to populate a sparse matrix.
        start_time: float = time.perf_counter()
        f_name: str = sys._getframe().f_code.co_name
        checkpoints_dir: str = os.path.join(output_dir, f"{f_name}-XX_{int(granularity*100)}pct")
        os.makedirs(checkpoints_dir, exist_ok=True)
        logger.info(f"Using the following directory to store intermediate chunks:\n\t{checkpoints_dir}")
        heatmap_coo, unique_lbas = process_in_chunks(valid_rows, num_bins, checkpoints_dir, compression=True)
        heatmap_coo: coo_matrix 
        unique_lbas: List[int]
        heatmap_dense: np.ndarray = heatmap_coo.toarray()
        end_time: float = time.perf_counter()
        logger.info(f"Efficiently aggregated counts into sparse matrix (took {end_time - start_time} s).")
        logger.info(f"Unrolled base LBAs w/ block sizes to a total of {len(unique_lbas)}")

        for lba_agg_factor in lba_aggregations:

            logger.info(f"Aggregating LBAs in groups of size {lba_agg_factor}...")

            # Aggregate LBAs.
            heatmap_dense_cp = heatmap_dense.copy()
            
            # Calculate the remainder.
            num_rows, num_cols = heatmap_dense_cp.shape
            remainder = num_rows % lba_agg_factor
            target_lbas: List[int] = copy.deepcopy(unique_lbas)

            # Handle remainders separately
            if remainder != 0:
                padded_rows = lba_agg_factor - remainder
                heatmap_dense_cp = np.pad(heatmap_dense_cp, ((0, padded_rows), (0, 0)), mode="constant")
                target_lbas.extend([f"Padding-{i}" for i in range(padded_rows)])  # Extend unique_lbas for padding


            #heatmap_dense_agg = heatmap_dense_cp.reshape(-1, lba_agg_factor, heatmap_dense_cp.shape[1]).sum(axis=1)
            # Reshape and aggregate
            heatmap_dense_agg = heatmap_dense_cp.reshape(-1, lba_agg_factor, num_cols).sum(axis=1)

            # Update unique LBAs and time bins.
            # unique_lbas_agg = [
            #     f"{unique_lbas[i]}-{unique_lbas[i+lba_agg_factor-1]}"
            #     for i in range(0, len(unique_lbas), lba_agg_factor)
            # ]
            unique_lbas_agg = [
                f"{target_lbas[i]}-{target_lbas[i+lba_agg_factor-1]}"
                for i in range(0, len(target_lbas), lba_agg_factor)
            ]

            # Normalize for plotting.
            start_time: float = time.perf_counter()
            heatmap_df_agg = pd.DataFrame(
                heatmap_dense_agg,
                index=unique_lbas_agg,
                columns=np.linspace(0, 100, num_bins)  # Percent execution
            )
            # heatmap_df: pd.DataFrame = pd.DataFrame(
            #     heatmap_dense,
            #     index=unique_lbas,
            #     columns=np.linspace(0, 100, num_bins)  # Percent execution
            # )
            end_time: float = time.perf_counter()
            logger.info(f"Normalized for plotting (took {end_time-start_time} s).")

            # Plot heatmap.
            start_time: float = time.perf_counter()
            plt.figure(figsize=(12, 8))
            seaborn.heatmap(
                #heatmap_df,
                heatmap_df_agg,
                cmap="YlGnBu",
                cbar=True,
                linewidths=0.5,
                linecolor="gray",
                vmin=0
            )

            # Add labels and title.
            plt.title(f"LBA Access Over Time ({int(granularity * 100)}% Time Intervals)")
            plt.xlabel(f"Time (% Execution) ({int(granularity * 100)}% steps)")
            plt.ylabel("Logical Block Address (LBA) ({lba_agg_factor} steps)")
            plt.tight_layout()

            # Save the plot.
            plot_functions.save_figure(output_dir, f"{plot_basename}_XX_{int(granularity*100)}pct_YY_{lba_agg_factor}", ["pdf", "png"])
            end_time: float = time.perf_counter()
            logger.info(f"Finished heat map image generation (took {end_time - start_time} s).")
