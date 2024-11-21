import math
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

def plot_bar_ts_bin(df: pd.DataFrame, plot_basename: str, output_dir: str, title: str = "placeholder_title") -> None:

    # HISTOGRAM: using (binning on unique) raw timestamps (original method).
    throughput_timestamp = df.groupby("timestamp").size()

    # Timestamp-based throughput.
    plt.figure(figsize=(10, 6))

    # We want integer YY axis values.
    max_y_value = math.ceil(max(throughput_timestamp.values))  # Ceiling of the maximum value.
    print(f"max y value was:\n\t{max(throughput_timestamp.values)}")
    plt.ylim(0, max_y_value + 1)  # Set Y-axis limits from 0 to max_y_value.
    plt.xlim(0, max(throughput_timestamp.index))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))  # Set tick intervals to 1.

    # Plot the histogram.
    plt.bar(
        x=throughput_timestamp.index,  # X values (time bins).
        height=throughput_timestamp.values,  # One bin per unique X value.
        width=0.01,  # Adjust bar width for better appearance.
        color="red",
        alpha=0.7,
        label="Throughput (IOPS)"
    )

    print("Throughput (timestamp-based):")
    print(throughput_timestamp.head())

    if throughput_timestamp.empty:
        print(f"No data to plot for timestamp-based throughput. Skipping...")
    else:
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Throughput (IOPS)")
        plt.legend()
        plt.grid()
        plt.xlim(left=0)
        png_out_path = os.path.join(output_dir, f"{plot_basename}.png")
        plt.savefig(png_out_path, dpi=300)
        pdf_out_path = os.path.join(output_dir, f"{plot_basename}.pdf")
        plt.savefig(pdf_out_path)
        plt.close()

def plot_bar_1s_bin(df: pd.DataFrame, plot_basename: str, output_dir: str, title: str = "placeholder_title") -> None:

    # HISTOGRAM: binning by 1 second.
    min_timestamp = max(df["timestamp"].min(), 0)
    df["time_bin_1s"] = ((df["timestamp"] - min_timestamp) // 1).astype(int)

    throughput_1s = df.groupby("time_bin_1s").size()
    x_values = throughput_1s.index * 1  # Scale bin indices to seconds.

    if throughput_1s.empty:
        print(f"No data to plot for 1-second binning. Skipping...")
    else:
        plt.figure(figsize=(10, 6))
        max_x = x_values.max() + 1
        plt.xlim(left=0, right=max_x)
        plt.xticks(np.arange(0, max_x + 1, 1))  # Explicitly set ticks at 1-second intervals.

        # Plot the histogram.
        plt.bar(
            x=x_values,  # X values (time bins).
            height=throughput_1s.values,  # One bin per unique X value.
            width=1,  # Adjust bar width for better appearance.
            color="red",
            alpha=0.7,
            align="edge",  # Align bars to the bin edges.
            label="Throughput (IOPS)"
        )

        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Throughput (IOPS)")
        plt.legend()
        plt.grid()
        png_out_path = os.path.join(output_dir, f"{plot_basename}.png")
        plt.savefig(png_out_path, dpi=300)
        pdf_out_path = os.path.join(output_dir, f"{plot_basename}.pdf")
        plt.savefig(pdf_out_path)
        plt.close()


def plot_bar_dynamic_bin(df: pd.DataFrame, plot_basename: str, output_dir: str, title="placeholder_title") -> None:

    # HISTOGRAM: Dynamic binning logic.
    min_timestamp = df["timestamp"].min()
    max_timestamp = df["timestamp"].max()

    if pd.isna(min_timestamp) or pd.isna(max_timestamp):
        print(f"No data available for dynamic binning. Skipping...")
        return

    time_range = max_timestamp - min_timestamp

    if time_range == 0:
        print(f"No variation in timestamps (all data at {min_timestamp}s). Skipping dynamic binning...")
        return

    # Determine dynamic bin size.
    num_bins = 50  # Default number of bins.
    bin_size = time_range / num_bins
    print(f"Dynamic bin size: {bin_size:.6f} seconds")

    # Create time bins and calculate throughput.
    df["time_bin_dynamic"] = ((df["timestamp"] - min_timestamp) / bin_size).astype(int)
    throughput_dynamic = df.groupby("time_bin_dynamic").size()
    x_values_dynamic = throughput_dynamic.index * bin_size + min_timestamp

    if throughput_dynamic.empty:
        print(f"No data to plot for dynamically binned throughput. Skipping...")
        return

    # Plot dynamically binned throughput as a bar plot.
    plt.figure(figsize=(10, 6))
    plt.bar(
        x=x_values_dynamic,  # Bin centers
        height=throughput_dynamic.values,  # Bin counts
        width=bin_size,  # Bar width matches the dynamic bin size
        color="blue",
        alpha=0.7,
        label="Throughput (IOPS)"
    )

    print("Throughput (dynamic binning):")
    print(throughput_dynamic.head())

    plt.title("Overall Throughput Over Time (Dynamic Binning)")
    plt.xlabel("Time (s)")
    plt.ylabel("Throughput (IOPS)")
    plt.legend()
    plt.grid(axis="y")  # Add horizontal gridlines for clarity

    # Save the plot.
    png_out_path = os.path.join(output_dir, f"{plot_basename}.png")
    plt.savefig(png_out_path, dpi=300)
    pdf_out_path = os.path.join(output_dir, f"{plot_basename}.pdf")
    plt.savefig(pdf_out_path)
    plt.close()

def plot_hist_static_50_bin_uniform(df: pd.DataFrame, plot_basename: str, output_dir: str, title="placeholder_title") -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(df, bins=50, edgecolor="black", label="Latency Distribution")
    plt.title("Overall Latency Distribution (Static Binning)")
    plt.xlabel("Latency (s)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()
    png_out_path = os.path.join(output_dir, f"{plot_basename}_50_static_binning.png")
    plt.savefig(png_out_path, dpi=300)
    pdf_out_path = os.path.join(output_dir, f"{plot_basename}_50_static_binning.pdf")
    plt.savefig(pdf_out_path)
    plt.close()

def plot_hist_dynamic_50_bin_uniform(df: pd.DataFrame, plot_basename: str, output_dir: str, title="placeholder_title") -> None:

    plt.figure(figsize=(10, 6))
    plt.hist(df, bins=50, range=(df.min(), df.max()), edgecolor="black", label="Latency Distribution")
    plt.title("Overall Latency Distribution (Dynamic Binning)")
    plt.xlabel("Latency (s)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()
    png_out_path = os.path.join(output_dir, f"{plot_basename}_50_dynamic_binning.png")
    plt.savefig(png_out_path, dpi=300)
    pdf_out_path = os.path.join(output_dir, f"{plot_basename}_50_dynamic_binning.pdf")
    plt.savefig(pdf_out_path)
    plt.close()

def plot_hist_logarithmic_bins(df: pd.DataFrame, bins, plot_basename: str, output_dir: str, title="placeholder_title") -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(df, bins=bins, edgecolor="black", label="Latency Distribution")
    plt.xscale("log")
    plt.title("Overall Latency Distribution (Logarithmic Binning)")
    plt.xlabel("Latency (s, log scale)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()
    png_out_path = os.path.join(output_dir, f"{plot_basename}_log.png")
    plt.savefig(png_out_path, dpi=300)
    pdf_out_path = os.path.join(output_dir, f"{plot_basename}_log.pdf")
    plt.savefig(pdf_out_path)
    plt.close()