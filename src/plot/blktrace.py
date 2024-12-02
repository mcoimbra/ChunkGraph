import math
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

import plot.util as plot_functions

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