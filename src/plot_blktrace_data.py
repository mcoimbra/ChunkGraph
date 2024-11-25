import argparse
import logging
import os
import sys
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from util.consts import BLKTRACE_OUTPUT_FILE_SUFFIX
# Logging import must come before other local project modules.
import util.logging as log_conf
logger: logging.Logger = log_conf.Logger.get_logger(__name__)
import util.functions as util_functions
import plot.blktrace as blktrace_plotting


SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))

def create_arg_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Plot blktrace data.")

    # parser.add_argument(
    #     "-d", "--device-path", required=True,
    #     help="Path to the device to be monitored by blktrace (e.g., /dev/sda)."
    # )
    parser.add_argument(
        "-o", "--output_dir", required=True,
        help="Directory to store blktrace output."
    )
    parser.add_argument(
        "-i", "--input_dir", required=True,
        help="Path to directory with blktrace outputs to plot."
    )
    

    return parser

def visualize_throughput_overview(df: pd.DataFrame, plot_basename: str, output_dir: str) -> None:
    """
    Generates an overview plot of throughput over time across all CPUs, using timestamp-based, dynamic binning, and 1-second binning.

    Args:
        df (pd.DataFrame): DataFrame with blkparse events.
        plot_basename (str): Basename for the plot output file.
        output_dir (str): Directory to save the plots.
    """

    # HISTOGRAM: using (binning on unique) raw timestamps (original method).
    title: str = "Overall Throughput Over Time (binning: timestamp)"
    blktrace_plotting.plot_throughput_bar_ts_bin(df, f"{plot_basename}_IOPS_timestamp_binning", output_dir, title)

    # HISTOGRAM: binning by 1 second.
    title = "Overall Throughput Over Time (binning: 1s)"
    blktrace_plotting.plot_throughput_bar_1s_bin(df, f"{plot_basename}_IOPS_1s_binning", output_dir, title)

    # HISTOGRAM: Dynamic binning logic.
    title = "Overall Throughput Over Time (Dynamic Binning)"
    blktrace_plotting.plot_throughput_bar_dynamic_bin(df, f"{plot_basename}_IOPS_dynamic_binning_bar", output_dir, title, default_bin_count=50)

def visualize_throughput_per_cpu(df: pd.DataFrame, plot_basename: str, output_dir: str) -> None:
    """
    Generates per-CPU visualizations of throughput over time using timestamp-based, 1-second binning, and dynamic binning.

    Args:
        df (pd.DataFrame): DataFrame with blkparse events.
        plot_basename (str): Basename for the plot output file.
        output_dir (str): Directory to save the plots.
    """
    # Group data by CPU.
    grouped = df.groupby("cpu")
    for cpu, cpu_data in grouped:
    
        # Set up per-CPU output directory.
        cpu_out_dir: str = os.path.join(output_dir, f"cpu-{cpu}")
        os.makedirs(cpu_out_dir, exist_ok=True)

        # HISTOGRAM: using (binning on unique) raw timestamps (original method).
        title: str = "Overall Throughput Over Time (binning: timestamp)"
        #blktrace_plotting.plot_throughput_bar_ts_bin(cpu_data, f"{plot_basename}.{cpu}_IOPS_timestamp_binning", cpu_out_dir, title)
        blktrace_plotting.plot_throughput_bar_ts_bin(cpu_data, f"{plot_basename}.{cpu}_IOPS_timestamp_binning", cpu_out_dir, title)

        # HISTOGRAM: binning by 1 second.
        title = "Overall Throughput Over Time (binning: 1s)"
        #blktrace_plotting.plot_throughput_bar_1s_bin(cpu_data, f"{plot_basename}.{cpu}_IOPS_1s_binning", cpu_out_dir, title)
        blktrace_plotting.plot_throughput_bar_1s_bin(cpu_data, f"{plot_basename}.{cpu}_IOPS_1s_binning", cpu_out_dir, title)

        # HISTOGRAM: Dynamic binning logic.
        title = "Overall Throughput Over Time (binning: 1s)"
        #blktrace_plotting.plot_throughput_bar_dynamic_bin(cpu_data, f"{plot_basename}.{cpu}_IOPS_dynamic_binning_bar", cpu_out_dir, title)
        blktrace_plotting.plot_throughput_bar_dynamic_bin(cpu_data, f"{plot_basename}.{cpu}_IOPS_dynamic_binning_bar", cpu_out_dir, title, default_bin_count=50)


def visualize_latency_overview(df: pd.DataFrame, plot_basename: str, output_dir: str, title: str = "placeholder_title") -> None:
    """
    Generates an overview plot of latency over time across all CPUs with three binning methods.

    Args:
        df (pd.DataFrame): DataFrame with blkparse events.
        plot_basename (str): Basename for the plot output file.
        output_dir (str): Directory to save the plots.
    """
    # Calculate latency (difference between 'Q' and 'C' events)

    # This is when an I/O request is added to the queue.
    # set_index(["lba", "cpu"]) creates a multi-index on both queued and 
    # completed DataFrames. 
    # This allows merging based on these two columns.
    queued = df[df["event_type"] == "Q"].set_index(["lba", "cpu"])

    # This is when an I/O request is completed.
    # set_index(["lba", "cpu"]) creates a multi-index on both queued and 
    # completed DataFrames. 
    # This allows merging based on these two columns.
    completed = df[df["event_type"] == "C"].set_index(["lba", "cpu"])

    # The goal is to pair each 'Q' event with its corresponding 'C' event.
    # Rows with matching lba and cpu values are combined.
    # Timestamp columns are suffixed with _q and _c to distinguish 
    # between the queued and completed events.
    merged = queued.merge(completed, on=["lba", "cpu"], suffixes=("_q", "_c"))

    # For each matched pair of Q and C events.
    # Latency is calculated as the time difference between the completed and 
    # queued timestamps.
    merged["latency"] = merged["timestamp_c"] - merged["timestamp_q"]

    # Only positive latencies are valid (merged["latency"] >= 0). 
    # Negative latencies are excluded.
    # Result - a pandas Series containing positive latency values 
    # for each lba and cpu:
    valid_latencies: pd.Series = merged[merged["latency"] >= 0]["latency"]

    if valid_latencies.empty:
        print(f"Did not print the following due to no data:\n\t{plot_basename}")
        return
    
    print(valid_latencies)
    print(f"Length of valid_latencies: {len(valid_latencies)}")
    print(f"Data type: {valid_latencies.dtype}")
    print(f"Min latency: {valid_latencies.min()}, Max latency: {valid_latencies.max()}")  

    # Generates 50 bins (default) dynamically from the data.
    # Uniform bins across the full range of latency values.
    title: str = "Overall Latency Distribution (Static Binning)"
    blktrace_plotting.plot_latency_bar_static_50_bin_uniform(valid_latencies, f"{plot_basename}_50_static_binning", output_dir, title, default_bin_count=50)

    # Logarithmic Binning.
    # Logarithmic bins for a more detailed view of large ranges.
    title = "Overall Latency Distribution (Logarithmic Binning)"

    bins = np.logspace(np.log10(valid_latencies.min()), np.log10(valid_latencies.max()), 50)
    blktrace_plotting.plot_latency_logarithmic_bins(valid_latencies, bins, f"{plot_basename}_log", output_dir, title=f"{plot_basename}_log")

def visualize_latency_per_cpu(df: pd.DataFrame, plot_basename: str, output_dir: str) -> None:
    """
    Generates per-CPU visualizations of latency with three binning methods.

    Args:
        df (pd.DataFrame): DataFrame with blkparse events.
        plot_basename (str): Basename for the plot output file.
        output_dir (str): Directory to save the plots.
    """
    grouped = df.groupby("cpu")
    for cpu, cpu_data in grouped:
        queued = cpu_data[cpu_data["event_type"] == "Q"].set_index("lba")
        completed = cpu_data[cpu_data["event_type"] == "C"].set_index("lba")
        merged = queued.merge(completed, on="lba", suffixes=("_q", "_c"))

        # Compute latency
        merged["latency"] = merged["timestamp_c"] - merged["timestamp_q"]
        valid_latencies = merged[merged["latency"] >= 0]["latency"]

        if valid_latencies.empty:
            print(f"Did not print the following due to no data:\n\t{plot_basename}.{cpu}")
            continue

        # Set up per-CPU output directory.
        cpu_out_dir: str = os.path.join(output_dir, f"cpu-{cpu}")
        os.makedirs(cpu_out_dir, exist_ok=True)

        # Static Binning (Uniform).
        # Uniform bins across the full range of latency values.
        blktrace_plotting.plot_latency_bar_static_50_bin_uniform(valid_latencies, f"{plot_basename}_50_static_binning", output_dir, f"{plot_basename}.{cpu}_50_static_binning")

        # Dynamic Binning (Range).
        # Uniform bins, but limited to the range of the latency values (minimum to maximum).
        # blktrace_plotting.plot_latency_hist_dynamic_50_bin_uniform(valid_latencies, plot_basename, output_dir, title=f"{plot_basename}.{cpu}_50_dynamic_binning")

        # Logarithmic Binning.
        # Logarithmic bins for a more detailed view of large ranges.
        bins = np.logspace(np.log10(valid_latencies.min()), np.log10(valid_latencies.max()), 50)
        blktrace_plotting.plot_latency_logarithmic_bins(valid_latencies, bins, plot_basename, output_dir, title = f"{plot_basename}.{cpu}_log")

def visualize_queue_depth_overview(df: pd.DataFrame, plot_basename: str, output_dir: str) -> None:
    """
    Generates an overview plot of queue depth over time across all CPUs, using timestamp-based, dynamic binning, and 1-second binning.

    Args:
        df (pd.DataFrame): DataFrame with blkparse events.
        plot_basename (str): Basename for the plot output file.
        output_dir (str): Directory to save the plots.
    """
    # Calculate queue depth for timestamp-based binning
    queued_timestamp = df[df["event_type"] == "Q"].groupby("timestamp").size()
    completed_timestamp = df[df["event_type"] == "C"].groupby("timestamp").size()
    queue_depth_timestamp = (queued_timestamp - completed_timestamp).cumsum()

    # Timestamp-based queue depth
    if queue_depth_timestamp.empty:
        print(f"No data to plot for timestamp-based queue depth. Skipping...")
    else:

        
        blktrace_plotting.plot_queue_depth(queue_depth_timestamp, f"{plot_basename}_queue_depth_timestamp_binning", output_dir)
        # plt.figure(figsize=(10, 6))
        # plt.plot(queue_depth_timestamp.index, queue_depth_timestamp.values, label="Queue Depth (Timestamp-Based)")
        # plt.title("Queue Depth Over Time (Timestamp-Based)")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Queue Depth")
        # plt.legend()
        # plt.grid()
        # plt.xlim(left=0)
        # png_out_path = os.path.join(output_dir, f"{plot_basename}_queue_depth_timestamp_binning.png")
        # plt.savefig(png_out_path, dpi=300)
        # pdf_out_path = os.path.join(output_dir, f"{plot_basename}_queue_depth_timestamp_binning.pdf")
        # plt.savefig(pdf_out_path)
        # plt.close()

    # 1-second binning logic
    df["time_bin_1s"] = ((df["timestamp"] - df["timestamp"].min()) // 1).astype(int)
    queued_1s = df[df["event_type"] == "Q"].groupby("time_bin_1s").size()
    completed_1s = df[df["event_type"] == "C"].groupby("time_bin_1s").size()
    queue_depth_1s = (queued_1s - completed_1s).cumsum()

    print("Queued events (1-second binning):")
    print(queued_1s.head())
    print("Completed events (1-second binning):")
    print(completed_1s.head())

    if queue_depth_1s.empty:
        print(f"No data to plot for 1-second binning queue depth. Skipping...")
    else:
        blktrace_plotting.plot_queue_depth(queue_depth_1s, f"{plot_basename}_queue_depth_1s_binning", output_dir)
        # plt.figure(figsize=(10, 6))
        # plt.plot(queue_depth_1s.index, queue_depth_1s.values, label="Queue Depth (1-Second Binning)")
        # plt.title("Queue Depth Over Time (1-Second Binning)")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Queue Depth")
        # plt.legend()
        # plt.grid()
        # png_out_path = os.path.join(output_dir, f"{plot_basename}_queue_depth_1s_binning.png")
        # plt.savefig(png_out_path, dpi=300)
        # pdf_out_path = os.path.join(output_dir, f"{plot_basename}_queue_depth_1s_binning.pdf")
        # plt.savefig(pdf_out_path)
        # plt.close()

    # Dynamic binning logic
    min_timestamp = df["timestamp"].min()
    max_timestamp = df["timestamp"].max()

    if pd.isna(min_timestamp) or pd.isna(max_timestamp):
        print(f"No data available for dynamic binning. Skipping...")
        return

    time_range = max_timestamp - min_timestamp

    if time_range == 0:
        print(f"No variation in timestamps (all data at {min_timestamp}s). Skipping dynamic binning...")
        return

    num_bins = 50
    bin_size = time_range / num_bins
    print(f"Dynamic bin size: {bin_size:.6f} seconds")

    df["time_bin_dynamic"] = ((df["timestamp"] - min_timestamp) / bin_size).astype(int)
    queued_dynamic = df[df["event_type"] == "Q"].groupby("time_bin_dynamic").size()
    completed_dynamic = df[df["event_type"] == "C"].groupby("time_bin_dynamic").size()
    queue_depth_dynamic = (queued_dynamic - completed_dynamic).cumsum()
    x_values_dynamic = queue_depth_dynamic.index * bin_size + min_timestamp

    if queue_depth_dynamic.empty:
        print(f"No data to plot for dynamically binned queue depth. Skipping...")
        return

    # Plot dynamically binned queue depth
    plt.figure(figsize=(10, 6))
    plt.plot(x_values_dynamic, queue_depth_dynamic.values, label="Queue Depth (Dynamic Binning)")
    plt.title("Queue Depth Over Time (Dynamic Binning)")
    plt.xlabel("Time (s)")
    plt.ylabel("Queue Depth")
    plt.legend()
    plt.grid()
    png_out_path = os.path.join(output_dir, f"{plot_basename}_queue_depth_dynamic_binning.png")
    plt.savefig(png_out_path, dpi=300)
    pdf_out_path = os.path.join(output_dir, f"{plot_basename}_queue_depth_dynamic_binning.pdf")
    plt.savefig(pdf_out_path)
    plt.close()

def visualize_queue_depth_per_cpu(df: pd.DataFrame, plot_basename: str, output_dir: str) -> None:
    """
    Generates per-CPU visualizations of queue depth over time.

    Args:
        df (pd.DataFrame): DataFrame with blkparse events.
    """
    grouped = df.groupby("cpu")
    for cpu, cpu_data in grouped:
        cpu_data["time_bin"] = (cpu_data["timestamp"] // 1).astype(int)
        queued = cpu_data[cpu_data["event_type"] == "Q"].groupby("time_bin").size()
        completed = cpu_data[cpu_data["event_type"] == "C"].groupby("time_bin").size()
        queue_depth = (queued - completed).cumsum()

        if queue_depth.empty:
            print(f"Did not print the following due to no data:\n\t{plot_basename}.{cpu}")
            continue

        # Set up per-CPU output directory.
        cpu_out_dir: str = os.path.join(output_dir, f"cpu-{cpu}")
        os.makedirs(cpu_out_dir, exist_ok=True)

        # Plot queue depth for this CPU
        plt.figure(figsize=(10, 6))
        plt.plot(queue_depth.index, queue_depth.values, label=f"Queue Depth (CPU {cpu})")
        plt.title(f"Queue Depth Over Time for CPU {cpu}")
        plt.xlabel("Time (s)")
        plt.ylabel("Queue Depth")
        plt.legend()
        plt.grid()

        png_out_path = os.path.join(cpu_out_dir, f"{plot_basename}.{cpu}.png")
        plt.savefig(png_out_path, dpi=300)
        pdf_out_path = os.path.join(cpu_out_dir, f"{plot_basename}.{cpu}.pdf")
        plt.savefig(pdf_out_path)
        plt.close()

def plot_blktrace_data() -> None:
    pass

def check_and_get_blktrace_file(directory: str) -> str:
    """
    Check if the given directory contains a non-empty file ending in 'blktrace.merged.txt'.

    Args:
        directory (str): Path to the directory.

    Returns:
        bool: True if such a file exists and is non-empty, False otherwise.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"The path {directory} is not a valid directory.")
    
    for filename in os.listdir(directory):
        filename: str
        if filename.endswith(BLKTRACE_OUTPUT_FILE_SUFFIX):
            full_path: str = os.path.join(directory, filename)
            if os.path.isfile(full_path) and os.path.getsize(full_path) > 0:
                return full_path
    return ""

# python3 -m src.plot_blktrace_data -i "Outputs/ChunkGraph-LiveJournal-test/blktrace" -o "Outputs/ChunkGraph-LiveJournal-test/blktrace" --device-path "/dev/nvme0n1"

# python3 -m src.plot_blktrace_data -i "Outputs/ChunkGraph-LiveJournal-test/blktrace" -o "Outputs/ChunkGraph-LiveJournal-test/blktrace"

def main():

    # Parse arguments.
    parser: argparse.ArgumentParser = create_arg_parser()
    args: argparse.Namespace = parser.parse_args()

    # Validate inputs.
    #util_functions.validate_device(args.device_path)
    util_functions.validate_output_directory(args.output_dir)

    # Argument validation.
    if not util_functions.dir_exists_and_not_empty(args.input_dir):
        logger.error("Problem with '-i'/'--input-dir' parameter. Exiting.")
        sys.exit(1)

    # Check that the input directory contains a `traces` directory.
    sub_dirs: List[str] = ["traces"]
    if not util_functions.dir_contains_elements(args.input_dir, sub_dirs):
        logger.error("Expected to find the following directories:")
        
    # Check that the input directory contains a file ending in `blktrace.merged.txt`.
    trace_file_path: str = check_and_get_blktrace_file(args.input_dir)

    plot_basename: str = trace_file_path[
        trace_file_path.rfind(os.path.sep) + 1:
        trace_file_path.rfind(".merged.txt")
    ]

    # Read the execution context.
    pids: List[int] = []

    # Read `blktrace` data into a pandas.DataFrame.
    df: pd.DataFrame = util_functions.parse_blkparse_output(trace_file_path, pids)
    # TODO: Filter by `program_handle.pid()`.


    #os.makedirs(blktrace_output_root_dir_path, exist_ok=True)
    #logger.info(f"Created output directory:\n\t{blktrace_output_root_dir_path}")

    blktrace_plots_dir: str = os.path.join(args.input_dir, "plots")
    os.makedirs(blktrace_plots_dir, exist_ok=True)

    # Visualize throughput.
    visualize_throughput_overview(df, f"{plot_basename}-throughput", blktrace_plots_dir)
    visualize_throughput_per_cpu(df, f"{plot_basename}-throughput", blktrace_plots_dir)

    # Visualize latencies.
    visualize_latency_overview(df, f"{plot_basename}-latency", blktrace_plots_dir)
    visualize_latency_per_cpu(df, f"{plot_basename}-latency", blktrace_plots_dir)

    # Visualize queue depth.
    #visualize_queue_depth_overview(df, f"{plot_basename}-q_depth", blktrace_plots_dir)
    #visualize_queue_depth_per_cpu(df, f"{plot_basename}-q_depth", blktrace_plots_dir)

if __name__ == "__main__":
    main()

