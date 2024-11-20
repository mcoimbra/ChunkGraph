import argparse
import logging
import os
import pprint
import subprocess
import sys
import time
from typing import Any, IO, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
FRAMEWORKS_DIR: str = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "frameworks"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)

def check_privileges(elevated_flag: bool) -> None:
    """
    Checks if the script is running with elevated privileges.
    If not, prompts the user for their password to elevate to sudo.

    Args:
        elevated_flag (bool): Indicates if the script is already running with elevated privileges.
    """
    current_euid: int = os.geteuid()
    #logger.info(f"Current PID: , Elevated: {elevated_flag}")

    if current_euid != 0 and not elevated_flag:
        logger.info(f"[{os.getpid()}] - Attempting to elevate privileges with sudo...")
        try:
            # Re-run the script with sudo and add the --elevated flag
            command: list[str] = ["sudo", sys.executable] + sys.argv + ["--elevated"]
            logger.debug(f"[{os.getpid()}] - Re-running command: {command}")
            result: subprocess.CompletedProcess = subprocess.run(command)
            logger.debug(f"[{os.getpid()}] - Command finished with return code: {result.returncode}")
            if result.returncode != 0:
                logger.info(f"[{os.getpid()}] - Original process exiting.")
                sys.exit(1)
            else:
                print()
                logger.info(f"Privileges elevated successfully {os.getpid()}. Exiting original instance.")
                sys.exit(0)  # Successfully elevated privileges, exit original instance
        except Exception as e:
            logger.error(f"Error while trying to elevate privileges: {e}")
            sys.exit(1)
        return  # This return is redundant because sys.exit() is above.
    elif current_euid == 0:
        logger.info(f"Launched {os.getpid()} with elevated privileges.")
    else:
        logger.info(f"Script {os.getpid()} is running as an elevated instance.")

def validate_device(device_path: str) -> None:
    if not os.path.exists(device_path):
        logger.error(f"Device '{device_path}' does not exist.")
        sys.exit(1)
    logger.info(f"Validated device:\n\t{device_path}")

def validate_output_directory(output_dir: str) -> None:
    if os.path.isfile(output_dir):
        logger.error(f"The path '{output_dir}' points to a file, not a directory.")
        sys.exit(1)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Created output directory:\n\t{output_dir}")
        except OSError as e:
            logger.error(f"Error creating directory\n\t'{output_dir}': {e}")
            sys.exit(1)
    else:
        logger.info(f"Validated output directory:\n\t{output_dir}")

def run_program_and_blktrace(framework: str, binary_args: List[str], device_path: str, output_dir: str) -> None:
    """
    Runs a program and starts blktrace on a specified device.

    Args:
        device (str): Path to the device to be monitored (e.g., /dev/sda).
        output_dir (str): Directory to save the blktrace output.
        program (str): Path to the program to be executed.

    Returns:
        None
    """
    # Start the program
    #args_str: str = "\n\t".join(binary_args)
    #logger.info(f"Starting the program:\n\t{args_str}\n")
    logger.info(f"Starting the program:\n\t{' '.join(binary_args)}\n")

    #pprint.pprint(binary_args)
    #sys.exit(0)
    #program_args: List[str] = program.split()
    program_stdout_path: str = os.path.join(output_dir, f"{framework}_stdout.log")
    program_stderr_path: str = os.path.join(output_dir, f"{framework}_stderr.log")
    program_stdout: IO = open(program_stdout_path, 'w')
    program_stderr: IO = open(program_stderr_path, 'w')
    try:
        program_proc: subprocess.Popen = subprocess.Popen(
            binary_args,
            stdout=program_stdout,
            stderr=program_stderr,
            text=True
        )
    except FileNotFoundError:
        logger.error(f"Program not found:\n\t'{binary_args[0]}'")
        sys.exit(1)
    
    logger.info(f"Program started with PID: {program_proc.pid}\n")

    # Start blktrace.
    #blktrace_output_file: str = os.path.join(output_dir, "blktrace_trace")
    bltrace_output_dir_path: str = output_dir
    blktrace_stdout_path: str = os.path.join(bltrace_output_dir_path, f"blktrace_stdout.log")
    blktrace_stderr_path: str = os.path.join(bltrace_output_dir_path, f"blktrace_stderr.log")
    blktrace_stdout: IO = open(blktrace_stdout_path, 'w')
    blktrace_stderr: IO = open(blktrace_stderr_path, 'w')
    try:
        logger.info(f"Effective UID for 'blktrace': {os.geteuid()}")
        blktrace_args: List[str] = ["blktrace", "-d", device_path, f"--output-dir={bltrace_output_dir_path}"]
        logger.info(f"Starting blktrace on device '{device_path}'...\n\t{' '.join(blktrace_args)}")
        blktrace_proc: subprocess.Popen = subprocess.Popen(
            #["sudo", "blktrace", "-d", device, "-o", blktrace_output_file],
            #["sudo", "blktrace", "-d", device, f"--output-dir={bltrace_output_dir_path}"],
            blktrace_args,
            stdout=blktrace_stdout,
            stderr=blktrace_stderr,
            text=True
        )
        logger.info(f"blktrace started with PID: {blktrace_proc.pid}")
    except FileNotFoundError:
        logger.error("blktrace not found. Ensure it is installed and available in the PATH.")
        program_proc.terminate()
        sys.exit(1)

    # Wait for the program to finish.
    program_proc.wait()
    program_stdout.close()
    program_stderr.close()
    os.system("stty sane")
    
    logger.info(f"Program finished with exit code: {program_proc.returncode}")

    # Stop blktrace.
    logger.info("Stopping blktrace...")

    blktrace_proc.terminate()
    #blktrace_proc.wait()

    try:
        blktrace_proc.wait(timeout=3)  # Wait up to 10 seconds for graceful termination
    except subprocess.TimeoutExpired:
        logger.warning("'blktrace' did not terminate gracefully. Forcing termination...")
        blktrace_proc.kill()  # Force terminate the process
        blktrace_proc.wait()

    blktrace_stdout.close()
    blktrace_stderr.close()

    if blktrace_proc.returncode == 0:
        logger.info(f"blktrace finished with exit code: {blktrace_proc.returncode}")
    else:
        logger.error(f"blktrace finished with exit code: {blktrace_proc.returncode}")
        logger.error(f"Exiting due to blktrace error.")
        sys.exit(1)

    logger.info(f"blktrace output saved to directory:\n\t{bltrace_output_dir_path}")

    # Output 'blktrace' statistics.
    # Step 1: Merge per-CPU files
    print()
    logger.info(f"Effective UID for 'blkparse': {os.geteuid()}")
    device: str = device_path[device_path.rfind(os.path.sep) + 1 :]
    merge_target_path: str = os.path.join(output_dir, f"{device}.blktrace.merged.txt")
    logger.info(f"Starting blkparse to create:\n\t{merge_target_path}")
    merge_blktrace_files(device, merge_target_path, output_dir)

    # Step 2: Parse the merged output
    pids: List[int] = [program_proc.pid]
    df: pd.DataFrame = parse_blkparse_output(merge_target_path, pids)

    # Filter by relevant process IDs.


    # Step 3: Visualize 'blktrace' data.
    plot_basename: str = f"{device}.blktrace"

    # Visualize throughput
    f"{plot_basename}-"
    visualize_throughput_overview(df, f"{plot_basename}-throughput", output_dir)
    visualize_throughput_per_cpu(df, f"{plot_basename}-throughput", output_dir)

    #sys.exit()

    # Visualize latencies.
    # TODO: print these plots with histogram bars rather than a curve
    visualize_latency_overview(df, f"{plot_basename}-latency", output_dir)
    visualize_latency_per_cpu(df, f"{plot_basename}-latency", output_dir)

    # Visualize queue depth.
    # TODO: add dynamic binning as well and check if histograms make sense. 
    visualize_queue_depth_overview(df, f"{plot_basename}-q_depth", output_dir)
    visualize_queue_depth_per_cpu(df, f"{plot_basename}-q_depth", output_dir)

def merge_blktrace_files(device: str, output_file: str, blktrace_dir: str) -> None:
    """
    Merges per-CPU blktrace files into a unified trace file using blkparse.

    Args:
        blktrace_dir (str): Directory containing blktrace per-CPU files.
        output_file (str): Path to save the merged output.
    """
    try:
        # Merge all blktrace files (e.g., blktrace.cpu0, blktrace.cpu1, ...)
        blkparse_args: List[str] = ["blkparse", "-i", f"{blktrace_dir}/{device}.blktrace.*", "-o", output_file]
        subprocess.run(
            blkparse_args,
            #["blkparse", "-i", f"{blktrace_dir}/blktrace.cpu*", "-o", output_file],
            check=True,
            text=True
        )
        logger.info(f"Merged blktrace output saved to\n\t{output_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"blkparse exception:\n\t{e}\n\t{' '.join(blkparse_args)}")

def parse_blkparse_output(file_path: str, pids: List[int]) -> pd.DataFrame:
    """
    Parses blkparse output into a DataFrame.

    Args:
        file_path (str): Path to the blkparse text output file.

    Returns:
        pd.DataFrame: Parsed data with columns for timestamp, CPU, operation, etc.
    """
    data: pd.DataFrame = []

    with open(file_path, "r") as f:
        for line in f:
            line: str
            parts: List[str] = line.split()
            if len(parts) > 6 and parts[3].replace(".", "").isdigit():
                try:
                    timestamp: float = float(parts[3])
                    cpu: int = int(parts[1])  # CPU ID
                    event_type: str = parts[5]
                    operation: str = parts[6]
                    
                    # Ensure LBA is a valid integer
                    lba: Optional[int] = int(parts[7]) if len(parts) > 7 and parts[7].isdigit() else None
                    
                    # Ensure blocks are valid
                    blocks: Optional[int] = int(parts[9][1:]) if len(parts) > 9 and parts[9].startswith("+") else None

                    # Append the parsed data
                    data.append({
                        "timestamp": timestamp,
                        "cpu": cpu,
                        "event_type": event_type,
                        "operation": operation,
                        "lba": lba,
                        "blocks": blocks,
                    })
                except ValueError as e:
                    # Log or print the line causing the issue (optional for debugging)
                    print(f"Skipping line due to error: {line.strip()} - {e}")
                    continue
    return pd.DataFrame(data)

def visualize_throughput_overview(df: pd.DataFrame, plot_basename: str, output_dir: str) -> None:
    """
    Generates an overview plot of throughput over time across all CPUs, using timestamp-based, dynamic binning, and 1-second binning.

    Args:
        df (pd.DataFrame): DataFrame with blkparse events.
        plot_basename (str): Basename for the plot output file.
        output_dir (str): Directory to save the plots.
    """
    # Plot using raw timestamps (original method)
    throughput_timestamp = df.groupby("timestamp").size()

    # Timestamp-based throughput
    plt.figure(figsize=(10, 6))
    plt.plot(throughput_timestamp.index, throughput_timestamp.values, label="Throughput (IOPS)")
    print("Throughput (timestamp-based):")
    print(throughput_timestamp.head())

    if throughput_timestamp.empty:
        print(f"No data to plot for timestamp-based throughput. Skipping...")
    else:
        plt.title("Overall Throughput Over Time (Timestamp-Based)")
        plt.xlabel("Time (s)")
        plt.ylabel("Throughput (IOPS)")
        plt.legend()
        plt.grid()
        plt.xlim(left=0)
        png_out_path = os.path.join(output_dir, f"{plot_basename}_IOPS_timestamp_binning.png")
        plt.savefig(png_out_path, dpi=300)
        pdf_out_path = os.path.join(output_dir, f"{plot_basename}_IOPS_timestamp_binning.pdf")
        plt.savefig(pdf_out_path)
        plt.close()

    # 1-second binning logic
    #df["time_bin_1s"] = (df["timestamp"] // 1).astype(int)  # Floor timestamps to nearest second

    # TODO: check why this is true: The negative X-axis values indicate an issue with how the binning for time_bin_1s is being calculated. Specifically, the timestamp column may contain negative or very small values that result in negative bin indices when using (df["timestamp"] // 1).
    df["time_bin_1s"] = ((df["timestamp"] - df["timestamp"].min()) // 1).astype(int)

    throughput_1s = df.groupby("time_bin_1s").size()

    if throughput_1s.empty:
        print(f"No data to plot for 1-second binning. Skipping...")
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(throughput_1s.index, throughput_1s.values, label="Throughput (IOPS, 1s Binning)")
        print("Throughput (1-second binning):")
        print(throughput_1s.head())

        plt.title("Overall Throughput Over Time (1-Second Binning)")
        plt.xlabel("Time (s)")
        plt.ylabel("Throughput (IOPS)")
        plt.legend()
        plt.grid()
        png_out_path = os.path.join(output_dir, f"{plot_basename}_IOPS_1s_binning.png")
        plt.savefig(png_out_path, dpi=300)
        pdf_out_path = os.path.join(output_dir, f"{plot_basename}_IOPS_1s_binning.pdf")
        plt.savefig(pdf_out_path)
        plt.close()

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

    # Determine dynamic bin size
    #     The dynamic binning logic groups timestamps into wider intervals (bins) based on the range of data and the chosen number of bins (num_bins=50). Wider bins aggregate events that fall within the same time interval, which can result in varying throughput values for each bin:

    #     If a bin has multiple events, the throughput will be higher.
    #     If a bin has fewer events, the throughput will be lower.

    # This variability produces the observed curve in the "dynamic_binning" plot.
    num_bins = 50  # Default number of bins
    bin_size = time_range / num_bins
    print(f"Dynamic bin size: {bin_size:.6f} seconds")

    # Create time bins and calculate throughput
    df["time_bin_dynamic"] = ((df["timestamp"] - min_timestamp) / bin_size).astype(int)
    throughput_dynamic = df.groupby("time_bin_dynamic").size()
    x_values_dynamic = throughput_dynamic.index * bin_size + min_timestamp

    if throughput_dynamic.empty:
        print(f"No data to plot for dynamically binned throughput. Skipping...")
        return

    # Plot dynamically binned throughput
    plt.figure(figsize=(10, 6))
    plt.plot(x_values_dynamic, throughput_dynamic.values, label="Throughput (IOPS)")
    print("Throughput (dynamic binning):")
    print(throughput_dynamic.head())

    plt.title("Overall Throughput Over Time (Dynamic Binning)")
    plt.xlabel("Time (s)")
    plt.ylabel("Throughput (IOPS)")
    plt.legend()
    plt.grid()
    png_out_path = os.path.join(output_dir, f"{plot_basename}_IOPS_dynamic_binning.png")
    plt.savefig(png_out_path, dpi=300)
    pdf_out_path = os.path.join(output_dir, f"{plot_basename}_IOPS_dynamic_binning.pdf")
    plt.savefig(pdf_out_path)
    plt.close()

# def visualize_throughput_overview(df: pd.DataFrame, plot_basename: str, output_dir: str) -> None:
#     """
#     Generates an overview plot of throughput over time across all CPUs, using both timestamp-based and dynamic binning.

#     Args:
#         df (pd.DataFrame): DataFrame with blkparse events.
#         plot_basename (str): Basename for the plot output file.
#         output_dir (str): Directory to save the plots.
#     """
#     # Plot using raw timestamps (original method)
#     throughput_timestamp = df.groupby("timestamp").size()

#     # Original method: Throughput plot by timestamp
#     plt.figure(figsize=(10, 6))
#     plt.plot(throughput_timestamp.index, throughput_timestamp.values, label="Throughput (IOPS)")
#     print("Throughput (timestamp-based):")
#     print(throughput_timestamp.head())

#     if throughput_timestamp.empty:
#         print(f"No data to plot for timestamp-based throughput. Skipping...")
#     else:
#         plt.title("Overall Throughput Over Time (Timestamp-Based)")
#         plt.xlabel("Time (s)")
#         plt.ylabel("Throughput (IOPS)")
#         plt.legend()
#         plt.grid()
#         png_out_path = os.path.join(output_dir, f"{plot_basename}_IOPS_timestamp_binning.png")
#         plt.savefig(png_out_path, dpi=300)
#         pdf_out_path = os.path.join(output_dir, f"{plot_basename}_IOPS_timestamp_binning.pdf")
#         plt.savefig(pdf_out_path)
#         plt.close()

#     # Dynamic binning logic
#     min_timestamp = df["timestamp"].min()
#     max_timestamp = df["timestamp"].max()

#     if pd.isna(min_timestamp) or pd.isna(max_timestamp):
#         print(f"No data available for dynamic binning. Skipping...")
#         return

#     time_range = max_timestamp - min_timestamp

#     if time_range == 0:
#         print(f"No variation in timestamps (all data at {min_timestamp}s). Skipping dynamic binning...")
#         return

#     # Determine dynamic bin size
#     num_bins = 50  # Default number of bins
#     bin_size = time_range / num_bins
#     print(f"Dynamic bin size: {bin_size:.6f} seconds")

#     # Create time bins and calculate throughput
#     df["time_bin"] = ((df["timestamp"] - min_timestamp) / bin_size).astype(int)
#     throughput_dynamic = df.groupby("time_bin").size()
#     x_values_dynamic = throughput_dynamic.index * bin_size + min_timestamp

#     if throughput_dynamic.empty:
#         print(f"No data to plot for dynamically binned throughput. Skipping...")
#         return

#     # Plot dynamically binned throughput
#     plt.figure(figsize=(10, 6))
#     plt.plot(x_values_dynamic, throughput_dynamic.values, label="Throughput (IOPS)")
#     print("Throughput (dynamic binning):")
#     print(throughput_dynamic.head())

#     plt.title("Overall Throughput Over Time (Dynamic Binning)")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Throughput (IOPS)")
#     plt.legend()
#     plt.grid()
#     png_out_path = os.path.join(output_dir, f"{plot_basename}_IOPS_dynamic_binning.png")
#     plt.savefig(png_out_path, dpi=300)
#     pdf_out_path = os.path.join(output_dir, f"{plot_basename}_IOPS_dynamic_binning.pdf")
#     plt.savefig(pdf_out_path)
#     plt.close()


# def visualize_throughput_overview(df: pd.DataFrame, plot_basename: str, output_dir: str) -> None:
#     """
#     Generates an overview plot of throughput over time across all CPUs.

#     Args:
#         df (pd.DataFrame): DataFrame with blkparse events.
#     """
#     # Group by time (in seconds) for throughput
#     df["time_bin"] = (df["timestamp"] // 1).astype(int)

#     # For plotting throughput over time, always group by time intervals using 
#     # time_bin to align with the definition of throughput (e.g., IOPS).
#     #throughput: Any[pd.DataFrame, pd.Series[int]] = df.groupby("time_bin").size()
#     throughput: Any[pd.DataFrame, pd.Series[int]] = df.groupby("timestamp").size()

#     # Plot.
#     plt.figure(figsize=(10, 6))
#     plt.plot(throughput.index, throughput.values, label="Throughput (IOPS)")
#     print(throughput.head())  # Check first few rows
#     print("X-axis values (Time bins):")
#     print(throughput.index)
#     print("Y-axis values (IOPS):")
#     print(throughput.values)

#     # Debugging: Print data and grouping results
#     print("Parsed DataFrame:")
#     print(df.head())
#     print("Throughput Data:")
#     print(throughput.head())

#     if throughput.empty:
#         print(f"No data to plot. Skipping...")
#         return

#     # Make sure the X-axis starts from 0.
#     plt.xlim(left=0)
    
#     plt.title("Overall Throughput Over Time")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Throughput (IOPS)")
#     plt.legend()
#     plt.grid()

#     png_out_path: str = os.path.join(output_dir, f"{plot_basename}_per_second_(IOPS).png")
#     plt.savefig(png_out_path, dpi=300)
#     pdf_out_path: str = os.path.join(output_dir, f"{plot_basename}_per_second_(IOPS).pdf")
#     plt.savefig(pdf_out_path)
#     plt.close()

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
        # Plot using raw timestamps (original method)
        throughput_timestamp = cpu_data.groupby("timestamp").size()

        # Timestamp-based throughput
        plt.figure(figsize=(10, 6))
        plt.plot(throughput_timestamp.index, throughput_timestamp.values, label=f"CPU {cpu} Throughput (IOPS)")
        print(f"Throughput (timestamp-based) for CPU {cpu}:")
        print(throughput_timestamp.head())

        if throughput_timestamp.empty:
            print(f"No data to plot for CPU {cpu} (timestamp-based). Skipping...")
        else:
            plt.title(f"Throughput Over Time for CPU {cpu} (Timestamp-Based)")
            plt.xlabel("Time (s)")
            plt.ylabel("Throughput (IOPS)")
            plt.legend()
            plt.grid()
            png_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}_IOPS_timestamp_binning.png")
            plt.savefig(png_out_path, dpi=300)
            pdf_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}_IOPS_timestamp_binning.pdf")
            plt.savefig(pdf_out_path)
            plt.close()

        # 1-second binning logic
        cpu_data["time_bin_1s"] = (cpu_data["timestamp"] // 1).astype(int)  # Floor timestamps to nearest second
        throughput_1s = cpu_data.groupby("time_bin_1s").size()

        if throughput_1s.empty:
            print(f"No data to plot for CPU {cpu} (1-second binning). Skipping...")
        else:
            plt.figure(figsize=(10, 6))
            plt.plot(throughput_1s.index, throughput_1s.values, label=f"CPU {cpu} Throughput (IOPS, 1s Binning)")
            print(f"Throughput (1-second binning) for CPU {cpu}:")
            print(throughput_1s.head())

            plt.title(f"Throughput Over Time for CPU {cpu} (1-Second Binning)")
            plt.xlabel("Time (s)")
            plt.ylabel("Throughput (IOPS)")
            plt.legend()
            plt.grid()
            png_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}_IOPS_1s_binning.png")
            plt.savefig(png_out_path, dpi=300)
            pdf_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}_IOPS_1s_binning.pdf")
            plt.savefig(pdf_out_path)
            plt.close()

        # Dynamic binning logic
        min_timestamp = cpu_data["timestamp"].min()
        max_timestamp = cpu_data["timestamp"].max()

        if pd.isna(min_timestamp) or pd.isna(max_timestamp):
            print(f"No data available for CPU {cpu} (dynamic binning). Skipping...")
            continue

        time_range = max_timestamp - min_timestamp

        if time_range == 0:
            print(f"No variation in timestamps for CPU {cpu} (all data at {min_timestamp}s). Skipping dynamic binning...")
            continue

        # Determine dynamic bin size
        num_bins = 50  # Default number of bins
        bin_size = time_range / num_bins
        print(f"Dynamic bin size for CPU {cpu}: {bin_size:.6f} seconds")

        # Create time bins and calculate throughput
        cpu_data["time_bin_dynamic"] = ((cpu_data["timestamp"] - min_timestamp) / bin_size).astype(int)
        throughput_dynamic = cpu_data.groupby("time_bin_dynamic").size()
        x_values_dynamic = throughput_dynamic.index * bin_size + min_timestamp

        if throughput_dynamic.empty:
            print(f"No data to plot for CPU {cpu} (dynamic binning). Skipping...")
            continue

        # Plot dynamically binned throughput
        plt.figure(figsize=(10, 6))
        plt.plot(x_values_dynamic, throughput_dynamic.values, label=f"CPU {cpu} Throughput (IOPS)")
        print(f"Throughput (dynamic binning) for CPU {cpu}:")
        print(throughput_dynamic.head())

        plt.title(f"Throughput Over Time for CPU {cpu} (Dynamic Binning)")
        plt.xlabel("Time (s)")
        plt.ylabel("Throughput (IOPS)")
        plt.legend()
        plt.grid()
        png_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}_IOPS_dynamic_binning.png")
        plt.savefig(png_out_path, dpi=300)
        pdf_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}_IOPS_dynamic_binning.pdf")
        plt.savefig(pdf_out_path)
        plt.close()


# def visualize_throughput_per_cpu(df: pd.DataFrame, plot_basename: str, output_dir: str) -> None:
#     """
#     Generates per-CPU visualizations of throughput over time using both timestamp-based and dynamic binning.

#     Args:
#         df (pd.DataFrame): DataFrame with blkparse events.
#         plot_basename (str): Basename for the plot output file.
#         output_dir (str): Directory to save the plots.
#     """
#     # Group data by CPU.
#     grouped = df.groupby("cpu")
#     for cpu, cpu_data in grouped:
#         # Plot using raw timestamps (original method)
#         throughput_timestamp = cpu_data.groupby("timestamp").size()

#         # Original method: Throughput plot by timestamp
#         plt.figure(figsize=(10, 6))
#         plt.plot(throughput_timestamp.index, throughput_timestamp.values, label=f"CPU {cpu} Throughput (IOPS)")
#         print(f"Throughput (timestamp-based) for CPU {cpu}:")
#         print(throughput_timestamp.head())

#         if throughput_timestamp.empty:
#             print(f"No data to plot for CPU {cpu} (timestamp-based). Skipping...")
#         else:
#             plt.title(f"Throughput Over Time for CPU {cpu} (Timestamp-Based)")
#             plt.xlabel("Time (s)")
#             plt.ylabel("Throughput (IOPS)")
#             plt.legend()
#             plt.grid()
#             png_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}_IOPS_timestamp_binning.png")
#             plt.savefig(png_out_path, dpi=300)
#             pdf_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}_IOPS_timestamp_binning.pdf")
#             plt.savefig(pdf_out_path)
#             plt.close()

#         # Dynamic binning logic
#         min_timestamp = cpu_data["timestamp"].min()
#         max_timestamp = cpu_data["timestamp"].max()

#         if pd.isna(min_timestamp) or pd.isna(max_timestamp):
#             print(f"No data available for CPU {cpu} (dynamic binning). Skipping...")
#             continue

#         time_range = max_timestamp - min_timestamp

#         if time_range == 0:
#             print(f"No variation in timestamps for CPU {cpu} (all data at {min_timestamp}s). Skipping dynamic binning...")
#             continue

#         # Determine dynamic bin size
#         num_bins = 50  # Default number of bins
#         bin_size = time_range / num_bins
#         print(f"Dynamic bin size for CPU {cpu}: {bin_size:.6f} seconds")

#         # Create time bins and calculate throughput
#         cpu_data["time_bin"] = ((cpu_data["timestamp"] - min_timestamp) / bin_size).astype(int)
#         throughput_dynamic = cpu_data.groupby("time_bin").size()
#         x_values_dynamic = throughput_dynamic.index * bin_size + min_timestamp

#         if throughput_dynamic.empty:
#             print(f"No data to plot for CPU {cpu} (dynamic binning). Skipping...")
#             continue

#         # Plot dynamically binned throughput
#         plt.figure(figsize=(10, 6))
#         plt.plot(x_values_dynamic, throughput_dynamic.values, label=f"CPU {cpu} Throughput (IOPS)")
#         print(f"Throughput (dynamic binning) for CPU {cpu}:")
#         print(throughput_dynamic.head())

#         plt.title(f"Throughput Over Time for CPU {cpu} (Dynamic Binning)")
#         plt.xlabel("Time (s)")
#         plt.ylabel("Throughput (IOPS)")
#         plt.legend()
#         plt.grid()
#         png_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}_IOPS_dynamic_binning.png")
#         plt.savefig(png_out_path, dpi=300)
#         pdf_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}_IOPS_dynamic_binning.pdf")
#         plt.savefig(pdf_out_path)
#         plt.close()


# def visualize_throughput_per_cpu(df: pd.DataFrame, plot_basename: str, output_dir: str) -> None:
#     """
#     Generates per-CPU visualizations of throughput over time.

#     Args:
#         df (pd.DataFrame): DataFrame with blkparse events.
#     """
#     # Group data by CPU.
#     grouped = df.groupby("cpu")
#     for cpu, cpu_data in grouped:
#         cpu_data["time_bin"] = (cpu_data["timestamp"] // 1).astype(int)

#         # For plotting throughput over time, always group by time intervals using 
#         # time_bin to align with the definition of throughput (e.g., IOPS).
#         #throughput: Any[pd.DataFrame, pd.Series[int]] = cpu_data.groupby("time_bin").size()
#         throughput: Any[pd.DataFrame, pd.Series[int]] = df.groupby("timestamp").size()

#         # TODO: operate only on data from the PID of the graph processing out-of-core framework.

#         # Plot.
#         plt.figure(figsize=(10, 6))
#         plt.plot(throughput.index, throughput.values, label=f"CPU {cpu} Throughput (IOPS)")
#         #print(throughput.head())  # Check first few rows

#         if throughput.empty:
#             print(f"No data to plot for CPU {cpu}. Skipping...")
#             continue  # Skip if no data is available

#         # Make sure the X-axis starts from 0.
#         plt.xlim(left=0)

#         plt.title(f"Throughput Over Time for CPU {cpu}")
#         plt.xlabel("Time (s)")
#         plt.ylabel("Throughput (IOPS)")
#         plt.legend()
#         plt.grid()

#         png_out_path: str = os.path.join(output_dir, f"{plot_basename}.{cpu}.png")
#         plt.savefig(png_out_path, dpi=300)
#         pdf_out_path: str = os.path.join(output_dir, f"{plot_basename}.{cpu}.pdf")
#         plt.savefig(pdf_out_path)
#         plt.close()

def visualize_latency_overview(df: pd.DataFrame, plot_basename: str, output_dir: str) -> None:
    """
    Generates an overview plot of latency over time across all CPUs with three binning methods.

    Args:
        df (pd.DataFrame): DataFrame with blkparse events.
        plot_basename (str): Basename for the plot output file.
        output_dir (str): Directory to save the plots.
    """
    # Calculate latency (difference between 'Q' and 'C' events)
    queued = df[df["event_type"] == "Q"].set_index(["lba", "cpu"])
    completed = df[df["event_type"] == "C"].set_index(["lba", "cpu"])
    merged = queued.merge(completed, on=["lba", "cpu"], suffixes=("_q", "_c"))

    # Compute latency
    merged["latency"] = merged["timestamp_c"] - merged["timestamp_q"]
    valid_latencies = merged[merged["latency"] >= 0]["latency"]

    if valid_latencies.empty:
        print(f"Did not print the following due to no data:\n\t{plot_basename}")
        return
    
    # Static Binning (Uniform).
    # Uniform bins across the full range of latency values.
    plt.figure(figsize=(10, 6))
    plt.hist(valid_latencies, bins=50, edgecolor="black", label="Latency Distribution")
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

    # Dynamic Binning (Range).
    # Dynamic Binning: Uniform bins, but limited to the range of the latency values (minimum to maximum).
    plt.figure(figsize=(10, 6))
    plt.hist(valid_latencies, bins=50, range=(valid_latencies.min(), valid_latencies.max()), edgecolor="black", label="Latency Distribution")
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

    # Logarithmic Binning.
    # Logarithmic bins for a more detailed view of large ranges.
    bins = np.logspace(np.log10(valid_latencies.min()), np.log10(valid_latencies.max()), 50)
    plt.figure(figsize=(10, 6))
    plt.hist(valid_latencies, bins=bins, edgecolor="black", label="Latency Distribution")
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


# def visualize_latency_overview(df: pd.DataFrame, plot_basename: str, output_dir: str) -> None:
#     """
#     Generates an overview plot of latency over time across all CPUs.

#     Args:
#         df (pd.DataFrame): DataFrame with blkparse events.
#         plot_basename (str): Basename for the plot output file.
#         output_dir (str): Directory to save the plots.
#     """
#     # Calculate latency (difference between 'Q' and 'C' events)
#     queued = df[df["event_type"] == "Q"].set_index(["lba", "cpu"])
#     completed = df[df["event_type"] == "C"].set_index(["lba", "cpu"])
#     merged = queued.merge(completed, on=["lba", "cpu"], suffixes=("_q", "_c"))

#     # Compute latency
#     merged["latency"] = merged["timestamp_c"] - merged["timestamp_q"]
#     valid_latencies = merged[merged["latency"] >= 0]["latency"]

#     if valid_latencies.empty:
#         print(f"Did not print the following due to no data:\n\t{plot_basename}")
#         return

#     # Plot latency distribution
#     plt.figure(figsize=(10, 6))
#     plt.hist(valid_latencies, bins=50, edgecolor="black", label="Latency Distribution")
#     plt.title("Overall Latency Distribution")
#     plt.xlabel("Latency (s)")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.grid()

#     # Save the plots
#     png_out_path = os.path.join(output_dir, f"{plot_basename}.png")
#     plt.savefig(png_out_path, dpi=300)
#     pdf_out_path = os.path.join(output_dir, f"{plot_basename}.pdf")
#     plt.savefig(pdf_out_path)
#     plt.close()


# def visualize_latency_overview(df: pd.DataFrame, plot_basename: str, output_dir: str) -> None:
#     """
#     Generates an overview plot of latency over time across all CPUs.

#     Args:
#         df (pd.DataFrame): DataFrame with blkparse events.
#     """
#     # Calculate latency (difference between 'Q' and 'C' events)
#     queued = df[df["event_type"] == "Q"].set_index(["lba", "cpu"])
#     completed = df[df["event_type"] == "C"].set_index(["lba", "cpu"])
#     latency = (completed["timestamp"] - queued["timestamp"]).dropna()

#     if latency.empty:
#         print(f"Did not print the following due to no data:\n\t{plot_basename}")
#         return

#     # Plot latency distribution
#     plt.figure(figsize=(10, 6))
#     plt.hist(latency, bins=50, edgecolor="black", label="Latency Distribution")
#     plt.title("Overall Latency Distribution")
#     plt.xlabel("Latency (s)")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.grid()

#     png_out_path = os.path.join(output_dir, f"{plot_basename}.png")
#     plt.savefig(png_out_path, dpi=300)
#     pdf_out_path = os.path.join(output_dir, f"{plot_basename}.pdf")
#     plt.savefig(pdf_out_path)
#     plt.close()

# def visualize_latency_per_cpu(df: pd.DataFrame, plot_basename: str, output_dir: str) -> None:
#     """
#     Generates per-CPU visualizations of latency.

#     Args:
#         df (pd.DataFrame): DataFrame with blkparse events.
#     """
#     grouped = df.groupby("cpu")
#     for cpu, cpu_data in grouped:
#         queued = cpu_data[cpu_data["event_type"] == "Q"].set_index("lba")
#         completed = cpu_data[cpu_data["event_type"] == "C"].set_index("lba")
#         latency = (completed["timestamp"] - queued["timestamp"]).dropna()

#         if latency.empty:
#             print(f"Did not print the following due to no data:\n\t{plot_basename}.{cpu}")
#             continue

#         # Plot latency distribution for this CPU
#         plt.figure(figsize=(10, 6))
#         plt.hist(latency, bins=50, edgecolor="black", label=f"Latency Distribution (CPU {cpu})")
#         plt.title(f"Latency Distribution for CPU {cpu}")
#         plt.xlabel("Latency (s)")
#         plt.ylabel("Frequency")
#         plt.legend()
#         plt.grid()

#         png_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}.png")
#         plt.savefig(png_out_path, dpi=300)
#         pdf_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}.pdf")
#         plt.savefig(pdf_out_path)
#         plt.close()

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

        # Static Binning (Uniform).
        # Uniform bins across the full range of latency values.
        plt.figure(figsize=(10, 6))
        plt.hist(valid_latencies, bins=50, edgecolor="black", label=f"Latency Distribution (CPU {cpu})")
        plt.title(f"Latency Distribution for CPU {cpu} (Static Binning)")
        plt.xlabel("Latency (s)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid()
        png_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}_50_static_binning.png")
        plt.savefig(png_out_path, dpi=300)
        pdf_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}_50_static_binning.pdf")
        plt.savefig(pdf_out_path)
        plt.close()

        # Dynamic Binning (Range).
        # Uniform bins, but limited to the range of the latency values (minimum to maximum).
        plt.figure(figsize=(10, 6))
        plt.hist(valid_latencies, bins=50, range=(valid_latencies.min(), valid_latencies.max()), edgecolor="black", label=f"Latency Distribution (CPU {cpu})")
        plt.title(f"Latency Distribution for CPU {cpu} (Dynamic Binning)")
        plt.xlabel("Latency (s)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid()
        png_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}_50_dynamic_binning.png")
        plt.savefig(png_out_path, dpi=300)
        pdf_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}_50_dynamic_binning.pdf")
        plt.savefig(pdf_out_path)
        plt.close()

        # Logarithmic Binning.
        # Logarithmic bins for a more detailed view of large ranges.
        bins = np.logspace(np.log10(valid_latencies.min()), np.log10(valid_latencies.max()), 50)
        plt.figure(figsize=(10, 6))
        plt.hist(valid_latencies, bins=bins, edgecolor="black", label=f"Latency Distribution (CPU {cpu})")
        plt.xscale("log")
        plt.title(f"Latency Distribution for CPU {cpu} (Logarithmic Binning)")
        plt.xlabel("Latency (s, log scale)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid()
        png_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}_log.png")
        plt.savefig(png_out_path, dpi=300)
        pdf_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}_log.pdf")
        plt.savefig(pdf_out_path)
        plt.close()


# def visualize_latency_per_cpu(df: pd.DataFrame, plot_basename: str, output_dir: str) -> None:
#     """
#     Generates per-CPU visualizations of latency.

#     Args:
#         df (pd.DataFrame): DataFrame with blkparse events.
#         plot_basename (str): Basename for the plot output file.
#         output_dir (str): Directory to save the plots.
#     """
#     grouped = df.groupby("cpu")
#     for cpu, cpu_data in grouped:
#         queued = cpu_data[cpu_data["event_type"] == "Q"].set_index("lba")
#         completed = cpu_data[cpu_data["event_type"] == "C"].set_index("lba")
#         merged = queued.merge(completed, on="lba", suffixes=("_q", "_c"))

#         # Compute latency
#         merged["latency"] = merged["timestamp_c"] - merged["timestamp_q"]
#         valid_latencies = merged[merged["latency"] >= 0]["latency"]

#         if valid_latencies.empty:
#             print(f"Did not print the following due to no data:\n\t{plot_basename}.{cpu}")
#             continue

#         # Plot latency distribution for this CPU
#         plt.figure(figsize=(10, 6))
#         plt.hist(valid_latencies, bins=50, edgecolor="black", label=f"Latency Distribution (CPU {cpu})")
#         plt.title(f"Latency Distribution for CPU {cpu}")
#         plt.xlabel("Latency (s)")
#         plt.ylabel("Frequency")
#         plt.legend()
#         plt.grid()

#         # Save the plots
#         png_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}.png")
#         plt.savefig(png_out_path, dpi=300)
#         pdf_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}.pdf")
#         plt.savefig(pdf_out_path)
#         plt.close()

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
        plt.figure(figsize=(10, 6))
        plt.plot(queue_depth_timestamp.index, queue_depth_timestamp.values, label="Queue Depth (Timestamp-Based)")
        plt.title("Queue Depth Over Time (Timestamp-Based)")
        plt.xlabel("Time (s)")
        plt.ylabel("Queue Depth")
        plt.legend()
        plt.grid()
        plt.xlim(left=0)
        png_out_path = os.path.join(output_dir, f"{plot_basename}_queue_depth_timestamp_binning.png")
        plt.savefig(png_out_path, dpi=300)
        pdf_out_path = os.path.join(output_dir, f"{plot_basename}_queue_depth_timestamp_binning.pdf")
        plt.savefig(pdf_out_path)
        plt.close()

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
        plt.figure(figsize=(10, 6))
        plt.plot(queue_depth_1s.index, queue_depth_1s.values, label="Queue Depth (1-Second Binning)")
        plt.title("Queue Depth Over Time (1-Second Binning)")
        plt.xlabel("Time (s)")
        plt.ylabel("Queue Depth")
        plt.legend()
        plt.grid()
        png_out_path = os.path.join(output_dir, f"{plot_basename}_queue_depth_1s_binning.png")
        plt.savefig(png_out_path, dpi=300)
        pdf_out_path = os.path.join(output_dir, f"{plot_basename}_queue_depth_1s_binning.pdf")
        plt.savefig(pdf_out_path)
        plt.close()

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


# def visualize_queue_depth_overview(df: pd.DataFrame, plot_basename: str, output_dir: str) -> None:
#     """
#     Generates an overview plot of queue depth over time across all CPUs.

#     Args:
#         df (pd.DataFrame): DataFrame with blkparse events.
#     """
#     # Calculate queue depth (number of outstanding 'Q' events)
#     df["time_bin"] = (df["timestamp"] // 1).astype(int)
#     queued = df[df["event_type"] == "Q"].groupby("time_bin").size()
#     completed = df[df["event_type"] == "C"].groupby("time_bin").size()
#     queue_depth = (queued - completed).cumsum()

#     if queue_depth.empty:
#         print(f"Did not print the following due to no data:\n\t{plot_basename}")
#         return

#     # Plot queue depth over time
#     plt.figure(figsize=(10, 6))
#     plt.plot(queue_depth.index, queue_depth.values, label="Queue Depth Over Time")
#     plt.title("Overall Queue Depth Over Time")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Queue Depth")
#     plt.legend()
#     plt.grid()

#     png_out_path = os.path.join(output_dir, f"{plot_basename}.png")
#     plt.savefig(png_out_path, dpi=300)
#     pdf_out_path = os.path.join(output_dir, f"{plot_basename}.pdf")
#     plt.savefig(pdf_out_path)
#     plt.close()

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

        # Plot queue depth for this CPU
        plt.figure(figsize=(10, 6))
        plt.plot(queue_depth.index, queue_depth.values, label=f"Queue Depth (CPU {cpu})")
        plt.title(f"Queue Depth Over Time for CPU {cpu}")
        plt.xlabel("Time (s)")
        plt.ylabel("Queue Depth")
        plt.legend()
        plt.grid()

        png_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}.png")
        plt.savefig(png_out_path, dpi=300)
        pdf_out_path = os.path.join(output_dir, f"{plot_basename}.{cpu}.pdf")
        plt.savefig(pdf_out_path)
        plt.close()


def create_arg_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Launch a program and monitor device I/O with blktrace.")

    parser.add_argument(
        "-d", "--device-path", required=True,
        help="Path to the device to be monitored by blktrace (e.g., /dev/sda)."
    )
    parser.add_argument(
        "--elevated", action="store_true",
        help="Indicates that the script is running with elevated privileges."
    )
    framework_choices: List[str] = ["Blaze", "ChunkGraph", "CSRGraph"]
    parser.add_argument(
        "-f", "--framework",
        type=str,
        choices=framework_choices,  # List of allowed frameworks.
        required=True,
        help=f"Mode of operation. Must be one of: {', '.join(framework_choices)}."
    )
    parser.add_argument(
        "-o", "--output_dir", required=True,
        help="Directory to store blktrace output."
    )
    parser.add_argument(
        "-p", "--program", help="Path to the program to be executed."
    )
    

    return parser

# python3 -m src.run_platform_benchmark --framework "ChunkGraph" -o "$HOME/chunk_graph_blktrace_test" --device-path "/dev/nvme0n1p1" --program "BFS -b -chunk -r 12 -t 48 Dataset/LiveJournal/chunk/livejournal"

# python3 -m src.run_platform_benchmark --framework "ChunkGraph" -o "Outputs/ChunkGraph-LiveJournal-test" --device-path "/dev/nvme0n1" --program "BFS -b -chunk -r 12 -t 48 Dataset/LiveJournal/chunk/livejournal"

# python3 -m src.run_platform_benchmark --framework "ChunkGraph" -o "Outputs/ChunkGraph-LiveJournal-test" --device-path "/dev/sdb2" --program "BFS -b -chunk -r 12 -t 48 Dataset/LiveJournal/chunk/livejournal"


def main():

    # Parse arguments.
    parser: argparse.ArgumentParser = create_arg_parser()
    args: argparse.Namespace = parser.parse_args()

    # Check for elevated privileges.
    check_privileges(args.elevated)
    
    print(repr(sys.argv))

    # Validate inputs.
    logger.info(f"Current PID: {os.getpid()}, Elevated: {args.elevated}")
    validate_device(args.device_path)
    logger.info(f"Current PID: {os.getpid()}, Elevated: {args.elevated}")
    validate_output_directory(args.output_dir)

    # Run program and blktrace.
    logger.info(f"Current PID: {os.getpid()}, Elevated: {args.elevated}")
    program_args: List[str] = args.program.split()
    binary_path: str = os.path.join(FRAMEWORKS_DIR, args.framework, "apps", program_args[0]) 
    binary_args: List[str] = [binary_path]
    binary_args.extend(program_args[1:])

    logger.info(f"Framework: {args.framework}")
    logger.info(f"Graph algorithm program:\n\t{binary_path}")
    
    run_program_and_blktrace(args.framework, binary_args, args.device_path, args.output_dir)

    # TODO: chmod out dir to have user-level permissions, not root

if __name__ == "__main__":
    main()
