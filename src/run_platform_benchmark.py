import argparse
import logging
import math
import os
import pprint
import subprocess
import sys
import time
from typing import Any, IO, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

print(sys.path)

import plot.blktrace as blktrace_plotting

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
FRAMEWORKS_DIR: str = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "frameworks"))

# Configure logging.
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
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory:\n\t{output_dir}")
        except OSError as e:
            logger.error(f"Error creating directory\n\t'{output_dir}': {e}")
            sys.exit(1)
    else:
        logger.info(f"Validated output directory:\n\t{output_dir}")


class ProgramHandle:
    """
    Encapsulates a subprocess.Popen object along with its stdout and stderr file descriptors.
    """
    def __init__(self, program: str, proc: subprocess.Popen, stdout: IO, stderr: IO):
        self.program: str = program
        self.proc: subprocess.Popen = proc
        self.stdout: IO = stdout
        self.stderr: IO = stderr

    def close(self) -> None:
        """
        Closes the associated stdout and stderr file descriptors.
        """
        if not self.stdout.closed:
            self.stdout.close()
        if not self.stderr.closed:
            self.stderr.close()
        logger.info("Program stdout and stderr have been closed.")

    def pid(self) -> int:
        return self.proc.pid

    def terminate(self) -> None:
        """
        Terminates the process and closes the associated file descriptors.
        """
        if self.proc.poll() is None:  # Check if the process is still running
            self.proc.terminate()
            logger.info(f"Terminated program with PID: {self.proc.pid}")
        self.close()

    def wait(self, timeout: int = 0) -> int:
        """
        Waits for the process to complete and returns its exit code.

        Returns:
            int: The exit code of the process.
        """

        if timeout == 0:
            exit_code: int = self.proc.wait()
            logger.info(f"Program with PID {self.proc.pid} exited with code {exit_code}")
            self.close()
            return exit_code
        elif timeout > 0:
            try:
                exit_code: int = self.proc.wait(timeout=3)  # Wait up to 10 seconds for graceful termination
                logger.info(f"Program with PID {self.proc.pid} exited with code {exit_code}")
                self.close()
                return exit_code
            except subprocess.TimeoutExpired:
                logger.warning(f"'{self.program}' did not terminate gracefully. Forcing termination...")
                self.proc.kill()  # Force terminate the process
                exit_code: int = self.proc.wait()
                self.close()
                return exit_code
        else:
            logger.error(f"ProgramHandle.wait() - timeout must be a positive integer")
            sys.exit(1)

def launch_program(program: str, binary_args: List[str], device_path: str, output_dir: str = "", log_dir: str = "",targets: List[ProgramHandle] = []) -> ProgramHandle: #subprocess.Popen:

    if len(output_dir) == 0:
        output_dir = os.getcwd()

    if len(log_dir) == 0:
        log_dir = output_dir

    if not (os.path.exists(output_dir) and os.path.isdir(output_dir)):
        logger.error(f"Directory does not exist, exiting:\n\t{output_dir}")
        sys.exit(1)

    if not (os.path.exists(log_dir) and os.path.isdir(log_dir)):
        logger.error(f"Directory does not exist, exiting:\n\t{log_dir}")
        sys.exit(1)
        
    program_stdout_path: str = os.path.join(log_dir, f"{program}_stdout.log")
    program_stderr_path: str = os.path.join(log_dir, f"{program}_stderr.log")
    program_stdout: IO = open(program_stdout_path, 'w')
    program_stderr: IO = open(program_stderr_path, 'w')
    try:
        logger.info(f"Starting {program} on device '{device_path}'...\n\t{' '.join(binary_args)}")
        program_proc: subprocess.Popen = subprocess.Popen(
            binary_args,
            stdout=program_stdout,
            stderr=program_stderr,
            text=True
        )
    except FileNotFoundError:
        logger.error(f"Program not found:\n\t'{binary_args[0]}'. Make sure it is installed and/or the provided binary path exists.")
        for t in targets:
            t.terminate()
        sys.exit(1)
    
    logger.info(f"{program} started with PID: {program_proc.pid}\n")

    return ProgramHandle(program=program, proc=program_proc, stdout=program_stdout, stderr=program_stderr)

def launch_monitoring_tool(tool: str, output_dir: str) -> None:
    pass

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

    program_handle: ProgramHandle = launch_program(framework, binary_args, device_path, output_dir=output_dir)

    # Start blktrace.
    # Create 'blktrace' root output directory.
    blktrace_output_root_dir_path: str = os.path.join(output_dir, "blktrace")
    os.makedirs(blktrace_output_root_dir_path, exist_ok=True)

    # Create 'blkparse' trace files output directory.
    blktrace_output_trace_dir_path: str = os.path.join(blktrace_output_root_dir_path, "traces")
    os.makedirs(blktrace_output_trace_dir_path, exist_ok=True)
    
    blktrace_args: List[str] = ["blktrace", "-d", device_path, f"--output-dir={blktrace_output_trace_dir_path}"]
    logger.info(f"Effective UID for 'blktrace': {os.geteuid()}")
    blktrace_handle: ProgramHandle = launch_program("blktrace", blktrace_args, device_path, output_dir=blktrace_output_trace_dir_path, log_dir=blktrace_output_root_dir_path)

    # Wait for the program to finish.
    prog_ret: int = program_handle.wait()
    os.system("stty sane")
    
    logger.info(f"Program finished with exit code: {prog_ret}")

    # Stop blktrace.
    logger.info("Stopping blktrace...")
    timeout: int = 3
    blktrace_handle.terminate()
    blktrace_exit_code: int = blktrace_handle.wait(timeout)

    if blktrace_exit_code == 0:
        logger.info(f"blktrace finished with exit code: {blktrace_exit_code}")
    else:
        logger.error(f"blktrace finished with exit code: {blktrace_exit_code}")
        logger.error(f"Exiting due to blktrace error.")
        sys.exit(1)

    logger.info(f"blktrace output saved to directory:\n\t{blktrace_output_root_dir_path}")

    # Output 'blktrace' statistics.
    # Step 1: Merge per-CPU files
    print()
    logger.info(f"Effective UID for 'blkparse': {os.geteuid()}")
    device: str = device_path[device_path.rfind(os.path.sep) + 1 :]
    merge_target_path: str = os.path.join(blktrace_output_root_dir_path, f"{device}.blktrace.merged.txt")
    logger.info(f"Starting blkparse to create:\n\t{merge_target_path}")
    merge_blktrace_files(device, merge_target_path, blktrace_output_trace_dir_path)

    # Step 2: Parse the merged output
    #pids: List[int] = [program_proc.pid]
    pids: List[int] = [program_handle.pid()]
    df: pd.DataFrame = parse_blkparse_output(merge_target_path, pids)

    # TODO: plot-generating code should be called from another Python script, not here...

    # TODO: Filter by `program_handle.pid()`.

    # Step 3: Visualize 'blktrace' data.
    plot_basename: str = f"{device}.blktrace"
    os.makedirs(blktrace_output_root_dir_path, exist_ok=True)
    logger.info(f"Created output directory:\n\t{blktrace_output_root_dir_path}")

    blktrace_plots_dir: str = os.path.join(blktrace_output_root_dir_path, "plots")
    os.makedirs(blktrace_plots_dir, exist_ok=True)

    # Visualize throughput.
    visualize_throughput_overview(df, f"{plot_basename}-throughput", blktrace_plots_dir)
    visualize_throughput_per_cpu(df, f"{plot_basename}-throughput", blktrace_plots_dir)

    # Visualize latencies.
    visualize_latency_overview(df, f"{plot_basename}-latency", blktrace_plots_dir)

    sys.exit(0)
    visualize_latency_per_cpu(df, f"{plot_basename}-latency", blktrace_plots_dir)

    # Visualize queue depth.
    # TODO: reuse the functions  used for throughput (add paramters / or class object for common properties...)
    visualize_queue_depth_overview(df, f"{plot_basename}-q_depth", blktrace_plots_dir)
    visualize_queue_depth_per_cpu(df, f"{plot_basename}-q_depth", blktrace_plots_dir)

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

            # Check that there are at least 7 parts (len(parts) > 6) and 
            # that the 4th part (timestamp) is a valid number. 
            # Remove lines that don’t meet these criteria.
            if len(parts) > 6 and parts[3].replace(".", "").isdigit():
                try:
                    # Time of the event.
                    timestamp: float = float(parts[3])

                    # CPU core ID where the event occurred.
                    cpu: int = int(parts[1])

                    # The type of the event (e.g., 'Q' for queued, 'C' for completed).
                    event_type: str = parts[5]

                    # The performed operation (e.g., read or write).
                    operation: str = parts[6]
                    
                    # (Logical Block Address): optional integer representing the 
                    # address of the block being accessed.
                    # Ensure LBA is a valid integer.
                    lba: Optional[int] = int(parts[7]) if len(parts) > 7 and parts[7].isdigit() else None
                    
                    # Optional integer representing the number of blocks involved in 
                    # the operation.
                    # Ensure blocks are valid.
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
                    # If a field cannot be parsed (e.g., a non-numeric lba), the line is 
                    # skipped, and an error message is printed.
                    # Log or print the line causing the issue (optional for debugging).
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
    
    # print(valid_latencies)
    #print(valid_latencies.columns)
    #sys.exit(0)

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
    
    blktrace_plotting.plot_latency_logarithmic_bins(valid_latencies, f"{plot_basename}_log", output_dir, title=f"{plot_basename}_log")

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
        blktrace_plotting.plot_latency_logarithmic_bins(valid_latencies, bins, plot_basename, output_dir, title=f"{plot_basename}.{cpu}_log")

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

    # TODO: check executing user before elevating permissions.

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

    # TODO: chmod out dir and its content to belong to the original non-root user.

if __name__ == "__main__":
    main()
