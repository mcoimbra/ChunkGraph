import argparse
import logging
import os
import pprint
import subprocess
import sys
import time
from typing import IO, List

import pandas as pd
import matplotlib.pyplot as plt

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
                logger.info(f"Privileges elevated successfully {os.getpid()}. Exiting original instance.")
                sys.exit(0)  # Successfully elevated privileges, exit original instance
        except Exception as e:
            logger.error(f"Error while trying to elevate privileges: {e}")
            sys.exit(1)
        return  # This return is redundant because sys.exit() is above.
    elif current_euid == 0:
        logger.info(f"Script {os.getpid()} is running with elevated privileges.")
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

def run_program_and_blktrace(framework: str, binary_args: List[str], device: str, output_dir: str) -> None:
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
    
    logger.info(f"Program started with PID: {program_proc.pid}")

    # Start blktrace.
    blktrace_output_file: str = os.path.join(output_dir, "blktrace_trace")
    logger.info(f"Starting blktrace on device '{device}'...")
    blktrace_stdout_path: str = os.path.join(output_dir, f"blktrace_stdout.log")
    blktrace_stderr_path: str = os.path.join(output_dir, f"blktrace_stderr.log")
    blktrace_stdout: IO = open(blktrace_stdout_path, 'w')
    blktrace_stderr: IO = open(blktrace_stderr_path, 'w')
    try:
        logger.info(f"Effective UID for 'blktrace': {os.geteuid()}")
        blktrace_proc: subprocess.Popen = subprocess.Popen(
            #["sudo", "blktrace", "-d", device, "-o", blktrace_output_file],
            ["sudo", "blktrace", "-d", device, f"--output-dir={output_dir}"],
            stdout=blktrace_stdout,
            stderr=blktrace_stderr,
            text=True
        )
    except FileNotFoundError:
        logger.error("blktrace not found. Ensure it is installed and available in the PATH.")
        program_proc.terminate()
        sys.exit(1)
    
    logger.info(f"blktrace started with PID: {blktrace_proc.pid}")

    # Wait for the program to finish
    program_proc.wait()
    program_stdout.close()
    program_stderr.close()
    os.system("stty sane")
    
    logger.info(f"Program finished with exit code: {program_proc.returncode}")

    # Stop blktrace
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

    logger.info(f"blktrace finished with exit code: {blktrace_proc.returncode}")

    logger.info(f"blktrace output saved to: {blktrace_output_file}")

    # Output 'blktrace' statistics.
    # Step 1: Merge per-CPU files
    merge_target_path: str = os.path.join(output_dir, "blktrace_merged_trace.txt")
    merge_blktrace_files(output_dir, merge_target_path)

    # Step 2: Parse the merged output
    df = parse_blkparse_output(merge_target_path)

    # Step 3: Visualize overview
    visualize_overview(df)

    # Step 4: Visualize per CPU
    visualize_per_cpu(df)

def merge_blktrace_files(blktrace_dir: str, output_file: str) -> None:
    """
    Merges per-CPU blktrace files into a unified trace file using blkparse.

    Args:
        blktrace_dir (str): Directory containing blktrace per-CPU files.
        output_file (str): Path to save the merged output.
    """
    try:
        # Merge all blktrace files (e.g., blktrace.cpu0, blktrace.cpu1, ...)
        subprocess.run(
            ["blkparse", "-i", f"{blktrace_dir}/blktrace.cpu*", "-o", output_file],
            check=True
        )
        print(f"[INFO] Merged blktrace output saved to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to merge blktrace files: {e}")

def parse_blkparse_output(file_path: str) -> pd.DataFrame:
    """
    Parses blkparse output into a DataFrame.

    Args:
        file_path (str): Path to the blkparse text output file.

    Returns:
        pd.DataFrame: Parsed data with columns for timestamp, CPU, operation, etc.
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) > 6 and parts[3].replace(".", "").isdigit():
                timestamp = float(parts[3])
                cpu = int(parts[1])  # CPU ID
                operation = parts[6]
                data.append({"timestamp": timestamp, "cpu": cpu, "operation": operation})
    return pd.DataFrame(data)

def visualize_overview(df: pd.DataFrame) -> None:
    """
    Generates an overview plot of throughput over time across all CPUs.

    Args:
        df (pd.DataFrame): DataFrame with blkparse events.
    """
    # Group by time (in seconds) for throughput
    df["time_bin"] = (df["timestamp"] // 1).astype(int)
    throughput = df.groupby("time_bin").size()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(throughput.index, throughput.values, label="Throughput (IOPS)")
    plt.title("Overall Throughput Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Throughput (IOPS)")
    plt.legend()
    plt.grid()
    plt.show()

def visualize_per_cpu(df: pd.DataFrame) -> None:
    """
    Generates per-CPU visualizations of throughput over time.

    Args:
        df (pd.DataFrame): DataFrame with blkparse events.
    """
    # Group data by CPU
    grouped = df.groupby("cpu")
    for cpu, cpu_data in grouped:
        cpu_data["time_bin"] = (cpu_data["timestamp"] // 1).astype(int)
        throughput = cpu_data.groupby("time_bin").size()

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(throughput.index, throughput.values, label=f"CPU {cpu} Throughput (IOPS)")
        plt.title(f"Throughput Over Time for CPU {cpu}")
        plt.xlabel("Time (s)")
        plt.ylabel("Throughput (IOPS)")
        plt.legend()
        plt.grid()
        plt.show()


def create_arg_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Launch a program and monitor device I/O with blktrace.")

    parser.add_argument(
        "-d", "--device", required=True,
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

def main():

    # Parse arguments.
    parser: argparse.ArgumentParser = create_arg_parser()
    args: argparse.Namespace = parser.parse_args()

    # Check for elevated privileges.
    check_privileges(args.elevated)
    
    print(repr(sys.argv))

    # Validate inputs.
    logger.info(f"Current PID: {os.getpid()}, Elevated: {args.elevated}")
    validate_device(args.device)
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
    
    run_program_and_blktrace(args.framework, binary_args, args.device, args.output_dir)

if __name__ == "__main__":
    main()