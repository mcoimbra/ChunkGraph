import argparse
import json
import logging
import math
import os
import pprint
import subprocess
import sys
import time
from typing import Any, Dict, IO, List, Optional

import pandas as pd

from util.consts import BLKTRACE_OUTPUT_FILE_SUFFIX, GRAPH_FRAMEWORKS

# Logging import must come before other local project modules.
import util.logging as log_conf
logger: logging.Logger = log_conf.Logger.get_logger(__name__)

import util.functions as util_functions
import plot.blktrace as blktrace_plotting

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
FRAMEWORKS_DIR: str = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "frameworks"))

# # Configure logging.
# logging.basicConfig(
#     level=logging.INFO,
#     format="[%(levelname)s] - %(message)s",
# )
# logger = logging.getLogger(__name__)

# logger: logging.Logger = log_conf.get_logger(__name__)


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

class ProgramHandle:
    """
    Encapsulates a subprocess.Popen object along with its stdout and stderr file descriptors.
    """
    def __init__(self, program: str, proc: subprocess.Popen, cwd: str, stdout: IO, stdout_path: str, stderr: IO, stderr_path: str):
        self.program: str = program
        self.proc: subprocess.Popen = proc
        self.stdout: IO = stdout
        self.stdout_path: str = stdout_path
        self.stderr: IO = stderr
        self.stderr_path: str = stderr_path
        self.cwd: str = os.getcwd()

    def close(self) -> None:
        """
        Closes the associated stdout and stderr file descriptors.
        """
        if not self.stdout.closed:
            self.stdout.close()
        if not self.stderr.closed:
            self.stderr.close()
        logger.info("Program stdout and stderr have been closed.")

    def cwd(self) -> str:
        return self.cwd

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
            logger.info(f"Program '{self.program}' with PID {self.proc.pid} exited with code {exit_code}")
            self.close()
            return exit_code
        elif timeout > 0:
            try:
                exit_code: int = self.proc.wait(timeout=3)  # Wait up to 10 seconds for graceful termination
                logger.info(f"Program '{self.program}' with PID {self.proc.pid} exited with code {exit_code}")
                self.close()
                return exit_code
            except subprocess.TimeoutExpired:
                logger.warning(f"'{self.program}' with PID {self.proc.pid} did not terminate gracefully. Forcing termination...")
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
    if program in GRAPH_FRAMEWORKS:
        cwd: str = os.path.join(FRAMEWORKS_DIR, program)
    else:
        cwd: str = os.getcwd()
        
    try:
        logger.info(f"Starting {program} on device '{device_path}'...\n\t{' '.join(binary_args)}")

        

        program_proc: subprocess.Popen = subprocess.Popen(
            binary_args,
            stdout=program_stdout,
            stderr=program_stderr,
            text=True
        )

        # if program == "ChunkGraph":
        #     cwd: str = os.path.join(FRAMEWORKS_DIR, program)
        #     program_proc: subprocess.Popen = subprocess.Popen(
        #         binary_args,
        #         stdout=program_stdout,
        #         stderr=program_stderr,
        #         text=True
        #     )
        # else:
        #     program_proc: subprocess.Popen = subprocess.Popen(
        #         binary_args,
        #         stdout=program_stdout,
        #         stderr=program_stderr,
        #         text=True
        #     )
    except FileNotFoundError:
        logger.error(f"Program not found:\n\t'{binary_args[0]}'. Make sure it is installed and/or the provided binary path exists.")
        for t in targets:
            t.terminate()
        sys.exit(1)
    
    logger.info(f"{program} started with PID: {program_proc.pid}\n")

    return ProgramHandle(
        program=program, 
        proc=program_proc, 
        cwd=cwd,
        stdout=program_stdout, 
        stdout_path=program_stdout_path,
        stderr=program_stderr,
        stderr_path=program_stderr_path)

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
    # Start the program.
    logger.info(f"Starting the program:\n\t{' '.join(binary_args)}\n")
    # Record start time
    framework_start_time: float = time.time()
    program_handle: ProgramHandle = launch_program(framework, binary_args, device_path, output_dir=output_dir)

    # Create `blkparse` root output directory.
    blktrace_output_root_dir_path: str = os.path.join(output_dir, "blktrace")
    os.makedirs(blktrace_output_root_dir_path, exist_ok=True)

    # Create `blkparse` trace files output directory.
    blktrace_output_trace_dir_path: str = os.path.join(blktrace_output_root_dir_path, "traces")
    os.makedirs(blktrace_output_trace_dir_path, exist_ok=True)
    
    # Run `blktrace`.
    blktrace_args: List[str] = ["blktrace", "-d", device_path, f"--output-dir={blktrace_output_trace_dir_path}"]
    logger.info(f"Effective UID for 'blktrace': {os.geteuid()}")
    blktrace_start_time: float = time.time()
    blktrace_handle: ProgramHandle = launch_program("blktrace", blktrace_args, device_path, output_dir=blktrace_output_trace_dir_path, log_dir=blktrace_output_root_dir_path)

    # Wait for the program to finish.
    prog_ret: int = program_handle.wait()
    framework_end_time: float = time.time()
    os.system("stty sane")
    
    logger.info(f"Program finished with exit code: {prog_ret}")

    # Stop blktrace.
    logger.info("Stopping blktrace...")
    timeout: int = 3
    blktrace_handle.terminate()
    blktrace_exit_code: int = blktrace_handle.wait(timeout)
    blktrace_end_time: float = time.time()

    if blktrace_exit_code == 0:
        logger.info(f"blktrace finished with exit code: {blktrace_exit_code}")
    else:
        logger.error(f"blktrace finished with exit code: {blktrace_exit_code}")
        logger.error(f"Exiting due to blktrace error.")
        sys.exit(1)

    logger.info(f"blktrace output saved to directory:\n\t{blktrace_output_root_dir_path}")

    # Output 'blktrace' statistics (merge per-CPU files).
    print()
    logger.info(f"Effective UID for 'blkparse': {os.geteuid()}")
    device: str = device_path[device_path.rfind(os.path.sep) + 1 :]
    merge_target_path: str = os.path.join(blktrace_output_root_dir_path, f"{device}.{BLKTRACE_OUTPUT_FILE_SUFFIX}")
    logger.info(f"Starting blkparse to create:\n\t{merge_target_path}")
    merge_blktrace_files(device, merge_target_path, blktrace_output_trace_dir_path)

    # Save the execution context to a JSON file in the output directory.
    # TODO: add program time and blktrace's time.
    pids: List[int] = [program_handle.pid()]
    data: Dict = {
        "blktrace_root_dir": blktrace_output_root_dir_path,
        "blktrace_stderr_path": blktrace_handle.stderr_path,
        "blktrace_stdout_path": blktrace_handle.stdout_path,
        "blktrace_time": f"{(blktrace_end_time - blktrace_start_time):.2f}",
        "blktrace_traces_dir": blktrace_output_trace_dir_path,
        "framework": framework,
        "framework_args": binary_args,
        "framework_call": " ".join(binary_args),
        "framework_pids": pids,
        "framework_cwd": program_handle.cwd(),
        "framework_stderr_path": program_handle.stderr_path,
        "framework_stdout_path": program_handle.stdout_path,
        "framework_time": f"{(framework_end_time - framework_start_time):.2f}"
    }
    context_target_path: str = os.path.join(output_dir, "context.json")
    with open(context_target_path, 'w') as f:
        json.dump(data, f, indent=4)

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
    framework_choices: List[str] = GRAPH_FRAMEWORKS
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

# ./frameworks/ChunkGraph/apps/BFS -b -chunk -r 12 -t 20 Dataset/LiveJournal/chunk/livejournal

# python3 -m src.run_platform_benchmark --framework "ChunkGraph" -o "$HOME/chunk_graph_blktrace_test" --device-path "/dev/nvme0n1p1" --program "BFS -b -chunk -r 12 -t 48 Dataset/LiveJournal/chunk/livejournal"

# python3 -m src.run_platform_benchmark --framework "ChunkGraph" -o "Outputs/ChunkGraph-LiveJournal-test" --device-path "/dev/nvme0n1" --program "BFS -b -chunk -r 12 -t 48 Dataset/LiveJournal/chunk/livejournal"

# python3 -m src.run_platform_benchmark --framework "ChunkGraph" -o "Outputs/ChunkGraph-LiveJournal-test" --device-path "/dev/nvme0n1" --program "BFS -b -chunk -r 48 -t 48 Dataset/LiveJournal/chunk/livejournal"

# python3 -m src.run_platform_benchmark --framework "ChunkGraph" -o "Outputs/ChunkGraph-dimacs10-uk-2007-05-test" --device-path "/dev/nvme0n1" --program "BFS -b -chunk -r 48 -t 48 Dataset/UKdomain/chunk/dimacs10-uk-2007-05"

# python3 -m src.run_platform_benchmark --framework "ChunkGraph" -o "Outputs/ChunkGraph-dimacs10-uk-2007-05-test" --device-path "/dev/nvme0n1" --program "BFS -b -chunk -r 5699262 -t 48 Dataset/UKdomain/chunk/dimacs10-uk-2007-05"

# BFS -b -chunk -r 5699262 -t 48 Dataset/UKdomain/chunk/dimacs10-uk-2007-05
# ./frameworks/ChunkGraph/apps/BFS -b -chunk -r 5699262 -t 48 Dataset/UKdomain/chunk/dimacs10-uk-2007-05
# stress
# strace -e trace=mmap,open,read,write,fsync,close -o strace_output.txt ./frameworks/ChunkGraph/apps/BFS -b -chunk -r 5699262 -t 48 Dataset/UKdomain/chunk/dimacs10-uk-2007-05

# strace -e trace=open,read,write,fsync,close -o strace_output.txt python3 -m src.run_platform_benchmark --framework "ChunkGraph" -o "Outputs/ChunkGraph-dimacs10-uk-2007-05-test" --device-path "/dev/nvme0n1" --program "BFS -b -chunk -r 5699262 -t 48 Dataset/UKdomain/chunk/dimacs10-uk-2007-05"

# strace -e trace=open,read,write,fsync,close -o strace_output.txt  ./frameworks/ChunkGraph/apps/BFS -b -chunk -r 5699262 -t 48 Dataset/UKdomain/chunk/dimacs10-uk-2007-05

# ./frameworks/ChunkGraph/apps/BFS -b -chunk -r 5699262 -t 48 Dataset/UKdomain/chunk/dimacs10-uk-2007-05

# python3 -m src.run_platform_benchmark --framework "ChunkGraph" -o "Outputs/ChunkGraph-LiveJournal-test" --device-path "/dev/sdb2" --program "BFS -b -chunk -r 12 -t 48 Dataset/LiveJournal/chunk/livejournal"

# strace -e trace=open,read,write,fsync,close -o strace_output.txt

def main():

    # Parse arguments.
    parser: argparse.ArgumentParser = create_arg_parser()
    args: argparse.Namespace = parser.parse_args()

    # Get the original user and group before privilege escalation.
    original_user: str = util_functions.get_original_user()
    original_group: str = util_functions.get_original_group()

    # Attempt to elevate user privileges.
    check_privileges(args.elevated)
    
    print(repr(sys.argv))

    # Validate inputs.
    logger.info(f"Current PID: {os.getpid()}, Elevated: {args.elevated}")
    util_functions.validate_device(args.device_path)
    logger.info(f"Current PID: {os.getpid()}, Elevated: {args.elevated}")
    util_functions.validate_output_directory(args.output_dir)

    # Run program and blktrace.
    logger.info(f"Current PID: {os.getpid()}, Elevated: {args.elevated}")
    program_args: List[str] = args.program.split()
    binary_path: str = os.path.join(FRAMEWORKS_DIR, args.framework, "apps", program_args[0]) 
    binary_args: List[str] = [binary_path]
    binary_args.extend(program_args[1:])

    logger.info(f"Framework: {args.framework}")
    logger.info(f"Graph algorithm program:\n\t{binary_path}")
    
    run_program_and_blktrace(args.framework, binary_args, args.device_path, args.output_dir)

    # Change ownership back to the original user.
    util_functions.change_ownership_recursively(args.output_dir, original_user, original_group)

if __name__ == "__main__":
    main()
