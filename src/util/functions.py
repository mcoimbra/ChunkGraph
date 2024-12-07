from abc import ABC, abstractmethod
import grp
import os
import pathlib
import pprint
import pwd
import subprocess
import sys
from typing import Dict, List, Optional

import pandas as pd

import logging
import util.logging as log_conf
logger: logging.Logger = log_conf.Logger.get_logger(__name__)

def get_original_user() -> str:
    """
    Get the username of the user who invoked the script.
    Returns:
        str: The username of the original user.
    """
    uid: int = os.getenv("SUDO_UID", os.getuid())
    return pwd.getpwuid(int(uid)).pw_name

def get_original_group() -> str:
    """
    Get the group name of the user who invoked the script.
    Returns:
        str: The group name of the original user.
    """
    gid: int = os.getenv("SUDO_GID", os.getgid())
    return grp.getgrgid(int(gid)).gr_name

def change_ownership_recursively(path: str, user: str, group: str):
    """
    Change ownership of the specified directory (and its contents) to the given user and group.
    Args:
        path (str): The path to the directory.
        user (str): The username to set as the owner.
        group (str): The group to set as the owner.
    """
    # Get UID and GID for the user and group.
    uid: int = pwd.getpwnam(user).pw_uid
    gid: int = grp.getgrnam(group).gr_gid

    # Walk through the directory.
    for root, dirs, files in os.walk(path):
        # Change ownership for directories.
        for dir_name in dirs:
            dir_path: str = os.path.join(root, dir_name)
            os.chown(dir_path, uid, gid)
        # Change ownership for files.
        for file_name in files:
            file_path: str = os.path.join(root, file_name)
            os.chown(file_path, uid, gid)
    # Change ownership for the root directory itself.
    os.chown(path, uid, gid)

def dir_contains_elements(dir_path: str, sub_dir_paths: List[str]) -> bool:
    for dir_name in sub_dir_paths:

        # Construct the full path for each directory.
        full_path: str = os.path.join(dir_path, dir_name)
        if not (os.path.exists(full_path) and os.path.isdir(full_path)):
            return False
    return True

def dir_exists_and_not_empty(dir_path: str) -> bool:
    # Check if it exists.
    if not (os.path.exists(dir_path) and os.path.isdir(dir_path)):
        logger.error(f"Expected this directory path to exist and and not be empty:\n\t{dir_path}")
        return False
    
    # Check if it is empty.
    try:
        has_content: bool = any(pathlib.Path(dir_path).iterdir())
        if not has_content:
            logger.error(f"Expected this directory to not be empty:\n\t{dir_path}")
        else:
            logger.info(f"Directory has content:\n\t{dir_path}")
        return has_content
    except FileNotFoundError:
        logger.error(f"The directory does not exist:\n\t{dir_path}")
        return False
    except PermissionError:
        logger.error(f"Permission denied when accessing:\n\t{dir_path}")
        return False

def blkparse_output_to_tsv(file_path: str, csv_path: str) -> None:
    """
    Parses blkparse output into a CSV file.

    Args:
        file_path (str): Path to the `blkparse` text output file.
        csv_path (str): Path to the CSV file that will be created with `blktrace` data.

    Returns:
        pd.DataFrame: Parsed data with columns for timestamp, CPU, operation, etc.
    """

    headers: List[str] = [
        "event_seq_num",
        "pid",
        "timestamp",
        "dev_mjr",
        "dev_mnr",
        "cpu",
        "event_type",
        "operation",
        "lba",
        "lba_end",
        "blocks",
    ]

    with open(file_path, "r") as f, open(csv_path, 'w') as csv_out:

        # Write the column headers first.
        header_str: str = "\t".join(headers) + "\n"
        csv_out.write(header_str)

        for line in f:
            line: str
            parts: List[str] = line.split()

            # Check that there are at least 7 parts (len(parts) > 6) and 
            # that the 4th part (timestamp) is a valid number. 
            # Remove lines that don’t meet these criteria.
            if len(parts) > 6 and parts[3].replace(".", "").isdigit():
                try:
                    
                    # Store device indication.
                    dev_part: List[str] = parts[0].strip().split(",")
                    dev_mjr: int = int(dev_part[0])
                    dev_mnr: int = int(dev_part[1])

                    # Time of the event.
                    timestamp: float = float(parts[3])

                    # Event sequence number.
                    event_seq_num: int = int(parts[2])

                    # CPU core ID where the event occurred.
                    cpu: int = int(parts[1])

                    # Process PID.
                    pid: int = int(parts[4])

                    # The type of the event (e.g., 'Q' for queued, 'C' for completed).
                    event_type: str = parts[5]

                    # The performed operation (e.g., read or write).
                    operation: str = parts[6]
                    
                    # (Logical Block Address): optional integer representing the 
                    # address of the block being accessed.
                    # Ensure LBA is a valid integer.
                    lba: Optional[int] = int(parts[7]) if len(parts) > 7 and parts[7].isdigit() else None

                    # Check the 'lba' address operator, if any.
                    # Need it to compute block sizes.
                    lba_operator: Optional[str] = parts[8] if len(parts) > 8 else None

                    block_count: Optional[int] = None
                    lba_end: Optional[int] = None

                    if (not lba == None) and (not lba_operator == None):
                        # Optional integer representing the number of blocks involved in 
                        # the operation.
                        # Parse the blocks depending on the event type.
                        
                        #if event_type in ["D", "M", "Q"]:
                        if lba_operator == "+":
                            block_count = int(parts[9]) if len(parts) > 9 and parts[9].isdigit() else None
                            if not block_count == None:
                                lba_end = lba + block_count
                        #elif event_type == "X":
                        elif lba_operator == "/":
                            lba_end = int(parts[9]) if len(parts) >= 9 and parts[9].isdigit() else None
                            if not lba_end == None:
                                block_count = lba_end - lba

                        # blocks: Optional[int] = int(parts[9][1:]) if len(parts) > 9 and parts[9].startswith("+") else None

                    # Read the parsed data.
                    line_data: Dict = {
                        "event_seq_num": event_seq_num,
                        "pid": pid,
                        "timestamp": timestamp,
                        "dev_mjr": dev_mjr,
                        "dev_mnr": dev_mnr,
                        "cpu": cpu,
                        "event_type": event_type,
                        "operation": operation,
                        "lba": lba,
                        "lba_end": lba_end,
                        "blocks": block_count,
                    }

                    #pprint.pprint(line_data)
                    #sys.exit(0)

                    # Write it to the CSV file.
                    out_line: str = ""
                    for h in headers:
                        out_line += str(line_data[h]) + "\t"
                    out_line = out_line[:-1] # Remove the last tab character.
                    out_line += "\n"

                    csv_out.write(out_line)
                except ValueError as e:
                    # If a field cannot be parsed (e.g., a non-numeric lba), the line is 
                    # skipped, and an error message is printed.
                    # Log or print the line causing the issue (optional for debugging).
                    print(f"Skipping line due to error: {line.strip()} - {e}")
                    continue

def parse_blkparse_output(file_path: str, pids: List[int]) -> pd.DataFrame:
    if file_path.endswith(".tsv"):
        logger.info("Reading blktrace TSV data into pandas.")
        return parse_blkparse_tsv_output(file_path, pids)
    else:
        logger.info("Reading blktrace native output data into pandas.")
        return parse_blkparse_native_output(file_path, pids)
    
def parse_blkparse_tsv_output(file_path: str, pids: List[int]) -> pd.DataFrame:

    # Define column names and types (adjust based on your blkparse format).
    col_types = {
        "event_seq_num": int,
        "pid": int,
        "timestamp": float,
        "dev_mjr": int,
        "dev_mnr": int,
        "cpu": int,
        "event_type": str,
        "operation": str,
        "lba": "Int64",
        "lba_end": "Int64",
        "blocks": "Int64",
    }

    # def custom_converter(blocks):
    #     # Handle blocks starting with '+'
    #     return int(blocks[1:]) if blocks.startswith("+") else None

    return pd.read_csv(
        file_path,
        sep="\t",
        header=0,
        usecols=["event_seq_num", "pid", "timestamp", "dev_mjr", "dev_mnr", "cpu", "event_type", "operation", "lba", "lba_end", "blocks"],
        dtype=col_types,
        #converters={"blocks": custom_converter},
        engine="c",
        low_memory=False,
        na_filter=False,
    )

def parse_blkparse_native_output(file_path: str, pids: List[int]) -> pd.DataFrame:
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
                    # Store device indication.
                    dev_part: List[str] = parts[0].strip().split(",")
                    dev_mjr: int = int(dev_part[0])
                    dev_mnr: int = int(dev_part[1])

                    # Time of the event.
                    timestamp: float = float(parts[3])

                    # Event sequence number.
                    event_seq_num: int = int(parts[2])

                    # CPU core ID where the event occurred.
                    cpu: int = int(parts[1])

                    # Process PID.
                    pid: int = int(parts[4])

                    # The type of the event (e.g., 'Q' for queued, 'C' for completed).
                    event_type: str = parts[5]

                    # The performed operation (e.g., read or write).
                    operation: str = parts[6]
                    
                    # (Logical Block Address): optional integer representing the 
                    # address of the block being accessed.
                    # Ensure LBA is a valid integer.
                    lba: Optional[int] = int(parts[7]) if len(parts) > 7 and parts[7].isdigit() else None

                    # Check the 'lba' address operator, if any.
                    # Need it to compute block sizes.
                    lba_operator: Optional[str] = parts[8] if len(parts) > 8 else None

                    block_count: Optional[int] = None
                    lba_end: Optional[int] = None

                    if (not lba == None) and (not lba_operator == None):
                        # Optional integer representing the number of blocks involved in 
                        # the operation.
                        # Parse the blocks depending on the event type.
                        
                        #if event_type in ["D", "M", "Q"]:
                        if lba_operator == "+":
                            block_count = int(parts[9]) if len(parts) > 9 and parts[9].isdigit() else None
                            if not block_count == None:
                                lba_end = lba + block_count
                        #elif event_type == "X":
                        elif lba_operator == "/":
                            lba_end = int(parts[9]) if len(parts) >= 9 and parts[9].isdigit() else None
                            if not lba_end == None:
                                block_count = lba_end - lba

                    # Append the parsed data
                    data.append({
                        "event_seq_num": event_seq_num,
                        "pid": pid,
                        "timestamp": timestamp,
                        "dev_mjr": dev_mjr,
                        "dev_mnr": dev_mnr,
                        "cpu": cpu,
                        "event_type": event_type,
                        "operation": operation,
                        "lba": lba,
                        "lba_end": lba_end,
                        "blocks": block_count,
                    })
                except ValueError as e:
                    # If a field cannot be parsed (e.g., a non-numeric lba), the line is 
                    # skipped, and an error message is printed.
                    # Log or print the line causing the issue (optional for debugging).
                    print(f"Skipping line due to error: {line.strip()} - {e}")
                    continue
    return pd.DataFrame(data)

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

class MetricsToolProfile(ABC):
    """
    Encapsulates the logic of a monitoring tool such as `blktrace` or `pidstat`.
    """
    def __init__(self, tool_name: str, tool_args: List[str]):
        self.tool_name: str = tool_name
        self.tool_args: List[str] = tool_args
        self.is_setup: bool = False

    """
    Checks if `self.tool_name` is available in the current environment.
    """
    def is_tool_available(self) -> bool:
        pass

    @abstractmethod
    def setup(self) -> None:
        self.is_setup = True

class BlktraceToolProfile(MetricsToolProfile):
    """
    Encapsulates the logic of `blktrace`.
    """
    def __init__(self, device_path: str, output_dir: str):
        self.tool_name = "blktrace"

        self.output_dir: str = os.path.join(output_dir, self.tool_name)
        self.blktrace_traces_dir: str = os.path.join(self.output_dir, "traces")
        
        tool_args: List[str] = [self.tool_name, "-d", device_path, f"--output-dir={self.blktrace_traces_dir}"]

        super().__init__(self.tool_name, tool_args)

    def setup(self) -> None:
        os.makedirs(self.blktrace_traces_dir, exist_ok=True)
        super().setup()

class PidstatToolProfile(MetricsToolProfile):
    """
    Encapsulates the logic of `pidstat`.
    """
    def __init__(self, framework_pid: int, output_dir: str, sampling_interval: int = 1):
        self.tool_name = "pidstat"

        self.output_dir: str = os.path.join(output_dir, self.tool_name)
        
        # # pidstat -t -p $pid 1 > pidstat.1.txt

        tool_args: List[str] = [self.tool_name, "-t", "-p", f"{framework_pid}", sampling_interval]

        super().__init__(self.tool_name, tool_args)

    def setup(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        super().setup()