from abc import ABC, abstractmethod
import datetime
import grp
import os
import pathlib
import pprint
import pwd
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

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

def is_device_raid(device: str) -> bool:
    """
    Checks if the given device is part of a RAID configuration by looking at /proc/mdstat.
    Provides detailed diagnostics if /proc/mdstat is missing.

    Args:
        device (str): The device path, e.g., "/dev/sda1".

    Returns:
        bool: True if the device is part of a RAID, False otherwise.
    """
    mdstat_path: str = "/proc/mdstat"

    # Check if /proc/mdstat exists
    if not os.path.exists(mdstat_path):
        logger.info(f"Did not find: {mdstat_path}")

        # Check if the md (multiple devices) kernel module is loaded
        try:
            result = subprocess.run(["lsmod"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if "md" not in result.stdout:
                logger.info(f"The 'md' driver is not loaded. Try loading it with:\n\tsudo modprobe md.")
            else:
                logger.info(f"The 'md' driver is loaded, but {mdstat_path} is missing. Check kernel configuration.")
        except Exception as e:
            logger.error(f"Could not check kernel modules:\n\t{e}")
            # logger.error("Exiting.")
            # sys.exit(1)

        # Check if the kernel supports software RAID
        try:
            kernel_config_path: str = "/boot/config-$(uname -r)"
            if os.path.exists(kernel_config_path):
                with open(kernel_config_path, "r") as config:
                    raid_support = any(line.strip() == "CONFIG_MD=y" for line in config)
                    if not raid_support:
                        logger.info("The kernel does not have RAID support enabled.")
                    else:
                        logger.info(f"RAID support is enabled in the kernel, but {mdstat_path} is missing.")
            else:
                logger.info("Could not locate kernel configuration. Ensure the system supports software RAID.")
        except Exception as e:
            logger.error(f"Error checking kernel configuration:\n\t{e}")
            # logger.error("Exiting.")
            # sys.exit(1)
        
        return False
    
    logger.info(f"Found {mdstat_path}")

    # If /proc/mdstat exists, check if the device is listed
    try:
        with open(mdstat_path, "r") as f:
            mdstat_content: List[str] = f.readlines()
            line_ctr: int = 1
            for l in mdstat_content:
                stripped: str = l.strip()
                logger.info(f"Current line {stripped}")
                if ':' in stripped:
                    tkns: List[str] = stripped.split(':')
                    curr_device: str = tkns[0].strip()
                    if curr_device in device:
                        logger.info(f"Found {device} in {mdstat_path} output line {line_ctr}:\n\t{stripped}")
                        return True 
                line_ctr =+ 1
        return False
        #return device in mdstat_content
    except Exception as e:
        logger.error(f"Error reading {mdstat_path}:\n\t{e}")
        return False

def convert_to_unix_ts(timestamp: str) -> int:
    """
    Converts a timestamp string to a Unix timestamp.

    Args:
        timestamp (str): A timestamp string in the format 'Mon Dec  4 15:20:31 2023'.

    Returns:
        int: The corresponding Unix timestamp.
    """
    return int(datetime.datetime.strptime(timestamp, "%a %b %d %H:%M:%S %Y").timestamp())

def parse_mdadm_output(file_path: str) -> Dict[str, Any]:
    """
    Parses the output of a successful 'mdadm --detail' call from a file.

    Args:
        file_path (str): Path to the file containing the 'mdadm --detail' output.

    Returns:
        Dict[str, Any]: A dictionary containing the parsed RAID details.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    mdadm_data: Dict[str, Any] = {}
    raid_devices: Dict[str, Dict[str, Any]] = {}
    current_device: str = ""

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            # Match single-value fields
            if line.startswith("Version :"):
                mdadm_data['version'] = line.split(":", 1)[1].strip()
            elif line.startswith("Creation Time :"):
                creation_time = line.split(":", 1)[1].strip()
                mdadm_data['creation_time'] = creation_time
                mdadm_data['creation_time_ts'] = convert_to_unix_ts(creation_time)
            elif line.startswith("Raid Level :"):
                mdadm_data['raid_level'] = line.split(":", 1)[1].strip()
            elif line.startswith("Raid Devices :"):
                mdadm_data['raid_devices'] = int(line.split(":", 1)[1].strip())
            elif line.startswith("Total Devices :"):
                mdadm_data['total_devices'] = int(line.split(":", 1)[1].strip())
            elif line.startswith("Consistency Policy :"):
                mdadm_data['consistency_policy'] = line.split(":", 1)[1].strip()
            elif line.startswith("UUID :"):
                mdadm_data['uuid'] = line.split(":", 1)[1].strip()
            elif line.startswith("Name :"):
                mdadm_data['name'] = line.split(":", 1)[1].strip()
            elif line.startswith("Layout :"):
                mdadm_data['layout'] = line.split(":", 1)[1].strip()
            elif line.startswith("Events :"):
                mdadm_data['events'] = int(line.split(":", 1)[1].strip())
            elif line.startswith("Chunk Size :"):
                mdadm_data['chunk_sz_bytes'] = line.split(":", 1)[1].strip()
                unit: str = mdadm_data['chunk_sz_bytes'][-1].upper()
                value: int = int(mdadm_data['chunk_sz_bytes'][:-1])
                if unit == "K":
                    mdadm_data['chunk_sz_bytes'] = value * 1024
                elif unit == "M":
                    mdadm_data['chunk_sz_bytes'] = value * 1024 * 1024
                elif unit == "G":
                    mdadm_data['chunk_sz_bytes'] = value * 1024 * 1024 * 1024
                else:
                    logger.error(f"Unsupported chunk size from mdadm: {mdadm_data['chunk_sz_bytes']}")

            # Match RAID device entries
            elif re.match(r"^\s*\d+\s+\d+\s+\d+\s+\d+\s+\S+", line):
                parts = line.split()
                current_device = parts[-1]  # Last part is the device name (e.g., '/dev/sda1')
                raid_devices[current_device] = {
                    'number': int(parts[0]),
                    'major': int(parts[1]),
                    'minor': int(parts[2]),
                    'raid_device': int(parts[3]),
                    'state': " ".join(parts[4:-1]),  # Combine the state description
                }

    # Add parsed RAID devices to the dictionary
    if raid_devices:
        mdadm_data['raid_devices_details'] = raid_devices

    return mdadm_data

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

