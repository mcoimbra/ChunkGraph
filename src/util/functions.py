import os
import grp
import pathlib
import pwd
import subprocess
import sys
from typing import List, Optional

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