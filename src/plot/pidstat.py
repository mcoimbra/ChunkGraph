from datetime import datetime
import math
import os
import pprint
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Load pidstat data from the file.
def parse_pidstat(file_path: str) -> pd.DataFrame:
    """
    Parses a pidstat output file into a pandas DataFrame with appropriate column mapping and data types.
    
    Args:
        file_path (str): The path to the pidstat output file.
    
    Returns:
        pd.DataFrame: A DataFrame containing the parsed data.
    """
    data: List[Dict[str, Optional[str]]] = []
    headers: Optional[List[str]] = None  # Store current header mapping
    
    with open(file_path, 'r') as f:
        for line in f:
            line: str = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Detect and parse header lines
            if "PID" in line and "Command" in line:
                headers = line.split()

                #print(headers[0])
                #pprint.pprint(headers)
                #sys.exit(0)
                headers[0] = "Time_s"
                headers[1] = "Time"
                headers = headers[1:]
                continue


            
            # Parse data rows if headers are detected.
            if headers:
                #parts: List[str] = line.split(maxsplit=len(headers) - 1)
                parts: List[str] = line.split()

                # Merge together the `time`` and `AM/PM marker` into a single
                # 24-hour format column.
                time_str: str = parts[0]
                am_pm_marker: str = parts[1]

                # Parse the 12-hour format time string
                time_obj: datetime = datetime.strptime(f"{time_str} {am_pm_marker}", "%I:%M:%S %p")

                # Convert to 24-hour format
                time_24hr: str = time_obj.strftime("%H:%M:%S")

                # Remove the column at index 0 which is now irrelevant.
                parts[1] = time_24hr

                # Store the 24-hour format string as a number of seconds.
                # Parse the time string to a datetime object
                time_obj: datetime = datetime.strptime(time_24hr, "%H:%M:%S")

                # Calculate the total number of seconds since midnight
                total_seconds: int = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

                parts[0] = 
                #parts = parts[1:]

                # The header was appropriately corrected as well.
                if len(parts) == len(headers):  # Ensure row matches header length
                    row: Dict[str, Optional[str]] = {
                        headers[i]: parts[i] for i in range(len(headers))
                    }

                    # pprint.pprint("header row:")
                    # pprint.pprint(parts)
                    # sys.exit(0)

                    data.append(row)

    # Convert to DataFrame
    df: pd.DataFrame = pd.DataFrame(data)
    
    # Convert numeric columns to appropriate types if they exist
    numeric_columns: List[str] = [
        "Time_s", "UID", "PID", "%usr", "%system", "%guest", "%wait", "%CPU", "CPU"
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

def save_to_tsv(dataframe: pd.DataFrame, output_file: str) -> None:
    """
    Saves a pandas DataFrame to a .tsv file with column titles and values separated by tabs.
    
    Args:
        dataframe (pd.DataFrame): The DataFrame to be saved.
        output_file (str): The path to the output .tsv file.
    """
    # Save the DataFrame as a tab-separated values file
    dataframe.to_csv(output_file, sep="\t", index=False, header=True)

def plot_iostat_by_cpu(df: pd.DataFrame, output_dir: str, plot_basename: str, dpi=300) -> None:

    plt.figure()

    plt.xlabel("Time")
    plt.ylabel("%CPU")
    plt.title("CPU Usage by Core")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Plot by CPU ID.
    for cpu_id in df['CPU'].unique():
        cpu_data = df[df['CPU'] == cpu_id]
        plt.plot(cpu_data['Time'], cpu_data['%CPU'], label=f'CPU {cpu_id}')
    
    for ext in ["pdf", "png"]:
        fig_output_path = os.path.join(output_dir, f"{plot_basename}.{ext}")
        if ext == "pdf":
            plt.savefig(fig_output_path)
        else:
            plt.savefig(fig_output_path, dpi=dpi)

def plot_iostat_by_cpu_v2(df: pd.DataFrame, output_dir: str, plot_basename: str, dpi=300, bins: int = 20) -> None:
    """
    Plots aggregated CPU usage by core over time, grouping into bins for readability.
    
    Args:
        df (pd.DataFrame): DataFrame containing columns "Time", "%CPU", and "CPU".
        output_dir (str): Directory to save the plot.
        plot_basename (str): Basename for the plot files.
        dpi (int, optional): DPI for the plot. Default is 300.
        bins (int, optional): Number of bins to aggregate data into. Default is 20.
    """
    # Convert "Time" column to pandas datetime if necessary
    if not pd.api.types.is_datetime64_any_dtype(df["Time"]):
        df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S")

    # Aggregate data into bins
    bin_edges = np.linspace(df["Time"].min().value, df["Time"].max().value, bins + 1)
    bin_labels = pd.to_datetime(bin_edges).strftime('%H:%M:%S')
    df['Time_Binned'] = pd.cut(df["Time"], bins=pd.to_datetime(bin_edges), labels=bin_labels[:-1], right=False)

    # Group by bins and CPU, then aggregate
    grouped = df.groupby(["Time_Binned", "CPU"])["%CPU"].mean().reset_index()

    # Plot the data
    plt.figure(figsize=(12, 6))
    for cpu_id in df["CPU"].unique():
        cpu_data = grouped[grouped["CPU"] == cpu_id]
        plt.plot(cpu_data["Time_Binned"], cpu_data["%CPU"], label=f"CPU {cpu_id}")

    # Improve x-axis readability
    plt.xlabel("Time")
    plt.ylabel("%CPU")
    plt.title("CPU Usage by Core (Binned)")
    plt.legend()
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()

    # Save the plot in multiple formats
    for ext in ["pdf", "png"]:
        fig_output_path = os.path.join(output_dir, f"{plot_basename}_binned.{ext}")
        plt.savefig(fig_output_path, dpi=dpi if ext == "png" else None)

    plt.close()


# def plot_cpu_metrics(df: pd.DataFrame, output_dir: str, plot_basename: str, dpi=300, bins: int = 20) -> None:
#     """
#     Plots the metrics %CPU, %system, %usr, and %wait over time, optionally grouping into bins.

#     Args:
#         df (pd.DataFrame): DataFrame containing the columns "Time", "%CPU", "%system", "%usr", "%wait".
#         output_dir (str): Directory to save the plot.
#         plot_basename (str): Basename for the plot files.
#         dpi (int, optional): DPI for the plot. Default is 300.
#         bins (int, optional): Number of bins to aggregate data into. Default is 20.
#     """
#     # Convert "Time" column to pandas datetime if necessary
#     if not pd.api.types.is_datetime64_any_dtype(df["Time"]):
#         df["Time"] = pd.to_datetime(df["Time"], format="%I:%M:%S %p")

#     # Aggregate data into bins
#     bin_edges = np.linspace(df["Time"].min().value, df["Time"].max().value, bins + 1)
#     bin_labels = pd.to_datetime(bin_edges).strftime('%H:%M:%S')
#     df['Time_Binned'] = pd.cut(df["Time"], bins=pd.to_datetime(bin_edges), labels=bin_labels[:-1], right=False)

#     # Group by bins and calculate the mean for each metric
#     metrics = ["%CPU", "%system", "%usr", "%wait"]
#     grouped = df.groupby("Time_Binned")[metrics].mean().reset_index()

#     # Plot the data
#     plt.figure(figsize=(12, 6))
#     for metric in metrics:
#         plt.plot(grouped["Time_Binned"], grouped[metric], label=metric)

#     # Improve x-axis readability
#     plt.xlabel("Time")
#     plt.ylabel("Percentage")
#     plt.title("CPU Metrics Over Time")
#     plt.legend()
#     plt.xticks(rotation=45, fontsize=8)
#     plt.tight_layout()

#     # Save the plot in multiple formats
#     for ext in ["pdf", "png"]:
#         fig_output_path = os.path.join(output_dir, f"{plot_basename}_metrics.{ext}")
#         plt.savefig(fig_output_path, dpi=dpi if ext == "png" else None)

#     plt.close()

def plot_cpu_metrics_with_markers(df: pd.DataFrame, output_dir: str, plot_basename: str, dpi=300, bins: int = 20) -> None:
    """
    Plots the metrics %CPU, %system, %usr, and %wait over time with distinct markers, optionally grouping into bins.

    Args:
        df (pd.DataFrame): DataFrame containing the columns "Time", "%CPU", "%system", "%usr", "%wait".
        output_dir (str): Directory to save the plot.
        plot_basename (str): Basename for the plot files.
        dpi (int, optional): DPI for the plot. Default is 300.
        bins (int, optional): Number of bins to aggregate data into. Default is 20.
    """
    # Convert "Time" column to pandas datetime if necessary
    if not pd.api.types.is_datetime64_any_dtype(df["Time"]):
        df["Time"] = pd.to_datetime(df["Time"], format="%I:%M:%S %p")

    # Aggregate data into bins
    bin_edges = np.linspace(df["Time"].min().value, df["Time"].max().value, bins + 1)
    bin_labels = pd.to_datetime(bin_edges).strftime('%H:%M:%S')
    df['Time_Binned'] = pd.cut(df["Time"], bins=pd.to_datetime(bin_edges), labels=bin_labels[:-1], right=False)

    # Group by bins and calculate the mean for each metric
    metrics = ["%CPU", "%system", "%usr", "%wait"]
    markers = ["o", "s", "D", "^"]  # Circle, square, diamond, triangle
    grouped = df.groupby("Time_Binned")[metrics].mean().reset_index()

    # Plot the data
    plt.figure(figsize=(12, 6))
    for metric, marker in zip(metrics, markers):
        plt.plot(grouped["Time_Binned"], grouped[metric], label=metric, marker=marker)

    # Improve x-axis readability
    plt.xlabel("Time")
    plt.ylabel("Percentage")
    plt.title("CPU Metrics Over Time with Markers")
    plt.legend()
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()

    # Save the plot in multiple formats
    for ext in ["pdf", "png"]:
        fig_output_path = os.path.join(output_dir, f"{plot_basename}_metrics_with_markers.{ext}")
        plt.savefig(fig_output_path, dpi=dpi if ext == "png" else None)

    plt.close()


def plot_cpu_metrics_with_transparent_markers(
    df: pd.DataFrame, output_dir: str, plot_basename: str, dpi=300, bins: int = 20
) -> None:
    """
    Plots the metrics %CPU, %system, %usr, and %wait over time with distinct and transparent markers,
    optionally grouping into bins.

    Args:
        df (pd.DataFrame): DataFrame containing the columns "Time", "%CPU", "%system", "%usr", "%wait".
        output_dir (str): Directory to save the plot.
        plot_basename (str): Basename for the plot files.
        dpi (int, optional): DPI for the plot. Default is 300.
        bins (int, optional): Number of bins to aggregate data into. Default is 20.
    """
    # Convert "Time" column to pandas datetime if necessary
    if not pd.api.types.is_datetime64_any_dtype(df["Time"]):
        df["Time"] = pd.to_datetime(df["Time"], format="%I:%M:%S %p")

    # Aggregate data into bins
    bin_edges = np.linspace(df["Time"].min().value, df["Time"].max().value, bins + 1)
    bin_labels = pd.to_datetime(bin_edges).strftime('%H:%M:%S')
    df['Time_Binned'] = pd.cut(df["Time"], bins=pd.to_datetime(bin_edges), labels=bin_labels[:-1], right=False)

    # Group by bins and calculate the mean for each metric
    metrics = ["%CPU", "%system", "%usr", "%wait"]
    markers = ["o", "s", "D", "^"]  # Circle, square, diamond, triangle
    grouped = df.groupby("Time_Binned")[metrics].mean().reset_index()

    # Plot the data
    plt.figure(figsize=(12, 6))
    for metric, marker in zip(metrics, markers):
        plt.plot(
            grouped["Time_Binned"], grouped[metric],
            label=metric,
            marker=marker,
            alpha=0.7  # Set marker transparency
        )

    # Improve x-axis readability
    plt.xlabel("Time")
    plt.ylabel("Percentage")
    plt.title("CPU Metrics Over Time with Transparent Markers")
    plt.legend()
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()

    # Save the plot in multiple formats
    for ext in ["pdf", "png"]:
        fig_output_path = os.path.join(output_dir, f"{plot_basename}_metrics_with_transparent_markers.{ext}")
        plt.savefig(fig_output_path, dpi=dpi if ext == "png" else None)

    plt.close()