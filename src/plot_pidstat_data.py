import argparse
import logging
import os
import sys
import time
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Logging import must come before other local project modules.
import util.logging as log_conf
logger: logging.Logger = log_conf.Logger.get_logger(__name__)
import util.functions as util_functions
import plot.pidstat as pidstat_plotting

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))

def create_arg_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Plot pidstat data.")

    parser.add_argument(
        "-o", "--output_dir", required=True,
        help="Directory to store pidstat plotting output."
    )
    parser.add_argument(
        "-i", "--input_dir", required=True,
        help="Path to directory with pidstat outputs to plot."
    )
    parser.add_argument(
        "--generate_tsv", required=False, action="store_true",
        help="Generate a TSV file from the `pidstat` output?"
    )

    parser.add_argument(
        "--per_cpu_plots", required=False, action="store_true",
        help="Should per-CPU visualizations be plotted?"
    )
    

    return parser


# call_code
# pid=$!
# pidstat -t -p $pid 1 > pidstat.txt

# python3 -m src.plot_pidstat_data -i "Outputs/ChunkGraph_UKdomain-2007/pidstat" -o "Outputs/ChunkGraph_UKdomain-2007/pidstat"

def main():

    # Parse arguments.
    parser: argparse.ArgumentParser = create_arg_parser()
    args: argparse.Namespace = parser.parse_args()

    # Validate inputs.
    util_functions.validate_output_directory(args.output_dir)

    # Argument validation.
    if not util_functions.dir_exists_and_not_empty(args.input_dir):
        logger.error("Problem with '-i'/'--input-dir' parameter. Exiting.")
        sys.exit(1)

    # Check that the input directory contains a `traces` directory.
    # sub_dirs: List[str] = ["traces"]
    # if not util_functions.dir_contains_elements(args.input_dir, sub_dirs):
    #     logger.error("Expected to find the following directories:")
        
    # Check that the input directory contains a file ending in `pidstat.merged.txt`.
    #trace_file_path: str = check_and_get_pidstat_file(args.input_dir)

    target_pidstat_file_path: str = os.path.join(args.input_dir, "pidstat.txt")
    if not (os.path.exists(target_pidstat_file_path) and os.path.isfile(target_pidstat_file_path)):
        logger.error(f"Exiting because 'pidstat' file was not found:\n\t{target_pidstat_file_path}")
        sys.exit(1)

    # Read `pidstat` data into a pandas.DataFrame.
    start_time: float = time.perf_counter()
    df: pd.DataFrame = pidstat_plotting.parse_pidstat(target_pidstat_file_path)
    end_time: float = time.perf_counter()
    logger.info(f"Finished reading pidstat data ({end_time-start_time}).")

    # Generate a TSV file from the output of `pidstat`.
    target_tsv_path: str = target_pidstat_file_path.replace(".txt", ".tsv")
    if args.generate_tsv or (not (os.path.exists(target_tsv_path) and os.path.isfile(target_tsv_path))):
        logger.info(f"Generating TSV file:\n\t{target_tsv_path}")
        start_time: float = time.perf_counter()
        pidstat_plotting.save_to_tsv(df, target_tsv_path)
        end_time: float = time.perf_counter()
        logger.info(f"Finished generating TSV file ({end_time-start_time})")
    
    count_row: int = df.shape[0]  # Gives number of rows
    count_col: int = df.shape[1]  # Gives number of columns
    logger.info(f"'pidstat' dataframe has {count_row} lines and {count_col} columns.")

    pidstat_plots_dir: str = os.path.join(args.input_dir, "plots")
    os.makedirs(pidstat_plots_dir, exist_ok=True)

    # Visualize throughput.
    # start_time: float = time.perf_counter()
    # visualize_throughput_overview(df, f"{plot_basename}-throughput", pidstat_plots_dir)
    # end_time: float = time.perf_counter()
    # logger.info(f"Throughput plotting total time: {end_time-start_time}")

    plot_basename: str = "pidstat"
    if args.per_cpu_plots:
        #pidstat_plotting.plot_iostat_by_cpu(df, pidstat_plots_dir, plot_basename, 300)
        pidstat_plotting.plot_iostat_by_cpu_v2(df, pidstat_plots_dir, plot_basename, 300, 20)
    pidstat_plotting.plot_cpu_metrics_with_transparent_markers(df, pidstat_plots_dir, plot_basename, 300, 20)
        

if __name__ == "__main__":
    main()

