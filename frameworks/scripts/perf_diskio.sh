#!/bin/bash

# This script is used to monitor disk I/O performance

pid=$1

#result_dir="../results/logs"
result_dir="Outputs/logs"

# get the newest directory in ../results/logs/
result_dir=$(ls -td ${result_dir}/*/ | head -1)
# get the newest file in result_dir, and remove the suffix
#filename=$(ls -t ${result_dir} | head -1 | cut -d'.' -f1)

# If result_dir is empty, generate a new directory and file name
if [ -z "$result_dir" ]; then
    result_dir="Outputs/logs/new_log"
    mkdir -p "$result_dir" # Create the new directory
    filename="log_$(date +%Y%m%d_%H%M%S)" # Generate a timestamped file name
else
    # Check if result_dir is empty
    if [ -z "$(ls -A "$result_dir")" ]; then
        filename="log_$(date +%Y%m%d_%H%M%S)" # Generate a timestamped file name
    else
        # Get the newest file in result_dir and remove the suffix
        filename=$(ls -t ${result_dir} | head -1 | cut -d'.' -f1)
    fi
fi

echo "Disk I/O Performance Test" $pid > ${result_dir}/${filename}.diskio

# Get the disk I/O performance for pid
sudo pidstat -d -p $pid 1 >> ${result_dir}/${filename}.diskio &