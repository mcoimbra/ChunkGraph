#!/bin/bash

# Define log files.
CPU_LOG="cpu_log.txt"
MEMORY_LOG="memory_log.txt"
BLKTRACE_LOG="blktrace_log.txt"
IOSTAT_LOG="iostat_log.txt"
VMSTAT_LOG="vmstat_log.txt"

# Define workload (e.g., a sample command).
workload_command="./BFS -b -chunk -r 12 -t 48 ../Dataset/LiveJournal/chunk/livejournal"

# Start workload in the background.
$workload_command &  # Adjust as necessary
# Get its PID (the last command that was executed).
WORKLOAD_PID=$!

# Run performance monitoring tools in background
# CPU monitoring with mpstat (update every 1 second).
mpstat -P ALL 1 > $CPU_LOG &

# Memory monitoring with vmstat (update every 1 second).
vmstat 1 > $MEMORY_LOG &

# Storage monitoring with iostat (update every 1 second).
iostat -d -m -x 1 > $IOSTAT_LOG &

# (Optional) Run blktrace if specific block-level tracing is needed (requires root privileges)
# sudo blktrace -d /dev/sdX -o - | blkparse -i - > blktrace_log.txt &

# Wait for the workload to finish
wait $WORKLOAD_PID

# Kill background monitoring processes
pkill mpstat
pkill vmstat
pkill iostat
# (Optional) pkill blktrace

echo "Workload and monitoring completed. Logs saved to $CPU_LOG, $MEMORY_LOG, and $IOSTAT_LOG."
