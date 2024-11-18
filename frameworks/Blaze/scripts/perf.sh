#!/bin/bash

# This script is used to monitor cache miss performance

pid=$1

result_dir="../scripts/results/"

# get the newest directory in ../results/logs/
result_dir=$(ls -td ${result_dir}/*/ | head -1)

# get the newset directory in result_dir
result_dir=$(ls -td ${result_dir}/*/ | head -1)

# get the newest file in result_dir, and remove the suffix
filename=$(ls -t ${result_dir} | head -1 | cut -d'.' -f1)

L1=false
LLC=false

# CACHE_ITEM="cache-references,cache-misses"
# CACHE_ITEM="cache-references,cache-misses,cycles,instructions,branches,migrations,faults,minor-faults,major-faults"
CACHE_ITEM="cycles,instructions,branches,migrations"

if [ "$L1" = true ]; then
    CACHE_L1=",L1-dcache-load-misses,L1-dcache-loads,L1-dcache-stores,L1-icache-load-misses"
fi

if [ "$LLC" = true ]; then
    CACHE_LLC=",LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses"
fi

# Get the cache miss performance for pid
echo "zwx.1005" | sudo -S perf stat -e ${CACHE_ITEM}${CACHE_L1}${CACHE_LLC} -p $pid -o ${result_dir}/${filename}.compute &