#!/bin/bash

# Make ChunkGraph and run different algorithms
# see README.md


# mkdir -p results/fig8

# Figure 8-10 Ligra-mmap
cur_time=$(date "+%Y-%m-%d %H:%M:%S")
echo $cur_time "Test Ligra-mmap Query Performace" >> results/progress.txt
echo "[ Expected completion time: around 1.5 hours ]" >> results/progress.txt
bash scripts/ligra_mmap.sh
sleep 10s
cur_time=$(date "+%Y-%m-%d %H:%M:%S")
echo $cur_time "Test Ligra-mmap Query Performace End" >> results/progress.txt

# Figure 8-10 ChunkGraph
cur_time=$(date "+%Y-%m-%d %H:%M:%S")
echo $cur_time "Test ChunkGraph Query Performace" >> results/progress.txt
echo "[ Expected completion time: around xx hours ]" >> results/progress.txt
bash scripts/chunkgraph.sh
sleep 10s
cur_time=$(date "+%Y-%m-%d %H:%M:%S")
echo $cur_time "Test ChunkGraph Query Performace End" >> results/progress.txt

# Figure 8-10 Blaze
cur_time=$(date "+%Y-%m-%d %H:%M:%S")
echo $cur_time "Test Blaze Query Performace" >> results/progress.txt
echo "[ Expected completion time: around xx hours ]" >> results/progress.txt
bash scripts/blaze.sh
sleep 10s
cur_time=$(date "+%Y-%m-%d %H:%M:%S")
echo $cur_time "Test Blaze Query Performace End" >> results/progress.txt