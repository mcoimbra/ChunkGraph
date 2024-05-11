#!/bin/bash

cur_time=$(date "+%Y%m%d_%H%M%S")
result_dir=../../results/blaze/
disks=/mnt/nvme2/blaze
threads=48

# Run workloads
# roots:        TT   FS     UK    K29       K30       YW
declare -a rts=(12 801109 5699262 310059974 233665123 35005211)

name[0]=twitter
name[1]=friendster
name[2]=ukdomain
name[3]=kron29
name[4]=kron30
name[5]=yahoo

cd blaze
mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j 16 && cd ../scripts/


# BFS
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k bfs -d ${name[0]} --start_node ${rts[0]}
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k bfs -d ${name[1]} --start_node ${rts[1]}
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k bfs -d ${name[2]} --start_node ${rts[2]}
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k bfs -d ${name[3]} --start_node ${rts[3]}
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k bfs -d ${name[4]} --start_node ${rts[4]}
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k bfs -d ${name[5]} --start_node ${rts[5]}

# BellmanFord
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k BellmanFord -d ${name[0]} --start_node ${rts[0]}
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k BellmanFord -d ${name[1]} --start_node ${rts[1]}
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k BellmanFord -d ${name[2]} --start_node ${rts[2]}
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k BellmanFord -d ${name[3]} --start_node ${rts[3]}
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k BellmanFord -d ${name[4]} --start_node ${rts[4]}
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k BellmanFord -d ${name[5]} --start_node ${rts[5]}

# BC
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k bc -d ${name[0]} --start_node ${rts[0]}
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k bc -d ${name[1]} --start_node ${rts[1]}
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k bc -d ${name[2]} --start_node ${rts[2]}
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k bc -d ${name[3]} --start_node ${rts[3]}
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k bc -d ${name[4]} --start_node ${rts[4]}
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k bc -d ${name[5]} --start_node ${rts[5]}

# KCore
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k kcore -d ${name[0]} --maxK 10
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k kcore -d ${name[1]} --maxK 10
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k kcore -d ${name[2]} --maxK 10
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k kcore -d ${name[3]} --maxK 10
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k kcore -d ${name[4]} --maxK 10
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k kcore -d ${name[5]} --maxK 3

# Radii
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k Radii -d ${name[0]}
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k Radii -d ${name[1]}
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k Radii -d ${name[2]}
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k Radii -d ${name[3]}
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k Radii -d ${name[4]}
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k Radii -d ${name[5]}

# PageRank 
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k pagerank -d ${name[0]} --max_iterations 10
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k pagerank -d ${name[1]} --max_iterations 10
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k pagerank -d ${name[2]} --max_iterations 10
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k pagerank -d ${name[3]} --max_iterations 10
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k pagerank -d ${name[4]} --max_iterations 10
./run.py --result_dir ${result_dir} --disks ${disks} -t ${threads} -k pagerank -d ${name[5]} --max_iterations 10
