#!/bin/bash

export CHUNK=1

DATA_PATH=/mnt/nvme1/zorax/chunks/

# CPU:use NUMA 0 node, with id 0-23 and 48-71, with taskset command
# TEST_CPU_SET="taskset --cpu-list 0-95:1"
TEST_CPU_SET="taskset -c 0-23,48-71:1"

export OMP_PROC_BIND=true

name[0]=twitter
name[1]=friendster
name[2]=ukdomain
name[3]=kron29
name[4]=kron30
name[5]=yahoo

data[0]=${DATA_PATH}twitter/${name[0]}
data[1]=${DATA_PATH}friendster/${name[1]}
data[2]=${DATA_PATH}ukdomain/${name[2]}
data[3]=${DATA_PATH}kron29/${name[3]}
data[4]=${DATA_PATH}kron30/${name[4]}
data[5]=${DATA_PATH}yahoo/${name[5]}

# roots:        TT   FS     UK    K29       K30       YW
declare -a rts=(12 801109 5699262 310059974 233665123 35005211)
declare -a reorder_rts=(0 0 0 0 0 0 0)
declare -a kcore_iter=(10 10 10 10 10 3)

function set_schedule {
	SCHEDULE=$1
	export OMP_SCHEDULE="${SCHEDULE}"
}

set_schedule "dynamic"

clear_pagecaches() { 
    echo "zwx.1005" | sudo -S sysctl -w vm.drop_caches=3;
}



# Test Process
log_time=$(date "+%Y%m%d_%H%M%S")
mkdir -p results/logs/${log_time}

cd apps && make clean   

outputFile="../results/chunkgraph_query_time.csv"
title="ChunkGraph"
cur_time=$(date "+%Y-%m-%d %H:%M:%S")
echo $cur_time "Test ${title} Query Performace" >> ${outputFile}

len=1

profile_performance() {
    eval commandargs="$1"
    eval filename="$2"

    commandname=$(echo $commandargs | awk '{print $1}')

    commandargs="${TEST_CPU_SET} ${commandargs}"

    cur_time=$(date "+%Y-%m-%d %H:%M:%S")

    log_dir="../results/logs/${log_time}"

    echo $cur_time "profile run with command: " $commandargs >> ../results/command.log
    echo $cur_time "profile run with command: " $commandargs > ${log_dir}/${filename}.txt

    nohup $commandargs &>> ${log_dir}/${filename}.txt &
    pid=$(ps -ef | grep $commandname | grep -v grep | awk '{print $2}')

    echo "pid: " $pid >> ../results/command.log

    sudo perf stat -e cycles,instructions,branches,migrations -p $pid -o ${log_dir}/${filename}.compute &

    wait $pid

    res=$(awk 'NR>5 {sum+=$5} END {printf "%.0f\n", sum}' ${log_dir}/${filename}.diskio)
    # echo total bytes read in KB and convert to GB
    echo "total bytes read during compute: " $res "KB ("$(echo "scale=2; $res/1024/1024" | bc) "GB)" >> ${log_dir}/${filename}.txt
    echo "total bytes read during compute: " $res "KB ("$(echo "scale=2; $res/1024/1024" | bc) "GB)" >> ${log_dir}/${filename}.diskio

    sleep 1s
    echo >> ${log_dir}/${filename}.txt

}

for idx in {0,1,2,3,4,5};
do
    echo -n "Data: "
    echo ${data[$idx]}
    echo -n "Root: "
    echo ${rts[$idx]}

    make BFS
    for ((mem=0;mem<$len;mem++))
    do
        clear_pagecaches
        commandargs="./BFS -b -r ${rts[$idx]} -chunk -threshold 20  ${data[${idx}]}"
        filename="${name[${idx}]}_chunkgraph_bfs"
        profile_performance "\${commandargs}" "\${filename}"
        wait
    done

    make BellmanFord
    for ((mem=0;mem<$len;mem++))
    do
        clear_pagecaches
        commandargs="./BellmanFord -b -r ${rts[$idx]} -chunk -threshold 5 ${data[${idx}]}"
        filename="${name[${idx}]}_chunkgraph_bf"
        profile_performance "\${commandargs}" "\${filename}"
        wait
    done

    make BC
    for ((mem=0;mem<$len;mem++))
    do
        clear_pagecaches
        commandargs="./BC -b -r ${rts[$idx]} -chunk -threshold 20 ${data[${idx}]}"
        filename="${name[${idx}]}_chunkgraph_bc"
        profile_performance "\${commandargs}" "\${filename}"
        wait
    done
    
    make KCore
    for ((mem=0;mem<$len;mem++))
    do
        clear_pagecaches
        commandargs="./KCore -b -maxk ${kcore_iter[$idx]} -chunk -threshold 20 ${data[${idx}]}"
        filename="${name[${idx}]}_chunkgraph_kcore"
        profile_performance "\${commandargs}" "\${filename}"
        wait
    done

    make Radii
    for ((mem=0;mem<$len;mem++))
    do
        clear_pagecaches
        commandargs="./Radii -b -chunk -threshold 20 ${data[${idx}]} "
        filename="${name[${idx}]}_chunkgraph_radii"
        profile_performance "\${commandargs}" "\${filename}"
        wait
    done

    make PageRankDelta
    for ((mem=0;mem<$len;mem++))
    do
        clear_pagecaches
        commandargs="./PageRankDelta -b -chunk -threshold 20 ${data[${idx}]}"
        filename="${name[${idx}]}_chunkgraph_pagerank"
        profile_performance "\${commandargs}" "\${filename}"
        wait
    done
done