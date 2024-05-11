[![DOI](https://zenodo.org/badge/477875804.svg)](https://zenodo.org/badge/latestdoi/477875804)

# Blaze

This repo contains artifacts for our SC '22 paper *Blaze: Fast Graph Processing for Fast SSDs*.

> More details will be updated

# Getting inside the docker container

As Blaze binaries are available in the provided docker image, it is necessary to run and get into the container as follows.

```bash
docker run --rm -it -v "/mnt/nvme2/blaze":"/mnt/nvme1/blaze" junokim8/blaze:1.0 /bin/bash
```

Inside the docker container console, the Blaze binaries are available at /home/zorax/GraphSSD/blaze

# Running each workload explicitly

For instance, run the following command to run BFS on rmat27 graph. This example calculates BFS using 17 threads 
(16 for computation and 1 for IO) starting from vertex 0.

```bash
# run on Twitter
./bin/bfs -computeWorkers 16 -startNode 12 /mnt/nvme2/blaze/twitter/twitter.gr.index /mnt/nvme2/blaze/twitter/twitter.gr.adj.0
./bin/bc -computeWorkers 16 -startNode 12 /mnt/nvme2/blaze/twitter/twitter.gr.index /mnt/nvme2/blaze/twitter/twitter.gr.adj.0 -inIndexFilename /mnt/nvme2/blaze/twitter/twitter.tgr.index -inAdjFilenames /mnt/nvme2/blaze/twitter/twitter.tgr.adj.0
```

```bash
mkdir -p build && cd build && cmake .. && make -j
```
```