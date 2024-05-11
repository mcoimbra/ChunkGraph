#include <iostream>
#include <string>
#include <queue>
#include <deque>
#include "llvm/Support/CommandLine.h"
#include "galois/Galois.h"
#include "galois/Bag.h"
#include "Type.h"
#include "Graph.h"
#include "atomics.h"
#include "Array.h"
#include "Util.h"
#include "EdgeMap.h"
#include "VertexMap.h"
#include "boilerplate.h"
#include "Runtime.h"

using namespace blaze;
namespace cll = llvm::cl;

#define MOD 16

static cll::opt<unsigned int>
    startNode("startNode",
            cll::desc("Node to start search from (default value 0)"),
            cll::init(0));

static cll::opt<unsigned int>
        binSpace("binSpace",
                cll::desc("Size of bin space in MB (default: 256)"),
                cll::init(256));

static cll::opt<int>
        binCount("binCount",
                cll::desc("Number of bins (default: 4096)"),
                cll::init(BIN_COUNT));

static cll::opt<int>
        binBufSize("binBufSize",
                cll::desc("Size of a bin buffer (default: 128)"),
                cll::init(BIN_BUF_SIZE));

static cll::opt<float>
        binningRatio("binningRatio",
                cll::desc("Binning worker ratio (default: 0.67)"),
                cll::init(BINNING_WORKER_RATIO));


struct BF_F : public EDGEMAP_F<uint32_t> {
    Array<uint32_t>& visited;
    Array<uint32_t>& splen;

    BF_F(Array<uint32_t>& v, Array<uint32_t>& s, Bins* b): visited(v), splen(s), EDGEMAP_F(b) {}

    inline bool cond(VID dst) {
        return true;
    }

    inline uint32_t scatter(VID src, VID dst) {
        return splen[src] + (src + dst) % MOD + 1;
    }

    inline bool gather(VID dst, uint32_t val) {
		if(splen[dst] > val) {
            splen[dst] = val;
            if(!visited[dst]) {visited[dst] = 1; return true;}
        }
        return false;
    }
};

struct BF_Vertex_F {
    Array<uint32_t>& visited;

    BF_Vertex_F(Array<uint32_t>& v): visited(v) {}

    inline bool operator() (const VID& node) {
        visited[node] = 0;
        return 1;
    }
};

struct BF_Vertex_Init {
    Array<uint32_t>& visited;
    Array<uint32_t>& splen;

    BF_Vertex_Init(Array<uint32_t>& v, Array<uint32_t>& s): visited(v), splen(s) {}

    inline bool operator() (const VID& node) {
        visited[node] = 0;
        splen[node] = INT_MAX >> 1;
        return 1;
    }
};

int main(int argc, char **argv) {
    AgileStart(argc, argv);

#ifdef PAGECACHE
    pid_t ppid = getpid();
    std::string pcache_command = "bash ../scripts/pagecache.sh " + std::to_string(ppid);
    int pcache_res = std::system(pcache_command.c_str());
    if (pcache_res == -1) {
    std::cout << "pagecache.sh failed" << std::endl;
    } else {
    std::cout << "pagecache.sh succeeded" << std::endl;
      }
#endif

    Runtime runtime(numComputeThreads, numIoThreads, ioBufferSize * MB);
    runtime.initBinning(binningRatio);

    Graph outGraph;
    outGraph.BuildGraph(outIndexFilename, outAdjFilenames);

#ifdef CACHEMISS
      pid_t pid = getpid();
      std::string cache_command = "bash ../scripts/perf.sh " + std::to_string(pid);
      int cache_res = std::system(cache_command.c_str());
      if (cache_res == -1) {
        std::cout << "perf.sh failed" << std::endl;
      } else {
        std::cout << "perf.sh succeeded" << std::endl;
      }
#endif

    uint64_t n = outGraph.NumberOfNodes();

    Array<uint32_t> visited, splen;
    visited.allocate(n);
    splen.allocate(n);

    // Allocate bins
    unsigned nthreads = galois::getActiveThreads();
    uint64_t binSpaceBytes = (uint64_t)binSpace * MB;
    Bins *bins = new Bins(outGraph, nthreads, binSpaceBytes,
                          binCount, binBufSize, binningRatio);

    vertexMap(outGraph, BF_Vertex_Init(visited, splen));

    splen[startNode] = 0;

    Worklist<VID>* frontier = new Worklist<VID>(n);
    frontier->activate(startNode);

    galois::StatTimer time("Time", "BF_MAIN");
    time.start();

    while (!frontier->empty()) {
        Worklist<VID>* output = edgeMap(outGraph, frontier, BF_F(visited, splen, bins), prop_blocking);
        vertexMap(output, BF_Vertex_F(visited));
        delete frontier;
        frontier = output;
    }

    delete frontier;

    time.stop();

    delete bins;

    return 0;
}
