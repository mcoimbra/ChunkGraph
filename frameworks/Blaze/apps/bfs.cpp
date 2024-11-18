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


struct BFS_F : public EDGEMAP_F<uint32_t> {
    Array<VID>& parents;

    BFS_F(Array<VID>& p, Bins* b): parents(p), EDGEMAP_F(b) {}

    inline bool cond(VID dst) {
        return parents[dst] == UINT_MAX;
    }

    inline uint32_t scatter(VID src, VID dst) {
        return src;
    }

    inline bool gather(VID dst, uint32_t val) {
		if (parents[dst] == UINT_MAX) {
			parents[dst] = val;
            return true;
		}
        return false;
    }
};

struct BFS_Vertex_Init {
    Array<VID>& parents;

    BFS_Vertex_Init(Array<VID>& p): parents(p) {}

    inline bool operator() (const VID& node) {
        parents[node] = UINT_MAX;
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

    Array<VID> parents;
    parents.allocate(n);

    // Allocate bins
    unsigned nthreads = galois::getActiveThreads();
    uint64_t binSpaceBytes = (uint64_t)binSpace * MB;
    Bins *bins = new Bins(outGraph, nthreads, binSpaceBytes,
                          binCount, binBufSize, binningRatio);

    vertexMap(outGraph, BFS_Vertex_Init(parents));

    parents[startNode] = startNode;

    Worklist<VID>* frontier = new Worklist<VID>(n);
    frontier->activate(startNode);

    galois::StatTimer time("Time", "BFS_MAIN");
    time.start();

    while (!frontier->empty()) {
        Worklist<VID>* output = edgeMap(outGraph, frontier, BFS_F(parents, bins), prop_blocking);
        delete frontier;
        frontier = output;
    }

    delete frontier;

    time.stop();

    delete bins;

    return 0;
}
