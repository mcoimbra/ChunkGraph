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

enum {UNDECIDED,CONDITIONALLY_IN,OUT,IN};

using namespace blaze;
namespace cll = llvm::cl;

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


struct MIS_Update : public EDGEMAP_F<uint32_t> {
    Array<int32_t>& flags;

    MIS_Update(Array<int32_t>& f, Bins* b): flags(f), EDGEMAP_F(b) {}

    inline bool cond(VID dst) {
        return true;
    }

    inline uint32_t scatter(VID src, VID dst) {
        //if neighbor is in MIS, then we are out
        if(flags[dst] == IN) {if(flags[src] != OUT) flags[src] = OUT;}
        //if neighbor has higher priority (lower ID) and is undecided, then so are we
        else if(dst < src && flags[src] == CONDITIONALLY_IN && flags[dst] < OUT)
            flags[src] = UNDECIDED;
        return src;
    }

    inline bool gather(VID dst, uint32_t val) {
        return 1;
    }
};

struct MIS_Filter {
    Array<int32_t>& flags;

    MIS_Filter(Array<int32_t>& f): flags(f) {}

    inline bool operator() (const VID& node) {
        if(flags[node] == CONDITIONALLY_IN) { flags[node] = IN; return 0; } //vertex in MIS
        else if(flags[node] == OUT) return 0; //vertex not in MIS
        else { flags[node] = CONDITIONALLY_IN; return 1; } //vertex undecided, move to next round
    }
};

struct MIS_Vertex_Init {
    Array<int32_t>& flags;

    MIS_Vertex_Init(Array<int32_t>& f): flags(f) {}

    inline bool operator() (const VID& node) {
        flags[node] = CONDITIONALLY_IN;
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

    Array<int32_t> flags;
    flags.allocate(n);

    // Allocate bins
    unsigned nthreads = galois::getActiveThreads();
    uint64_t binSpaceBytes = (uint64_t)binSpace * MB;
    Bins *bins = new Bins(outGraph, nthreads, binSpaceBytes,
                          binCount, binBufSize, binningRatio);

    vertexMap(outGraph, MIS_Vertex_Init(flags));

    Worklist<VID>* frontier = new Worklist<VID>(n);
    frontier->activate_all();

    galois::StatTimer time("Time", "MIS_MAIN");
    time.start();

    while (!frontier->empty()) {
        Worklist<VID>* output = edgeMap(outGraph, frontier, MIS_Update(flags, bins), prop_blocking);
        vertexMap(output, MIS_Filter(flags));
        delete frontier;
        frontier = output;
    }

    delete frontier;

    time.stop();

    delete bins;

    return 0;
}
