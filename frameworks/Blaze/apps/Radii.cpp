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

template <class ET>
inline bool CAS(ET *ptr, ET oldv, ET newv) {
  if (sizeof(ET) == 1) {
    return __sync_bool_compare_and_swap((bool*)ptr, *((bool*)&oldv), *((bool*)&newv));
  } else if (sizeof(ET) == 4) {
    return __sync_bool_compare_and_swap((int*)ptr, *((int*)&oldv), *((int*)&newv));
  } else if (sizeof(ET) == 8) {
    return __sync_bool_compare_and_swap((long*)ptr, *((long*)&oldv), *((long*)&newv));
  }
  else {
    std::cout << "CAS bad length : " << sizeof(ET) << std::endl;
    abort();
  }
}

template <class ET>
inline void writeOr(ET *a, ET b) {
  volatile ET newV, oldV; 
  do {oldV = *a; newV = oldV | b;}
  while ((oldV != newV) && !CAS(a, oldV, newV));
}

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

inline uint64_t hashInt(uint64_t a) {
   a = (a+0x7ed55d166bef7a1d) + (a<<12);
   a = (a^0xc761c23c510fa2dd) ^ (a>>9);
   a = (a+0x165667b183a9c0e1) + (a<<59);
   a = (a+0xd3a2646cab3487e3) ^ (a<<49);
   a = (a+0xfd7046c5ef9ab54c) + (a<<3);
   a = (a^0xb55a4f090dd4a67b) ^ (a>>32);
   return a;
}

struct Radii_F : public EDGEMAP_F<uint64_t> {
    Array<uint64_t>& radii;
    Array<uint64_t>& visited;
    Array<uint64_t>& nextVisited;
    int round;

    Radii_F(Array<uint64_t>& r, Array<uint64_t>& v, Array<uint64_t>& n, int ro, Bins* b):
        radii(r), visited(v), nextVisited(n), round(ro), EDGEMAP_F(b) {}

    inline bool cond(VID dst) {
        return true;
    }

    inline uint64_t scatter(VID src, VID dst) {
        return visited[src];
    }

    // inline bool gather(VID dst, uint64_t val) {
		// uint64_t toWrite = visited[dst] | val;
    //     if(visited[dst] != toWrite){
    //         writeOr(&nextVisited[dst], toWrite);
    //         uint64_t oldRadii = radii[dst];
    //         if(radii[dst] != round) { return CAS(&radii[dst],oldRadii,(uint64_t)round); }
    //     }
    //     return false;
    // }

    inline bool gather(VID dst, uint64_t val) {
		uint64_t toWrite = visited[dst] | val;
        if(visited[dst] != toWrite){
            nextVisited[dst] |= toWrite;
            if(radii[dst] != round) { radii[dst] = round; return true; }
        }
        return false;
    }
};

struct Radii_Vertex_F {
    Array<uint64_t>& visited;
    Array<uint64_t>& nextVisited;

    Radii_Vertex_F(Array<uint64_t>& v, Array<uint64_t>& n): visited(v), nextVisited(n) {}

    inline bool operator() (const VID& node) {
        visited[node] = nextVisited[node];
        return 1;
    }
};

struct Radii_Vertex_Init {
    Array<uint64_t>& radii;
    Array<uint64_t>& visited;
    Array<uint64_t>& nextVisited;

    Radii_Vertex_Init(Array<uint64_t>& r, Array<uint64_t>& v, Array<uint64_t>& n): 
        radii(r), visited(v), nextVisited(n) {}

    inline bool operator() (const VID& node) {
        radii[node] = -1;
        visited[node] = nextVisited[node] = 0;
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

    Array<uint64_t> radii;
    Array<uint64_t> visited, nextVisited;
    radii.allocate(n);
    visited.allocate(n);
    nextVisited.allocate(n);

    // Allocate bins
    unsigned nthreads = galois::getActiveThreads();
    uint64_t binSpaceBytes = (uint64_t)binSpace * MB;
    Bins *bins = new Bins(outGraph, nthreads, binSpaceBytes,
                          binCount, binBufSize, binningRatio);

    vertexMap(outGraph, Radii_Vertex_Init(radii, visited, nextVisited));

    Worklist<VID>* frontier = new Worklist<VID>(n);
    
    int sampleSize = n < 64 ? n : 64;  
    for(int i = 0; i < sampleSize; i++) { //initial set of vertices
        uint64_t v = hashInt(i) % n;
        radii[v] = 0;
        nextVisited[v] = (uint64_t) 1<<i;
        frontier->activate(v);
    }

    galois::StatTimer time("Time", "Radii_MAIN");
    time.start();

    int round = 0;
    while (!frontier->empty()) {
        round++;
        vertexMap(frontier, Radii_Vertex_F(visited, nextVisited));
        Worklist<VID>* output = edgeMap(outGraph, frontier, Radii_F(radii, visited, nextVisited, round, bins), prop_blocking);
        delete frontier;
        frontier = output;
    }

    delete frontier;

    time.stop();

    delete bins;

    return 0;
}
