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
#include "boilerplate.h"
#include "Runtime.h"
#include "VertexMap.h"
#include "VertexFilter.h"
#include "PageRank.h"
#include "Bin.h"
#include "Param.h"
#include "gettime.h"

// #define LIGRA_ALIGN

using namespace blaze;
namespace cll = llvm::cl;

// All PageRank algorithm variants use the same constants for ease of
// comparison.
constexpr static const float DAMPING        = 0.85;
constexpr static const float EPSILON        = 1.0e-2;
constexpr static const float EPSILON2       = 1.0e-7;
constexpr static const unsigned MAX_ITER    = 3;

static cll::opt<float>
        damping("damping", cll::desc("damping"),
                cll::init(DAMPING));

static cll::opt<float>
        epsilon("epsilon", cll::desc("epsilon"),
                cll::init(EPSILON));

static cll::opt<float>
        epsilon2("epsilon2", cll::desc("epsilon2"),
                cll::init(EPSILON2));

static cll::opt<unsigned int>
        maxIterations("maxIterations",
                cll::desc("Maximum iterations"),
                cll::init(MAX_ITER));

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

#ifndef LIGRA_ALIGN
struct Node {
    float score, delta, ngh_sum;
};
#else
struct Node {
    float score, p_next;
};
#endif

// EdgeMap functions

#ifndef LIGRA_ALIGN
struct PR_F : public EDGEMAP_F<float> {
    Graph& graph;
    Array<Node>& data;

    PR_F(Graph& g, Array<Node>& d, Bins* b):
        graph(g), data(d), EDGEMAP_F(b)
    {}

    inline float scatter(VID src, VID dst) {
        return data[src].delta / graph.GetDegree(src);
    }

    inline bool gather(VID dst, float val) {
        data[dst].ngh_sum += val;
        return true;
    }
};
#else
struct PR_F : public EDGEMAP_F<float> {
    Graph& graph;
    Array<Node>& data;

    PR_F(Graph& g, Array<Node>& d, Bins* b):
        graph(g), data(d), EDGEMAP_F(b)
    {}

    // inline VID scatter(VID src, VID dst) {
    //     return src;
    // }

    // inline bool gather(VID dst, VID val) {
    //     data[dst].p_next += data[val].score / graph.GetDegree(val);
    //     return true;
    // }
    inline bool update(VID s, VID d){ //update function applies PageRank equation
    data[d].p_next += data[s].score / graph.GetDegree(s);
    return 1;
  }
  inline bool updateAtomic (VID s, VID d) { //atomic Update
    atomic_add(&data[d].p_next, data[s].score / graph.GetDegree(s));
    return 1;
  }
  inline bool cond (VID d) { return 1; }
};
#endif

// VertexMap functions
#ifndef LIGRA_ALIGN
struct PR_Vertex_Init {
    Array<Node>& data;
    float one_over_n;

    PR_Vertex_Init(Array<Node>& d, float o)
        : data(d), one_over_n(o) {}

    inline bool operator() (const VID& node) {
        data[node].score = 0.0;
        data[node].delta = one_over_n;
        data[node].ngh_sum = 0.0;
        return 1;
    }
};
#else
struct PR_Vertex_Init {
    Array<Node>& data;
    float one_over_n;

    PR_Vertex_Init(Array<Node>& d, float o)
        : data(d), one_over_n(o) {}

    inline bool operator() (const VID& node) {
        data[node].p_next = 0.0;
        data[node].score = one_over_n;
        return 1;
    }
};
#endif

#ifndef LIGRA_ALIGN
struct PR_VertexApply_FirstRound {
    float damping;
    float added_constant;
    float one_over_n;
    float epsilon;
    Array<Node>& data;

    PR_VertexApply_FirstRound(Array<Node>& _d, float _dmp, float _o, float _eps):
        data(_d), damping(_dmp), one_over_n(_o), added_constant((1 - _dmp) * _o), epsilon(_eps) {}

    inline bool operator() (const VID& node) {
        auto& dnode = data[node];
        dnode.delta = damping * dnode.ngh_sum + added_constant;
        dnode.score += dnode.delta;
        dnode.delta -= one_over_n;
        dnode.ngh_sum = 0.0;
        return (std::fabs(dnode.delta) > epsilon * dnode.score);
    }
};
#endif

#ifndef LIGRA_ALIGN
struct PR_VertexApply {
    float damping;
    float epsilon;
    Array<Node>& data;

    PR_VertexApply(Array<Node>& _d, float _dmp, float _eps):
        data(_d), damping(_dmp), epsilon(_eps) {}

    inline bool operator() (const VID& node) {
        auto& dnode = data[node];
        dnode.delta = dnode.ngh_sum * damping;
        dnode.ngh_sum = 0.0;
        if (std::fabs(dnode.delta) > epsilon * dnode.score) {
            dnode.score += dnode.delta;
            return 1;

        } else {
            return 0;
        }
    }
};
#else
struct PR_VertexApply {
    float damping;
    float added_constant;
    float one_over_n;
    Array<Node>& data;

    PR_VertexApply(Array<Node>& _d, float _dmp, float _o):
        data(_d), damping(_dmp), one_over_n(_o), added_constant((1 - _dmp) * _o) {}

    inline bool operator() (const VID& node) {
        auto& dnode = data[node];
        dnode.p_next = damping * dnode.p_next + added_constant;
        return 1;
    }
};
#endif

#ifndef LIGRA_ALIGN
struct PR_TotalDelta {
    Array<Node>& data;
    galois::GAccumulator<float>& total_delta;

    PR_TotalDelta(Array<Node>& d, galois::GAccumulator<float>& t)
        : data(d), total_delta(t) {}

    inline bool operator() (const VID& node) {
        auto& dnode = data[node];
        total_delta += std::fabs(dnode.delta);
        return 1;
    }
};
#else
struct PR_TotalDelta {
    Array<Node>& data;
    galois::GAccumulator<float>& total_delta;

    PR_TotalDelta(Array<Node>& d, galois::GAccumulator<float>& t)
        : data(d), total_delta(t) {}

    inline bool operator() (const VID& node) {
        auto& dnode = data[node];
        total_delta += std::fabs(dnode.score - dnode.p_next);
        return 1;
    }
};
#endif

#ifdef LIGRA_ALIGN
struct PR_Reset {
    Array<Node>& data;

    PR_Reset(Array<Node>& d)
        : data(d) {}

    inline bool operator() (const VID& node) {
        auto& dnode = data[node];
        dnode.score = dnode.p_next;
        dnode.p_next = 0;
        return 1;
    }
};
#endif

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
    // system("iostat > iostat_mid.txt");

    uint64_t n = outGraph.NumberOfNodes();
    float one_over_n = 1 / (float)n;

    Array<Node> data;
    data.allocate(n);

    // Allocate bins
    unsigned nthreads = galois::getActiveThreads();
    uint64_t binSpaceBytes = (uint64_t)binSpace * MB;
    Bins *bins = new Bins(outGraph, nthreads, binSpaceBytes,
                          binCount, binBufSize, binningRatio);

    Worklist<VID>* frontier = new Worklist<VID>(n);
    frontier->activate_all();

    // Initialize values
    vertexMap(outGraph, PR_Vertex_Init(data, one_over_n));

    galois::GAccumulator<float> totalDelta;

    galois::StatTimer time("Time", "PAGERANK_MAIN");
    time.start();

    reportInit();
    startTime();

    long iter = 0;

    while (iter++ < maxIterations) {
        edgeMap(outGraph, frontier, PR_F(outGraph, data, bins), no_output | prop_blocking);
#ifndef LIGRA_ALIGN
        Worklist<VID>* active = (iter == 1) ?
                                vertexFilter(outGraph, PR_VertexApply_FirstRound(data, damping, one_over_n, epsilon)) :
                                vertexFilter(outGraph, PR_VertexApply(data, damping, epsilon));
#else
        Worklist<VID>* active = vertexFilter(outGraph, PR_VertexApply(data, damping, one_over_n));
#endif

        vertexMap(outGraph, PR_TotalDelta(data, totalDelta));
        float L1_norm = totalDelta.reduce();
        printf("L1 norm: %.5f\n", L1_norm);
        if (L1_norm < epsilon2) break;
        totalDelta.reset();

#ifdef LIGRA_ALIGN
        vertexMap(outGraph, PR_Reset(data));
#endif
        delete frontier;
        frontier = active;

        bins->reset();
    }

    delete frontier;

    time.stop();
    double time_ligra = nextTime("Running time");
    reportTimeToFile(time_ligra);
    reportEnd();

    printTop(data);

    delete bins;

    return 0;
}
