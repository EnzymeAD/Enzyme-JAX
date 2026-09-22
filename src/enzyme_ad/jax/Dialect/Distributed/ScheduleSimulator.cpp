#include "ScheduleSimulator.h"

#include "CollectiveCost.h"
#include "MeshCostMetadata.h"
#include "ScheduleGraph.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <queue>

namespace mlir::enzyme::distributed {

namespace {

constexpr double kEpsilon = 1e-9;

// One event's demand normalized into a per-resource rate: how fast it would
// consume each resource it needs, running alone, to finish exactly at
// isolatedDuration - latency on every resource simultaneously. This is what
// max-min fair sharing scales down when an event is not alone.
struct RateNeeded {
  std::vector<std::pair<size_t, double>> rates;
};

std::vector<RateNeeded> computeRateNeeded(const SimGraph &graph) {
  std::vector<RateNeeded> rateNeeded(graph.events.size());
  for (auto [event, rate] : llvm::zip(graph.events, rateNeeded)) {
    double denom = std::max(event.isolatedDuration - event.latency, kEpsilon);
    for (auto [resource, amount] : event.demand)
      rate.rates.emplace_back(resource, amount / denom);
  }
  return rateNeeded;
}

// Longest path to a sink, weighted by isolatedDuration, computed in one
// reverse pass over `graph.events` (SimGraph precondition: predecessors
// only reference lower indices, so a single backward sweep already sees
// every successor's final priority before it needs it). `bestSuccessor[i]`
// records which successor achieves that path, for reconstructing the
// reported critical path; npos means i is a sink.
struct CriticalPathInfo {
  std::vector<double> priority;
  std::vector<size_t> bestSuccessor;
};

CriticalPathInfo computeCriticalPath(
    const SimGraph &graph,
    const std::vector<std::vector<SimPredecessor>> &successors) {
  size_t n = graph.events.size();
  CriticalPathInfo info;
  info.priority.assign(n, 0.0);
  info.bestSuccessor.assign(n, std::numeric_limits<size_t>::max());
  for (size_t i = n; i-- > 0;) {
    double best = 0.0;
    size_t bestSucc = std::numeric_limits<size_t>::max();
    bool haveSucc = false;
    for (const SimPredecessor &edge : successors[i]) {
      if (!haveSucc || info.priority[edge.node] > best) {
        best = info.priority[edge.node];
        bestSucc = edge.node;
        haveSucc = true;
      }
    }
    info.priority[i] = graph.events[i].isolatedDuration + best;
    info.bestSuccessor[i] = bestSucc;
  }
  return info;
}

std::vector<size_t> reconstructCriticalPath(const CriticalPathInfo &info) {
  if (info.priority.empty())
    return {};
  size_t start = 0;
  for (size_t i = 1; i < info.priority.size(); ++i)
    if (info.priority[i] > info.priority[start])
      start = i;
  std::vector<size_t> path;
  for (size_t node = start; node != std::numeric_limits<size_t>::max();
       node = info.bestSuccessor[node])
    path.push_back(node);
  return path;
}

// Strict-priority dispatch: `contenders` must already be sorted
// highest-priority first (the same order admission draws from `ready`, see
// priorityLess in simulate). Each contender in turn takes as much of every
// resource it needs as still remains, capped at scale 1.0 (its full,
// unthrottled rate) rather than split with anyone behind it, and what it
// takes is subtracted from the running capacity before the next contender is
// considered. A resource a higher-priority contender bottlenecks on is
// exactly zeroed out for everyone queued behind it on that resource; a
// resource it doesn't touch is untouched, so a lower-priority contender can
// still run at full rate on an axis nothing ahead of it needs. Unlike max-min
// fair sharing, a contender can be left with scale 0 (fully starved this
// boundary) if everything ahead of it has already exhausted its bottleneck
// resource; `contenders` holds resource-consuming events only (an
// empty-demand event is resolved by the caller before this runs, see the
// "instant finish" handling in simulate).
std::vector<double> priorityDispatch(const std::vector<size_t> &contenders,
                                     const std::vector<RateNeeded> &rateNeeded,
                                     const std::vector<double> &capacity) {
  std::vector<double> remainingCap = capacity;
  std::vector<double> scale(contenders.size(), 0.0);
  for (size_t k = 0; k < contenders.size(); ++k) {
    double s = 1.0;
    for (auto [resource, rate] : rateNeeded[contenders[k]].rates) {
      if (rate <= 0)
        continue;
      s = std::min(s, remainingCap[resource] / rate);
    }
    s = std::max(s, 0.0);
    scale[k] = s;
    for (auto [resource, rate] : rateNeeded[contenders[k]].rates)
      remainingCap[resource] -= s * rate;
  }
  return scale;
}

enum class State { Blocked, Ready, LatencyWait, Transferring, Done };

} // namespace

TimelineResult simulate(const SimGraph &graph, const SimResources &resources,
                        size_t kMaxInFlight) {
  size_t n = graph.events.size();
  TimelineResult result;
  result.start.assign(n, 0.0);
  result.finish.assign(n, 0.0);
  if (n == 0)
    return result;

  // Successor adjacency, built once from every event's (already
  // deduplicated, multiplicity-annotated) predecessor list: successors[p]
  // holds every (s, multiplicity) edge whose predecessor is p, i.e. exactly
  // what waking p's dependents needs to decrement their blocked counters by
  // the same multiplicity they were initialized with.
  std::vector<std::vector<SimPredecessor>> successors(n);
  std::vector<size_t> blocked(n, 0);
  for (size_t s = 0; s < n; ++s)
    for (const SimPredecessor &pred : graph.events[s].predecessors) {
      assert(pred.node < s && "SimGraph precondition: predecessors precede "
                              "their event in build order");
      successors[pred.node].push_back({s, pred.multiplicity});
      blocked[s] += pred.multiplicity;
    }

  std::vector<RateNeeded> rateNeeded = computeRateNeeded(graph);
  CriticalPathInfo criticalPath = computeCriticalPath(graph, successors);

  auto priorityLess = [&](size_t a, size_t b) {
    // std::priority_queue is a max-heap ordered by this predicate; higher
    // remaining critical path goes first, original build order breaks ties
    // so admission order is reproducible across runs of the same module.
    if (criticalPath.priority[a] != criticalPath.priority[b])
      return criticalPath.priority[a] < criticalPath.priority[b];
    return graph.events[a].sequenceIndex > graph.events[b].sequenceIndex;
  };
  std::priority_queue<size_t, std::vector<size_t>, decltype(priorityLess)>
      ready(priorityLess);
  std::vector<size_t> noopReady;

  std::vector<State> state(n, State::Blocked);
  std::vector<double> waitUntil(n, 0.0);
  std::vector<double> remainingWork(n, 0.0);
  size_t doneCount = 0;

  auto isNoop = [&](size_t i) {
    return graph.events[i].isolatedDuration <= kEpsilon;
  };
  auto becomeReady = [&](size_t i) {
    state[i] = State::Ready;
    if (isNoop(i))
      noopReady.push_back(i);
    else
      ready.push(i);
  };
  // Every event still in LatencyWait or Transferring, unordered. Bounded by
  // kMaxInFlight, so scanning it each iteration (to find newly released
  // events, the next latency release, and the fair-share contenders) keeps
  // the simulator's per-boundary cost a true constant rather than O(N).
  std::vector<size_t> active;
  auto removeFromActive = [&](size_t i) {
    auto it = llvm::find(active, i);
    *it = active.back();
    active.pop_back();
  };
  // True while some exclusiveCompute event is active (LatencyWait or
  // Transferring): the compute engine can only ever run one kernel at a
  // time, so admission must refuse a second exclusiveCompute event even if
  // kMaxInFlight has headroom (see the admission loop below).
  bool computeEngineBusy = false;
  auto finishEvent = [&](size_t i, double finishTime) {
    result.finish[i] = finishTime;
    state[i] = State::Done;
    ++doneCount;
    // Only a Compute event that actually went through admission ever set
    // computeEngineBusy; a zero-duration one finishes via the noop fast path
    // without acquiring the gate, so it must not release it here.
    if (graph.events[i].exclusiveCompute && !isNoop(i))
      computeEngineBusy = false;
    for (const SimPredecessor &edge : successors[i]) {
      blocked[edge.node] -= edge.multiplicity;
      if (blocked[edge.node] == 0)
        becomeReady(edge.node);
    }
  };

  for (size_t i = 0; i < n; ++i)
    if (blocked[i] == 0)
      becomeReady(i);

  double time = 0.0;
  while (doneCount < n) {
    // Noop fast path: drain every zero-cost ready event before admission,
    // looping since finishing one can make another noop ready. These never
    // occupy a kMaxInFlight slot or wait for a fair-share resolve.
    while (!noopReady.empty()) {
      size_t i = noopReady.back();
      noopReady.pop_back();
      result.start[i] = time;
      finishEvent(i, time);
    }

    // Admission is a one-time snapshot: at most this many ready events enter
    // `active` this iteration, and any capacity they leave unused is not
    // backfilled until the next admission point. An exclusiveCompute event
    // is skipped (left in `ready` for a later iteration) whenever the
    // compute engine is already busy, independent of this slot count: it is
    // the single-issue compute engine, not kMaxInFlight, that limits it here.
    // Skipped events don't consume a slot, so admission keeps examining
    // `ready` past them for other, non-conflicting events to fill the
    // iteration's slots; they're restored to `ready` once the snapshot is
    // done so a higher-priority event never loses its place because it
    // happened to be tried while blocked.
    size_t slots =
        kMaxInFlight > active.size() ? kMaxInFlight - active.size() : 0;
    std::vector<size_t> deferredCompute;
    for (size_t admitted = 0; admitted < slots && !ready.empty();) {
      size_t i = ready.top();
      ready.pop();
      if (graph.events[i].exclusiveCompute && computeEngineBusy) {
        deferredCompute.push_back(i);
        continue;
      }
      result.start[i] = time;
      state[i] = State::LatencyWait;
      waitUntil[i] = time + graph.events[i].latency;
      active.push_back(i);
      if (graph.events[i].exclusiveCompute)
        computeEngineBusy = true;
      ++admitted;
    }
    for (size_t i : deferredCompute)
      ready.push(i);

    for (size_t i : active)
      if (state[i] == State::LatencyWait && waitUntil[i] <= time + kEpsilon) {
        state[i] = State::Transferring;
        remainingWork[i] =
            graph.events[i].isolatedDuration - graph.events[i].latency;
      }

    // An event with a latency floor but no resource demand (a "latency
    // floor not consuming bandwidth" event, and every FunctionCall barrier
    // event) has nothing left for max-min fair share to resolve once its
    // latency clears: its remaining work is already zero. Progressive
    // filling has no fixed point for such an event (it never appears in any
    // resource's contender list), so it is finished here directly rather
    // than passed into priorityDispatch, which assumes every contender needs
    // at least one resource. `active` is snapshotted before this loop since
    // finishEvent mutates it.
    std::vector<size_t> transferring;
    for (size_t i : active)
      if (state[i] == State::Transferring)
        transferring.push_back(i);
    std::vector<size_t> contenders;
    for (size_t i : transferring) {
      if (remainingWork[i] <= kEpsilon) {
        finishEvent(i, time);
        removeFromActive(i);
      } else {
        contenders.push_back(i);
      }
    }

    // The instant finishes above may have emptied `active` without the
    // simulation being done: either every event is finished (the outer loop
    // condition ends it next iteration) or a finish just woke a successor
    // that is still only `ready`, not yet admitted (admission is a
    // once-per-iteration snapshot, see the top of this loop). Either way
    // there is no resource boundary to wait for this iteration; loop back
    // (without advancing `time`) so the next iteration's admission and noop
    // drain pick it up.
    if (active.empty()) {
      assert((doneCount == n || !ready.empty() || !noopReady.empty()) &&
             "deadlock: no event can make progress; the schedule graph has a "
             "cycle or an unmodeled dependency, which ScheduleGraph.h should "
             "never produce from valid IR");
      continue;
    }

    double nextTime = std::numeric_limits<double>::infinity();
    for (size_t i : active)
      if (state[i] == State::LatencyWait)
        nextTime = std::min(nextTime, waitUntil[i]);

    std::vector<double> scale;
    if (!contenders.empty()) {
      // Highest-priority-first, matching the order admission drew these
      // events from `ready`: priorityDispatch gives resources to a
      // contender only after everyone ranked ahead of it has taken what it
      // needs, so a starved contender (scale 0) simply sits out this
      // boundary rather than forcing a wait.
      llvm::sort(contenders,
                 [&](size_t a, size_t b) { return priorityLess(b, a); });
      scale = priorityDispatch(contenders, rateNeeded, resources.capacity);
      for (size_t k = 0; k < contenders.size(); ++k)
        if (scale[k] > 0)
          nextTime = std::min(nextTime,
                              time + remainingWork[contenders[k]] / scale[k]);
    }

    assert(std::isfinite(nextTime) &&
           "no event can make progress: the schedule graph has a cycle or an "
           "unmodeled dependency, which ScheduleGraph.h should never produce "
           "from valid IR");
    double dt = nextTime - time;
    time = nextTime;
    for (size_t k = 0; k < contenders.size(); ++k) {
      remainingWork[contenders[k]] -= scale[k] * dt;
      if (remainingWork[contenders[k]] <= kEpsilon) {
        finishEvent(contenders[k], time);
        removeFromActive(contenders[k]);
      }
    }
  }

  result.makespan = time;
  result.criticalPath = reconstructCriticalPath(criticalPath);

  // Communication time is the union of every Communication event's
  // [start, finish) interval, computed with a merge-sorted-intervals pass:
  // independent axes or fair-shared time on one axis can make those
  // intervals genuinely overlap.
  //
  // Compute time is a plain sum of Compute events' durations: exclusiveCompute
  // already guarantees at most one is ever active, so no interval union is
  // needed.
  std::vector<std::pair<double, double>> commIntervals;
  for (size_t i = 0; i < n; ++i) {
    const SimEvent &event = graph.events[i];
    if (event.isCommunication)
      commIntervals.emplace_back(result.start[i], result.finish[i]);
    if (event.exclusiveCompute)
      result.totalComputeTime += result.finish[i] - result.start[i];
  }
  llvm::sort(commIntervals);
  double mergedStart = 0, mergedEnd = -1;
  for (auto &[start, finish] : commIntervals) {
    if (start > mergedEnd) {
      if (mergedEnd >= mergedStart)
        result.totalCommunicationTime += mergedEnd - mergedStart;
      mergedStart = start;
      mergedEnd = finish;
    } else {
      mergedEnd = std::max(mergedEnd, finish);
    }
  }
  if (mergedEnd >= mergedStart)
    result.totalCommunicationTime += mergedEnd - mergedStart;

  result.averageComputeUtilization =
      result.makespan > kEpsilon ? result.totalComputeTime / result.makespan
                                 : 0.0;
  result.totalComputeIdleTime = result.makespan - result.totalComputeTime;

  return result;
}

SimGraph buildSimGraph(const ScheduleGraph &graph, const MeshCostParams &mesh,
                       const DeviceCostParams &device, size_t kMaxInFlight) {
  assert(graph.feasible && "buildSimGraph requires a feasible ScheduleGraph; "
                           "callers must check ScheduleGraph::feasible first");
  SimGraph sim;
  sim.numResources = kFirstAxisResource + mesh.numAxes();
  sim.events.reserve(graph.nodes.size());
  for (auto [index, node] : llvm::enumerate(graph.nodes)) {
    SimEvent event;
    event.sequenceIndex = index;
    for (auto [predecessor, multiplicity] :
         llvm::zip(node.predecessors, node.predecessorMultiplicity))
      event.predecessors.push_back({predecessor, multiplicity});

    switch (node.kind) {
    case ScheduleEventKind::Compute:
      // Demand is registered exactly as before, including memBW: only the
      // compute *engine* is single-issue (see exclusiveCompute's comment in
      // the header), not the memory bus, so memBW stays a normal,
      // capacity-limited, fair-shared resource. Once the admission gate
      // below keeps at most one Compute event active at a time, this event
      // is never actually sharing either resource with another Compute
      // event in practice; the demand stays populated so the resource model
      // remains honest if that ever changes (e.g. a future non-exclusive
      // compute-engine model).
      if (node.computeCost.flops > 0)
        event.demand.emplace_back(kComputeResource, node.computeCost.flops);
      if (node.computeCost.bytes > 0)
        event.demand.emplace_back(kMemBandwidthResource,
                                  node.computeCost.bytes);
      event.latency = 0;
      event.isolatedDuration = node.isolatedDuration;
      event.exclusiveCompute = true;
      break;
    case ScheduleEventKind::Communication:
      for (size_t axis = 0; axis < node.step.volume.size(); ++axis)
        if (node.step.volume[axis] > 0)
          event.demand.emplace_back(kFirstAxisResource + axis,
                                    node.step.volume[axis]);
      event.latency = node.step.latency;
      event.isolatedDuration = node.isolatedDuration;
      event.isCommunication = true;
      break;
    case ScheduleEventKind::FunctionCall: {
      // Recursively simulate the callee's own scope to get a real makespan,
      // rather than ScheduleGraph.h's serial-sum proxy, for the barrier
      // event's duration: nothing else runs while a call is in flight (the
      // barrier edges already enforce that), so modeling it as a single
      // latency floor with no resource demand, equal to that makespan,
      // exactly reproduces "the callee runs to completion, on its own,
      // before anything after the call may start".
      TimelineResult calleeResult =
          simulateScheduleGraph(*node.calleeGraph, mesh, device, kMaxInFlight);
      event.latency = calleeResult.makespan;
      event.isolatedDuration = calleeResult.makespan;
      break;
    }
    }
    sim.events.push_back(std::move(event));
  }
  return sim;
}

SimResources simResourcesFromCostParams(const MeshCostParams &mesh,
                                        const DeviceCostParams &device) {
  SimResources resources;
  resources.capacity.assign(kFirstAxisResource + mesh.numAxes(), 0.0);
  resources.capacity[kComputeResource] = device.flops;
  resources.capacity[kMemBandwidthResource] = device.memBandwidth;
  for (size_t axis = 0; axis < mesh.numAxes(); ++axis)
    resources.capacity[kFirstAxisResource + axis] = mesh.bandwidth[axis];
  return resources;
}

TimelineResult simulateScheduleGraph(const ScheduleGraph &graph,
                                     const MeshCostParams &mesh,
                                     const DeviceCostParams &device,
                                     size_t kMaxInFlight) {
  SimGraph sim = buildSimGraph(graph, mesh, device, kMaxInFlight);
  SimResources resources = simResourcesFromCostParams(mesh, device);
  return simulate(sim, resources, kMaxInFlight);
}

} // namespace mlir::enzyme::distributed
