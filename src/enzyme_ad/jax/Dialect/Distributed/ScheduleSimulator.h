#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_SCHEDULE_SIMULATOR_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_SCHEDULE_SIMULATOR_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace mlir::enzyme::distributed {

// Discrete-event, strict-priority-dispatch timeline simulator over an event
// DAG.
// ScheduleGraph.h builds the DAG from a lowered candidate module; this file
// turns it into a predicted wall-clock makespan, accounting for events on
// independent resources overlapping each other and for events sharing one
// bottlenecked resource splitting its bandwidth between them.
//
// The engine below (SimEvent, SimGraph, SimResources, simulate) is plain
// C++ with no MLIR dependency: an event is nothing more than a sparse
// resource-demand vector, a latency floor and a set of predecessor edges, so
// the algorithm can be read, tested and reasoned about on its own. The MLIR-
// facing boundary (translating a ScheduleGraph, with its MeshCostParams-
// derived costs, into this representation) is declared at
// the bottom of this file behind a forward declaration and implemented in
// ScheduleSimulator.cpp, which is the only part of this feature that
// actually includes ScheduleGraph.h.
//
// Resource model
// --------------
// Communication events contend for one bandwidth resource per physical mesh
// axis; resource a is axis a. One device's timeline stands in for the whole
// SPMD program (see ScheduleGraph.h).
//
// A Compute event demands no resource: it is a latency floor of its
// duration (the cost backend's latency, see CostBackend.h). The device has
// no multithreaded or event-based kernel dispatch, so its compute engine
// runs one kernel at a time. Compute events are therefore gated at
// admission: at most one may be active (in LATENCY_WAIT/TRANSFERRING) at a
// time, regardless of kMaxInFlight headroom (SimEvent::exclusiveCompute,
// enforced in simulate()'s admission loop).
//
// Algorithm
// ---------
// At every event boundary (an event's latency floor releasing it into
// resource contention, or an event finishing), the simulator redispatches
// everything currently contending for a resource in strict priority order
// (priorityDispatch): the highest-priority contender takes as much of each
// resource it needs as remains, up to its own full unthrottled rate, before
// the next contender is even considered, so a lower-priority contender can
// be left running at rate 0 until something ahead of it frees the resource.
// A contender always eventually becomes highest-priority among what's left
// once everything ahead of it finishes, so this still guarantees forward
// progress; it just never blends two contenders' rates the way max-min fair
// sharing would. Concurrency is bounded by kMaxInFlight: at most that many
// events are ever admitted out of latency wait at once, a deliberate model
// of limited issue width (not just a complexity knob) that also bounds each
// dispatch resolve to a constant size (the noop fast path and snapshot
// admission are each implemented in ScheduleSimulator.cpp).

// Bound on how many events the simulator admits out of latency wait at
// once. This is an unvalidated modeling constant, not a measured issue
// width: a small default to calibrate later once real search runs show
// whether it binds (candidates artificially serialized that should overlap)
// or is irrelevant (DAG width rarely approaches it).
constexpr size_t kDefaultMaxInFlight = 16;

// One event's dependence on a predecessor, carrying the number of operand
// slots that edge collapses (ScheduleGraph.h's predecessorMultiplicity):
// the simulator's readiness counter must be decremented by `multiplicity`,
// not by 1, when `node` finishes, to match how it was initialized (once per
// real operand occurrence, not once per distinct producer).
struct SimPredecessor {
  size_t node;
  size_t multiplicity = 1;
};

struct TimelineResult;

// One simulated event. A plain value type: nothing here refers back to the
// op, collective step or callee graph it was built from (see buildSimGraph
// for that mapping).
struct SimEvent {
  // Sparse per-resource demand: (resource index, amount) for every resource
  // this event actually consumes. A resource absent here is never contended
  // for by this event. Demand is in the same units as SimResources::capacity
  // (bytes).
  std::vector<std::pair<size_t, double>> demand;
  // Nanoseconds this event waits before it starts consuming any resource
  // (a collective's launch-plus-round latency; a Compute event's whole
  // duration).
  int64_t latency = 0;
  // Nanoseconds this event would take running alone at full resource
  // allocation: latency plus however long its bottleneck resource needs.
  // Zero here (implying both zero latency and zero demand) is the noop
  // fast-path condition: such an event is drained immediately when ready,
  // without occupying a kMaxInFlight slot or entering a fair-share resolve.
  int64_t isolatedDuration = 0;
  // True for Compute-kind events with nonzero isolatedDuration: at most one
  // such event may be admitted into `active` at a time (see simulate()'s
  // admission loop), modeling that a device dispatches kernels one at a
  // time rather than opportunistically overlapping them. Irrelevant for
  // Communication/FunctionCall events, and never consulted for a Compute
  // event with isolatedDuration == 0 (it takes the noop fast path before
  // admission is ever considered).
  bool exclusiveCompute = false;
  // True for Communication-kind events, which select the intervals that
  // feed TimelineResult::totalCommunicationTime's interval-union computation
  // (see simulate()): unlike Compute events, these can genuinely overlap
  // (independent axes, or fair-shared time on one axis).
  bool isCommunication = false;
  // FunctionCall only (null otherwise): the callee scope's own, already
  // simulated timeline. The event itself is scheduled as a pure latency
  // floor of the callee's makespan; simulate() only reads this to fold the
  // callee's time breakdown into its own summary metrics and to expose it
  // through TimelineResult::calleeTimelines.
  std::shared_ptr<const TimelineResult> callee;
  std::vector<SimPredecessor> predecessors;
  // This event's position in the order the graph was built (ScheduleGraph's
  // node index, or a nested graph's own node index): used only as a
  // deterministic secondary key when the ready queue's priority ties.
  size_t sequenceIndex = 0;
};

struct SimGraph {
  std::vector<SimEvent> events;
  size_t numResources = 0;
};

// Per-resource capacity, indexed the same way as SimEvent::demand.
struct SimResources {
  std::vector<double> capacity;
};

struct TimelineResult {
  // The simulated wall-clock time, in nanoseconds, every event has finished
  // by. The score: lower is better.
  int64_t makespan = 0;
  // Per-event start (admission out of latency wait onto the timeline) and
  // finish time, in nanoseconds, parallel to the SimGraph's events.
  std::vector<int64_t> start;
  std::vector<int64_t> finish;
  // The DAG's own longest weighted path (by isolatedDuration, ignoring
  // resource contention), source to sink, ending at whichever event this
  // result's makespan search identified as critical. This is a static,
  // contention-free lower bound used for ready-queue prioritization, not a
  // trace of the live simulation, so it can differ from the path that
  // actually determined `makespan` when contention adds delay.
  std::vector<size_t> criticalPath;

  // Per-event callee timelines, parallel to the SimGraph's events: the
  // FunctionCall event's SimEvent::callee, null for every other event. Lets
  // a consumer walk a whole-program breakdown that needs per-event detail
  // the aggregates below do not keep (e.g. SearchStrategies.cpp's
  // schedule.json node list) through each call's own timeline.
  std::vector<std::shared_ptr<const TimelineResult>> calleeTimelines;

  // Summary metrics derived from the start/finish arrays above. Each covers
  // the whole program under this scope: a FunctionCall event adds its
  // callee's own totals, once per call site, so a callee called twice
  // counts twice. A call is a hard barrier (ScheduleGraph.h), so nothing
  // else in the caller's scope runs while it is in flight, and a callee's
  // intervals never overlap the caller's own; adding the callee's totals
  // therefore keeps every interval-union metric below an exact union.

  // Nanoseconds during which at least one Communication event was active,
  // i.e. the union of every Communication event's [start, finish) interval
  // (not the sum of their durations, which would double-count overlap
  // between independent axes or fair-shared time on one axis).
  int64_t totalCommunicationTime = 0;
  // Nanoseconds the compute engine was occupied: the sum of every Compute
  // event's own duration. exclusiveCompute guarantees at most one Compute
  // event is ever active, so no interval union is needed here.
  int64_t totalComputeTime = 0;
  // makespan minus totalComputeTime, in nanoseconds.
  int64_t totalComputeIdleTime = 0;
};

// Runs the simulation. `graph.events[i].predecessors` must only reference
// indices less than `i` (i.e. `graph.events` must already be in a valid
// topological order; ScheduleGraph.h's build order guarantees this and
// buildSimGraph below preserves it) and every resource index appearing in a
// demand or predecessor list must be in range for `resources.capacity` /
// `graph.events.size()` respectively. A cyclic or otherwise malformed graph
// is a precondition violation (asserted), not a reported failure: by the
// time a schedule graph reaches this simulator it has already been built by
// ScheduleGraph.h from valid IR, which cannot produce a cycle.
TimelineResult simulate(const SimGraph &graph, const SimResources &resources,
                        size_t kMaxInFlight = kDefaultMaxInFlight);

// The MLIR-facing translation boundary. Forward declared here so this
// header stays free of MLIR includes; ScheduleSimulator.cpp includes
// ScheduleGraph.h and MeshCostMetadata.h to implement these.
struct ScheduleGraph;
struct MeshCostParams;

// Translates one ScheduleGraph scope into a SimGraph, using `mesh` for axis
// bandwidths (the same parameters buildScheduleGraph already resolved its
// collective costs from, so the two never disagree). `graph` must be feasible
// (ScheduleGraph::feasible); an infeasible graph has no nodes to translate and
// callers already special-case that before reaching here. A FunctionCall
// event's callee graph is recursively simulated to obtain a real makespan for
// the call's own duration (rather than ScheduleGraph.h's own
// sum-of-isolated-durations proxy), so translating a graph with nested calls
// also runs nested simulations; each nested result is kept on the event
// (SimEvent::callee).
SimGraph buildSimGraph(const ScheduleGraph &graph, const MeshCostParams &mesh,
                       size_t kMaxInFlight = kDefaultMaxInFlight);

// Resource capacities for one mesh, indexed the same way as the demand this
// same mesh would produce in buildSimGraph.
SimResources simResourcesFromCostParams(const MeshCostParams &mesh);

// Convenience wrapper: buildSimGraph followed by simulate.
TimelineResult simulateScheduleGraph(const ScheduleGraph &graph,
                                     const MeshCostParams &mesh,
                                     size_t kMaxInFlight = kDefaultMaxInFlight);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_SCHEDULE_SIMULATOR_H
