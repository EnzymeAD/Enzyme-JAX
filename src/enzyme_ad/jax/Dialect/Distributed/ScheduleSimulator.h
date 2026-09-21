#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_SCHEDULE_SIMULATOR_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_SCHEDULE_SIMULATOR_H

#include <cstddef>
#include <utility>
#include <vector>

namespace mlir::enzyme::distributed {

// Discrete-event, max-min fair-share timeline simulator over an event DAG.
// ScheduleGraph.h builds the DAG from a lowered candidate module; this file
// turns it into a predicted wall-clock makespan, accounting for events on
// independent resources overlapping each other and for events sharing one
// bottlenecked resource splitting its bandwidth between them.
//
// The engine below (SimEvent, SimGraph, SimResources, simulate) is plain
// C++ with no MLIR dependency: an event is nothing more than a sparse
// resource-demand vector, a latency floor and a set of predecessor edges, so
// the algorithm can be read, tested and reasoned about on its own. The MLIR-
// facing boundary (translating a ScheduleGraph, with its MeshCostParams/
// DeviceCostParams-derived costs, into this representation) is declared at
// the bottom of this file behind a forward declaration and implemented in
// ScheduleSimulator.cpp, which is the only part of this feature that
// actually includes ScheduleGraph.h.
//
// Resource model
// --------------
// Every event contends for some subset of a fixed resource list: one
// "compute" resource and one "memBW" resource shared by every event on the
// representative device (see ScheduleGraph.h/ComputeCost.h for why one
// device's timeline stands in for the whole SPMD program), plus one
// bandwidth resource per physical mesh axis. Resources are addressed by a
// small integer index; kComputeResource and kMemBandwidthResource name the
// first two, axis a is kFirstAxisResource + a.
//
// kMemBandwidthResource is a genuinely divisible physical resource, like a
// network link, and stays an ordinary capacity-limited resource in the
// max-min fair-share pool ("Algorithm" below): Compute events register
// demand against it exactly as they do for any other resource.
// kComputeResource is different: this device model has no multithreaded or
// event-based kernel dispatch, so the compute engine runs one kernel at a
// time and literally cannot be double-booked, even during a memory-bound
// kernel's "idle compute" stretch (there is no second stream to dispatch a
// second kernel onto). Fair-sharing kComputeResource the way the other
// resources are shared would let two concurrently-active Compute events
// each claim partial, simultaneous use of the device and finish sooner than
// either could alone, which is impossible on real hardware. So Compute
// events are additionally gated at admission: at most one may be active (in
// LATENCY_WAIT/TRANSFERRING) at a time, regardless of kMaxInFlight headroom
// (SimEvent::exclusiveCompute, enforced in simulate()'s admission loop).
// Once that gate holds, a Compute event is never actually contending against
// another Compute event on either kComputeResource or kMemBandwidthResource
// in practice (there is only ever one active), so this is mostly about
// keeping the resource model conceptually honest rather than a change in
// near-term memBW behavior.
//
// Algorithm
// ---------
// At every event boundary (an event's latency floor releasing it into
// resource contention, or an event finishing), the simulator recomputes a
// max-min fair split of each resource's capacity among everything currently
// contending for it (maxMinFairShare), so an event always runs at the rate
// its tightest resource can currently spare and a finishing event's freed
// capacity is picked up by the next boundary. Concurrency is bounded by
// kMaxInFlight: at most that many events are ever admitted out of latency
// wait at once, a deliberate model of limited issue width (not just a
// complexity knob) that also bounds each fair-share resolve to a constant
// size (progressive filling, the noop fast path and snapshot admission are
// each implemented in ScheduleSimulator.cpp).
enum : size_t {
  kComputeResource = 0,
  kMemBandwidthResource = 1,
  kFirstAxisResource = 2,
};

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

// One simulated event. A plain value type: nothing here refers back to the
// op, collective step or callee graph it was built from (see buildSimGraph
// for that mapping).
struct SimEvent {
  // Sparse per-resource demand: (resource index, amount) for every resource
  // this event actually consumes. A resource absent here is never contended
  // for by this event. Demand is in the same units as SimResources::capacity
  // (bytes for memBW/axis resources, FLOPs for compute).
  std::vector<std::pair<size_t, double>> demand;
  // Fixed time this event waits before it starts consuming any resource
  // (a collective's launch-plus-round latency; zero for compute).
  double latency = 0;
  // Time this event would take running alone at full resource allocation:
  // latency plus however long its bottleneck resource needs. Zero here
  // (implying both zero latency and zero demand) is the noop fast-path
  // condition: such an event is drained immediately when ready, without
  // occupying a kMaxInFlight slot or entering a fair-share resolve.
  double isolatedDuration = 0;
  // True for Compute-kind events with nonzero isolatedDuration: at most one
  // such event may be admitted into `active` at a time (see simulate()'s
  // admission loop), modeling that a device dispatches kernels one at a
  // time rather than opportunistically overlapping them. Irrelevant for
  // Communication/FunctionCall events, and never consulted for a Compute
  // event with isolatedDuration == 0 (it takes the noop fast path before
  // admission is ever considered).
  bool exclusiveCompute = false;
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
  // The simulated wall-clock time every event has finished by. The score:
  // lower is better.
  double makespan = 0;
  // Per-event start (admission out of latency wait onto the timeline) and
  // finish time, parallel to the SimGraph's events.
  std::vector<double> start;
  std::vector<double> finish;
  // The DAG's own longest weighted path (by isolatedDuration, ignoring
  // resource contention), source to sink, ending at whichever event this
  // result's makespan search identified as critical. This is a static,
  // contention-free lower bound used for ready-queue prioritization, not a
  // trace of the live simulation, so it can differ from the path that
  // actually determined `makespan` when contention adds delay.
  std::vector<size_t> criticalPath;
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
struct DeviceCostParams;

// Translates one ScheduleGraph scope into a SimGraph plus its resource
// capacities, using `mesh`/`device` for both axis bandwidth/latency and
// device compute/memory-bandwidth capacities (the same parameters
// buildScheduleGraph already resolved its node costs from, so the two never
// disagree). `graph` must be feasible (ScheduleGraph::feasible); an
// infeasible graph has no nodes to translate and callers already special-
// case that before reaching here. A FunctionCall event's callee graph is
// recursively simulated to obtain a real makespan for the call's own
// duration (rather than ScheduleGraph.h's own sum-of-isolated-durations
// proxy), so translating a graph with nested calls also runs nested
// simulations.
SimGraph buildSimGraph(const ScheduleGraph &graph, const MeshCostParams &mesh,
                       const DeviceCostParams &device,
                       size_t kMaxInFlight = kDefaultMaxInFlight);

// Resource capacities for one mesh/device pair, indexed the same way as the
// demand this same mesh/device would produce in buildSimGraph.
SimResources simResourcesFromCostParams(const MeshCostParams &mesh,
                                        const DeviceCostParams &device);

// Convenience wrapper: buildSimGraph followed by simulate.
TimelineResult simulateScheduleGraph(const ScheduleGraph &graph,
                                     const MeshCostParams &mesh,
                                     const DeviceCostParams &device,
                                     size_t kMaxInFlight = kDefaultMaxInFlight);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_SCHEDULE_SIMULATOR_H
