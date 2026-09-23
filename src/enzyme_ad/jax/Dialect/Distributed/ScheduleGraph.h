#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_SCHEDULE_GRAPH_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_SCHEDULE_GRAPH_H

#include "CollectiveCost.h"
#include "CollectiveDecomposer.h"
#include "ComputeCost.h"
#include "Dialect.h"

#include "mlir/IR/BuiltinOps.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace mlir::enzyme::distributed {

// The event DAG of one fully-lowered search candidate: one node per op inside
// a distributed.DistributedKernel body, one node per primitive step of every
// atomic distributed.Collective's decomposed chain (CollectiveDecomposer.h),
// and one opaque barrier node per call to another function (see
// ScheduleEventKind::FunctionCall below). A future timeline-simulation
// scheduler will run over this graph; here it is only built and exposed for
// inspection, with no scheduling or IR rewriting attached.

enum class ScheduleEventKind { Compute, Communication, FunctionCall };

struct ScheduleGraph;

// One event. Fields not meaningful for the event's kind are left at their
// default.
struct ScheduleEvent {
  ScheduleEventKind kind;

  // Compute: the costed op itself. Communication: the DistributedCollectiveOp
  // the step's chain belongs to (every step of one chain shares this pointer;
  // chainIndex tells them apart). FunctionCall: the call op itself (a
  // CallOpInterface, e.g. distributed.DistributedCall or func.call).
  Operation *op = nullptr;

  // Communication only: this step's position in its collective's chain, and
  // the step's own footprint.
  size_t chainIndex = 0;
  PrimitiveStep step;

  // Compute only: the op's roofline footprint.
  OpCost computeCost;

  // FunctionCall only: the callee's own event graph, built independently as
  // its own fresh scheduling scope (see the "hard barrier" note below).
  // Always non-null and feasible for a FunctionCall event: an unresolved
  // callee, a call cycle, or an infeasible callee graph instead fails the
  // whole enclosing buildScheduleGraph call, matching how an unplannable
  // collective already fails the whole graph rather than producing a
  // partial one.
  //
  // A call whose resolved callee carries an act.cost attribute (ACT's
  // convention for a bodyless accelerator-launch declaration) never becomes
  // a FunctionCall event at all: ScheduleGraph.cpp costs it as an ordinary
  // atomic Compute event instead, since there is no callee body to build a
  // nested graph from.
  std::unique_ptr<ScheduleGraph> calleeGraph;

  // max(flops/F, bytes/M) for a Compute event, step.isolatedDuration() for a
  // Communication event: the time the event would take running alone, at the
  // device/mesh parameters buildScheduleGraph resolved. Kept on the event
  // itself, rather than recomputed later, since a future timeline simulation
  // needs it as-is per event and it is cheap to compute once, here, from the
  // same OpCost/PrimitiveStep this node already carries.
  //
  // For a FunctionCall event this is only a placeholder: the sum of the
  // callee graph's own node durations, i.e. what the callee would take
  // running fully serially. This ignores any overlap the callee's own
  // internal events could exploit, so it is an overestimate whenever the
  // callee has such overlap. A real duration needs a timeline simulator that
  // actually schedules the callee body and reports its makespan.
  double isolatedDuration = 0;

  // Direct predecessors, within this event's own scheduling scope (see
  // ScheduleGraph): SSA operand producers; for a step after the first in its
  // chain, that chain's previous step (chain steps are strictly sequential,
  // per CollectiveDecomposer.h's D6, so this is not itself a data dependency
  // but has to be respected the same way); and, for any event built after a
  // FunctionCall event in the same scope, that call event, whether or not
  // there is a real data dependency on it. A call is a hard scheduling
  // barrier: nothing live before it may still be running when it starts, and
  // nothing after it may start before it finishes. Every event on either
  // side of a call is wired to it directly rather than relying on data
  // dependencies alone to enforce that ordering.
  //
  // Deduplicated: one entry per distinct predecessor node, which is what a
  // human-facing report (describeEvent, the print-schedule-graph debug pass)
  // wants. `predecessorMultiplicity` (parallel, same length) carries the
  // operand-slot count collapsed into each entry, for a consumer (the
  // timeline simulator) whose readiness bookkeeping has to match how many
  // times a predecessor's value is actually used rather than how many
  // distinct predecessors there are: an op referencing the same producer
  // through two operand slots (e.g. `add %x, %x`) has multiplicity 2 at that
  // predecessor's entry, since two OpOperand occurrences must each be
  // accounted for once. Always 1 for a chain-internal or call-barrier edge,
  // since those are never multi-use.
  std::vector<size_t> predecessors;
  std::vector<size_t> predecessorMultiplicity;
};

// One function's event DAG (the entry function, dissolved together with the
// module-level ops around it, see buildScheduleGraph, or one callee reached
// through a call), or why it could not be built.
//
// A ScheduleGraph is exactly one scheduling scope: `distributed.kernel`
// boundaries within it are dissolved (kernel body ops and the ops around the
// kernel read as one flat scope, aliased through shared SSA values) for
// maximum schedule mobility, but a call to another function is a hard
// barrier that stays a scope boundary. The callee gets its own, separate
// ScheduleGraph (ScheduleEvent::calleeGraph) rather than being flattened into
// this one. A caller wanting finer-grained overlap across a call should
// inline the callee before running this analysis; this builder does not do
// that for them.
struct ScheduleGraph {
  std::vector<ScheduleEvent> nodes;

  // False if any atomic distributed.Collective could not be planned, any
  // call's callee could not be resolved to a single-block function-like op,
  // a call cycle was found, or a callee's own graph was itself infeasible;
  // `nodes` is then empty rather than a partial graph, matching how
  // StrategyScorer::scoreLowered already treats an unplannable collective as
  // infeasible rather than scoring around it.
  bool feasible = true;
  std::string failureReason;
};

// Builds the event DAG of `module`'s entry function (the function named
// "main", func.func or distributed.DistributedFunction), which must have run
// the lowering pipeline's atomize-collectives step and contain a unique
// physical mesh: its optional performance attributes give the mesh/device
// cost parameters (see MeshCostMetadata.h), the same ones
// StrategyScorer::scoreLowered already reads, so this graph's costs cannot
// disagree with the existing isolated score.
//
// Module-level ops outside any function (axis-algebra metadata, and, in
// practice, the atomic collectives and kernels themselves once
// MaterializeDistributedCollectives has run) are walked as part of this same
// top-level scope, interleaved with "main"'s own body in module order; any
// other function definition sitting at module scope is not walked directly
// here and only enters the graph (as its own nested ScheduleGraph) if main,
// or something main calls, transitively calls it.
ScheduleGraph buildScheduleGraph(ModuleOp module,
                                 const PlanOptions &options = {});

const char *toString(ScheduleEventKind kind);

// Multi-line text report of one event: kind, cost/demand summary, and the
// originating op (Compute) or collective and step index (Communication).
std::string describeEvent(const ScheduleEvent &event);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_SCHEDULE_GRAPH_H
