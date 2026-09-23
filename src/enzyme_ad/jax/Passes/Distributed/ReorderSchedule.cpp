#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/MeshCostMetadata.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/ScheduleGraph.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/ScheduleSimulator.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"

#include "mlir/IR/RegionKindInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_REORDERSCHEDULEPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

#ifndef NDEBUG
// Every operand defined in `block` must be defined by an op that already
// precedes its use, i.e. reordering never turned a valid def-before-use
// block into an invalid one. Cheap enough (one pass over the block) to run
// unconditionally in a debug build as a direct check of the claim that
// admission-order sorting is topologically valid by construction.
//
// Only meaningful for an SSACFG block (a function body): ScheduleGraph.h
// walks module-scope ops (axis-algebra metadata, and, once
// MaterializeDistributedCollectives has run, atomic collectives/kernels
// themselves) as part of the same top-level scope, but ModuleOp's body is a
// graph region, where forward SSA references are legal and op order carries
// no dominance guarantee to begin with.
void assertBlockOrderValid(Block &block) {
  if (mayBeGraphRegion(*block.getParent()))
    return;
  llvm::DenseMap<Operation *, size_t> position;
  size_t index = 0;
  for (Operation &op : block)
    position[&op] = index++;
  for (Operation &op : block)
    for (Value operand : op.getOperands())
      if (Operation *def = operand.getDefiningOp())
        if (def->getBlock() == &block)
          assert(position[def] < position[&op] &&
                 "distributed-reorder-schedule moved a use before its def");
}
#endif

// One anchor op (an op with a ScheduleEvent of its own, living in `block`)
// together with the run of non-anchor ops immediately following it in the
// original block, and the admission time to sort the anchor by. A run of
// non-anchor ops before the first anchor (axis-algebra/mesh metadata, always
// sitting at true module scope ahead of any real computation) has no anchor
// to attach to and is kept as a fixed prefix instead, never reordered.
struct Chunk {
  Operation *anchor;
  SmallVector<Operation *> trailer;
  double start;
  size_t originalIndex;
};

// Physically reorders `block`'s non-terminator operations to match the
// admission order `result` predicts for whichever of `nodes` live in this
// block, leaving every other op's position relative to its immediately
// preceding anchor unchanged (see the Chunk comment above). No-op if `block`
// has no anchor ops at all.
void reorderBlockByAdmission(Block &block, ArrayRef<ScheduleEvent> nodes,
                             const TimelineResult &result) {
  llvm::DenseMap<Operation *, size_t> anchorNode;
  for (auto [index, node] : llvm::enumerate(nodes))
    if (node.op && node.op->getBlock() == &block)
      anchorNode.try_emplace(node.op, index);
  if (anchorNode.empty())
    return;

  // A block only has a terminator if its region requires one (e.g. a
  // func.func body); ScheduleGraph.h's "module scope and a function body are
  // one flattened scope" treatment (see reorderScopeRecursively below) means
  // `block` here can be a bare ModuleOp body, which has none.
  Operation *terminator =
      block.mightHaveTerminator() ? block.getTerminator() : nullptr;
  // Moves `op` to just before `terminator`, or to the end of `block` if it
  // has none.
  auto moveToEnd = [&](Operation *op) {
    if (terminator)
      op->moveBefore(terminator);
    else
      op->moveBefore(&block, block.end());
  };

  SmallVector<Operation *> prefix;
  SmallVector<Chunk> chunks;
  for (Operation &opRef : block) {
    Operation *op = &opRef;
    if (op == terminator)
      continue;
    auto it = anchorNode.find(op);
    if (it != anchorNode.end())
      chunks.push_back({op, {}, result.start[it->second], it->second});
    else if (chunks.empty())
      prefix.push_back(op);
    else
      chunks.back().trailer.push_back(op);
  }

  llvm::stable_sort(chunks, [](const Chunk &a, const Chunk &b) {
    if (a.start != b.start)
      return a.start < b.start;
    // A tie can only occur between chunks with no predecessor relationship
    // between their anchors (the simulator never admits a successor before
    // its predecessor finishes, and finish >= start), so original op order
    // is always a valid, deterministic tiebreak.
    return a.originalIndex < b.originalIndex;
  });

  for (Operation *op : prefix)
    moveToEnd(op);
  for (Chunk &chunk : chunks) {
    moveToEnd(chunk.anchor);
    for (Operation *op : chunk.trailer)
      moveToEnd(op);
  }

#ifndef NDEBUG
  assertBlockOrderValid(block);
#endif
}

// Simulates `graph`'s own scope and reorders every physical block its
// nodes' ops live in, then recurses into each FunctionCall node's callee
// scope with a fresh simulation of its own.
//
// - Module scope and a function body are two separate blocks that
//   ScheduleGraph.h treats as one flattened scope; a nested callee scope
//   spans only its own single-block body.
// - Recursion into a callee is needed because the caller-level simulation
//   only records the callee's overall makespan: buildSimGraph's
//   FunctionCall case (ScheduleSimulator.cpp) reads `.makespan` off the
//   nested TimelineResult and discards the rest.
void reorderScopeRecursively(const ScheduleGraph &graph,
                             const MeshCostParams &mesh,
                             const DeviceCostParams &device) {
  TimelineResult result = simulateScheduleGraph(graph, mesh, device);

  llvm::SmallSetVector<Block *, 4> blocks;
  for (const ScheduleEvent &node : graph.nodes)
    if (node.op)
      blocks.insert(node.op->getBlock());
  for (Block *block : blocks)
    reorderBlockByAdmission(*block, graph.nodes, result);

  for (const ScheduleEvent &node : graph.nodes)
    if (node.kind == ScheduleEventKind::FunctionCall)
      reorderScopeRecursively(*node.calleeGraph, mesh, device);
}

struct ReorderSchedulePass
    : public impl::ReorderSchedulePassBase<ReorderSchedulePass> {
  using ReorderSchedulePassBase::ReorderSchedulePassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    const ScheduleGraph &graph =
        getAnalysis<ScheduleGraphAnalysis>().getGraph();
    if (!graph.feasible) {
      // This pass is meant to run winner-only, on a single search candidate
      // that has already been validated (planned successfully) during
      // scoring, so an infeasible graph here means an unexpected
      // precondition violation, not a normal outcome to skip over quietly.
      module.emitError()
          << "distributed-reorder-schedule: schedule graph infeasible ("
          << graph.failureReason << ")";
      signalPassFailure();
      return;
    }

    // buildScheduleGraph only succeeds after finding a unique physical mesh,
    // so re-resolving it here cannot fail.
    PhysicalMeshOp physicalMesh = *findUniquePhysicalMesh(module);
    MeshCostParams mesh = meshCostParamsFromPhysicalMesh(physicalMesh);
    DeviceCostParams device = deviceCostParamsFromPhysicalMesh(physicalMesh);

    reorderScopeRecursively(graph, mesh, device);
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
