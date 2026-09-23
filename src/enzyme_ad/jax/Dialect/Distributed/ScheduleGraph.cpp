#include "ScheduleGraph.h"

#include "MeshCostMetadata.h"
#include "Utilities.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::enzyme::distributed {

const char *toString(ScheduleEventKind kind) {
  switch (kind) {
  case ScheduleEventKind::Compute:
    return "compute";
  case ScheduleEventKind::Communication:
    return "communication";
  case ScheduleEventKind::FunctionCall:
    return "call";
  }
  llvm_unreachable("unknown ScheduleEventKind");
}

namespace {

// Aliases `value` to `node` if it does not already have one. Kernel and
// collective-chain boundaries are bridged this way (block arg <-> kernel
// operand's own node, kernel result <-> yielded value's node, await result
// <-> chain's last step) instead of by rewriting the IR: an alias just lets a
// later lookup on either value find the same producing event.
void alias(llvm::DenseMap<Value, size_t> &valueNode, Value value, size_t node) {
  valueNode.try_emplace(value, node);
}

// The node producing `value`, if any op in the graph produced it. A value
// with no entry was produced by an op this graph does not model (mesh/axis
// metadata, a plain top-level op outside any kernel, a block argument of
// main) and is simply not a dependency edge: the consuming event is ready
// with respect to it from the start.
std::optional<size_t> lookup(const llvm::DenseMap<Value, size_t> &valueNode,
                             Value value) {
  auto it = valueNode.find(value);
  if (it == valueNode.end())
    return std::nullopt;
  return it->second;
}

// Records `producer` as one of `event`'s predecessors, deduplicated against
// any predecessor already recorded (predecessorMultiplicity tracks how many
// operand slots collapsed into each entry; see ScheduleGraph.h).
void addPredecessor(ScheduleEvent &event, size_t producer) {
  auto it = llvm::find(event.predecessors, producer);
  if (it == event.predecessors.end()) {
    event.predecessors.push_back(producer);
    event.predecessorMultiplicity.push_back(1);
  } else {
    ++event.predecessorMultiplicity[it - event.predecessors.begin()];
  }
}

// Walks ViewLikeOpInterface's getViewSource() (memref.subview/cast/
// reinterpret_cast) back to the root value `value` is ultimately a view of:
// a memref.alloc result, a block argument, or any other non-view producer.
// This is the coarsest granularity that can tell two ops touching the same
// underlying buffer apart from two touching unrelated ones without full
// alias analysis: distinct views of one root are treated as conflicting
// even when their offsets provably don't overlap (see
// addMemrefSideEffectEdges).
Value memrefRoot(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    auto view = dyn_cast<ViewLikeOpInterface>(def);
    if (!view)
      break;
    value = view.getViewSource();
  }
  return value;
}

// Chains `op` after whichever previously built node last touched the same
// memref root, for every memref-typed operand `op` has, and records `op`'s
// own node as that root's new last-toucher.
//
// A memref op reads or writes through its operands as a side effect; there
// is no SSA result threading that effect the way there is for a tensor
// value, so two ops that only share a memref operand (e.g. one op's write
// into a subview and another op's later read of that same subview) have no
// SSA producer/consumer edge between them at all: both only depend on
// whatever produced the view, never on each other. Threading one linear
// chain per root through this graph's existing block-walk order recovers
// exactly the serialization real memory effects impose, which is what lets,
// for instance, ACT's void-returning stage/launch/fetch calls
// (CollectivePrimitiveBuilders.h's ACT convention) get ordered relative to
// each other despite passing no SSA value between them.
void addMemrefSideEffectEdges(Operation &op, size_t index, ScheduleEvent &event,
                              llvm::DenseMap<Value, size_t> &lastToucher) {
  // Reads every operand's previous toucher before writing any of them, so an
  // op with several operands aliasing the same root (e.g. an ACT launch
  // reading two staged views and writing a third, all views of one
  // allocation) never wires a spurious edge to itself.
  for (Value operand : op.getOperands())
    if (isa<MemRefType>(operand.getType()))
      if (std::optional<size_t> producer =
              lookup(lastToucher, memrefRoot(operand)))
        addPredecessor(event, *producer);
  for (Value operand : op.getOperands())
    if (isa<MemRefType>(operand.getType()))
      lastToucher[memrefRoot(operand)] = index;
}

// Appends one Compute node for `op`: costs it, wires its operand producers
// and memref side effects as predecessors, and aliases its results to the
// new node. Shared by a kernel body's ops (buildKernelNodes) and a bare
// compute op sitting directly in a block post-inlining (buildBlockNodes).
void appendComputeNode(Operation &op, const DeviceCostParams &device,
                       std::vector<ScheduleEvent> &nodes,
                       llvm::DenseMap<Value, size_t> &valueNode,
                       llvm::DenseMap<Value, size_t> &lastMemrefToucher) {
  ScheduleEvent event;
  event.kind = ScheduleEventKind::Compute;
  event.op = &op;
  // costOp returns nullopt in two cases, both treated as a zero-cost node
  // here rather than dropped, so it still carries through any dependency it
  // sits on:
  //   - a dynamic shape or unsized type (R6, R7 in ComputeCost.h), expected
  //     for a kernel body's own stablehlo ops since they're always lowered
  //     to local shapes by this point;
  //   - unconditionally, any op with a non-tensor operand or result, which
  //     covers every memref op ACT's real patched kernel bodies splice in
  //     around their bodyless stage/launch/fetch calls. A memref op's real
  //     cost, if any, is already attributed to the stage/launch/fetch call
  //     it stages memory for (see actCalleeCostCycles).
  event.computeCost = costOp(&op).value_or(OpCost{});
  event.isolatedDuration = event.computeCost.time(device);
  for (Value operand : op.getOperands())
    if (std::optional<size_t> producer = lookup(valueNode, operand))
      addPredecessor(event, *producer);
  size_t index = nodes.size();
  addMemrefSideEffectEdges(op, index, event, lastMemrefToucher);
  nodes.push_back(std::move(event));
  for (Value result : op.getResults())
    alias(valueNode, result, index);
}

// The cost, in cycles, of calling `op`'s resolved callee, if it is one of
// ACT's own bodyless declarations:
//   - an accelerator-launch declaration (@act_launch, per the ACT plan
//     doc's real sample) carries a real cycle count in its act.cost
//     attribute.
//   - a staging/fetch declaration (@act_stage_N/@act_fetch_N) carries
//     act.role ("stage"/"fetch") instead, with no act.cost: ACT costs the
//     memref copy it performs as part of its neighboring launch, not
//     separately, so zero is its real cost here, not a placeholder.
// Nullopt for any other call, including one whose callee cannot be resolved
// at all; each caller decides what to do with that case itself
// (buildKernelNodes falls back to costing the call as an ordinary op via
// appendComputeNode, buildBlockNodes falls back to the hard-barrier
// handling in buildCallNode).
std::optional<double> actCalleeCostCycles(Operation *op) {
  Operation *calleeOp = cast<CallOpInterface>(op).resolveCallable();
  if (!calleeOp)
    return std::nullopt;
  if (FloatAttr cost = calleeOp->getAttrOfType<FloatAttr>("act.cost"))
    return cost.getValueAsDouble();
  if (calleeOp->hasAttrOfType<StringAttr>("act.role"))
    return 0.0;
  return std::nullopt;
}

// Appends one atomic Compute node for a call to a bodyless ACT declaration
// (a costed launch or a zero-cost stage/fetch, per actCalleeCostCycles).
// Unlike an ordinary call, the callee has no region to build a nested
// ScheduleGraph from, and its cost (real or zero) is already known directly,
// so there is no reason to treat the call as a scope-wide hard barrier
// (buildCallNode) or recurse into it (appendComputeNode via costOp would
// just see an opaque, likely memref-typed, unknown op).
void appendActCalleeNode(Operation *op, double costCycles,
                         const DeviceCostParams &device,
                         std::vector<ScheduleEvent> &nodes,
                         llvm::DenseMap<Value, size_t> &valueNode,
                         llvm::DenseMap<Value, size_t> &lastMemrefToucher) {
  ScheduleEvent event;
  event.kind = ScheduleEventKind::Compute;
  event.op = op;
  event.isolatedDuration = costCycles / device.clockFrequency;
  // The simulator (ScheduleSimulator.cpp) drives kComputeResource occupancy
  // off computeCost.flops/.bytes, not isolatedDuration directly. Leaving
  // flops at 0 would make this node look like it claims no compute resource
  // at all despite occupying the device for isolatedDuration; back-solving a
  // synthetic flops that reproduces this duration through the same
  // OpCost::time() formula as an ordinary op keeps its resource accounting
  // consistent with every other Compute node's.
  event.computeCost.flops = device.flops * event.isolatedDuration;
  for (Value operand : op->getOperands())
    if (std::optional<size_t> producer = lookup(valueNode, operand))
      addPredecessor(event, *producer);
  size_t index = nodes.size();
  // A stage/launch/fetch call's own operands are exactly the memref views it
  // reads or writes as a side effect (see ActWorkerClient.cpp's spliced
  // kernel-body convention), so this is the case addMemrefSideEffectEdges
  // exists for: none of these calls return an SSA value the others consume,
  // so without this they would only ever be ordered relative to whichever
  // memref.subview produced their operand, never relative to each other.
  addMemrefSideEffectEdges(*op, index, event, lastMemrefToucher);
  nodes.push_back(std::move(event));
  for (Value result : op->getResults())
    alias(valueNode, result, index);
}

// Appends the Compute nodes for one kernel's body, aliasing the block args to
// whatever produced the kernel's operands and the kernel's results to
// whatever the terminator yields. Calls inside a kernel body are never
// barriers relative to each other (a kernel body has no notion of a "call
// barrier" the way a scope walked by buildBlockNodes does): a call to a
// bodyless ACT declaration (see actCalleeCostCycles) is costed as one atomic
// node (real cost for a launch, zero for a stage/fetch) instead of through
// the ordinary per-op roofline, and any other call is costed the same as
// any other op in the body.
void buildKernelNodes(DistributedKernelOp kernel,
                      const DeviceCostParams &device,
                      std::vector<ScheduleEvent> &nodes,
                      llvm::DenseMap<Value, size_t> &valueNode,
                      llvm::DenseMap<Value, size_t> &lastMemrefToucher) {
  Block &body = kernel.getBody().front();
  for (auto [blockArg, operand] :
       llvm::zip(body.getArguments(), kernel.getArguments()))
    if (std::optional<size_t> producer = lookup(valueNode, operand))
      alias(valueNode, blockArg, *producer);

  for (Operation &op : body) {
    if (isa<DistributedYieldOp>(op))
      continue;
    if (isa<CallOpInterface>(op))
      if (std::optional<double> costCycles = actCalleeCostCycles(&op)) {
        appendActCalleeNode(&op, *costCycles, device, nodes, valueNode,
                            lastMemrefToucher);
        continue;
      }
    appendComputeNode(op, device, nodes, valueNode, lastMemrefToucher);
  }

  auto yield = cast<DistributedYieldOp>(body.getTerminator());
  for (auto [result, yielded] :
       llvm::zip(kernel.getResults(), yield.getReturns()))
    if (std::optional<size_t> producer = lookup(valueNode, yielded))
      alias(valueNode, result, *producer);
}

// Appends the Communication nodes for one collective's decomposed chain and
// aliases its result port. Returns false (and sets `failureReason`) if the
// collective could not be planned.
bool buildCollectiveNodes(DistributedCollectiveOp collective,
                          ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                          const MeshCostParams &params,
                          const PlanOptions &options,
                          std::vector<ScheduleEvent> &nodes,
                          llvm::DenseMap<Value, size_t> &valueNode,
                          std::string &failureReason) {
  std::optional<CollectivePlan> plan =
      planCollective(collective, meshAxisTypes, params, failureReason, options);
  if (!plan)
    return false;

  std::optional<size_t> inputProducer = lookup(valueNode, plan->inputPort);
  size_t lastIndex = 0;
  for (size_t i = 0; i < plan->chain.size(); ++i) {
    ScheduleEvent event;
    event.kind = ScheduleEventKind::Communication;
    event.op = collective;
    event.chainIndex = i;
    event.step = plan->chain[i];
    event.isolatedDuration = event.step.isolatedDuration();
    if (i == 0) {
      if (inputProducer) {
        event.predecessors.push_back(*inputProducer);
        event.predecessorMultiplicity.push_back(1);
      }
    } else {
      event.predecessors.push_back(lastIndex);
      event.predecessorMultiplicity.push_back(1);
    }
    lastIndex = nodes.size();
    nodes.push_back(std::move(event));
  }

  // An empty chain moves nothing (CollectivePlan's own contract): the result
  // port is just the input port under a new name, so it aliases straight to
  // the input's producer, if any, instead of getting a node of its own.
  if (!plan->chain.empty())
    alias(valueNode, plan->resultPort, lastIndex);
  else if (inputProducer)
    alias(valueNode, plan->resultPort, *inputProducer);
  return true;
}

// The PrimitiveKind named by `name`, the inverse of toString(PrimitiveKind).
PrimitiveKind primitiveKindFromString(StringRef name) {
  for (PrimitiveKind kind :
       {PrimitiveKind::AllReduce, PrimitiveKind::ReduceScatter,
        PrimitiveKind::AllGather, PrimitiveKind::AllToAll,
        PrimitiveKind::Permute, PrimitiveKind::LocalSlice})
    if (toString(kind) == name)
      return kind;
  llvm_unreachable(
      "distributed.primitive_cost's \"kind\" field must be one produced by "
      "toString(PrimitiveKind), since buildPrimitiveCostAttr is the only "
      "writer of this attribute");
}

// Rebuilds a PrimitiveStep from the distributed.primitive_cost attribute
// (buildPrimitiveCostAttr, CollectivePrimitiveBuilders.cpp) stamped on a
// lowered step's stablehlo.async_start. This attribute is the only source of
// the step's cost at this point in the pipeline: distributed-lower-collectives
// replaces the DistributedCollectiveOp/CollectivePlan the step was planned
// from with the async op this attribute is attached to.
//
// `atoms` and `rho` are left empty: the simulator (ScheduleSimulator.cpp)
// never reads them, atoms only drive IR emission (already done by this
// point), and rho is informational/DP-only elsewhere.
PrimitiveStep primitiveStepFromCostAttr(DictionaryAttr costAttr) {
  PrimitiveStep step;
  step.kind = primitiveKindFromString(
      cast<StringAttr>(costAttr.get("kind")).getValue());
  step.payloadIn = cast<IntegerAttr>(costAttr.get("payload_in")).getInt();
  step.payloadOut = cast<IntegerAttr>(costAttr.get("payload_out")).getInt();
  step.latency = cast<FloatAttr>(costAttr.get("latency")).getValueAsDouble();
  step.transferTime =
      cast<FloatAttr>(costAttr.get("transfer_time")).getValueAsDouble();
  for (Attribute volumeAttr : cast<ArrayAttr>(costAttr.get("volume")))
    step.volume.push_back(cast<FloatAttr>(volumeAttr).getValueAsDouble());
  return step;
}

// Appends one Communication node for a distributed.primitive_cost-tagged
// stablehlo.async_start (one PrimitiveStep of a collective lowered by
// distributed-lower-collectives), wiring its operand producers as
// predecessors.
//
// `asyncStart`'s own result is a !stablehlo.future, not a payload value, so
// nothing aliases to it directly. Its paired async_done (the future
// result's sole use, per StableHLO's async contract) has its result aliased
// to this node instead, since that is the value the rest of the graph
// depends on.
void appendAsyncCommunicationNode(stablehlo::AsyncStartOp asyncStart,
                                  DictionaryAttr costAttr,
                                  std::vector<ScheduleEvent> &nodes,
                                  llvm::DenseMap<Value, size_t> &valueNode) {
  ScheduleEvent event;
  event.kind = ScheduleEventKind::Communication;
  event.op = asyncStart;
  event.step = primitiveStepFromCostAttr(costAttr);
  event.isolatedDuration = event.step.isolatedDuration();
  // A multi-step chain threads each step's async_done result into the next
  // step's async_start operand (distributed-lower-collectives), so the
  // chain position is recoverable from whichever operand traces back to a
  // prior Communication node, without any explicit chain-grouping in the IR.
  for (Value operand : asyncStart.getOperands())
    if (std::optional<size_t> producer = lookup(valueNode, operand)) {
      addPredecessor(event, *producer);
      if (nodes[*producer].kind == ScheduleEventKind::Communication)
        event.chainIndex = nodes[*producer].chainIndex + 1;
    }
  size_t index = nodes.size();
  nodes.push_back(std::move(event));

  Value future = asyncStart.getResult();
  assert(llvm::hasSingleElement(future.getUses()) &&
         "stablehlo.async_start's future result must have exactly one use, "
         "its paired async_done, per StableHLO's async contract");
  auto asyncDone =
      cast<stablehlo::AsyncDoneOp>(future.getUses().begin()->getOwner());
  alias(valueNode, asyncDone.getResult(), index);
}

// Read-only parameters shared by every scope built in one buildScheduleGraph
// call, threaded down through the recursion into callee scopes.
struct GraphBuildContext {
  ArrayRef<PhysicalCommAxisType> meshAxisTypes;
  const MeshCostParams &params;
  const DeviceCostParams &device;
  const PlanOptions &options;

  // Symbols of the callee chain currently being built, for cycle detection
  // (A calls B calls A, or a direct self-call): identity is the resolved
  // callee op itself, not its name, so this works the same whether the
  // callee is a func.func or a distributed.DistributedFunction.
  llvm::SmallPtrSetImpl<Operation *> &functionsInProgress;
};

bool buildBlockNodes(Block &block, GraphBuildContext &ctx, ScheduleGraph &graph,
                     llvm::DenseMap<Value, size_t> &valueNode,
                     llvm::DenseMap<Value, size_t> &lastMemrefToucher,
                     std::optional<size_t> &pendingBarrier);

// Builds one function-like op's own scope (module scope, or a callee's
// body), with fresh scheduling state: `distributed.kernel` boundaries
// dissolve within this scope, but nothing here is visible to, or depends on,
// any enclosing scope. That is exactly what makes a call a hard barrier
// rather than an ordinary dependency edge (see ScheduleGraph.h). The memref
// last-toucher map (see addMemrefSideEffectEdges) gets the same fresh,
// scope-local lifetime as valueNode: a callee's own allocations are never
// visible to its caller's memory ordering either.
ScheduleGraph buildScopeGraph(Block &block, GraphBuildContext &ctx) {
  ScheduleGraph graph;
  llvm::DenseMap<Value, size_t> valueNode;
  llvm::DenseMap<Value, size_t> lastMemrefToucher;
  std::optional<size_t> pendingBarrier;
  if (!buildBlockNodes(block, ctx, graph, valueNode, lastMemrefToucher,
                       pendingBarrier)) {
    graph.feasible = false;
    graph.nodes.clear();
  }
  return graph;
}

// True if `op` is the module's unique entry point ("main"), whether it is
// still a func.func or has already been converted to a
// distributed.DistributedFunction (ConvertMainToDistributedFunction.cpp
// converts every function transitively reachable from main, main included).
// Only this function's body gets flattened directly into its enclosing
// scope; every other function definition sitting at module scope is inert
// until a call reaches it (see buildBlockNodes).
bool isMainFunctionOp(Operation *op) {
  auto name = op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
  return name && name.getValue() == "main";
}

// Appends one FunctionCall barrier event for `op` (a CallOpInterface),
// recursively building the resolved callee's own scope as a nested,
// independent ScheduleGraph. Returns false (and sets `graph.failureReason`)
// if the callee cannot be resolved, is not a single-block function-like op,
// a call cycle is found, or the callee's own graph is infeasible.
bool buildCallNode(Operation *op, GraphBuildContext &ctx, ScheduleGraph &graph,
                   llvm::DenseMap<Value, size_t> &valueNode,
                   std::optional<size_t> &pendingBarrier) {
  auto call = cast<CallOpInterface>(op);
  Operation *calleeOp = call.resolveCallable();
  auto callee = dyn_cast_or_null<FunctionOpInterface>(calleeOp);
  Region *calleeRegion = callee ? callee.getCallableRegion() : nullptr;
  if (!callee || !calleeRegion || !calleeRegion->hasOneBlock()) {
    graph.failureReason = "call to an unresolved or non-single-block callee";
    return false;
  }

  if (!ctx.functionsInProgress.insert(calleeOp).second) {
    graph.failureReason =
        "call cycle through " + SymbolTable::getSymbolName(calleeOp).str();
    return false;
  }
  ScheduleGraph calleeGraph = buildScopeGraph(calleeRegion->front(), ctx);
  ctx.functionsInProgress.erase(calleeOp);
  if (!calleeGraph.feasible) {
    graph.failureReason = "infeasible callee " +
                          SymbolTable::getSymbolName(calleeOp).str() + ": " +
                          calleeGraph.failureReason;
    return false;
  }

  ScheduleEvent event;
  event.kind = ScheduleEventKind::FunctionCall;
  event.op = op;

  // A call is a hard barrier: every node in this scope with no successor
  // recorded yet (i.e. every "frontier" node, nothing has yet been built to
  // depend on it) must finish before the call's own body can start.
  std::vector<bool> hasSuccessor(graph.nodes.size(), false);
  for (const ScheduleEvent &node : graph.nodes)
    for (size_t predecessor : node.predecessors)
      hasSuccessor[predecessor] = true;
  for (size_t i = 0; i < graph.nodes.size(); ++i)
    if (!hasSuccessor[i]) {
      event.predecessors.push_back(i);
      event.predecessorMultiplicity.push_back(1);
    }

  // TODO: this is only a rough serial-execution proxy for the callee's
  // duration (sum of its own nodes' isolated durations), not a real
  // simulated one. An accurate duration needs a timeline simulator that
  // actually schedules the callee body with its own internal overlap and
  // reports a makespan, which does not exist yet.
  double proxyDuration = 0;
  for (const ScheduleEvent &calleeNode : calleeGraph.nodes)
    proxyDuration += calleeNode.isolatedDuration;
  event.isolatedDuration = proxyDuration;
  event.calleeGraph = std::make_unique<ScheduleGraph>(std::move(calleeGraph));

  size_t index = graph.nodes.size();
  graph.nodes.push_back(std::move(event));
  for (Value result : op->getResults())
    alias(valueNode, result, index);
  // Every event built after this one in the same scope must depend on it,
  // whether or not it has a real data dependency on the call's results.
  pendingBarrier = index;
  return true;
}

// Walks `block` in order, appending its events to `graph.nodes`. Kernel and
// collective-chain boundaries dissolve into this same flat scope via
// `valueNode` (see the file-level comment). A call to a walkable callee is a
// hard barrier (buildCallNode): once one is built, every subsequently built
// node in this same scope, regardless of what builds it, gets that call
// event wired in as an extra predecessor, which is what actually enforces
// "nothing after the call may start before it finishes". A call to a
// bodyless ACT declaration is not a barrier at all, since it has no body to
// wait on, and costs as an ordinary atomic Compute node (appendActCalleeNode)
// wired to its own SSA operand producers plus, via addMemrefSideEffectEdges,
// whichever earlier node last touched the same memref, the same as any
// other op built here; this matters for a real ACT-patched kernel's
// stage/fetch calls in particular, which sit as plain top-level ops in the
// dissolved block, sharing memref views with act_launch as a side effect
// rather than through an SSA result any of them return.
//
// The module's own top-level block reaches here as the outermost scope, with
// its "main" function's body flattened inline (isMainFunctionOp); any other
// function definition found directly in a block is left untouched. It is
// only ever walked by recursing into buildScopeGraph from buildCallNode, so
// a callee's own body never contains a nested function definition to
// (mis)trigger this branch.
bool buildBlockNodes(Block &block, GraphBuildContext &ctx, ScheduleGraph &graph,
                     llvm::DenseMap<Value, size_t> &valueNode,
                     llvm::DenseMap<Value, size_t> &lastMemrefToucher,
                     std::optional<size_t> &pendingBarrier) {
  for (Operation &opRef : block) {
    Operation *op = &opRef;
    size_t startIndex = graph.nodes.size();

    if (auto kernel = dyn_cast<DistributedKernelOp>(op)) {
      buildKernelNodes(kernel, ctx.device, graph.nodes, valueNode,
                       lastMemrefToucher);
    } else if (auto collective = dyn_cast<DistributedCollectiveOp>(op)) {
      if (!buildCollectiveNodes(collective, ctx.meshAxisTypes, ctx.params,
                                ctx.options, graph.nodes, valueNode,
                                graph.failureReason))
        return false;
    } else if (auto asyncStart = dyn_cast<stablehlo::AsyncStartOp>(op)) {
      // Only a distributed.primitive_cost-tagged async_start is a lowered
      // collective step (distributed-lower-collectives); an untagged one is
      // some other async op this graph does not model, the same as any
      // other unrecognized op.
      if (auto costAttr =
              op->getAttrOfType<DictionaryAttr>("distributed.primitive_cost"))
        appendAsyncCommunicationNode(asyncStart, costAttr, graph.nodes,
                                     valueNode);
      else
        continue;
    } else if (isa<stablehlo::AsyncDoneOp>(op)) {
      // Its result was already aliased to the paired async_start's node (if
      // that async_start was cost-tagged) by appendAsyncCommunicationNode;
      // it must never become a node of its own.
      continue;
    } else if (isa<CallOpInterface>(op)) {
      if (std::optional<double> costCycles = actCalleeCostCycles(op)) {
        appendActCalleeNode(op, *costCycles, ctx.device, graph.nodes, valueNode,
                            lastMemrefToucher);
        // Falls through to the ordinary pendingBarrier wiring below, the
        // same as any other Compute node built in this loop: a bodyless ACT
        // declaration is never a hard barrier.
      } else {
        // The barrier event is itself wired to the previous barrier (if
        // any) by the frontier scan in buildCallNode, not by the post-loop
        // pendingBarrier patch-up below, so no further action is needed
        // here.
        if (!buildCallNode(op, ctx, graph, valueNode, pendingBarrier))
          return false;
        continue;
      }
    } else if (auto func = dyn_cast<FunctionOpInterface>(op)) {
      if (isMainFunctionOp(op)) {
        Region *body = func.getCallableRegion();
        if (body && body->hasOneBlock())
          if (!buildBlockNodes(body->front(), ctx, graph, valueNode,
                               lastMemrefToucher, pendingBarrier))
            return false;
      }
      continue;
    } else if (StringRef ns = op->getName().getDialectNamespace();
               ns == "stablehlo" || ns == "memref") {
      // A bare stablehlo op sitting directly in a block: the shape
      // distributed-inline-kernels (D4) produces when it splices a kernel
      // body into the surrounding block, dissolving the DistributedKernelOp
      // wrapper. A bare memref op alongside it: the memref.alloc/subview/
      // dealloc scaffolding ACT's own patched kernel bodies (spliced in by
      // ensureActPatchedKernel) build around their bodyless stage/launch/
      // fetch calls, dissolved into the block the same way.
      //
      // Restricting this branch to these two dialects, rather than costing
      // every unrecognized op the way buildKernelNodes costs a whole kernel
      // body, keeps a module-scope non-computational value producer (e.g. a
      // placeholder tensor.empty feeding a collective in a hand-written
      // test) from becoming a spurious node: nothing outside a kernel body
      // or its dissolved form is expected to be either dialect otherwise.
      appendComputeNode(*op, ctx.device, graph.nodes, valueNode,
                        lastMemrefToucher);
    } else {
      // Axis-algebra / declarative distributed metadata (PhysicalMesh,
      // GetPhysicalMeshAxes, axis.getaxis/factor/product/map, ...), a
      // terminator (func.return, or a distributed.Yield that somehow
      // reached block scope), and any other value producer sitting directly
      // in a block outside these two dialects (e.g. tensor.empty) carry no
      // runtime cost this graph models: none of them is a node here.
      continue;
    }

    if (pendingBarrier)
      for (size_t i = startIndex; i < graph.nodes.size(); ++i)
        if (!llvm::is_contained(graph.nodes[i].predecessors, *pendingBarrier)) {
          graph.nodes[i].predecessors.push_back(*pendingBarrier);
          graph.nodes[i].predecessorMultiplicity.push_back(1);
        }
  }
  return true;
}

} // namespace

ScheduleGraph buildScheduleGraph(ModuleOp module, const PlanOptions &options) {
  FailureOr<PhysicalMeshOp> physicalMesh = findUniquePhysicalMesh(module);
  if (failed(physicalMesh)) {
    ScheduleGraph graph;
    graph.feasible = false;
    graph.failureReason = "no unique physical mesh";
    return graph;
  }
  SmallVector<PhysicalCommAxisType> meshAxisTypes;
  for (Attribute axisAttr : physicalMesh->getAxesAttr())
    meshAxisTypes.push_back(
        cast<PhysicalCommAxisType>(cast<TypeAttr>(axisAttr).getValue()));
  MeshCostParams params = meshCostParamsFromPhysicalMesh(*physicalMesh);
  DeviceCostParams device = deviceCostParamsFromPhysicalMesh(*physicalMesh);

  llvm::SmallPtrSet<Operation *, 8> functionsInProgress;
  GraphBuildContext ctx{meshAxisTypes, params, device, options,
                        functionsInProgress};
  return buildScopeGraph(*module.getBody(), ctx);
}

std::string describeEvent(const ScheduleEvent &event) {
  std::string message;
  llvm::raw_string_ostream os(message);
  if (event.kind == ScheduleEventKind::Compute) {
    os << "compute " << event.op->getName()
       << ": flops=" << llvm::format("%g", event.computeCost.flops)
       << " bytes=" << llvm::format("%g", event.computeCost.bytes)
       << " time=" << llvm::format("%g", event.isolatedDuration);
  } else if (event.kind == ScheduleEventKind::Communication) {
    os << "communication " << toString(event.step.kind) << " (step "
       << event.chainIndex << "): "
       << "time=" << llvm::format("%g", event.isolatedDuration);
  } else {
    os << "call " << event.op->getName()
       << ": callee_nodes=" << event.calleeGraph->nodes.size()
       << " time=" << llvm::format("%g", event.isolatedDuration)
       << " (proxy: serial sum of callee node durations, pending D2 timeline "
          "simulator)";
  }
  return message;
}

ScheduleGraphAnalysis::ScheduleGraphAnalysis(ModuleOp module)
    : graph(buildScheduleGraph(module)) {}

} // namespace mlir::enzyme::distributed
