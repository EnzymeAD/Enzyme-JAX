#include "ScheduleGraph.h"

#include "MeshCostMetadata.h"
#include "Utilities.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
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

// Appends the Compute nodes for one kernel's body, aliasing the block args to
// whatever produced the kernel's operands and the kernel's results to
// whatever the terminator yields.
void buildKernelNodes(DistributedKernelOp kernel,
                      const DeviceCostParams &device,
                      std::vector<ScheduleEvent> &nodes,
                      llvm::DenseMap<Value, size_t> &valueNode) {
  Block &body = kernel.getBody().front();
  for (auto [blockArg, operand] :
       llvm::zip(body.getArguments(), kernel.getArguments()))
    if (std::optional<size_t> producer = lookup(valueNode, operand))
      alias(valueNode, blockArg, *producer);

  for (Operation &op : body) {
    if (isa<DistributedYieldOp>(op))
      continue;
    ScheduleEvent event;
    event.kind = ScheduleEventKind::Compute;
    event.op = &op;
    // costOp returns nullopt only for a dynamic shape or unsized type (R6,
    // R7 in ComputeCost.h), which should not occur in a kernel body lowered
    // to local shapes; treat it as a zero-cost node rather than dropping it,
    // so it still carries through any dependency it sits on.
    event.computeCost = costOp(&op).value_or(OpCost{});
    event.isolatedDuration = event.computeCost.time(device);
    // An op using the same value for more than one operand (e.g. `add %x,
    // %x`) must not turn into a duplicated dependency edge, but the number of
    // operand slots it occupied still has to be recoverable (see
    // predecessorMultiplicity in ScheduleGraph.h), so track it alongside the
    // deduplicated list instead of discarding it.
    for (Value operand : op.getOperands())
      if (std::optional<size_t> producer = lookup(valueNode, operand)) {
        auto it = llvm::find(event.predecessors, *producer);
        if (it == event.predecessors.end()) {
          event.predecessors.push_back(*producer);
          event.predecessorMultiplicity.push_back(1);
        } else {
          ++event.predecessorMultiplicity[it - event.predecessors.begin()];
        }
      }
    size_t index = nodes.size();
    nodes.push_back(std::move(event));
    for (Value result : op.getResults())
      alias(valueNode, result, index);
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
                     std::optional<size_t> &pendingBarrier);

// Builds one function-like op's own scope (module scope, or a callee's
// body), with fresh scheduling state: `distributed.kernel` boundaries
// dissolve within this scope, but nothing here is visible to, or depends on,
// any enclosing scope. That is exactly what makes a call a hard barrier
// rather than an ordinary dependency edge (see ScheduleGraph.h).
ScheduleGraph buildScopeGraph(Block &block, GraphBuildContext &ctx) {
  ScheduleGraph graph;
  llvm::DenseMap<Value, size_t> valueNode;
  std::optional<size_t> pendingBarrier;
  if (!buildBlockNodes(block, ctx, graph, valueNode, pendingBarrier)) {
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
// `valueNode` (see the file-level comment). A call is instead a hard barrier
// (buildCallNode): once one is built, every subsequently built node in this
// same scope, regardless of what builds it, gets that call event wired in as
// an extra predecessor, which is what actually enforces "nothing after the
// call may start before it finishes".
//
// The module's own top-level block reaches here as the outermost scope, with
// its "main" function's body flattened inline (isMainFunctionOp); any other
// function definition found directly in a block is left untouched. It is
// only ever walked by recursing into buildScopeGraph from buildCallNode, so
// a callee's own body never contains a nested function definition to
// (mis)trigger this branch.
bool buildBlockNodes(Block &block, GraphBuildContext &ctx, ScheduleGraph &graph,
                     llvm::DenseMap<Value, size_t> &valueNode,
                     std::optional<size_t> &pendingBarrier) {
  for (Operation &opRef : block) {
    Operation *op = &opRef;
    size_t startIndex = graph.nodes.size();

    if (auto kernel = dyn_cast<DistributedKernelOp>(op)) {
      buildKernelNodes(kernel, ctx.device, graph.nodes, valueNode);
    } else if (auto collective = dyn_cast<DistributedCollectiveOp>(op)) {
      if (!buildCollectiveNodes(collective, ctx.meshAxisTypes, ctx.params,
                                ctx.options, graph.nodes, valueNode,
                                graph.failureReason))
        return false;
    } else if (isa<CallOpInterface>(op)) {
      // The barrier event is itself wired to the previous barrier (if any)
      // by the frontier scan in buildCallNode, not by the post-loop
      // pendingBarrier patch-up below, so no further action is needed here.
      if (!buildCallNode(op, ctx, graph, valueNode, pendingBarrier))
        return false;
      continue;
    } else if (auto func = dyn_cast<FunctionOpInterface>(op)) {
      if (isMainFunctionOp(op)) {
        Region *body = func.getCallableRegion();
        if (body && body->hasOneBlock())
          if (!buildBlockNodes(body->front(), ctx, graph, valueNode,
                               pendingBarrier))
            return false;
      }
      continue;
    } else {
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

} // namespace mlir::enzyme::distributed
