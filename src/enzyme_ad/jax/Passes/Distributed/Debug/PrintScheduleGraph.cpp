#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/ScheduleGraph.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_PRINTSCHEDULEGRAPHPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// `describeEvent(event)`, plus this node's own index and its predecessors',
// so a lit test can check edges without relying on op order matching node
// order by coincidence. `scope` prefixes the message (e.g. "callee node 0:"
// for a node inside a FunctionCall event's nested graph, nested once more per
// further level of call nesting) so a lit test can tell which graph a line
// belongs to.
std::string describeNode(size_t index, const ScheduleEvent &event,
                         StringRef scope) {
  std::string message;
  llvm::raw_string_ostream os(message);
  os << scope << "node " << index << ": " << describeEvent(event) << " preds=[";
  llvm::interleaveComma(event.predecessors, os);
  os << "]";
  return message;
}

// Emits one remark per node in `graph`, recursing into a FunctionCall
// event's nested callee graph so its internal structure is fully visible in
// the report too, even though ScheduleGraph itself keeps it as an opaque
// nested graph rather than flattening it into the caller's own node list.
void printGraph(const ScheduleGraph &graph, StringRef scope, size_t &numCompute,
                size_t &numCommunication, size_t &numCalls) {
  for (auto [index, event] : llvm::enumerate(graph.nodes)) {
    switch (event.kind) {
    case ScheduleEventKind::Compute:
      ++numCompute;
      break;
    case ScheduleEventKind::Communication:
      ++numCommunication;
      break;
    case ScheduleEventKind::FunctionCall:
      ++numCalls;
      break;
    }
    event.op->emitRemark() << describeNode(index, event, scope);
    if (event.kind == ScheduleEventKind::FunctionCall)
      printGraph(*event.calleeGraph, (scope + "callee ").str(), numCompute,
                 numCommunication, numCalls);
  }
}

struct PrintScheduleGraphPass
    : public impl::PrintScheduleGraphPassBase<PrintScheduleGraphPass> {
  using PrintScheduleGraphPassBase::PrintScheduleGraphPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    const ScheduleGraph &graph =
        getAnalysis<ScheduleGraphAnalysis>().getGraph();
    if (!graph.feasible) {
      module.emitRemark() << "schedule graph: infeasible ("
                          << graph.failureReason << ")";
      return;
    }

    // Counts include nested callee-graph nodes (printGraph recurses into
    // them), so they can exceed the top-level graph.nodes.size() whenever a
    // FunctionCall event's own callee contributes further nodes.
    size_t numCompute = 0, numCommunication = 0, numCalls = 0;
    printGraph(graph, /*scope=*/"", numCompute, numCommunication, numCalls);
    module.emitRemark() << "schedule graph: "
                        << (numCompute + numCommunication + numCalls)
                        << " nodes (" << numCompute << " compute, "
                        << numCommunication << " communication, " << numCalls
                        << " call)";
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
