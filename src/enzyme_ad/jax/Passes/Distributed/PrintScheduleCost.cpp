#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/MeshCostMetadata.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/ScheduleGraph.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/ScheduleSimulator.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_PRINTSCHEDULECOSTPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

std::string shortest(double value) {
  std::string text;
  llvm::raw_string_ostream(text) << llvm::format("%g", value);
  return text;
}

struct PrintScheduleCostPass
    : public impl::PrintScheduleCostPassBase<PrintScheduleCostPass> {
  using PrintScheduleCostPassBase::PrintScheduleCostPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    ScheduleGraph graph = buildScheduleGraph(module);
    if (!graph.feasible) {
      module.emitRemark() << "schedule cost: infeasible ("
                          << graph.failureReason << ")";
      return;
    }

    // Same mesh/device resolution as buildScheduleGraph itself used to cost
    // the graph's own nodes (MeshCostMetadata.h), so the simulator's
    // resource capacities can never disagree with the demand the graph
    // already carries.
    FailureOr<PhysicalMeshOp> physicalMesh = findUniquePhysicalMesh(module);
    assert(succeeded(physicalMesh) &&
           "a feasible ScheduleGraph implies buildScheduleGraph already "
           "resolved a unique physical mesh");
    MeshCostParams mesh = meshCostParamsFromPhysicalMesh(*physicalMesh);
    DeviceCostParams device = deviceCostParamsFromPhysicalMesh(*physicalMesh);

    TimelineResult result = simulateScheduleGraph(graph, mesh, device);

    // One remark per top-level node with its simulated start/finish (not
    // just the critical path's), so a lit test can pin down contention
    // effects on any node, e.g. an event that finishes early because its own
    // resource never contended with the makespan's bottleneck. FunctionCall
    // events are reported the same way as the others: their own nested
    // callee graph's internal schedule is not re-printed here (it already
    // has its own remarks the next time this pass, or
    // distributed-print-schedule-graph, is run directly on it).
    for (auto [index, event] : llvm::enumerate(graph.nodes))
      event.op->emitRemark() << "node " << index << ": " << toString(event.kind)
                             << " start=" << shortest(result.start[index])
                             << " finish=" << shortest(result.finish[index]);

    std::string criticalPath;
    llvm::raw_string_ostream criticalPathStream(criticalPath);
    llvm::interleaveComma(result.criticalPath, criticalPathStream);
    module.emitRemark() << "schedule cost: makespan="
                        << shortest(result.makespan) << " over "
                        << graph.nodes.size() << " nodes, critical path=["
                        << criticalPath << "]";
    module.emitRemark() << "schedule cost summary: communicating="
                        << shortest(result.totalCommunicationTime)
                        << " computing=" << shortest(result.totalComputeTime)
                        << " utilization="
                        << shortest(result.averageComputeUtilization)
                        << " idle=" << shortest(result.totalComputeIdleTime)
                        << " activeUtilization="
                        << shortest(result.averageActiveComputeUtilization);
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
