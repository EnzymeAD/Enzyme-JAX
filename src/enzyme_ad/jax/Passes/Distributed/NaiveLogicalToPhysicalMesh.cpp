#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_NAIVELOGICALTOPHYSICALMESHPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

static LogicalResult rewriteLogicalMeshToPhysicalAxesInComputation(
    MeshComputationOp meshComputation) {
  SmallVector<LogicalMeshAxesOp> logicalMeshOps;
  meshComputation.walk(
      [&](LogicalMeshAxesOp op) { logicalMeshOps.push_back(op); });
  if (logicalMeshOps.empty()) {
    return meshComputation.emitOpError()
           << "expected exactly one distributed.LogicalMeshAxes op in mesh "
              "computation, found 0";
  }
  if (logicalMeshOps.size() != 1) {
    return meshComputation.emitOpError()
           << "expected exactly one distributed.LogicalMeshAxes op in mesh "
              "computation, found "
           << logicalMeshOps.size();
  }

  SmallVector<GetPhysicalMeshAxesOp> getPhysicalAxesOps;
  meshComputation.walk(
      [&](GetPhysicalMeshAxesOp op) { getPhysicalAxesOps.push_back(op); });
  if (getPhysicalAxesOps.empty()) {
    return meshComputation.emitOpError()
           << "expected exactly one distributed.GetPhysicalMeshAxes op in "
              "mesh computation, found 0";
  }
  if (getPhysicalAxesOps.size() != 1) {
    return meshComputation.emitOpError()
           << "expected exactly one distributed.GetPhysicalMeshAxes op in "
              "mesh computation, found "
           << getPhysicalAxesOps.size();
  }

  LogicalMeshAxesOp logicalMesh = logicalMeshOps.front();
  GetPhysicalMeshAxesOp getPhysicalAxes = getPhysicalAxesOps.front();

  auto logicalAxes = logicalMesh.getAxes();
  auto physicalAxes = getPhysicalAxes.getAxes();

  if (logicalAxes.size() != physicalAxes.size()) {
    return logicalMesh.emitOpError()
           << "logical mesh rank does not match physical mesh rank ("
           << logicalAxes.size() << " != " << physicalAxes.size() << ")";
  }

  for (auto [idx, logicalAxisValue, physicalAxisValue] :
       llvm::enumerate(logicalAxes, physicalAxes)) {
    auto logicalAxisType =
        dyn_cast<LogicalMeshAxisType>(logicalAxisValue.getType());
    auto physicalAxisType =
        dyn_cast<PhysicalCommAxisType>(physicalAxisValue.getType());

    if (!logicalAxisType || !physicalAxisType) {
      return logicalMesh.emitOpError()
             << "expected logical/physical mesh axis types at index " << idx;
    }

    if (logicalAxisType.getExtent() != physicalAxisType.getExtent()) {
      return logicalMesh.emitOpError()
             << "mesh axis extent mismatch at index " << idx
             << " (logical=" << logicalAxisType.getExtent()
             << ", physical=" << physicalAxisType.getExtent() << ")";
    }
  }

  for (auto [logicalAxisValue, physicalAxisValue] :
       llvm::zip(logicalAxes, physicalAxes)) {
    if (failed(axis::replaceAndTypePropagate(logicalAxisValue,
                                             physicalAxisValue))) {
      return failure();
    }
  }

  logicalMesh.erase();
  return success();
}

struct NaiveLogicalToPhysicalMeshPass
    : public impl::NaiveLogicalToPhysicalMeshPassBase<
          NaiveLogicalToPhysicalMeshPass> {
  using NaiveLogicalToPhysicalMeshPassBase::NaiveLogicalToPhysicalMeshPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    bool sawMeshComputation = false;
    for (MeshComputationOp meshComputation :
         module.getOps<MeshComputationOp>()) {
      sawMeshComputation = true;
      if (failed(
              rewriteLogicalMeshToPhysicalAxesInComputation(meshComputation))) {
        signalPassFailure();
        return;
      }
    }

    if (!sawMeshComputation) {
      module.emitError() << "expected at least one distributed.MeshComputation"
                            " in module";
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
