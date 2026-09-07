#include "src/enzyme_ad/jax/Passes/Distributed/LogicalAxisOverlap.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"

namespace mlir::enzyme::distributed {

namespace {
// Walks factor groups down to their factors and collects the distinct logical
// mesh axes backing them. Values that are not factor groups, factors without
// resolvable provenance, and non-logical axes (physical, replication, device
// local, shape) are skipped.
llvm::SmallVector<TypedValue<LogicalMeshAxisType>>
resolveLogicalAxes(ValueRange factorGroups) {
  llvm::SmallSetVector<TypedValue<LogicalMeshAxisType>, 4> logicalAxes;
  for (Value groupValue : factorGroups) {
    auto group = dyn_cast<TypedValue<axis::FactorGroupType>>(groupValue);
    if (!group)
      continue;
    auto factors = axis::getProductProvenanceFactors(group);
    if (failed(factors))
      continue;
    for (auto factor : *factors) {
      auto provenanceAxis = axis::getFactorProvenanceAxis(factor);
      if (failed(provenanceAxis))
        continue;
      if (auto logicalAxis =
              dyn_cast<TypedValue<LogicalMeshAxisType>>(Value(*provenanceAxis)))
        logicalAxes.insert(logicalAxis);
    }
  }
  return logicalAxes.takeVector();
}
} // namespace

void LogicalAxisOverlap::addMutualOverlaps(
    llvm::ArrayRef<TypedValue<LogicalMeshAxisType>> axes) {
  for (size_t i = 0; i < axes.size(); ++i)
    for (size_t j = i + 1; j < axes.size(); ++j)
      addOverlap(axes[i], axes[j]);
}

LogicalAxisOverlap::LogicalAxisOverlap(ModuleOp moduleOp) {
  // Overlaps happen two places: kernels and collectives
  moduleOp.walk([&](DistributedCollectiveOp collective) {
    // LHS and RHS meshes are separate index spaces, so axes only overlap
    // within their own side.
    addMutualOverlaps(resolveLogicalAxes(collective.getInputMesh()));
    addMutualOverlaps(resolveLogicalAxes(collective.getOutputMesh()));
  });

  moduleOp.walk([&](DistributedKernelOp kernel) {
    addMutualOverlaps(resolveLogicalAxes(kernel.getPartitioningAxes()));
  });
}

std::optional<const llvm::ArrayRef<TypedValue<LogicalMeshAxisType>>>
LogicalAxisOverlap::getOverlaps(TypedValue<LogicalMeshAxisType> axis) const {
  auto it = overlaps.find(axis);
  if (it != overlaps.end()) {
    return it->second.getArrayRef();
  }
  return std::nullopt;
}
} // namespace mlir::enzyme::distributed