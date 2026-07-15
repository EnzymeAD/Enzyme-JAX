#include "Dialect.h"

#include "llvm/Support/ErrorHandling.h"

namespace mlir::enzyme::distributed {

// Physical-axis alias semantics are intentionally not implemented yet.
bool PhysicalCommAxisType::aliases(Value selfAxis, Value otherAxis) const {
  (void)selfAxis;
  (void)otherAxis;
  llvm_unreachable(
      "alias semantics are not implemented for PhysicalCommAxisType");
}

// Logical-mesh-axis alias semantics are intentionally not implemented yet.
bool LogicalMeshAxisType::aliases(Value selfAxis, Value otherAxis) const {
  (void)selfAxis;
  (void)otherAxis;
  llvm_unreachable(
      "alias semantics are not implemented for LogicalMeshAxisType");
}

// Replication axes are modeled as always disjoint.
bool ReplicationAxisType::aliases(Value selfAxis, Value otherAxis) const {
  (void)selfAxis;
  (void)otherAxis;
  return false;
}

// Tensor-axis equivalence is currently based on typed index-space coordinates.
bool TensorAxisType::aliases(Value selfAxis, Value otherAxis) const {
  (void)selfAxis;
  auto other = dyn_cast<TensorAxisType>(otherAxis.getType());
  assert(other && "TensorAxisType aliasing requires tensor axis values");
  // TODO: include tensor object/value provenance once index-space identity
  // is represented beyond the current type payload.
  return getAxisIndex() == other.getAxisIndex() &&
         getTensorType() == other.getTensorType();
}

} // namespace mlir::enzyme::distributed
