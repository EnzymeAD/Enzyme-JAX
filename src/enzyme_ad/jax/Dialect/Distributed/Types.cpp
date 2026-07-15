#include "Dialect.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Dialect.h"

#include "llvm/Support/ErrorHandling.h"

namespace mlir::enzyme::distributed {

bool PhysicalCommAxisType::aliases(Value ax1, Value ax2) const {
  (void)ax1;
  (void)ax2;
  llvm::report_fatal_error(
      "PhysicalCommAxisType::aliases is not implemented yet");
}

bool LogicalMeshAxisType::aliases(Value ax1, Value ax2) const {
  (void)ax1;
  (void)ax2;
  llvm::report_fatal_error(
      "LogicalMeshAxisType::aliases is not implemented yet");
}

// Replication axes are modeled as always disjoint.
bool ReplicationAxisType::aliases(Value ax1, Value ax2) const {
  (void)ax1;
  (void)ax2;
  return false;
}

} // namespace mlir::enzyme::distributed
