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
  // alias iff they are the same result of the same op
  auto result1 = dyn_cast<OpResult>(ax1);
  auto result2 = dyn_cast<OpResult>(ax2);
  if (!result1 || !result2) {
    llvm::report_fatal_error(
        "LogicalMeshAxisType::aliases requires both axes to be OpResults");
  }
  return result1.getOwner() == result2.getOwner() &&
         result1.getResultNumber() == result2.getResultNumber();
}

// Replication axes are modeled as always disjoint.
bool ReplicationAxisType::aliases(Value ax1, Value ax2) const {
  (void)ax1;
  (void)ax2;
  return false;
}

} // namespace mlir::enzyme::distributed
