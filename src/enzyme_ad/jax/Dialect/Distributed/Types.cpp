#include "Dialect.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Dialect.h"

#include "llvm/Support/ErrorHandling.h"

namespace mlir::enzyme::distributed {

bool PhysicalCommAxisType::aliases(Value ax1, Value ax2) const {
  auto result1 = dyn_cast<OpResult>(ax1);
  auto result2 = dyn_cast<OpResult>(ax2);
  if (!result1 || !result2) {
    assert(result1 && result2 &&
           "PhysicalCommAxisType::aliases requires both axes to be OpResults");
    return false;
  }

  auto getMeshAxes1 = dyn_cast<GetPhysicalMeshAxesOp>(result1.getOwner());
  auto getMeshAxes2 = dyn_cast<GetPhysicalMeshAxesOp>(result2.getOwner());
  if (!getMeshAxes1 || !getMeshAxes2) {
    assert(getMeshAxes1 && getMeshAxes2 &&
           "PhysicalCommAxisType::aliases requires both axes to be produced "
           "by distributed.GetPhysicalMeshAxes");
    return false;
  }

  // Physical axes alias iff they reference the same physical mesh symbol and
  // correspond to the same axis index.
  return getMeshAxes1.getPhysicalMeshAttr() ==
             getMeshAxes2.getPhysicalMeshAttr() &&
         result1.getResultNumber() == result2.getResultNumber();
}

bool LogicalMeshAxisType::aliases(Value ax1, Value ax2) const {
  // alias iff they are the same result of the same op
  auto result1 = dyn_cast<OpResult>(ax1);
  auto result2 = dyn_cast<OpResult>(ax2);
  if (!result1 || !result2) {
    assert(result1 && result2 &&
           "LogicalMeshAxisType::aliases requires both axes to be OpResults");
    return false;
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
