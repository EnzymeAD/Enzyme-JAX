#include "Dialect.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Dialect.h"

#include "llvm/Support/ErrorHandling.h"

namespace mlir::enzyme::distributed {

bool PhysicalCommAxisType::equivalent(Value ax1, Value ax2) const {
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

bool PhysicalCommAxisType::disjoint(Value ax1, Value ax2) const {
  return !PhysicalCommAxisType::equivalent(ax1, ax2);
}

bool LogicalMeshAxisType::equivalent(Value ax1, Value ax2) const {
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

bool LogicalMeshAxisType::disjoint(Value ax1, Value ax2) const {
  return !LogicalMeshAxisType::equivalent(ax1, ax2);
}

// Replication axes are equivalent whenever they have the same extent
bool ReplicationAxisType::equivalent(Value ax1, Value ax2) const {
  axis::AxisExtentT extent1 =
      cast<ReplicationAxisType>(ax1.getType()).getExtent();
  axis::AxisExtentT extent2 =
      cast<ReplicationAxisType>(ax2.getType()).getExtent();
  return extent1 == extent2;
}

// But replication axes are always disjoint / non-interfering
bool ReplicationAxisType::disjoint(Value ax1, Value ax2) const {
  (void)ax1;
  (void)ax2;
  return true;
}

// Identity, not extent: two DeviceLocalAxis declarations with the same extent
// are NOT the same axis (mirrors LogicalMeshAxisType::equivalent -- see that
// type's comment, and DeviceLocalAxisOp's own doc comment for why this
// matters: an extent-only check can't tell "the same serialized chunk reused"
// apart from "two unrelated chunks that happen to have the same size").
bool DeviceLocalAxisType::equivalent(Value ax1, Value ax2) const {
  auto result1 = dyn_cast<OpResult>(ax1);
  auto result2 = dyn_cast<OpResult>(ax2);
  if (!result1 || !result2) {
    assert(result1 && result2 &&
           "DeviceLocalAxisType::equivalent requires both axes to be "
           "OpResults");
    return false;
  }
  return result1.getOwner() == result2.getOwner() &&
         result1.getResultNumber() == result2.getResultNumber();
}

// Also always disjoint: no collision for serializing everything
bool DeviceLocalAxisType::disjoint(Value ax1, Value ax2) const {
  (void)ax1;
  (void)ax2;
  return true;
}

} // namespace mlir::enzyme::distributed
