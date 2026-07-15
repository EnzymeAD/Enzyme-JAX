#include "Dialect.h"

#include "llvm/Support/ErrorHandling.h"

namespace mlir::enzyme::axis {

unsigned ShapeAxisType::getExtent() const {
  auto shaped = ::mlir::dyn_cast<::mlir::ShapedType>(getShapeType());
  assert(shaped && "ShapeAxisType expects a shaped type");
  assert(shaped.hasRank() && "ShapeAxisType expects a ranked shaped type");
  assert(getAxisIndex() < shaped.getRank() && "axis index out of range");
  assert(!shaped.isDynamicDim(getAxisIndex()) &&
         "dynamic shape dimensions are not yet supported");
  return static_cast<unsigned>(shaped.getDimSize(getAxisIndex()));
}

// Shape-axis equivalence is currently based on typed index-space coordinates.
bool ShapeAxisType::aliases(Value ax1, Value ax2) const {
  (void)ax1;
  auto other = dyn_cast<ShapeAxisType>(ax2.getType());
  assert(other && "ShapeAxisType aliasing requires shape axis values");
  // TODO: include object/value provenance once index-space identity
  // is represented beyond the current type payload.
  return getAxisIndex() == other.getAxisIndex() &&
         getShapeType() == other.getShapeType();
}

} // namespace mlir::enzyme::axis
