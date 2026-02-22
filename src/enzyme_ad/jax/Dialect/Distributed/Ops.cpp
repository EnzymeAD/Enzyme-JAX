#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/TypeSwitch.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "Dialect.h"

using mlir::OpTrait::enzyme::distributed::ChannelDefTrait;
using mlir::OpTrait::enzyme::distributed::DeviceDefTrait;
namespace mlir::enzyme::distributed {

LogicalResult AxisFactorOp::verify() {
  auto factors = getFactors();
  for (auto factor_attr : factors) {
    if (auto factor = dyn_cast<IntegerAttr>(factor_attr)) {
      if (factor.getValue().getSExtValue() <= 0) {
        return emitOpError() << "requires all factors to be > 0";
      }
    } else {
      return emitOpError() << "requires all factors to be integer attributes";
    }
  }

  if (getLogicalAxes().size() != factors.size()) {
    return emitOpError() << "requires one logical axis result per factor (got "
                         << getLogicalAxes().size() << " results for "
                         << factors.size() << " factors)";
  }

  // TODO: Verify that product(factors) equals physical_axis size.
  return mlir::success();
}

LogicalResult AxisFactorOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  AxisFactorOpAdaptor adaptor(operands, attributes, properties, regions);
  if (!adaptor.getFactors()) {
    if (location)
      mlir::emitError(*location) << "missing factors attribute";
    return mlir::failure();
  }

  inferredReturnTypes.reserve(adaptor.getFactors().size());
  for (auto _ : adaptor.getFactors()) {
    inferredReturnTypes.push_back(LogicalCommAxisType::get(context));
  }
  return mlir::success();
}

LogicalResult AxisProductOp::verify() {
  // TODO disjointness
  return mlir::success();
}

} // namespace mlir::enzyme::distributed
#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedOps.cpp.inc"