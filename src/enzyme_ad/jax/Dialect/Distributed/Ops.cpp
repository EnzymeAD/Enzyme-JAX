#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/TypeSwitch.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "Dialect.h"

using llvm::report_fatal_error;

namespace mlir::enzyme::distributed {

int64_t AxisAllToAllOp::getAxisSize(::mlir::Value axis) {
  // This op defines a single value, so just check if the
  // proper value is passed.
  if (getAxis() != axis) {
    report_fatal_error("axis not defined by this op");
  }
  auto axisSizeValues = getAxisSizeAttr().getValue();
  return axisSizeValues.getSExtValue();
}

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

int64_t AxisFactorOp::getAxisSize(::mlir::Value axis) {
  int i = 0;
  for (auto result : getLogicalAxes()) {
    if (result == axis) {
      auto factor_attr = getFactors()[i];
      auto factor = dyn_cast<IntegerAttr>(factor_attr);
      assert(factor && "factors must be integer attributes");
      return factor.getValue().getSExtValue();
    }
    i++;
  }
  report_fatal_error("axis not defined by this op");
}

LogicalResult AxisProductOp::verify() {
  for (auto operand : getLogicalAxes()) {
    auto defining_op = operand.getDefiningOp();
    if (!isa<CommAxisOpInterface>(defining_op)) {
      return emitOpError() << "requires all factors to be defined by ops "
                           << "implementing the CommAxisOpInterface";
    }
  }
  // TODO disjointness
  return mlir::success();
}

int64_t AxisProductOp::getAxisSize(::mlir::Value axis) {
  // This op defines a single value, so just check if the
  // proper value is passed.
  if (getLogicalAxis() != axis) {
    report_fatal_error("axis not defined by this op");
  }
  int64_t size = 1;
  for (auto operand : getLogicalAxes()) {
    // get the defining op and assert it should implement
    // the CommAxisOpInterface
    auto defining_op = operand.getDefiningOp();
    auto comm_axis_op = dyn_cast<CommAxisOpInterface>(defining_op);
    size *= comm_axis_op.getAxisSize(operand);
  }
  return size;
}

LogicalResult RegionComputationOp::verify() {
  // TODO
  return mlir::success();
}

} // namespace mlir::enzyme::distributed
#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedOps.cpp.inc"