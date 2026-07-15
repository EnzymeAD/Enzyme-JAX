#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/TypeSwitch.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "Dialect.h"

using llvm::report_fatal_error;

namespace mlir::enzyme::distributed {

LogicalResult PhysicalMeshOp::verify() {
  for (auto axisRef : getAxes()) {
    auto axisSymRef = dyn_cast<FlatSymbolRefAttr>(axisRef);
    if (!axisSymRef)
      return emitOpError() << "requires axes to be flat symbol refs";
    Operation *axisOp = SymbolTable::lookupNearestSymbolFrom(*this, axisSymRef);
    if (!axisOp)
      return emitOpError() << "references unknown physical axis symbol "
                           << axisSymRef;
    if (!isa<PhysicalCommAxisOpInterface>(axisOp)) {
      return emitOpError() << "requires all referenced axes to implement "
                           << "PhysicalCommAxisOpInterface";
    }
  }
  return mlir::success();
}

int64_t AxisAllToAllOp::getPhysicalAxisSize() {
  return static_cast<int64_t>(getAxisSize());
}

LogicalResult AxisFactorOp::verify() {
  auto physicalAxisRef = (*this)->getAttrOfType<FlatSymbolRefAttr>("physical_axis");
  if (!physicalAxisRef)
    return emitOpError() << "requires physical_axis symbol reference";

  Operation *physicalAxisOp =
      SymbolTable::lookupNearestSymbolFrom(*this, physicalAxisRef);
  if (!physicalAxisOp)
    return emitOpError() << "references unknown physical axis symbol "
                         << physicalAxisRef;

  auto physicalAxis = dyn_cast<PhysicalCommAxisOpInterface>(physicalAxisOp);
  if (!physicalAxis)
    return emitOpError() << "requires physical_axis to reference an op "
                         << "implementing PhysicalCommAxisOpInterface";

  auto factors = getFactors();
  int64_t factorProduct = 1;
  for (auto factor_attr : factors) {
    if (auto factor = dyn_cast<IntegerAttr>(factor_attr)) {
      int64_t factorValue = factor.getValue().getSExtValue();
      if (factorValue <= 0) {
        return emitOpError() << "requires all factors to be > 0";
      }
      factorProduct *= factorValue;
    } else {
      return emitOpError() << "requires all factors to be integer attributes";
    }
  }

  int64_t physicalAxisSize = physicalAxis.getPhysicalAxisSize();
  if (factorProduct != physicalAxisSize) {
    return emitOpError()
           << "requires product(factors) == referenced physical axis size ("
           << factorProduct << " != " << physicalAxisSize << ")";
  }

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
    if (!isa<LogicalCommAxisOpInterface>(defining_op)) {
      return emitOpError() << "requires all factors to be defined by ops "
                           << "implementing the LogicalCommAxisOpInterface";
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
    // the LogicalCommAxisOpInterface
    auto defining_op = operand.getDefiningOp();
    auto comm_axis_op = dyn_cast<LogicalCommAxisOpInterface>(defining_op);
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