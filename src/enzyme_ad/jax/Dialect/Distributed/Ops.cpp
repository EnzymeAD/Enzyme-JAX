#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/TypeSwitch.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "Dialect.h"

using llvm::report_fatal_error;

namespace mlir::enzyme::distributed {

LogicalResult PhysicalMeshOp::verify() {
  for (auto axisRef : getAxes()) {
    if (failed(resolvePhysicalAxisInterfaceFromAttr(*this, axisRef)))
      return failure();
  }
  return mlir::success();
}

LogicalResult LogicalMeshOp::verify() {
  if (!isLogicalMeshDisjoint(*this)) {
    return emitOpError()
           << "requires mesh factors to be disjoint, and all factors for the "
           << "same physical axis to come from a single factorization op";
  }
  return mlir::success();
}

int64_t AxisAllToAllOp::getPhysicalAxisSize() {
  return static_cast<int64_t>(getAxisSize());
}

LogicalResult AxisFactorOp::verify() {
  FailureOr<PhysicalCommAxisOpInterface> physicalAxis =
      resolvePhysicalAxisInterfaceFromAttr(*this,
                                           (*this)->getAttr("physical_axis"));
  if (failed(physicalAxis))
    return failure();

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

  int64_t physicalAxisSize = physicalAxis->getPhysicalAxisSize();
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
  for (int i = 0; i < adaptor.getFactors().size(); ++i) {
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

  llvm::SmallVector<Value> atomicFactors;
  if (failed(resolveLogicalAxisToAtomicFactors(getLogicalAxis(), atomicFactors))) {
    return emitOpError() << "logical axis does not resolve to atomic factors";
  }
  // Disjointness of factors: all factors refering to the same symbol/physical
  // axis should be defined by the same factor op and should be distinct values.
  llvm::SmallDenseMap<Attribute, llvm::SmallVector<Value>> factorGroups;

  for (auto atomicFactor : atomicFactors) {
    auto defining_op = atomicFactor.getDefiningOp();
    auto axisFactorOp = cast<AxisFactorOp>(defining_op);
    auto physicalAxisAttr = axisFactorOp.getPhysicalAxisAttr();
    if (factorGroups.count(physicalAxisAttr)) {
        // Check if the atomic factor is already in the group
        auto &group = factorGroups[physicalAxisAttr];
        for (auto existingFactor : group) {
          if (existingFactor == atomicFactor) {
            return emitOpError() << "logical axis has duplicate atomic factors "
                                 << "referring to the same physical axis";
          }
          if (existingFactor.getDefiningOp() != axisFactorOp) {
            return emitOpError() << "logical axis has atomic factors referring "
                                 << "to the same physical axis but defined by "
                                 << "different factorization ops";
          }
        }
        group.push_back(atomicFactor);
    } else {
      factorGroups[physicalAxisAttr].push_back(atomicFactor);
    }
  }

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

LogicalResult SubmeshCollectivePartsOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  SubmeshCollectivePartsOpAdaptor adaptor(operands, attributes, properties,
                                          regions);
  if (!adaptor.getSubmesh()) {
    if (location)
      mlir::emitError(*location) << "missing submesh operand";
    return failure();
  }

  auto submeshDefOp = adaptor.getSubmesh().getDefiningOp<LogicalMeshOp>();
  if (!submeshDefOp) {
    if (location) {
      mlir::emitError(*location)
          << "requires submesh operand to be defined by LogicalMeshOp";
    }
    return failure();
  }

  FailureOr<int64_t> meshSize = getLogicalMeshSize(submeshDefOp);
  if (failed(meshSize)) {
    if (location)
      mlir::emitError(*location) << "failed to determine submesh size";
    return failure();
  }

  inferredReturnTypes.reserve(static_cast<size_t>(2 * (*meshSize)));
  for (int64_t i = 0; i < *meshSize; ++i) {
    inferredReturnTypes.push_back(CollectiveTokenType::get(context));
  }

  return success();
}

} // namespace mlir::enzyme::distributed
#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedOps.cpp.inc"