#include "Dialect.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::enzyme::distributed {

using llvm::report_fatal_error;

int64_t AxisAllToAllOp::getPhysicalAxisSize() {
  return static_cast<int64_t>(getAxisSize());
}

LogicalResult AxisFactorOp::verify() {
  FailureOr<PhysicalCommAxisOpInterface> physicalAxis =
      resolveSymbolOpFromAttr<PhysicalCommAxisOpInterface>(
          *this, (*this)->getAttr("physical_axis"));
  if (failed(physicalAxis)) {
    return emitOpError() << "references unknown physical axis symbol "
                         << (*this)->getAttr("physical_axis");
  }

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
      mlir::emitError(*location) << "missing factor size attribute";
    return mlir::failure();
  }

  inferredReturnTypes.reserve(adaptor.getFactors().size());
  for (int i = 0; i < adaptor.getFactors().size(); ++i) {
    inferredReturnTypes.push_back(LogicalCommAxisType::get(context));
  }
  return mlir::success();
}

int64_t AxisFactorOp::getAxisSize(TypedOpResult<LogicalCommAxisType> axis) {
  auto ax = axis.asOpResult();
  if (ax.getDefiningOp() != getOperation()) {
    report_fatal_error("axis not defined by this op");
  }
  unsigned i = ax.getResultNumber();
  // zero, unless we add more results in the future.
  unsigned offset = getLogicalAxes()[0].getResultNumber();
  i -= offset;
  auto factor_attr = getFactors()[i];
  auto factor = dyn_cast<IntegerAttr>(factor_attr);
  assert(factor && "factors must be integer attributes");
  return factor.getValue().getSExtValue();
}

void AxisFactorOp::resolveToAtomicFactors(
    TypedOpResult<LogicalCommAxisType> typed_axis,
    llvm::SmallVectorImpl<TypedOpResult<LogicalCommAxisType>> &atomicFactors) {
  auto axis = typed_axis.asOpResult();
  assert(axis.getOwner() == getOperation() &&
         "cannot resolve atomic factors for axis not defined by this "
         "AxisFactorOp");
  atomicFactors.push_back(axis);
}

LogicalResult AxisProductOp::verify() {
  for (auto operand : getLogicalAxes()) {
    auto defining_op = operand.getDefiningOp();
    if (!isa<LogicalCommAxisOpInterface>(defining_op)) {
      return emitOpError() << "requires all factors to be defined by ops "
                           << "implementing the LogicalCommAxisOpInterface";
    }
  }

  llvm::SmallVector<TypedOpResult<LogicalCommAxisType>> atomicFactors;
  resolveLogicalAxisToAtomicFactors(getLogicalAxis(), atomicFactors);
  // Disjointness of factors: all factors refering to the same symbol/physical
  // axis should be defined by the same factor op and should be distinct values.
  llvm::SmallDenseMap<Attribute, llvm::SmallVector<TypedOpResult<LogicalCommAxisType>>> factorGroups;

  for (auto atomicFactor : atomicFactors) {
    auto atomicFactorResult = atomicFactor.asOpResult();
    auto defining_op = atomicFactorResult.getDefiningOp();
    auto axisFactorOp = cast<AxisFactorOp>(defining_op);
    auto physicalAxisAttr = axisFactorOp.getPhysicalAxisAttr();
    if (factorGroups.count(physicalAxisAttr)) {
      // Check if the atomic factor is already in the group
      auto &group = factorGroups[physicalAxisAttr];
      for (auto existingFactor : group) {
        auto existingFactorResult = existingFactor.asOpResult();
        if (existingFactorResult == atomicFactorResult) {
          return emitOpError() << "logical axis has duplicate atomic factors "
                               << "referring to the same physical axis";
        }
        if (existingFactorResult.getDefiningOp() != axisFactorOp) {
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

int64_t AxisProductOp::getAxisSize(TypedOpResult<LogicalCommAxisType> typed_axis) {
  auto axis = typed_axis.asOpResult();
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

void AxisProductOp::resolveToAtomicFactors(
    TypedOpResult<LogicalCommAxisType> typed_axis,
    llvm::SmallVectorImpl<TypedOpResult<LogicalCommAxisType>> &atomicFactors) {
  auto axis = typed_axis.asOpResult();
  if (getLogicalAxis() != axis) {
    emitOpError()
        << "cannot resolve atomic factors for axis not defined by this "
        << "AxisProductOp";
    return;
  }

  for (auto operandAxis : getLogicalAxes()) {
    resolveLogicalAxisToAtomicFactors(operandAxis, atomicFactors);
  }
}

} // namespace mlir::enzyme::distributed
