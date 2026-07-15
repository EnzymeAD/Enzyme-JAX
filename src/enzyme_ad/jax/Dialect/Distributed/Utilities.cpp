#include "Dialect.h"

namespace mlir::enzyme::distributed {

namespace {

bool areAtomicFactorsDisjoint(
    ArrayRef<TypedOpResult<LogicalCommAxisType>> atomicFactors) {
  llvm::SmallDenseSet<OpResult> seenAtomicFactors;
  llvm::SmallDenseMap<Attribute, Operation *> physicalAxisToFactorization;

  for (TypedOpResult<LogicalCommAxisType> atomicFactor : atomicFactors) {
    OpResult factorResult = atomicFactor.asOpResult();
    if (!seenAtomicFactors.insert(factorResult).second) {
      return false;
    }

    auto axisFactorOp =
        dyn_cast_or_null<AxisFactorOp>(factorResult.getDefiningOp());
    if (!axisFactorOp) {
      return false;
    }

    Attribute physicalAxis = axisFactorOp.getPhysicalAxisAttr();
    Operation *&firstFactorization = physicalAxisToFactorization[physicalAxis];
    if (!firstFactorization) {
      firstFactorization = axisFactorOp.getOperation();
      continue;
    }

    if (firstFactorization != axisFactorOp.getOperation()) {
      return false;
    }
  }

  return true;
}

} // namespace

void resolveLogicalAxisToAtomicFactors(
    TypedOpResult<LogicalCommAxisType> logicalAxis,
    SmallVectorImpl<TypedOpResult<LogicalCommAxisType>> &atomicFactors) {
  auto ax = logicalAxis.asOpResult();
  Operation *definingOp = ax.getDefiningOp();

  auto axisInterface = cast<LogicalCommAxisOpInterface>(definingOp);
  axisInterface.resolveToAtomicFactors(logicalAxis, atomicFactors);
}

int getAxisSize(TypedOpResult<LogicalCommAxisType> logicalAxis) {
  auto ax = logicalAxis.asOpResult();
  Operation *definingOp = ax.getDefiningOp();

  auto axisInterface = cast<LogicalCommAxisOpInterface>(definingOp);
  return axisInterface.getAxisSize(logicalAxis);
}

bool areLogicalAxesDisjoint(ValueRange logicalAxes) {
  llvm::SmallVector<TypedOpResult<LogicalCommAxisType>> atomicFactors;
  for (Value logicalAxis : logicalAxes) {
    if (!isa<LogicalCommAxisType>(logicalAxis.getType())) {
      return false;
    }
    resolveLogicalAxisToAtomicFactors(
        TypedOpResult<LogicalCommAxisType>(logicalAxis), atomicFactors);
  }
  return areAtomicFactorsDisjoint(atomicFactors);
}

} // namespace mlir::enzyme::distributed
