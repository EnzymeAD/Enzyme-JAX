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

TypedOpResult<CollectiveTokenType> resolveCollectiveTokenToRootCollective(
    TypedOpResult<CollectiveTokenType> token) {
  Operation *definingOp = token.asOpResult().getDefiningOp();
  assert(definingOp && "collective token must be defined by an operation");

  if (auto collectiveOp = dyn_cast<CollectiveOp>(definingOp))
    return TypedOpResult<CollectiveTokenType>(collectiveOp.getToken());

  if (auto partsOp = dyn_cast<SubmeshCollectivePartsOp>(definingOp)) {
    return resolveCollectiveTokenToRootCollective(
        TypedOpResult<CollectiveTokenType>(partsOp.getCollective()));
  }

  definingOp->emitOpError()
      << "does not define a collective token rooted in CollectiveOp";
  assert(false &&
         "collective token must be defined by CollectiveOp or "
         "SubmeshCollectivePartsOp");
  return token;
}
} // namespace mlir::enzyme::distributed
