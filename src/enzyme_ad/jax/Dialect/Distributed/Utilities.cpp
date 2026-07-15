#include "Dialect.h"

namespace mlir::enzyme::distributed {

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
