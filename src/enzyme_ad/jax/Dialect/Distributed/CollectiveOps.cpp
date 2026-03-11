#include "Dialect.h"

namespace mlir::enzyme::distributed {

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

  FailureOr<int64_t> meshSize = submeshDefOp.getMeshSize();
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

mlir::Value CollectiveOp::getHandle() { return getToken(); }

// Send, Recv, and SendRecv op interface
llvm::SmallVector<mlir::Value> SendOp::happensAfter() {
  return {}; // sending is first in the chain
}
llvm::SmallVector<mlir::Value> SendOp::simultaneousWith() {
  return {getToken()}; // satisfies its send token
}
bool SendOp::concurrentWith(Operation *other) { return false; }
llvm::SmallVector<mlir::Value> RecvOp::happensAfter() {
  return {getToken()}; // receiving happens after the token is satisfied
}
llvm::SmallVector<mlir::Value> RecvOp::simultaneousWith() {
  return {}; // receiving is last in the chain
}
bool RecvOp::concurrentWith(Operation *other) { return false; }

// SendRecv behaves differently since it is a synchronous protocol
llvm::SmallVector<mlir::Value> SendRecvOp::happensAfter() { return {}; }
llvm::SmallVector<mlir::Value> SendRecvOp::simultaneousWith() {
  // Need to walk the token back to its defining collective- it could be defined
  // by a parts op.
  auto tok = resolveCollectiveTokenToRootCollective(getToken()).asOpResult();
  assert(tok.getDefiningOp<CollectiveOp>() &&
         "SendRecv token must be ultimately defined by a CollectiveOp");
  return {tok}; // satisfies its send/recv token
}
bool SendRecvOp::concurrentWith(Operation *other) {
  // Must commute with other SendRecvOps on the same token so long as there is
  // no SSA dependency between them
  if (auto otherSendRecv = dyn_cast<SendRecvOp>(other)) {
    if (getToken() == otherSendRecv.getToken()) {
      // Check for SSA dependencies
      for (Value operand : getOperands()) {
        if (llvm::is_contained(otherSendRecv.getOperands(), operand)) {
          return false; // There is an SSA dependency, cannot commute
        }
      }
      for (Value result : getOperation()->getResults()) {
        if (llvm::is_contained(otherSendRecv.getOperation()->getResults(),
                               result)) {
          return false; // There is an SSA dependency, cannot commute
        }
      }
      return true; // No SSA dependencies, can commute
    }
  }
  return false; // Don't care about commuting with other ops
}

} // namespace mlir::enzyme::distributed
