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
    inferredReturnTypes.push_back(MessageTokenType::get(context));
  }

  return success();
}

mlir::Value CollectiveOp::getHandle() { return getToken(); }

// Send, Recv, and Transfer op interface
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

llvm::SmallVector<mlir::Value> TransferOp::happensAfter() { return {}; }
llvm::SmallVector<mlir::Value> TransferOp::simultaneousWith() {
  return {getToken()};
}
bool TransferOp::concurrentWith(Operation *other) { return false; }

LogicalResult TransferOp::verify() {
  auto meshComputation = (*this)->getParentOfType<MeshComputationOp>();
  if (!meshComputation) {
    return emitOpError() << "must be nested in a distributed.MeshComputation";
  }

  Region *parentRegion = (*this)->getParentRegion();
  for (uint32_t i = 0; i < meshComputation.getNumCommunicationBodies(); ++i) {
    if (&meshComputation.getCommunicationBody(i) == parentRegion) {
      return success();
    }
  }

  return emitOpError()
         << "must be placed in a communication region of its parent "
            "distributed.MeshComputation";
}

} // namespace mlir::enzyme::distributed
