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

} // namespace mlir::enzyme::distributed
