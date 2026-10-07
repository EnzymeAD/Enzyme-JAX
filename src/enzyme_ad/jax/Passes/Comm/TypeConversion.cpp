#include "TypeConversion.h"
#include "src/enzyme_ad/jax/Dialect/Comm/Dialect.h"

namespace mlir::comm {

Type convertMpiComm(MpiCommType type) {
  return RankedTensorType::get({}, IntegerType::get(type.getContext(), 64));
}

Type convertMpiOp(MpiOpType type) {
  return RankedTensorType::get({}, IntegerType::get(type.getContext(), 64));
}

Type convertMpiRequest(MpiRequestType type) {
  return RankedTensorType::get({}, IntegerType::get(type.getContext(), 64));
}

StablehloTypeConverter::StablehloTypeConverter() {
  addConversion([](Type type) { return type; });
  addConversion(convertMpiComm);
  addConversion(convertMpiOp);
  addConversion(convertMpiRequest);
}

} // namespace mlir::comm
