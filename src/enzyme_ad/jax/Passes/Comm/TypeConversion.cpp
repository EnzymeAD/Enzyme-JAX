#include "TypeConversion.h"
#include "src/enzyme_ad/jax/Dialect/Comm/Dialect.h"

namespace mlir::comm {

Type convertMpiComm(MpiCommType type) {
  return RankedTensorType::get({}, IntegerType::get(type.getContext(), 64));
}

Type convertMpiOp(MpiOpType type) {
  return RankedTensorType::get({}, IntegerType::get(type.getContext(), 64));
}

Type convertMpiRequest(MpiRequestType type, unsigned requestWidth) {
  return RankedTensorType::get(
      {}, IntegerType::get(type.getContext(), requestWidth));
}

StablehloTypeConverter::StablehloTypeConverter(unsigned requestWidth) {
  addConversion([](Type type) { return type; });
  addConversion(convertMpiComm);
  addConversion(convertMpiOp);
  addConversion([requestWidth](MpiRequestType type) {
    return convertMpiRequest(type, requestWidth);
  });
}

Type convertCommMpiToNccl(MpiCommType type) {
  return NcclCommType::get(type.getContext());
}

MpiToNcclTypeConverter::MpiToNcclTypeConverter() {
  addConversion([](Type type) { return type; });
  addConversion(convertCommMpiToNccl);
}

} // namespace mlir::comm
