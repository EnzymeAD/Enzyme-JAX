#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace enzyme {

template <typename T> inline Attribute makeAttr(mlir::Type elemType, T val) {
  if (auto TT = dyn_cast<RankedTensorType>(elemType))
    return SplatElementsAttr::get(
        TT, ArrayRef(makeAttr<T>(TT.getElementType(), val)));

  if (isa<FloatType>(elemType)) {
    return FloatAttr::get(elemType, val);
  } else if (isa<IntegerType>(elemType)) {
    return IntegerAttr::get(elemType, val);
  } else if (auto complexType = dyn_cast<ComplexType>(elemType)) {
    auto elementType = complexType.getElementType();
    auto realAttr = makeAttr(elementType, val);
    auto imagAttr = makeAttr(elementType, 0);

    SmallVector<Attribute> complexVals = {realAttr, imagAttr};
    return ArrayAttr::get(elemType.getContext(), complexVals);
  } else {
    llvm_unreachable("Unsupported type");
  }
}

template <> inline Attribute makeAttr(mlir::Type elemType, llvm::APFloat val) {
  if (auto TT = dyn_cast<RankedTensorType>(elemType))
    return SplatElementsAttr::get(
        TT, ArrayRef(makeAttr<llvm::APFloat>(TT.getElementType(), val)));

  return FloatAttr::get(elemType, val);
}

} // namespace enzyme
} // namespace mlir
