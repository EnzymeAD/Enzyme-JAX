//===- ConversionUtils.cpp - Utilities for conversions --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/ConversionUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace enzyme {

Value extractLimb(Value tensor, int limbIndex, OpBuilder &builder, Location loc,
                  StringRef concatDimension) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(tensor.getType())) {
    if (isa<ComplexType>(tensorTy.getElementType())) {
      if (limbIndex == 0)
        return builder.create<stablehlo::RealOp>(loc, tensor);
      else
        return builder.create<stablehlo::ImagOp>(loc, tensor);
    }
  }
  bool isTuple = concatDimension == "tuple";
  bool isFirst = concatDimension == "first";
  if (auto castOp = tensor.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (isTuple) {
      if (castOp.getNumOperands() > 1)
        return castOp.getOperand(limbIndex);
      if (castOp.getNumOperands() == 1 &&
          isa<TupleType>(castOp.getOperand(0).getType())) {
        Value tupleVal = castOp.getOperand(0);
        if (auto tupleOp = tupleVal.getDefiningOp<stablehlo::TupleOp>()) {
          return tupleOp.getOperand(limbIndex);
        }
        return builder.create<stablehlo::GetTupleElementOp>(
            loc, tupleVal, builder.getI32IntegerAttr(limbIndex));
      }
    } else {
      Value expanded = castOp.getOperand(0);
      auto expandedType = dyn_cast<RankedTensorType>(expanded.getType());
      auto resultType = dyn_cast<RankedTensorType>(tensor.getType());
      if (expandedType && resultType &&
          expandedType.getRank() > resultType.getRank()) {
        SmallVector<int64_t> sliceShape =
            llvm::to_vector(expandedType.getShape());
        int dimToSlice = isFirst ? 0 : expandedType.getRank() - 1;
        sliceShape[dimToSlice] = 1;

        SmallVector<int64_t> startIndices(expandedType.getRank(), 0);
        SmallVector<int64_t> limitIndices =
            llvm::to_vector(expandedType.getShape());
        startIndices[dimToSlice] = limbIndex;
        limitIndices[dimToSlice] = limbIndex + 1;

        SmallVector<int64_t> strides(expandedType.getRank(), 1);

        return builder.create<stablehlo::SliceOp>(
            loc,
            RankedTensorType::get(sliceShape, expandedType.getElementType()),
            expanded, builder.getDenseI64ArrayAttr(startIndices),
            builder.getDenseI64ArrayAttr(limitIndices),
            builder.getDenseI64ArrayAttr(strides));
      }
    }
  }

  if (isTuple) {
    if (auto tupleOp = tensor.getDefiningOp<stablehlo::TupleOp>()) {
      return tupleOp.getOperand(limbIndex);
    }
    return builder.create<stablehlo::GetTupleElementOp>(
        loc, tensor, builder.getI32IntegerAttr(limbIndex));
  }

  if (auto concatOp = tensor.getDefiningOp<stablehlo::ConcatenateOp>()) {
    auto type =
        cast<RankedTensorType>(tensor.getType()); // use result type to get rank
    int concatDim = isFirst ? 0 : type.getRank() - 1;
    if (concatOp.getDimension() == concatDim &&
        concatOp.getOperands().size() == 2) { // TODO expansionSize
      return concatOp.getOperand(limbIndex);
    }
  }

  // Fallback for normal tensors
  bool isTuple_fallback = concatDimension == "tuple";
  bool isFirst_fallback = concatDimension == "first";
  if (isTuple_fallback) {
    return builder.create<stablehlo::GetTupleElementOp>(
        loc, tensor, builder.getI32IntegerAttr(limbIndex));
  }

  auto type = cast<RankedTensorType>(tensor.getType());
  int concatDim = isFirst_fallback ? 0 : type.getRank() - 1;

  SmallVector<int64_t> sliceShape = llvm::to_vector(type.getShape());
  sliceShape[concatDim] = 1;

  SmallVector<int64_t> startIndices(type.getRank(), 0);
  SmallVector<int64_t> limitIndices = llvm::to_vector(type.getShape());
  startIndices[concatDim] = limbIndex;
  limitIndices[concatDim] = limbIndex + 1;

  SmallVector<int64_t> strides(type.getRank(), 1);

  return builder.create<stablehlo::SliceOp>(
      loc, RankedTensorType::get(sliceShape, type.getElementType()),
      tensor, builder.getDenseI64ArrayAttr(startIndices),
      builder.getDenseI64ArrayAttr(limitIndices),
      builder.getDenseI64ArrayAttr(strides));
}

Value packLimbs(ArrayRef<Value> limbs, OpBuilder &builder, Location loc,
                StringRef concatDimension) {
  bool isTuple = concatDimension == "tuple";
  bool isFirst = concatDimension == "first";
  if (isTuple) {
    SmallVector<Type> types;
    for (auto limb : limbs)
      types.push_back(limb.getType());
    return builder.create<stablehlo::TupleOp>(
        loc, TupleType::get(limbs[0].getContext(), types), limbs);
  }
  auto type = cast<RankedTensorType>(limbs[0].getType());

  if (type.getRank() == 0) {
    SmallVector<Value> expandedLimbs;
    SmallVector<int64_t> expandedShape = {1};
    auto expandedType =
        RankedTensorType::get(expandedShape, type.getElementType());
    for (auto limb : limbs) {
      expandedLimbs.push_back(builder.create<stablehlo::BroadcastInDimOp>(
          loc, expandedType, limb, builder.getDenseI64ArrayAttr({})));
    }
    SmallVector<int64_t> outShape = {static_cast<int64_t>(limbs.size())};
    return builder.create<stablehlo::ConcatenateOp>(
        loc, RankedTensorType::get(outShape, type.getElementType()),
        expandedLimbs, 0);
  }

  int concatDim = isFirst ? 0 : type.getRank() - 1;

  SmallVector<int64_t> outShape = llvm::to_vector(type.getShape());
  outShape[concatDim] = limbs.size();

  return builder.create<stablehlo::ConcatenateOp>(
      loc, RankedTensorType::get(outShape, type.getElementType()), limbs,
      concatDim);
}

Value packLimbs(Value high, Value low, OpBuilder &builder, Location loc,
                StringRef concatDimension) {
  return packLimbs({high, low}, builder, loc, concatDimension);
}

} // namespace enzyme
} // namespace mlir
