//===- MultiFloatConversion.cpp - Multi-Float Conversion Pass
//-----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "multi-float-conversion"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_MULTIFLOATCONVERSIONPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {

Type getFloatTypeFromString(StringRef typeStr, MLIRContext *context) {
  if (typeStr == "f64")
    return Float64Type::get(context);
  if (typeStr == "f32")
    return Float32Type::get(context);
  if (typeStr == "f16")
    return Float16Type::get(context);
  if (typeStr == "bf16")
    return BFloat16Type::get(context);
  if (typeStr == "f8E4M3FN" || typeStr == "fp8")
    return Float8E4M3FNType::get(context);
  if (typeStr == "f8E5M2")
    return Float8E5M2Type::get(context);
  return nullptr;
}

Value extractLimb(Value tensor, int limbIndex, OpBuilder &builder, Location loc,
                  StringRef concatDimension) {
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

  auto type = cast<RankedTensorType>(tensor.getType());
  SmallVector<int64_t> sliceShape = llvm::to_vector(type.getShape());
  SmallVector<int64_t> startIndices(type.getRank(), 0);
  SmallVector<int64_t> limitIndices = llvm::to_vector(type.getShape());

  int dimToSlice = isFirst ? 0 : type.getRank() - 1;

  sliceShape[dimToSlice] = 1;
  startIndices[dimToSlice] = limbIndex;
  limitIndices[dimToSlice] = limbIndex + 1;

  SmallVector<int64_t> strides(type.getRank(), 1);

  auto sliceOp = builder.create<stablehlo::SliceOp>(
      loc, RankedTensorType::get(sliceShape, type.getElementType()), tensor,
      builder.getDenseI64ArrayAttr(startIndices),
      builder.getDenseI64ArrayAttr(limitIndices),
      builder.getDenseI64ArrayAttr(strides));

  return sliceOp;
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

template <typename OpFunc>
Value applyElementwiseOpToLimbs(Value tensor, OpBuilder &builder, Location loc,
                                StringRef concatDimension, OpFunc opFunc) {
  bool isTuple = concatDimension == "tuple";
  if (isTuple) {
    Value hi = extractLimb(tensor, 0, builder, loc, concatDimension);
    Value lo = extractLimb(tensor, 1, builder, loc, concatDimension);
    Value res_hi = opFunc(hi);
    Value res_lo = opFunc(lo);
    return packLimbs(res_hi, res_lo, builder, loc, concatDimension);
  } else {
    return opFunc(tensor);
  }
}

Value convertToMultifloat(DenseElementsAttr val, OpBuilder &b, Location loc,
                          Type tgtTy, StringRef concatDimension,
                          int expansionSize) {
  SmallVector<Value> limbs;
  DenseElementsAttr rem = val;
  RankedTensorType tensorType = cast<RankedTensorType>(val.getType());
  RankedTensorType outType =
      RankedTensorType::get(tensorType.getShape(), tgtTy);

  SmallVector<int64_t> expandedShape;
  if (concatDimension == "first") {
    expandedShape.push_back(1);
    for (auto dim : tensorType.getShape())
      expandedShape.push_back(dim);
  } else {
    for (auto dim : tensorType.getShape())
      expandedShape.push_back(dim);
    expandedShape.push_back(1);
  }
  auto expandedType = RankedTensorType::get(expandedShape, tgtTy);

  for (int i = 0; i < expansionSize; ++i) {
    DenseElementsAttr limb = nullptr;
    if (auto splat = dyn_cast<SplatElementsAttr>(rem)) {
      auto newVal = splat.getSplatValue<APFloat>();
      bool losesInfo;
      newVal.convert(cast<FloatType>(tgtTy).getFloatSemantics(),
                     APFloat::rmNearestTiesToEven, &losesInfo);
      limb = SplatElementsAttr::get(outType, newVal);
    } else {
      SmallVector<Attribute> convertedAttrs;
      for (auto val : rem.getValues<APFloat>()) {
        bool losesInfo;
        APFloat newVal = val;
        newVal.convert(cast<FloatType>(tgtTy).getFloatSemantics(),
                       APFloat::rmNearestTiesToEven, &losesInfo);
        convertedAttrs.push_back(
            b.getFloatAttr(tgtTy, newVal.convertToDouble()));
      }
      limb = DenseElementsAttr::get(outType, convertedAttrs);
    }

    if (concatDimension == "tuple") {
      limbs.push_back(b.create<stablehlo::ConstantOp>(loc, limb));
    } else if (expansionSize > 1) {
      limbs.push_back(
          b.create<stablehlo::ConstantOp>(loc, limb.reshape(expandedType)));
    } else {
      limbs.push_back(b.create<stablehlo::ConstantOp>(loc, limb));
    }

    if (i < expansionSize - 1) {
      auto limbSplat = dyn_cast<SplatElementsAttr>(limb);
      auto remSplat = dyn_cast<SplatElementsAttr>(rem);
      if (limbSplat && remSplat) {
        auto limbBack = limbSplat.getSplatValue<APFloat>();
        bool losesInfo;
        limbBack.convert(
            cast<FloatType>(tensorType.getElementType()).getFloatSemantics(),
            APFloat::rmNearestTiesToEven, &losesInfo);
        auto remFlt = remSplat.getSplatValue<APFloat>();
        remFlt.subtract(limbBack, APFloat::rmNearestTiesToEven);
        rem = SplatElementsAttr::get(tensorType, remFlt);
      } else {
        SmallVector<Attribute> newRems;
        for (auto [limbBack, remFlt] : llvm::zip_equal(
                 limb.getValues<APFloat>(), rem.getValues<APFloat>())) {
          bool losesInfo;
          limbBack.convert(
              cast<FloatType>(tensorType.getElementType()).getFloatSemantics(),
              APFloat::rmNearestTiesToEven, &losesInfo);
          remFlt.subtract(limbBack, APFloat::rmNearestTiesToEven);
          newRems.push_back(
              FloatAttr::get(tensorType.getElementType(), remFlt));
        }
        rem = DenseElementsAttr::get(tensorType, newRems);
      }
    }
  }

  if (expansionSize == 1)
    return limbs[0];

  return packLimbs(limbs, b, loc, concatDimension);
}

Value convertToMultifloat(Value val, OpBuilder &b, Location loc, Type tgtTy,
                          StringRef concatDimension, int expansionSize) {
  auto tensorType = cast<RankedTensorType>(val.getType());
  SmallVector<Value> limbs;
  Value rem = val;
  for (int i = 0; i < expansionSize; ++i) {
    Value limb = b.create<stablehlo::ConvertOp>(
        loc, RankedTensorType::get(tensorType.getShape(), tgtTy), rem);
    limbs.push_back(limb);
    if (i < expansionSize - 1) {
      Value limbBack = b.create<stablehlo::ConvertOp>(loc, tensorType, limb);
      rem = b.create<stablehlo::SubtractOp>(loc, rem, limbBack);
    }
  }

  if (concatDimension != "tuple") {
    SmallVector<int64_t> expandedShape;
    if (concatDimension == "first") {
      expandedShape.push_back(1);
      for (auto dim : tensorType.getShape())
        expandedShape.push_back(dim);
    } else {
      for (auto dim : tensorType.getShape())
        expandedShape.push_back(dim);
      expandedShape.push_back(1);
    }
    auto expandedType = RankedTensorType::get(expandedShape, tgtTy);
    SmallVector<Value> reshapedLimbs;
    for (auto limb : limbs) {
      reshapedLimbs.push_back(
          b.create<stablehlo::ReshapeOp>(loc, expandedType, limb));
    }
    return packLimbs(reshapedLimbs, b, loc, concatDimension);
  }
  return packLimbs(limbs, b, loc, concatDimension);
}

Value convertFromMultifloat(Value packedVal, OpBuilder &b, Location loc,
                            Type srcTy, StringRef concatDimension,
                            Value &zero) {
  if (concatDimension != "tuple") {
    auto tensorType = cast<RankedTensorType>(packedVal.getType());
    auto f64PackedType = RankedTensorType::get(tensorType.getShape(), srcTy);
    Value packed64 =
        b.create<stablehlo::ConvertOp>(loc, f64PackedType, packedVal);

    int reduceDim =
        (concatDimension == "first") ? 0 : (tensorType.getRank() - 1);

    SmallVector<int64_t> resultShape;
    for (int i = 0; i < tensorType.getRank(); ++i) {
      if (i != reduceDim) {
        resultShape.push_back(tensorType.getShape()[i]);
      }
    }
    auto resultType = RankedTensorType::get(resultShape, srcTy);

    auto scalarType = RankedTensorType::get({}, srcTy);
    auto zeroAttr =
        DenseElementsAttr::get(scalarType, b.getFloatAttr(srcTy, 0.0));
    if (!zero)
      zero = b.create<stablehlo::ConstantOp>(loc, scalarType, zeroAttr);

    auto reduceOp = b.create<stablehlo::ReduceOp>(
        loc, resultType, packed64, zero, b.getDenseI64ArrayAttr({reduceDim}));

    {
      OpBuilder::InsertionGuard guard(b);
      auto &region = reduceOp.getRegion();
      auto *block =
          b.createBlock(&region, {}, {scalarType, scalarType}, {loc, loc});
      b.setInsertionPointToStart(block);
      Value added = b.create<stablehlo::AddOp>(loc, block->getArgument(0),
                                               block->getArgument(1));
      b.create<stablehlo::ReturnOp>(loc, added);
    }

    return reduceOp.getResult(0);
  }

  Value high = extractLimb(packedVal, 0, b, loc, concatDimension);
  Value low = extractLimb(packedVal, 1, b, loc, concatDimension);
  auto f64Type = RankedTensorType::get(
      cast<RankedTensorType>(high.getType()).getShape(), srcTy);
  Value high64 = b.create<stablehlo::ConvertOp>(loc, f64Type, high);
  Value low64 = b.create<stablehlo::ConvertOp>(loc, f64Type, low);
  return b.create<stablehlo::AddOp>(loc, high64, low64);
}

std::pair<Value, Value> twoSum(Value a, Value b, OpBuilder &builder,
                               Location loc) {
  Value sum = builder.create<stablehlo::AddOp>(loc, a, b);
  Value a_prime = builder.create<stablehlo::SubtractOp>(loc, sum, b);
  Value b_prime = builder.create<stablehlo::SubtractOp>(loc, sum, a_prime);
  Value a_err = builder.create<stablehlo::SubtractOp>(loc, a, a_prime);
  Value b_err = builder.create<stablehlo::SubtractOp>(loc, b, b_prime);
  Value err = builder.create<stablehlo::AddOp>(loc, a_err, b_err);
  return {sum, err};
}

std::pair<Value, Value> fastTwoSum(Value a, Value b, OpBuilder &builder,
                                   Location loc) {
  Value sum = builder.create<stablehlo::AddOp>(loc, a, b);
  Value b_prime = builder.create<stablehlo::SubtractOp>(loc, sum, a);
  Value b_err = builder.create<stablehlo::SubtractOp>(loc, b, b_prime);
  return {sum, b_err};
}

std::pair<int, int> getFloatProperties(Type type) {
  auto floatTy = dyn_cast<FloatType>(type);
  if (!floatTy)
    return {0, 0};
  int mantissaWidth = floatTy.getFPMantissaWidth();
  int expBits = floatTy.getWidth() - mantissaWidth;
  return {expBits, mantissaWidth};
}

int getFloatPrecision(Type type) { return getFloatProperties(type).second; }

bool isSubsetFloat(Type a, Type b) {
  auto pA = getFloatProperties(a);
  auto pB = getFloatProperties(b);
  return pA.first <= pB.first && pA.second <= pB.second;
}

Value getSplitConstant(Type type, OpBuilder &builder, Location loc) {
  auto tensorTy = cast<RankedTensorType>(type);
  auto floatTy = cast<FloatType>(tensorTy.getElementType());
  int precision = getFloatPrecision(floatTy);
  if (precision == 0)
    return nullptr;
  int k = (precision + 1) / 2;
  APInt api(256, 1);
  api <<= k;
  api += 1;
  APFloat apVal(floatTy.getFloatSemantics());
  apVal.convertFromAPInt(api, /*isSigned=*/false, APFloat::rmNearestTiesToEven);

  auto attr = builder.getFloatAttr(floatTy, apVal);
  auto splatAttr = SplatElementsAttr::get(tensorTy, attr);
  return builder.create<stablehlo::ConstantOp>(loc, splatAttr);
}

std::pair<Value, Value> split(Value a, OpBuilder &builder, Location loc) {
  Value c_const = getSplitConstant(a.getType(), builder, loc);
  Value c = builder.create<stablehlo::MulOp>(loc, a, c_const);
  Value a_big = builder.create<stablehlo::SubtractOp>(loc, c, a);
  Value a_hi = builder.create<stablehlo::SubtractOp>(loc, c, a_big);
  Value a_lo = builder.create<stablehlo::SubtractOp>(loc, a, a_hi);
  return {a_hi, a_lo};
}

std::pair<Value, Value> twoProdDekker(Value a, Value b, OpBuilder &builder,
                                      Location loc) {
  Value p = builder.create<stablehlo::MulOp>(loc, a, b);
  auto [a_hi, a_lo] = split(a, builder, loc);
  auto [b_hi, b_lo] = split(b, builder, loc);

  Value p1 = builder.create<stablehlo::MulOp>(loc, a_hi, b_hi);
  Value p2 = builder.create<stablehlo::MulOp>(loc, a_hi, b_lo);
  Value p3 = builder.create<stablehlo::MulOp>(loc, a_lo, b_hi);
  Value p4 = builder.create<stablehlo::MulOp>(loc, a_lo, b_lo);

  Value err1 = builder.create<stablehlo::SubtractOp>(loc, p1, p);
  Value err2 = builder.create<stablehlo::AddOp>(
      loc, p2, p3); // Group cross terms for accuracy
  Value err3 = builder.create<stablehlo::AddOp>(loc, err1, err2);
  Value err4 = builder.create<stablehlo::AddOp>(loc, err3, p4);

  return {p, err4};
}

std::pair<Value, Value> multiFloatMul(Value x1, Value x2, Value y1, Value y2,
                                      OpBuilder &builder, Location loc) {
  auto [p00, e00] = twoProdDekker(x1, y1, builder, loc);
  Value p01 = builder.create<stablehlo::MulOp>(loc, x1, y2);
  Value p10 = builder.create<stablehlo::MulOp>(loc, x2, y1);
  Value p01_p10 = builder.create<stablehlo::AddOp>(loc, p01, p10);
  Value e00_new = builder.create<stablehlo::AddOp>(loc, e00, p01_p10);
  return fastTwoSum(p00, e00_new, builder, loc);
}

std::pair<Value, Value> multiFloatAdd(Value x1, Value x2, Value y1, Value y2,
                                      OpBuilder &builder, Location loc) {
  auto [a, b] = twoSum(x1, y1, builder, loc);
  auto [c, d] = twoSum(x2, y2, builder, loc);
  auto [new_a, new_c] = fastTwoSum(a, c, builder, loc);
  Value b2 = builder.create<stablehlo::AddOp>(loc, b, d);
  Value b3 = builder.create<stablehlo::AddOp>(loc, b2, new_c);
  return fastTwoSum(new_a, b3, builder, loc);
}

std::pair<Value, Value> multiFloatDiv(Value x1, Value x2, Value y1, Value y2,
                                      OpBuilder &builder, Location loc) {
  auto tensorType = cast<RankedTensorType>(x1.getType());
  auto floatTy = cast<FloatType>(tensorType.getElementType());
  auto zeroAttr = builder.getFloatAttr(floatTy, 0.0);
  Value zero = builder.create<stablehlo::ConstantOp>(
      loc, SplatElementsAttr::get(tensorType, zeroAttr));

  Value q1 = builder.create<stablehlo::DivOp>(loc, x1, y1);
  auto [p_hi, p_lo] = multiFloatMul(q1, zero, y1, y2, builder, loc);

  Value neg_p_hi = builder.create<stablehlo::NegOp>(loc, p_hi);
  Value neg_p_lo = builder.create<stablehlo::NegOp>(loc, p_lo);
  auto [r_hi, r_lo] = multiFloatAdd(x1, x2, neg_p_hi, neg_p_lo, builder, loc);

  Value q2 = builder.create<stablehlo::DivOp>(loc, r_hi, y1);
  return fastTwoSum(q1, q2, builder, loc);
}

template <typename OpTy>
struct GenericOpConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getResults().empty())
      return failure();

    SmallVector<Value> operands;
    for (auto operand : adaptor.getOperands()) {
      if (!this->getTypeConverter()->isLegal(operand.getType())) {
        if (auto cast =
                operand.template getDefiningOp<UnrealizedConversionCastOp>()) {
          if (this->getTypeConverter()->isLegal(cast.getOperand(0).getType())) {
            operand = cast.getOperand(0);
          } else {
            return failure();
          }
        } else {
          return failure();
        }
      }
      operands.push_back(operand);
    }

    OperationState state(op.getLoc(), OpTy::getOperationName());
    state.addOperands(operands);
    state.addAttributes(op->getAttrs());
    for (unsigned i = 0; i < op->getNumRegions(); ++i) {
      Region &region = op->getRegion(i);
      Region *newRegion = state.addRegion();
      rewriter.inlineRegionBefore(region, *newRegion, newRegion->end());
    }
    state.addTypes(
        this->getTypeConverter()->convertType(op->getResult(0).getType()));
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct ConstantOpConversion
    : public OpConversionPattern<stablehlo::ConstantOp> {
  StringRef concatDimension;
  int expansionSize;
  Type sourceType;
  Type targetType;

  ConstantOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                       StringRef concatDimension, int expansionSize,
                       Type sourceType, Type targetType)
      : OpConversionPattern<stablehlo::ConstantOp>(typeConverter, context),
        concatDimension(concatDimension), expansionSize(expansionSize),
        sourceType(sourceType), targetType(targetType) {}

  LogicalResult
  matchAndRewrite(stablehlo::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto attr = op.getValue();

    auto elementsAttr = dyn_cast<DenseElementsAttr>(attr);
    if (!elementsAttr)
      return failure();

    Type elType = elementsAttr.getElementType();
    if (elType != sourceType)
      return failure();

    auto replacement =
        convertToMultifloat(elementsAttr, rewriter, loc, targetType,
                            concatDimension, expansionSize);
    rewriter.replaceOp(op, replacement);
    return success();
  }
};
std::pair<Value, Value> mfAdd(Value x_hi, Value x_lo, Value y_hi, Value y_lo,
                              OpBuilder &builder, Location loc) {
  auto [a, b] = twoSum(x_hi, y_hi, builder, loc);
  auto [c, d] = twoSum(x_lo, y_lo, builder, loc);
  auto [new_a, new_c] = fastTwoSum(a, c, builder, loc);
  Value b2 = builder.create<stablehlo::AddOp>(loc, b, d);
  Value b3 = builder.create<stablehlo::AddOp>(loc, b2, new_c);
  return fastTwoSum(new_a, b3, builder, loc);
}

std::pair<Value, Value> mfMul(Value x_hi, Value x_lo, Value y_hi, Value y_lo,
                              OpBuilder &builder, Location loc) {
  auto [p00, e00] = twoProdDekker(x_hi, y_hi, builder, loc);
  Value p01 = builder.create<stablehlo::MulOp>(loc, x_hi, y_lo);
  Value p10 = builder.create<stablehlo::MulOp>(loc, x_lo, y_hi);
  Value p01_p10 = builder.create<stablehlo::AddOp>(loc, p01, p10);
  Value e00_new = builder.create<stablehlo::AddOp>(loc, e00, p01_p10);
  return fastTwoSum(p00, e00_new, builder, loc);
}

struct AddOpConversion : public OpConversionPattern<stablehlo::AddOp> {
  AddOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                  Type sourceType, StringRef concatDimension)
      : OpConversionPattern<stablehlo::AddOp>(typeConverter, context),
        sourceType(sourceType), concatDimension(concatDimension) {}

  Type sourceType;
  StringRef concatDimension;

  LogicalResult
  matchAndRewrite(stablehlo::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "AddOpConversion called\n");

    if (isa<stablehlo::ReduceOp>(op->getParentOp())) {
      return failure();
    }
    // Check if single-limb by looking at the converted operand type
    bool isTuple = concatDimension == "tuple";
    if (isTuple) {
      if (!mlir::dyn_cast_or_null<TupleType>(
              adaptor.getOperands()[0].getType())) {
        LLVM_DEBUG(llvm::dbgs() << "AddOpConversion: Found single-limb operand "
                                   "in tuple mode, failing\n");
        return failure();
      }
    } else {
      auto convertedType = mlir::dyn_cast_or_null<RankedTensorType>(
          adaptor.getOperands()[0].getType());
      if (!convertedType)
        return failure();

      int limbDim =
          (concatDimension == "first") ? 0 : convertedType.getRank() - 1;
      if (convertedType.getShape()[limbDim] == 1) {
        LLVM_DEBUG(llvm::dbgs() << "AddOpConversion: Found single-limb operand "
                                   "in dimension mode, failing\n");
        return failure();
      }
    }

    Location loc = op.getLoc();

    Value x1 = extractLimb(adaptor.getOperands()[0], 0, rewriter, loc,
                           concatDimension);
    Value x2 = extractLimb(adaptor.getOperands()[0], 1, rewriter, loc,
                           concatDimension);
    Value y1 = extractLimb(adaptor.getOperands()[1], 0, rewriter, loc,
                           concatDimension);
    Value y2 = extractLimb(adaptor.getOperands()[1], 1, rewriter, loc,
                           concatDimension);

    auto [final_a, final_b] = mfAdd(x1, x2, y1, y2, rewriter, loc);

    Value packed = packLimbs(final_a, final_b, rewriter, loc, concatDimension);
    rewriter.replaceOp(op, packed);
    LLVM_DEBUG(llvm::dbgs() << "AddOpConversion succeeded\n");
    return success();
  }
};

struct MulOpConversion : public OpConversionPattern<stablehlo::MulOp> {
  MulOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                  StringRef concatDimension)
      : OpConversionPattern<stablehlo::MulOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  StringRef concatDimension;

  LogicalResult
  matchAndRewrite(stablehlo::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value x1 = extractLimb(adaptor.getOperands()[0], 0, rewriter, loc,
                           concatDimension);
    Value x2 = extractLimb(adaptor.getOperands()[0], 1, rewriter, loc,
                           concatDimension);
    Value y1 = extractLimb(adaptor.getOperands()[1], 0, rewriter, loc,
                           concatDimension);
    Value y2 = extractLimb(adaptor.getOperands()[1], 1, rewriter, loc,
                           concatDimension);

    auto [final_p, final_e] = mfMul(x1, x2, y1, y2, rewriter, loc);

    Value packed = packLimbs(final_p, final_e, rewriter, loc, concatDimension);
    rewriter.replaceOp(op, packed);
    return success();
  }
};
struct ReduceOpConversion : public OpConversionPattern<stablehlo::ReduceOp> {
  ReduceOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                     StringRef concatDimension, bool preciseReduce,
                     Type sourceType, Type targetType)
      : OpConversionPattern<stablehlo::ReduceOp>(typeConverter, context),
        concatDimension(concatDimension), preciseReduce(preciseReduce),
        sourceType(sourceType), targetType(targetType) {}

  StringRef concatDimension;
  bool preciseReduce;
  Type sourceType;
  Type targetType;

  LogicalResult
  matchAndRewrite(stablehlo::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Block &body = op.getBody().front();
    bool isMax =
        !body.getOperations().empty() && isa<stablehlo::MaxOp>(body.front());
    if (body.getOperations().empty() ||
        (!isa<stablehlo::AddOp>(body.front()) && !isMax)) {
      LLVM_DEBUG(llvm::dbgs() << "ReduceOp: unsupported reduction operation\n");
      return failure();
    }

    auto inputs = adaptor.getInputs();
    auto initValues = adaptor.getInitValues();

    if (inputs.size() != 1) {
      llvm::errs() << "ReduceOpConversion failed at line 695\n";
      return failure();
    }

    Value input = inputs[0];
    Value initValue = initValues[0];

    Value init_hi = extractLimb(initValue, 0, rewriter, loc, concatDimension);
    Value init_lo = extractLimb(initValue, 1, rewriter, loc, concatDimension);

    auto scalarType = RankedTensorType::get({}, targetType);
    if (concatDimension != "tuple") {
      init_hi =
          stablehlo::ReshapeOp::create(rewriter, loc, scalarType, init_hi);
      init_lo =
          stablehlo::ReshapeOp::create(rewriter, loc, scalarType, init_lo);
    }

    if (preciseReduce || isMax) {
      Value input_hi = extractLimb(input, 0, rewriter, loc, concatDimension);
      Value input_lo = extractLimb(input, 1, rewriter, loc, concatDimension);
      SmallVector<Value, 2> newInputs = {input_hi, input_lo};
      SmallVector<Value, 2> newInits = {init_hi, init_lo};
      SmallVector<int64_t> dims;
      for (auto dim : op.getDimensions()) {
        if (concatDimension == "first") {
          dims.push_back(dim + 1);
          continue;
        }
        dims.push_back(dim);
      }
      auto reduceOp =
          rewriter.create<stablehlo::ReduceOp>(loc, newInputs, newInits, dims);

      Block *reduceBlock = new Block();
      reduceOp.getBody().push_back(reduceBlock);
      reduceBlock->addArguments(
          {scalarType, scalarType, scalarType, scalarType},
          {loc, loc, loc, loc});

      auto blockBuilder = OpBuilder::atBlockBegin(reduceBlock);

      Value acc_hi = reduceBlock->getArgument(0);
      Value acc_lo = reduceBlock->getArgument(1);
      Value val_hi = reduceBlock->getArgument(2);
      Value val_lo = reduceBlock->getArgument(3);

      Value final_hi, final_lo;
      if (!isMax) {
        auto [h, l] = mfAdd(acc_hi, acc_lo, val_hi, val_lo, blockBuilder, loc);
        final_hi = h;
        final_lo = l;
      } else {
        // Lexicographical comparison for MaxOp
        Value cmp = blockBuilder.create<stablehlo::CompareOp>(
            loc, acc_hi, val_hi, stablehlo::ComparisonDirection::GT);
        Value eq = blockBuilder.create<stablehlo::CompareOp>(
            loc, acc_hi, val_hi, stablehlo::ComparisonDirection::EQ);
        Value lo_cmp = blockBuilder.create<stablehlo::CompareOp>(
            loc, acc_lo, val_lo, stablehlo::ComparisonDirection::GT);
        Value final_cmp =
            blockBuilder.create<stablehlo::SelectOp>(loc, eq, lo_cmp, cmp);

        final_hi = blockBuilder.create<stablehlo::SelectOp>(loc, final_cmp,
                                                            acc_hi, val_hi);
        final_lo = blockBuilder.create<stablehlo::SelectOp>(loc, final_cmp,
                                                            acc_lo, val_lo);
      }

      blockBuilder.create<stablehlo::ReturnOp>(loc,
                                               ValueRange{final_hi, final_lo});

      Value res_hi = reduceOp.getResult(0);
      Value res_lo = reduceOp.getResult(1);

      Value packed = packLimbs(res_hi, res_lo, rewriter, loc, concatDimension);
      rewriter.replaceOp(op, packed);

      return success();
    } else {
      SmallVector<int64_t> dims;
      for (auto dim : op.getDimensions()) {
        if (concatDimension == "first") {
          dims.push_back(dim + 1);
          continue;
        }
        dims.push_back(dim);
      }

      auto scalarType = RankedTensorType::get({}, targetType);

      auto createReducer = [&](stablehlo::ReduceOp reduceOp) {
        Block *reduceBlock = new Block();
        reduceOp.getBody().push_back(reduceBlock);
        reduceBlock->addArguments({scalarType, scalarType}, {loc, loc});

        auto blockBuilder = OpBuilder::atBlockBegin(reduceBlock);
        Value add = blockBuilder.create<stablehlo::AddOp>(
            loc, scalarType, reduceBlock->getArgument(0),
            reduceBlock->getArgument(1));
        blockBuilder.create<stablehlo::ReturnOp>(loc, add);
      };

      if (concatDimension == "tuple") {
        Value input_hi = extractLimb(input, 0, rewriter, loc, concatDimension);
        Value input_lo = extractLimb(input, 1, rewriter, loc, concatDimension);

        auto reduceOp_hi =
            rewriter.create<stablehlo::ReduceOp>(loc, input_hi, init_hi, dims);
        createReducer(reduceOp_hi);

        auto reduceOp_lo =
            rewriter.create<stablehlo::ReduceOp>(loc, input_lo, init_lo, dims);
        createReducer(reduceOp_lo);

        auto finalF64Type = op.getResult(0).getType();
        Value conv_hi = rewriter.create<stablehlo::ConvertOp>(
            loc, finalF64Type, reduceOp_hi.getResult(0));
        Value conv_lo = rewriter.create<stablehlo::ConvertOp>(
            loc, finalF64Type, reduceOp_lo.getResult(0));
        Value final_sum = rewriter.create<stablehlo::AddOp>(loc, finalF64Type,
                                                            conv_hi, conv_lo);

        rewriter.replaceOp(op, final_sum);
        return success();
      }

      auto reduceOp =
          rewriter.create<stablehlo::ReduceOp>(loc, input, init_hi, dims);
      createReducer(reduceOp);

      Value res_all = reduceOp.getResult(0);
      auto resType = cast<RankedTensorType>(res_all.getType());
      auto resShape = resType.getShape();

      Value res_hi, res_lo;
      if (concatDimension == "first") {
        SmallVector<int64_t> sliceOffsets(resType.getRank(), 0);
        SmallVector<int64_t> sliceLimits(resShape.begin(), resShape.end());
        SmallVector<int64_t> sliceStrides(resType.getRank(), 1);

        sliceLimits[0] = 1;
        res_hi = rewriter.create<stablehlo::SliceOp>(loc, res_all, sliceOffsets,
                                                     sliceLimits, sliceStrides);

        sliceOffsets[0] = 1;
        sliceLimits[0] = 2;
        res_lo = rewriter.create<stablehlo::SliceOp>(loc, res_all, sliceOffsets,
                                                     sliceLimits, sliceStrides);
      } else {
        int lastDim = resType.getRank() - 1;
        SmallVector<int64_t> sliceOffsets(resType.getRank(), 0);
        SmallVector<int64_t> sliceLimits(resShape.begin(), resShape.end());
        SmallVector<int64_t> sliceStrides(resType.getRank(), 1);

        sliceLimits[lastDim] = 1;
        res_hi = rewriter.create<stablehlo::SliceOp>(loc, res_all, sliceOffsets,
                                                     sliceLimits, sliceStrides);

        sliceOffsets[lastDim] = 1;
        sliceLimits[lastDim] = 2;
        res_lo = rewriter.create<stablehlo::SliceOp>(loc, res_all, sliceOffsets,
                                                     sliceLimits, sliceStrides);
      }

      auto finalF64Type = op.getResult(0).getType();
      auto finalF32Type = RankedTensorType::get(
          cast<RankedTensorType>(finalF64Type).getShape(), targetType);

      Value reshaped_hi =
          rewriter.create<stablehlo::ReshapeOp>(loc, finalF32Type, res_hi);
      Value reshaped_lo =
          rewriter.create<stablehlo::ReshapeOp>(loc, finalF32Type, res_lo);

      Value conv_hi =
          rewriter.create<stablehlo::ConvertOp>(loc, finalF64Type, reshaped_hi);
      Value conv_lo =
          rewriter.create<stablehlo::ConvertOp>(loc, finalF64Type, reshaped_lo);

      Value sum = rewriter.create<stablehlo::AddOp>(loc, finalF64Type, conv_hi,
                                                    conv_lo);
      rewriter.replaceOp(op, sum);
      return success();
    }
  }
};

struct SubOpConversion : public OpConversionPattern<stablehlo::SubtractOp> {
  SubOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                  StringRef concatDimension)
      : OpConversionPattern<stablehlo::SubtractOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  StringRef concatDimension;

  LogicalResult
  matchAndRewrite(stablehlo::SubtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value x1 = extractLimb(adaptor.getOperands()[0], 0, rewriter, loc,
                           concatDimension);
    Value x2 = extractLimb(adaptor.getOperands()[0], 1, rewriter, loc,
                           concatDimension);
    Value y1 = extractLimb(adaptor.getOperands()[1], 0, rewriter, loc,
                           concatDimension);
    Value y2 = extractLimb(adaptor.getOperands()[1], 1, rewriter, loc,
                           concatDimension);

    Value neg_y1 = rewriter.create<stablehlo::NegOp>(loc, y1);
    Value neg_y2 = rewriter.create<stablehlo::NegOp>(loc, y2);

    auto [a, b] = twoSum(x1, neg_y1, rewriter, loc);
    auto [c, d] = twoSum(x2, neg_y2, rewriter, loc);
    auto [new_a, new_c] = fastTwoSum(a, c, rewriter, loc);
    Value b2 = rewriter.create<stablehlo::AddOp>(loc, b, d);
    Value b3 = rewriter.create<stablehlo::AddOp>(loc, b2, new_c);
    auto [final_a, final_b] = fastTwoSum(new_a, b3, rewriter, loc);

    Value packed = packLimbs(final_a, final_b, rewriter, loc, concatDimension);
    rewriter.replaceOp(op, packed);
    return success();
  }
};

struct DivOpConversion : public OpConversionPattern<stablehlo::DivOp> {
  DivOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                  StringRef concatDimension, int divSubsteps)
      : OpConversionPattern<stablehlo::DivOp>(typeConverter, context),
        concatDimension(concatDimension), divSubsteps(divSubsteps) {}

  StringRef concatDimension;
  int divSubsteps;

  LogicalResult
  matchAndRewrite(stablehlo::DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value x1 = extractLimb(adaptor.getOperands()[0], 0, rewriter, loc,
                           concatDimension);
    Value x2 = extractLimb(adaptor.getOperands()[0], 1, rewriter, loc,
                           concatDimension);
    Value y1 = extractLimb(adaptor.getOperands()[1], 0, rewriter, loc,
                           concatDimension);
    Value y2 = extractLimb(adaptor.getOperands()[1], 1, rewriter, loc,
                           concatDimension);

    if (divSubsteps == 0) {
      // 1. q_hi = x_hi / y_hi
      Value q_hi = rewriter.create<stablehlo::DivOp>(loc, x1, y1);

      // 2. (p, e) = twoProdDekker(q_hi, y_hi)
      auto [p, e] = twoProdDekker(q_hi, y1, rewriter, loc);

      // 3. rem = x_hi - p - e + x_lo - q_hi * y_lo
      Value neg_p = rewriter.create<stablehlo::NegOp>(loc, p);
      Value neg_e = rewriter.create<stablehlo::NegOp>(loc, e);

      // x_hi - p
      Value rem1 = rewriter.create<stablehlo::AddOp>(loc, x1, neg_p);
      // (x_hi - p) - e
      Value rem2 = rewriter.create<stablehlo::AddOp>(loc, rem1, neg_e);
      // ((x_hi - p) - e) + x_lo
      Value rem3 = rewriter.create<stablehlo::AddOp>(loc, rem2, x2);

      // q_hi * y_lo
      Value q_hi_y_lo = rewriter.create<stablehlo::MulOp>(loc, q_hi, y2);
      Value neg_q_hi_y_lo = rewriter.create<stablehlo::NegOp>(loc, q_hi_y_lo);

      // rem = rem3 - q_hi * y_lo
      Value rem = rewriter.create<stablehlo::AddOp>(loc, rem3, neg_q_hi_y_lo);

      // 4. q_lo = rem / y_hi
      Value q_lo = rewriter.create<stablehlo::DivOp>(loc, rem, y1);

      // 5. Combine q_hi and q_lo into a normalized MultiFloat (fast_two_sum)
      Value final_h = rewriter.create<stablehlo::AddOp>(loc, q_hi, q_lo);
      Value h_diff = rewriter.create<stablehlo::SubtractOp>(loc, final_h, q_hi);
      Value q_lo_diff =
          rewriter.create<stablehlo::SubtractOp>(loc, q_lo, h_diff);
      Value final_l = q_lo_diff;

      Value packed =
          packLimbs(final_h, final_l, rewriter, loc, concatDimension);
      rewriter.replaceOp(op, packed);
    } else {
      auto tensorType = cast<RankedTensorType>(x1.getType());
      auto floatTy = cast<FloatType>(tensorType.getElementType());

      auto oneAttr = rewriter.getFloatAttr(floatTy, 1.0);
      Value one = rewriter.create<stablehlo::ConstantOp>(
          loc, SplatElementsAttr::get(tensorType, oneAttr));

      auto zeroAttr = rewriter.getFloatAttr(floatTy, 0.0);
      Value zero = rewriter.create<stablehlo::ConstantOp>(
          loc, SplatElementsAttr::get(tensorType, zeroAttr));

      auto twoAttr = rewriter.getFloatAttr(floatTy, 2.0);
      Value two_hi = rewriter.create<stablehlo::ConstantOp>(
          loc, SplatElementsAttr::get(tensorType, twoAttr));
      Value two_lo = zero;

      // u0 = 1 / y1
      Value u_hi = rewriter.create<stablehlo::DivOp>(loc, one, y1);
      Value u_lo = zero;

      for (int i = 0; i < divSubsteps; ++i) {
        // Y * u
        auto [Y_u_hi, Y_u_lo] =
            multiFloatMul(y1, y2, u_hi, u_lo, rewriter, loc);
        // 2 - Y * u
        Value neg_Y_u_hi = rewriter.create<stablehlo::NegOp>(loc, Y_u_hi);
        Value neg_Y_u_lo = rewriter.create<stablehlo::NegOp>(loc, Y_u_lo);
        auto [diff_hi, diff_lo] = multiFloatAdd(two_hi, two_lo, neg_Y_u_hi,
                                                neg_Y_u_lo, rewriter, loc);
        // u = u * (2 - Y * u)
        auto [next_u_hi, next_u_lo] =
            multiFloatMul(u_hi, u_lo, diff_hi, diff_lo, rewriter, loc);
        u_hi = next_u_hi;
        u_lo = next_u_lo;
      }

      // quotient = X * u
      auto [q_hi, q_lo] = multiFloatMul(x1, x2, u_hi, u_lo, rewriter, loc);

      Value packed = packLimbs(q_hi, q_lo, rewriter, loc, concatDimension);
      rewriter.replaceOp(op, packed);
    }

    return success();
  }
};

struct SelectOpConversion : public OpConversionPattern<stablehlo::SelectOp> {
  StringRef concatDimension;

  SelectOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                     StringRef concatDimension)
      : OpConversionPattern<stablehlo::SelectOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool isTuple = concatDimension == "tuple";
    bool isFirst = concatDimension == "first";

    if (isTuple) {
      Value onTrueHigh = extractLimb(adaptor.getOperands()[1], 0, rewriter, loc,
                                     concatDimension);
      Value onTrueLow = extractLimb(adaptor.getOperands()[1], 1, rewriter, loc,
                                    concatDimension);
      Value onFalseHigh = extractLimb(adaptor.getOperands()[2], 0, rewriter,
                                      loc, concatDimension);
      Value onFalseLow = extractLimb(adaptor.getOperands()[2], 1, rewriter, loc,
                                     concatDimension);

      auto selectHigh = rewriter.create<stablehlo::SelectOp>(
          loc, adaptor.getOperands()[0], onTrueHigh, onFalseHigh);
      auto selectLow = rewriter.create<stablehlo::SelectOp>(
          loc, adaptor.getOperands()[0], onTrueLow, onFalseLow);

      Value packed =
          packLimbs(selectHigh, selectLow, rewriter, loc, concatDimension);
      rewriter.replaceOp(op, packed);
      return success();
    }

    Type expectedType = getTypeConverter()->convertType(op.getType());
    Value onTrue = adaptor.getOperands()[1];
    Value onFalse = adaptor.getOperands()[2];

    if (onTrue.getType() != expectedType) {
      onTrue =
          rewriter.create<UnrealizedConversionCastOp>(loc, expectedType, onTrue)
              .getResult(0);
    }
    if (onFalse.getType() != expectedType) {
      onFalse =
          rewriter
              .create<UnrealizedConversionCastOp>(loc, expectedType, onFalse)
              .getResult(0);
    }

    auto valuesType = cast<RankedTensorType>(expectedType);
    auto predType = cast<RankedTensorType>(adaptor.getOperands()[0].getType());

    bool skipBroadcast =
        predType.getRank() == 0 || predType.getShape() == valuesType.getShape();
    Value pred = adaptor.getOperands()[0];

    if (!skipBroadcast) {
      SmallVector<int64_t> broadcastDims;
      if (isFirst) {
        for (int i = 0; i < predType.getRank(); ++i) {
          broadcastDims.push_back(i + 1);
        }
      } else {
        for (int i = 0; i < predType.getRank(); ++i) {
          broadcastDims.push_back(i);
        }
      }

      auto outType = RankedTensorType::get(valuesType.getShape(),
                                           predType.getElementType());
      pred = rewriter.create<stablehlo::BroadcastInDimOp>(
          loc, outType, adaptor.getOperands()[0],
          rewriter.getDenseI64ArrayAttr(broadcastDims));
    }

    LLVM_DEBUG(llvm::dbgs() << "SelectOpConversion creating SelectOp:\n");
    LLVM_DEBUG(llvm::dbgs() << "  valuesType: " << valuesType << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  pred type: " << pred.getType() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  onTrue type: " << onTrue.getType() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  onFalse type: " << onFalse.getType() << "\n");

    auto selectOp = rewriter.create<stablehlo::SelectOp>(loc, valuesType, pred,
                                                         onTrue, onFalse);

    rewriter.replaceOp(op, selectOp);
    return success();
  }
};

struct ConvertOpConversion : public OpConversionPattern<stablehlo::ConvertOp> {
  StringRef concatDimension;
  Type sourceType;
  Type targetType;
  int expansionSize;

  ConvertOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                      StringRef concatDimension, Type sourceType,
                      Type targetType, int expansionSize)
      : OpConversionPattern<stablehlo::ConvertOp>(typeConverter, context),
        concatDimension(concatDimension), sourceType(sourceType),
        targetType(targetType), expansionSize(expansionSize) {}

  LogicalResult
  matchAndRewrite(stablehlo::ConvertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "ConvertOpConversion called\n");
    Location loc = op.getLoc();
    Type outType = op.getResult().getType();
    Type inType = op.getOperand().getType();

    auto outTensorType = cast<RankedTensorType>(outType);
    auto inTensorType = cast<RankedTensorType>(inType);
    Type outElType = outTensorType.getElementType();
    Type inElType = inTensorType.getElementType();

    bool isTuple = concatDimension == "tuple";
    bool isFirst = concatDimension == "first";

    Type limbType;
    if (isTuple) {
      limbType = RankedTensorType::get(outTensorType.getShape(), targetType);
    } else {
      SmallVector<int64_t> limbShape =
          llvm::to_vector(outTensorType.getShape());
      limbShape.insert(limbShape.begin() + (isFirst ? 0 : limbShape.size()), 1);
      limbType = RankedTensorType::get(limbShape, targetType);
    }

    // Case 1: Source to Standard (Expanded to Standard)
    if (inElType == sourceType && outElType != sourceType &&
        !outElType.isIntOrIndex()) {
      if (isSubsetFloat(outElType, targetType)) {
        Value high = extractLimb(adaptor.getOperands()[0], 0, rewriter, loc,
                                 concatDimension);
        if (!isTuple) {
          high = rewriter.create<stablehlo::ReshapeOp>(
              loc, RankedTensorType::get(outTensorType.getShape(), targetType),
              high);
        }
        if (outElType != targetType) {
          high = rewriter.create<stablehlo::ConvertOp>(
              loc, RankedTensorType::get(outTensorType.getShape(), outElType),
              high);
        }
        rewriter.replaceOp(op, high);
        return success();
      } else {
        if (!isTuple) {
          Value zero = nullptr;
          Value converted =
              convertFromMultifloat(adaptor.getOperands()[0], rewriter, loc,
                                    outElType, concatDimension, zero);
          rewriter.replaceOp(op, converted);
          return success();
        }

        // Output is wider than a limb, sum limbs
        Value sum = nullptr;
        for (int i = 0; i < expansionSize; ++i) {
          Value limb = extractLimb(adaptor.getOperands()[0], i, rewriter, loc,
                                   concatDimension);
          Value convertedLimb = rewriter.create<stablehlo::ConvertOp>(
              loc, RankedTensorType::get(outTensorType.getShape(), outElType),
              limb);
          if (sum) {
            sum = rewriter.create<stablehlo::AddOp>(loc, sum, convertedLimb);
          } else {
            sum = convertedLimb;
          }
        }
        rewriter.replaceOp(op, sum);
        return success();
      }
    }

    // Case 2: Standard to Source (Standard to Expanded)
    if (outElType == sourceType && inElType != sourceType &&
        !inElType.isIntOrIndex()) {
      if (isSubsetFloat(inElType, targetType)) {
        Value high = adaptor.getOperands()[0];
        if (inElType != targetType) {
          high = rewriter.create<stablehlo::ConvertOp>(
              loc, RankedTensorType::get(inTensorType.getShape(), targetType),
              high);
        }
        Value highPacked = high;
        if (!isTuple) {
          highPacked =
              rewriter.create<stablehlo::ReshapeOp>(loc, limbType, high);
        }

        SmallVector<Value> packedLimbsList;
        packedLimbsList.push_back(highPacked);
        for (int i = 1; i < expansionSize; ++i) {
          packedLimbsList.push_back(rewriter.create<stablehlo::ConstantOp>(
              loc, rewriter.getZeroAttr(limbType)));
        }

        Value packed =
            packLimbs(packedLimbsList, rewriter, loc, concatDimension);
        rewriter.replaceOp(op, packed);
        return success();
      } else {
        Value packed =
            convertToMultifloat(adaptor.getOperands()[0], rewriter, loc,
                                targetType, concatDimension, expansionSize);
        rewriter.replaceOp(op, packed);
        return success();
      }
    }

    // Case 3: Integer to Source Type
    if (inElType.isIntOrIndex() && outElType == sourceType) {
      Value packed =
          convertToMultifloat(adaptor.getOperands()[0], rewriter, loc,
                              targetType, concatDimension, expansionSize);
      rewriter.replaceOp(op, packed);
      return success();
    }

    // Case 4: Source to Integer
    if (inElType == sourceType && outElType.isIntOrIndex()) {
      Value sum = nullptr;
      for (int i = 0; i < expansionSize; ++i) {
        Value limb = extractLimb(adaptor.getOperands()[0], i, rewriter, loc,
                                 concatDimension);
        if (!isTuple) {
          limb = rewriter.create<stablehlo::ReshapeOp>(
              loc, RankedTensorType::get(outTensorType.getShape(), targetType),
              limb);
        }
        Value convertedLimb =
            rewriter.create<stablehlo::ConvertOp>(loc, outType, limb);
        if (sum) {
          sum = rewriter.create<stablehlo::AddOp>(loc, sum, convertedLimb);
        } else {
          sum = convertedLimb;
        }
      }
      rewriter.replaceOp(op, sum);
      return success();
    }

    // Case 5: Un-converted Source -> Converted Source (expanded pair)
    if (inElType == sourceType && outElType == sourceType &&
        adaptor.getOperands()[0].getType() == inType) {
      Value packed =
          convertToMultifloat(adaptor.getOperands()[0], rewriter, loc,
                              targetType, concatDimension, expansionSize);
      rewriter.replaceOp(op, packed);
      return success();
    }
    return failure();
  }
};

struct ReverseOpConversion : public OpConversionPattern<stablehlo::ReverseOp> {
  StringRef concatDimension;

  ReverseOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                      StringRef concatDimension)
      : OpConversionPattern<stablehlo::ReverseOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::ReverseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool isTuple = concatDimension == "tuple";
    bool isFirst = concatDimension == "first";

    if (isTuple) {
      Value high = extractLimb(adaptor.getOperands()[0], 0, rewriter, loc,
                               concatDimension);
      Value low = extractLimb(adaptor.getOperands()[0], 1, rewriter, loc,
                              concatDimension);

      Value revHigh =
          rewriter.create<stablehlo::ReverseOp>(loc, high, op.getDimensions());
      Value revLow =
          rewriter.create<stablehlo::ReverseOp>(loc, low, op.getDimensions());

      Value packed = packLimbs(revHigh, revLow, rewriter, loc, concatDimension);
      rewriter.replaceOp(op, packed);
      return success();
    }

    SmallVector<int64_t> dims = llvm::to_vector(op.getDimensions());
    if (isFirst) {
      for (auto &dim : dims) {
        dim += 1;
      }
    }

    Value reversed = rewriter.create<stablehlo::ReverseOp>(
        loc, adaptor.getOperands()[0], rewriter.getDenseI64ArrayAttr(dims));
    rewriter.replaceOp(op, reversed);
    return success();
  }
};

struct SliceOpConversion : public OpConversionPattern<stablehlo::SliceOp> {
  StringRef concatDimension;

  SliceOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                    StringRef concatDimension)
      : OpConversionPattern<stablehlo::SliceOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool isTuple = concatDimension == "tuple";
    bool isFirst = concatDimension == "first";

    if (isTuple) {
      Value high = extractLimb(adaptor.getOperands()[0], 0, rewriter, loc,
                               concatDimension);
      Value low = extractLimb(adaptor.getOperands()[0], 1, rewriter, loc,
                              concatDimension);

      auto sliceHigh = rewriter.create<stablehlo::SliceOp>(
          loc, high, op.getStartIndices(), op.getLimitIndices(),
          op.getStrides());
      auto sliceLow = rewriter.create<stablehlo::SliceOp>(
          loc, low, op.getStartIndices(), op.getLimitIndices(),
          op.getStrides());

      Value packed =
          packLimbs(sliceHigh, sliceLow, rewriter, loc, concatDimension);
      rewriter.replaceOp(op, packed);
      return success();
    }

    SmallVector<int64_t> startIndices = llvm::to_vector(op.getStartIndices());
    SmallVector<int64_t> limitIndices = llvm::to_vector(op.getLimitIndices());
    SmallVector<int64_t> strides = llvm::to_vector(op.getStrides());

    if (isFirst) {
      startIndices.insert(startIndices.begin(), 0);
      limitIndices.insert(limitIndices.begin(), 2);
      strides.insert(strides.begin(), 1);
    } else {
      startIndices.push_back(0);
      limitIndices.push_back(2);
      strides.push_back(1);
    }

    auto outType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

    auto sliceOp = rewriter.create<stablehlo::SliceOp>(
        loc, outType, adaptor.getOperands()[0],
        rewriter.getDenseI64ArrayAttr(startIndices),
        rewriter.getDenseI64ArrayAttr(limitIndices),
        rewriter.getDenseI64ArrayAttr(strides));

    rewriter.replaceOp(op, sliceOp);
    return success();
  }
};

struct AbsOpConversion : public OpConversionPattern<stablehlo::AbsOp> {
  StringRef concatDimension;

  AbsOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                  StringRef concatDimension)
      : OpConversionPattern<stablehlo::AbsOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::AbsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value input = adaptor.getOperands()[0];
    Value hi = extractLimb(input, 0, rewriter, loc, concatDimension);

    auto tensorType = cast<RankedTensorType>(hi.getType());
    auto floatTy = cast<FloatType>(tensorType.getElementType());

    auto zeroAttr = rewriter.getFloatAttr(floatTy, 0.0);
    auto splatAttr = SplatElementsAttr::get(tensorType, zeroAttr);
    Value zero = rewriter.create<stablehlo::ConstantOp>(loc, splatAttr);

    Value isPositive = rewriter.create<stablehlo::CompareOp>(
        loc, hi, zero, stablehlo::ComparisonDirection::GE);

    bool isTuple = concatDimension == "tuple";

    if (isTuple) {
      Value lo = extractLimb(input, 1, rewriter, loc, concatDimension);
      Value neg_hi = rewriter.create<stablehlo::NegOp>(loc, hi);
      Value neg_lo = rewriter.create<stablehlo::NegOp>(loc, lo);

      Value res_hi =
          rewriter.create<stablehlo::SelectOp>(loc, isPositive, hi, neg_hi);
      Value res_lo =
          rewriter.create<stablehlo::SelectOp>(loc, isPositive, lo, neg_lo);

      Value packed = packLimbs(res_hi, res_lo, rewriter, loc, concatDimension);
      rewriter.replaceOp(op, packed);
    } else {
      auto fullType = cast<RankedTensorType>(input.getType());
      auto predType =
          RankedTensorType::get(fullType.getShape(), rewriter.getI1Type());

      SmallVector<int64_t> broadcastDims;
      for (int i = 0; i < fullType.getRank(); ++i) {
        broadcastDims.push_back(i);
      }
      Value broadcasted_isPositive =
          rewriter.create<stablehlo::BroadcastInDimOp>(
              loc, predType, isPositive,
              rewriter.getDenseI64ArrayAttr(broadcastDims));

      Value neg = rewriter.create<stablehlo::NegOp>(loc, input);
      Value res = rewriter.create<stablehlo::SelectOp>(
          loc, fullType, broadcasted_isPositive, input, neg);
      rewriter.replaceOp(op, res);
    }
    return success();
  }
};

struct FloorOpConversion : public OpConversionPattern<stablehlo::FloorOp> {
  StringRef concatDimension;

  FloorOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                    StringRef concatDimension)
      : OpConversionPattern<stablehlo::FloorOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::FloorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value input = adaptor.getOperands()[0];
    Value hi = extractLimb(input, 0, rewriter, loc, concatDimension);
    Value lo = extractLimb(input, 1, rewriter, loc, concatDimension);

    auto tensorType = cast<RankedTensorType>(hi.getType());
    auto floatTy = cast<FloatType>(tensorType.getElementType());

    // 1. fh = floor(xh)
    Value fh = rewriter.create<stablehlo::FloorOp>(loc, hi);

    // 2. diff = xh - fh
    Value diff = rewriter.create<stablehlo::SubtractOp>(loc, hi, fh);

    // 3. is_negative = compare(LT, diff, negate(xl))
    Value neg_lo = rewriter.create<stablehlo::NegOp>(loc, lo);
    Value is_negative = rewriter.create<stablehlo::CompareOp>(
        loc, diff, neg_lo, stablehlo::ComparisonDirection::LT);

    // 4. fh_minus_1 = fh - 1
    auto oneAttr = rewriter.getFloatAttr(floatTy, 1.0);
    auto splatAttr = SplatElementsAttr::get(tensorType, oneAttr);
    Value one = rewriter.create<stablehlo::ConstantOp>(loc, splatAttr);
    Value fh_minus_1 = rewriter.create<stablehlo::SubtractOp>(loc, fh, one);

    // 5. res_hi = select(is_negative, fh_minus_1, fh)
    Value res_hi =
        rewriter.create<stablehlo::SelectOp>(loc, is_negative, fh_minus_1, fh);

    // 6. res_lo = 0
    auto zeroAttr = rewriter.getFloatAttr(floatTy, 0.0);
    auto zeroSplatAttr = SplatElementsAttr::get(tensorType, zeroAttr);
    Value zero = rewriter.create<stablehlo::ConstantOp>(loc, zeroSplatAttr);

    // 7. Pack res_hi and res_lo
    Value packed = packLimbs(res_hi, zero, rewriter, loc, concatDimension);
    rewriter.replaceOp(op, packed);

    return success();
  }
};

struct CeilOpConversion : public OpConversionPattern<stablehlo::CeilOp> {
  StringRef concatDimension;

  CeilOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                   StringRef concatDimension)
      : OpConversionPattern<stablehlo::CeilOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::CeilOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value input = adaptor.getOperands()[0];
    Value hi = extractLimb(input, 0, rewriter, loc, concatDimension);
    Value lo = extractLimb(input, 1, rewriter, loc, concatDimension);

    auto tensorType = cast<RankedTensorType>(hi.getType());
    auto floatTy = cast<FloatType>(tensorType.getElementType());

    // 1. ch = ceil(xh)
    Value ch = rewriter.create<stablehlo::CeilOp>(loc, hi);

    // 2. is_int = compare(EQ, ch, hi)
    Value is_int = rewriter.create<stablehlo::CompareOp>(
        loc, ch, hi, stablehlo::ComparisonDirection::EQ);

    // 3. is_pos = compare(GT, lo, 0)
    auto zeroAttr = rewriter.getFloatAttr(floatTy, 0.0);
    auto zeroSplatAttr = SplatElementsAttr::get(tensorType, zeroAttr);
    Value zero = rewriter.create<stablehlo::ConstantOp>(loc, zeroSplatAttr);
    Value is_pos = rewriter.create<stablehlo::CompareOp>(
        loc, lo, zero, stablehlo::ComparisonDirection::GT);

    // 4. should_inc = and(is_int, is_pos)
    Value should_inc = rewriter.create<stablehlo::AndOp>(loc, is_int, is_pos);

    // 5. ch_plus_1 = ch + 1
    auto oneAttr = rewriter.getFloatAttr(floatTy, 1.0);
    auto splatAttr = SplatElementsAttr::get(tensorType, oneAttr);
    Value one = rewriter.create<stablehlo::ConstantOp>(loc, splatAttr);
    Value ch_plus_1 = rewriter.create<stablehlo::AddOp>(loc, ch, one);

    // 6. res_hi = select(should_inc, ch_plus_1, ch)
    Value res_hi =
        rewriter.create<stablehlo::SelectOp>(loc, should_inc, ch_plus_1, ch);

    // 7. Pack res_hi and zero (res_lo is 0)
    Value packed = packLimbs(res_hi, zero, rewriter, loc, concatDimension);
    rewriter.replaceOp(op, packed);

    return success();
  }
};

struct ExpOpConversion : public OpConversionPattern<stablehlo::ExpOp> {
  StringRef concatDimension;

  ExpOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                  StringRef concatDimension)
      : OpConversionPattern<stablehlo::ExpOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  Value polEvl(Value x, ArrayRef<double> coefs,
               ConversionPatternRewriter &rewriter, Location loc,
               RankedTensorType type) const {
    auto floatTy =
        cast<FloatType>(cast<RankedTensorType>(type).getElementType());
    Value res = rewriter.create<stablehlo::ConstantOp>(
        loc,
        SplatElementsAttr::get(type, rewriter.getFloatAttr(floatTy, coefs[0])));
    for (size_t i = 1; i < coefs.size(); ++i) {
      Value c = rewriter.create<stablehlo::ConstantOp>(
          loc, SplatElementsAttr::get(
                   type, rewriter.getFloatAttr(floatTy, coefs[i])));
      res = rewriter.create<stablehlo::MulOp>(loc, res, x);
      res = rewriter.create<stablehlo::AddOp>(loc, res, c);
    }
    return res;
  }

  LogicalResult
  matchAndRewrite(stablehlo::ExpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value input = adaptor.getOperands()[0];
    Value hi = extractLimb(input, 0, rewriter, loc, concatDimension);
    Value lo = extractLimb(input, 1, rewriter, loc, concatDimension);

    auto tensorType = cast<RankedTensorType>(hi.getType());
    auto floatTy = cast<FloatType>(tensorType.getElementType());

    // 1. Multiply by log2(e) in multi-float!
    // log2(e) approx C1 + C2
    // C1 = Float32(+0x1.715476p+000) = 1.4426950216293335
    // C2 = Float32(+0x1.4AE0C0p-026) = 1.9259629911678385e-8
    auto c1Attr = rewriter.getFloatAttr(floatTy, 1.4426950216293335);
    auto c1Splat = SplatElementsAttr::get(tensorType, c1Attr);
    Value c1 = rewriter.create<stablehlo::ConstantOp>(loc, c1Splat);

    auto c2Attr = rewriter.getFloatAttr(floatTy, 1.9259629911678385e-8);
    auto c2Splat = SplatElementsAttr::get(tensorType, c2Attr);
    Value c2 = rewriter.create<stablehlo::ConstantOp>(loc, c2Splat);

    auto [p00, e00] = twoProdDekker(hi, c1, rewriter, loc);
    Value p01 = rewriter.create<stablehlo::MulOp>(loc, hi, c2);
    Value p10 = rewriter.create<stablehlo::MulOp>(loc, lo, c1);
    Value p01_p10 = rewriter.create<stablehlo::AddOp>(loc, p01, p10);
    Value e00_new = rewriter.create<stablehlo::AddOp>(loc, e00, p01_p10);
    auto [y_hi, y_lo] = fastTwoSum(p00, e00_new, rewriter, loc);

    // 2. n = floor(y_hi + 0.5)
    auto halfAttr = rewriter.getFloatAttr(floatTy, 0.5);
    auto halfSplat = SplatElementsAttr::get(tensorType, halfAttr);
    Value half = rewriter.create<stablehlo::ConstantOp>(loc, halfSplat);
    Value add_half = rewriter.create<stablehlo::AddOp>(loc, y_hi, half);
    Value n = rewriter.create<stablehlo::FloorOp>(loc, add_half);

    // 3. r = y - n in multi-float!
    Value neg_n = rewriter.create<stablehlo::NegOp>(loc, n);
    auto [r_hi, r_lo_err] = twoSum(y_hi, neg_n, rewriter, loc);
    Value r_lo = rewriter.create<stablehlo::AddOp>(loc, y_lo, r_lo_err);
    auto [r_hi_final, r_lo_final] = fastTwoSum(r_hi, r_lo, rewriter, loc);

    // 4. Scale r by 1/8
    auto one_eighthAttr = rewriter.getFloatAttr(floatTy, 0.125);
    auto one_eighthSplat = SplatElementsAttr::get(tensorType, one_eighthAttr);
    Value one_eighth =
        rewriter.create<stablehlo::ConstantOp>(loc, one_eighthSplat);
    Value r_prime_hi =
        rewriter.create<stablehlo::MulOp>(loc, r_hi_final, one_eighth);
    Value r_prime_lo =
        rewriter.create<stablehlo::MulOp>(loc, r_lo_final, one_eighth);

    // 5. Evaluate polynomial of degree 7 on r_prime in multi-float!
    // Coefficients from MultiFloats.jl for Float32, Val{2}:
    // 1: (1.0, -1.62C458p-59) -> we only use 1.0 for high part!
    // 2: (+0x1.62E430p-001, -0x1.05C610p-029)
    // 3: (+0x1.EBFBE0p-003, -0x1.F4DEB0p-033)
    // 4: (+0x1.C6B08Ep-005, -0x1.1F6B9Cp-030)
    // 5: (+0x1.3B2AB6p-007, +0x1.DB9286p-032)
    // 6: (+0x1.5D87FEp-010)
    // 7: (+0x1.430E9Ep-013)
    // 8: (+0x1.FFD486p-017)

    constexpr double kCoefs[][2] = {
        {1.0, 0.0},
        {0.6931471824645996, -1.9046542121259336e-09},
        {0.24022650718688965, -2.27769247906906e-10},
        {0.05550410971045494, -1.0456291388294403e-09},
        {0.009618128649890423, 4.3253053916281203e-10},
        {0.0013333557872101665, 0.0},
        {0.00015404562873300165, 0.0},
        {1.5253727724484634e-05, 0.0},
    };

    // Horner's method in multi-float!
    // Start with last coefficient!
    auto getConst = [&](double val) {
      return rewriter.create<stablehlo::ConstantOp>(
          loc, SplatElementsAttr::get(tensorType,
                                      rewriter.getFloatAttr(floatTy, val)));
    };

    Value p_hi = getConst(kCoefs[7][0]);
    Value p_lo = getConst(kCoefs[7][1]);

    for (int i = 6; i >= 0; --i) {
      auto [mul_hi, mul_lo] =
          mfMul(p_hi, p_lo, r_prime_hi, r_prime_lo, rewriter, loc);
      auto [add_hi, add_lo] = mfAdd(mul_hi, mul_lo, getConst(kCoefs[i][0]),
                                    getConst(kCoefs[i][1]), rewriter, loc);
      p_hi = add_hi;
      p_lo = add_lo;
    }

    // 6. Square 3 times in multi-float!
    auto [s1_hi, s1_lo] = mfMul(p_hi, p_lo, p_hi, p_lo, rewriter, loc);
    auto [s2_hi, s2_lo] = mfMul(s1_hi, s1_lo, s1_hi, s1_lo, rewriter, loc);
    auto [res_hi, res_lo] = mfMul(s2_hi, s2_lo, s2_hi, s2_lo, rewriter, loc);

    // 7. Scale by 2^n
    // We can use stablehlo.pow(2.0, n) and multiply!
    auto twoAttr = rewriter.getFloatAttr(floatTy, 2.0);
    auto twoSplat = SplatElementsAttr::get(tensorType, twoAttr);
    Value two = rewriter.create<stablehlo::ConstantOp>(loc, twoSplat);
    Value scale = rewriter.create<stablehlo::PowOp>(loc, two, n);

    res_hi = rewriter.create<stablehlo::MulOp>(loc, res_hi, scale);
    res_lo = rewriter.create<stablehlo::MulOp>(loc, res_lo, scale);

    // Pack limbs
    Value packed = packLimbs(res_hi, res_lo, rewriter, loc, concatDimension);
    rewriter.replaceOp(op, packed);

    return success();
  }
};

struct NegOpConversion : public OpConversionPattern<stablehlo::NegOp> {
  StringRef concatDimension;

  NegOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                  StringRef concatDimension)
      : OpConversionPattern<stablehlo::NegOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::NegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value packed = applyElementwiseOpToLimbs(
        adaptor.getOperands()[0], rewriter, loc, concatDimension,
        [&](Value v) { return rewriter.create<stablehlo::NegOp>(loc, v); });
    rewriter.replaceOp(op, packed);
    return success();
  }
};

struct SineOpConversion : public OpConversionPattern<stablehlo::SineOp> {
  StringRef concatDimension;

  SineOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                   StringRef concatDimension)
      : OpConversionPattern<stablehlo::SineOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::SineOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value hi = extractLimb(adaptor.getOperands()[0], 0, rewriter, loc,
                           concatDimension);
    Value lo = extractLimb(adaptor.getOperands()[0], 1, rewriter, loc,
                           concatDimension);

    auto tensorType = cast<RankedTensorType>(hi.getType());
    auto floatTy = cast<FloatType>(tensorType.getElementType());

    // 1/pi constants
    auto hiInvPiAttr = rewriter.getFloatAttr(floatTy, 0.31830987334251403809);
    auto loInvPiAttr = rewriter.getFloatAttr(floatTy, 0.00000001284127648660);
    Value hi_inv_pi = rewriter.create<stablehlo::ConstantOp>(
        loc, SplatElementsAttr::get(tensorType, hiInvPiAttr));
    Value lo_inv_pi = rewriter.create<stablehlo::ConstantOp>(
        loc, SplatElementsAttr::get(tensorType, loInvPiAttr));

    // x_pi = x * (1/pi)
    auto [x_pi_hi, x_pi_lo] =
        multiFloatMul(hi, lo, hi_inv_pi, lo_inv_pi, rewriter, loc);

    // Absolute value of x_pi
    auto zeroAttr = rewriter.getFloatAttr(floatTy, 0.0);
    Value zero = rewriter.create<stablehlo::ConstantOp>(
        loc, SplatElementsAttr::get(tensorType, zeroAttr));
    Value lt_zero = rewriter.create<stablehlo::CompareOp>(
        loc, x_pi_hi, zero, stablehlo::ComparisonDirection::LT);
    Value neg_x_pi_hi = rewriter.create<stablehlo::NegOp>(loc, x_pi_hi);

    Value abs_x_hi = rewriter.create<stablehlo::SelectOp>(loc, lt_zero,
                                                          neg_x_pi_hi, x_pi_hi);

    // n = round(2 * abs_x_hi)
    // Using trunc(2 * abs_x_hi + 0.5) as a proxy for round for positive
    // numbers.
    auto twoAttr = rewriter.getFloatAttr(floatTy, 2.0);
    Value two = rewriter.create<stablehlo::ConstantOp>(
        loc, SplatElementsAttr::get(tensorType, twoAttr));
    Value two_abs_x_hi = rewriter.create<stablehlo::MulOp>(loc, two, abs_x_hi);

    auto halfAttr = rewriter.getFloatAttr(floatTy, 0.5);
    Value half = rewriter.create<stablehlo::ConstantOp>(
        loc, SplatElementsAttr::get(tensorType, halfAttr));
    Value two_abs_x_hi_plus_half =
        rewriter.create<stablehlo::AddOp>(loc, two_abs_x_hi, half);

    auto intType =
        RankedTensorType::get(tensorType.getShape(), rewriter.getI32Type());
    Value n_int = rewriter.create<stablehlo::ConvertOp>(loc, intType,
                                                        two_abs_x_hi_plus_half);
    Value n_float =
        rewriter.create<stablehlo::ConvertOp>(loc, tensorType, n_int);

    // rx = abs_x - 0.5 * n
    Value half_n = rewriter.create<stablehlo::MulOp>(loc, half, n_float);
    Value neg_half_n = rewriter.create<stablehlo::NegOp>(loc, half_n);

    Value neg_x_pi_lo = rewriter.create<stablehlo::NegOp>(loc, x_pi_lo);
    Value abs_x_lo = rewriter.create<stablehlo::SelectOp>(loc, lt_zero,
                                                          neg_x_pi_lo, x_pi_lo);

    auto [rx_hi, rx_lo] =
        multiFloatAdd(abs_x_hi, abs_x_lo, neg_half_n, zero, rewriter, loc);

    // quadrant = n_int & 3
    auto threeAttr = rewriter.getIntegerAttr(rewriter.getI32Type(), 3);
    Value three = rewriter.create<stablehlo::ConstantOp>(
        loc, SplatElementsAttr::get(intType, threeAttr));
    Value quadrant = rewriter.create<stablehlo::AndOp>(loc, n_int, three);

    // Polynomial evaluation constants
    auto getConstant = [&](float val) -> Value {
      auto attr = rewriter.getFloatAttr(floatTy, val);
      return rewriter.create<stablehlo::ConstantOp>(
          loc, SplatElementsAttr::get(tensorType, attr));
    };

    SmallVector<std::pair<float, float>> sin_coefs = {
        {3.14159274101257324219f, -0.00000008742277657348f},
        {-5.16771268844604492188f, -0.00000009160392266949f},
        {2.55016398429870605469f, 0.00000005557863858030f},
        {-0.59926450252532958984f, -0.00000002679546184936f},
        {0.08214588463306427002f, 0.00000000197806393487f},
        {-0.00737043097615242004f, 0.00000000003043807220f},
        {0.00046630279393866658f, 0.00000000001182894581f},
        {-0.00002191535349993501f, 0.00000000000005210480f}};

    SmallVector<std::pair<float, float>> cos_coefs = {
        {1.00000000000000000000f, 0.00000000000000000000f},
        {-4.93480205535888671875f, -0.00000014518579405376f},
        {4.05871200561523437500f, 0.00000012080153055649f},
        {-1.33526277542114257812f, 0.00000000656655352316f},
        {0.23533062636852264404f, 0.00000000399037070054f},
        {-0.02580689080059528351f, -0.00000000058941879155f},
        {0.00192957429680973291f, 0.00000000001259418958f},
        {-0.00010463810758665204f, 0.00000000000266180632f}};

    auto evaluatePolynomial =
        [&](Value z_h, Value z_l,
            const SmallVector<std::pair<float, float>> &coefs)
        -> std::pair<Value, Value> {
      Value acc_h = getConstant(coefs.back().first);
      Value acc_l = getConstant(coefs.back().second);
      for (int i = coefs.size() - 2; i >= 0; --i) {
        auto [mul_h, mul_l] =
            multiFloatMul(acc_h, acc_l, z_h, z_l, rewriter, loc);
        Value c_h = getConstant(coefs[i].first);
        Value c_l = getConstant(coefs[i].second);
        std::tie(acc_h, acc_l) =
            multiFloatAdd(mul_h, mul_l, c_h, c_l, rewriter, loc);
      }
      return {acc_h, acc_l};
    };

    // z = rx^2
    auto [z_hi, z_lo] =
        multiFloatMul(rx_hi, rx_lo, rx_hi, rx_lo, rewriter, loc);

    // Evaluate Sine and Cosine polynomials
    auto [poly_sine_hi, poly_sine_lo] =
        evaluatePolynomial(z_hi, z_lo, sin_coefs);
    auto [poly_cosine_hi, poly_cosine_lo] =
        evaluatePolynomial(z_hi, z_lo, cos_coefs);

    // Sine result = rx * evalpoly(rx^2, sin_coefs)
    auto [res_sine_hi, res_sine_lo] =
        multiFloatMul(rx_hi, rx_lo, poly_sine_hi, poly_sine_lo, rewriter, loc);

    // Cosine result = evalpoly(rx^2, cos_coefs)
    Value res_cosine_hi = poly_cosine_hi;
    Value res_cosine_lo = poly_cosine_lo;

    // Selection based on quadrant
    auto zeroIntAttr = rewriter.getIntegerAttr(rewriter.getI32Type(), 0);
    Value zero_int = rewriter.create<stablehlo::ConstantOp>(
        loc, SplatElementsAttr::get(intType, zeroIntAttr));

    auto oneIntAttr = rewriter.getIntegerAttr(rewriter.getI32Type(), 1);
    Value one_int = rewriter.create<stablehlo::ConstantOp>(
        loc, SplatElementsAttr::get(intType, oneIntAttr));
    Value quadrant_and_1 =
        rewriter.create<stablehlo::AndOp>(loc, quadrant, one_int);
    Value is_sine_quad = rewriter.create<stablehlo::CompareOp>(
        loc, quadrant_and_1, zero_int, stablehlo::ComparisonDirection::EQ);

    Value final_raw_hi = rewriter.create<stablehlo::SelectOp>(
        loc, is_sine_quad, res_sine_hi, res_cosine_hi);
    Value final_raw_lo = rewriter.create<stablehlo::SelectOp>(
        loc, is_sine_quad, res_sine_lo, res_cosine_lo);

    bool isTuple = concatDimension == "tuple";

    // Apply quadrant sign
    auto twoIntAttr = rewriter.getIntegerAttr(rewriter.getI32Type(), 2);
    Value two_int = rewriter.create<stablehlo::ConstantOp>(
        loc, SplatElementsAttr::get(intType, twoIntAttr));
    Value quadrant_and_2 =
        rewriter.create<stablehlo::AndOp>(loc, quadrant, two_int);
    Value is_neg_quad = rewriter.create<stablehlo::CompareOp>(
        loc, quadrant_and_2, zero_int, stablehlo::ComparisonDirection::NE);

    if (isTuple) {
      Value neg_final_hi = rewriter.create<stablehlo::NegOp>(loc, final_raw_hi);
      Value neg_final_lo = rewriter.create<stablehlo::NegOp>(loc, final_raw_lo);

      Value final_hi = rewriter.create<stablehlo::SelectOp>(
          loc, is_neg_quad, neg_final_hi, final_raw_hi);
      Value final_lo = rewriter.create<stablehlo::SelectOp>(
          loc, is_neg_quad, neg_final_lo, final_raw_lo);

      // Apply original sign (lt_zero)
      Value neg_final2_hi = rewriter.create<stablehlo::NegOp>(loc, final_hi);
      Value neg_final2_lo = rewriter.create<stablehlo::NegOp>(loc, final_lo);

      Value res_h = rewriter.create<stablehlo::SelectOp>(
          loc, lt_zero, neg_final2_hi, final_hi);
      Value res_l = rewriter.create<stablehlo::SelectOp>(
          loc, lt_zero, neg_final2_lo, final_lo);

      Value packed = packLimbs(res_h, res_l, rewriter, loc, concatDimension);
      rewriter.replaceOp(op, packed);
    } else {
      Value final_raw =
          packLimbs(final_raw_hi, final_raw_lo, rewriter, loc, concatDimension);
      auto fullType = cast<RankedTensorType>(final_raw.getType());
      auto predType =
          RankedTensorType::get(fullType.getShape(), rewriter.getI1Type());

      SmallVector<int64_t> broadcastDims;
      for (int i = 0; i < fullType.getRank(); ++i) {
        broadcastDims.push_back(i);
      }

      // Broadcast is_neg_quad
      Value bcast_is_neg_quad = rewriter.create<stablehlo::BroadcastInDimOp>(
          loc, predType, is_neg_quad,
          rewriter.getDenseI64ArrayAttr(broadcastDims));

      Value neg_final = rewriter.create<stablehlo::NegOp>(loc, final_raw);
      Value final_signed = rewriter.create<stablehlo::SelectOp>(
          loc, fullType, bcast_is_neg_quad, neg_final, final_raw);

      // Broadcast lt_zero
      Value bcast_lt_zero = rewriter.create<stablehlo::BroadcastInDimOp>(
          loc, predType, lt_zero, rewriter.getDenseI64ArrayAttr(broadcastDims));

      Value neg_final2 = rewriter.create<stablehlo::NegOp>(loc, final_signed);
      Value res = rewriter.create<stablehlo::SelectOp>(
          loc, fullType, bcast_lt_zero, neg_final2, final_signed);

      rewriter.replaceOp(op, res);
    }
    return success();
  }
};

struct SqrtOpConversion : public OpConversionPattern<stablehlo::SqrtOp> {
  StringRef concatDimension;

  SqrtOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                   StringRef concatDimension)
      : OpConversionPattern<stablehlo::SqrtOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::SqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value x_hi = extractLimb(adaptor.getOperands()[0], 0, rewriter, loc,
                             concatDimension);
    Value x_lo = extractLimb(adaptor.getOperands()[0], 1, rewriter, loc,
                             concatDimension);

    auto tensorType = cast<RankedTensorType>(x_hi.getType());
    auto floatTy = cast<FloatType>(tensorType.getElementType());

    auto zeroAttr = rewriter.getFloatAttr(floatTy, 0.0);
    Value zero = rewriter.create<stablehlo::ConstantOp>(
        loc, SplatElementsAttr::get(tensorType, zeroAttr));

    auto oneAttr = rewriter.getFloatAttr(floatTy, 1.0);
    Value one = rewriter.create<stablehlo::ConstantOp>(
        loc, SplatElementsAttr::get(tensorType, oneAttr));

    // Ensure input is positive to avoid NaN in rsqrt
    Value is_le_zero = rewriter.create<stablehlo::CompareOp>(
        loc, x_hi, zero, stablehlo::ComparisonDirection::LE);

    // x_hi_safe = is_le_zero ? 1.0 : x_hi
    Value x_hi_safe =
        rewriter.create<stablehlo::SelectOp>(loc, is_le_zero, one, x_hi);

    // u0 = rsqrt(x_hi_safe)
    Value u0 = rewriter.create<stablehlo::RsqrtOp>(loc, x_hi_safe);

    // root = X * u0
    auto [root_hi, root_lo] =
        multiFloatMul(x_hi, x_lo, u0, zero, rewriter, loc);

    // residual = root^2 - X
    auto [root_sq_hi, root_sq_lo] =
        multiFloatMul(root_hi, root_lo, root_hi, root_lo, rewriter, loc);
    Value neg_x_hi = rewriter.create<stablehlo::NegOp>(loc, x_hi);
    Value neg_x_lo = rewriter.create<stablehlo::NegOp>(loc, x_lo);
    auto [res_hi, res_lo] = multiFloatAdd(root_sq_hi, root_sq_lo, neg_x_hi,
                                          neg_x_lo, rewriter, loc);

    // u_over_2 = 0.5 * u0
    auto halfAttr = rewriter.getFloatAttr(floatTy, 0.5);
    Value half = rewriter.create<stablehlo::ConstantOp>(
        loc, SplatElementsAttr::get(tensorType, halfAttr));
    Value u_over_2 = rewriter.create<stablehlo::MulOp>(loc, half, u0);

    // correction = residual * u_over_2
    auto [corr_hi, corr_lo] =
        multiFloatMul(res_hi, res_lo, u_over_2, zero, rewriter, loc);

    // result = root - correction
    Value neg_corr_hi = rewriter.create<stablehlo::NegOp>(loc, corr_hi);
    Value neg_corr_lo = rewriter.create<stablehlo::NegOp>(loc, corr_lo);
    auto [final_h, final_l] = multiFloatAdd(root_hi, root_lo, neg_corr_hi,
                                            neg_corr_lo, rewriter, loc);

    bool isTuple = concatDimension == "tuple";

    if (isTuple) {
      // If is_le_zero, result is zero
      Value res_h =
          rewriter.create<stablehlo::SelectOp>(loc, is_le_zero, zero, final_h);
      Value res_l =
          rewriter.create<stablehlo::SelectOp>(loc, is_le_zero, zero, final_l);

      Value packed = packLimbs(res_h, res_l, rewriter, loc, concatDimension);
      rewriter.replaceOp(op, packed);
    } else {
      Value final_packed =
          packLimbs(final_h, final_l, rewriter, loc, concatDimension);
      auto fullType = cast<RankedTensorType>(final_packed.getType());
      auto predType =
          RankedTensorType::get(fullType.getShape(), rewriter.getI1Type());

      SmallVector<int64_t> broadcastDims;
      for (int i = 0; i < fullType.getRank(); ++i) {
        broadcastDims.push_back(i);
      }

      // Broadcast is_le_zero
      Value bcast_is_zero = rewriter.create<stablehlo::BroadcastInDimOp>(
          loc, predType, is_le_zero,
          rewriter.getDenseI64ArrayAttr(broadcastDims));

      // Create a full zero tensor of the same shape
      auto floatTy = cast<FloatType>(fullType.getElementType());
      auto zeroAttr = rewriter.getFloatAttr(floatTy, 0.0);
      auto splatAttr = SplatElementsAttr::get(fullType, zeroAttr);
      Value full_zero = rewriter.create<stablehlo::ConstantOp>(loc, splatAttr);

      Value res = rewriter.create<stablehlo::SelectOp>(
          loc, fullType, bcast_is_zero, full_zero, final_packed);

      rewriter.replaceOp(op, res);
    }
    return success();
  }
};

struct WhileOpConversion : public OpConversionPattern<stablehlo::WhileOp> {
  StringRef concatDimension;

  WhileOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                    StringRef concatDimension)
      : OpConversionPattern<stablehlo::WhileOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "WhileOpConversion called\n");
    Location loc = op.getLoc();

    bool isTuple = concatDimension == "tuple";
    SmallVector<Type> flatConvertedTypes;
    SmallVector<Value> flatOperands;

    TypeConverter::SignatureConversion signatureConversion(op.getNumOperands());

    for (unsigned i = 0; i < adaptor.getOperands().size(); ++i) {
      Value operand = adaptor.getOperands()[i];
      Type type = getTypeConverter()->convertType(op.getOperandTypes()[i]);
      if (isTuple && dyn_cast<TupleType>(type)) {
        auto tupleTy = cast<TupleType>(type);
        SmallVector<Type> types;
        for (unsigned j = 0; j < tupleTy.size(); ++j) {
          flatConvertedTypes.push_back(tupleTy.getType(j));
          flatOperands.push_back(
              rewriter.create<stablehlo::GetTupleElementOp>(loc, operand, j));
          types.push_back(tupleTy.getType(j));
        }
        signatureConversion.addInputs(i, types);
      } else {
        flatConvertedTypes.push_back(type);
        flatOperands.push_back(operand);
        signatureConversion.addInputs(i, type);
      }
    }

    auto newWhileOp = rewriter.create<stablehlo::WhileOp>(
        loc, flatConvertedTypes, flatOperands);
    newWhileOp.getOperation()->setAttrs(op.getOperation()->getAttrs());

    rewriter.inlineRegionBefore(op.getCond(), newWhileOp.getCond(),
                                newWhileOp.getCond().end());
    if (failed(rewriter.convertRegionTypes(
            &newWhileOp.getCond(), *getTypeConverter(), &signatureConversion)))
      return failure();

    rewriter.inlineRegionBefore(op.getBody(), newWhileOp.getBody(),
                                newWhileOp.getBody().end());
    if (failed(rewriter.convertRegionTypes(
            &newWhileOp.getBody(), *getTypeConverter(), &signatureConversion)))
      return failure();

    SmallVector<Value> replacements;
    unsigned flatResIdx = 0;
    for (auto origRes : op.getResults()) {
      Type convResTy = getTypeConverter()->convertType(origRes.getType());
      if (isTuple && dyn_cast<TupleType>(convResTy)) {
        auto tupleTy = cast<TupleType>(convResTy);
        SmallVector<Value> tupleElements;
        for (unsigned j = 0; j < tupleTy.size(); ++j) {
          tupleElements.push_back(newWhileOp.getResult(flatResIdx++));
        }
        replacements.push_back(
            rewriter.create<stablehlo::TupleOp>(loc, tupleElements));
      } else {
        replacements.push_back(newWhileOp.getResult(flatResIdx++));
      }
    }

    rewriter.replaceOp(op, replacements);
    return success();
  }
};

struct ReturnOpConversion : public OpConversionPattern<stablehlo::ReturnOp> {
  StringRef concatDimension;
  bool convertSignatures;
  int expansionSize;
  Type sourceType;
  Type targetType;

  ReturnOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                     StringRef concatDimension, bool convertSignatures,
                     int expansionSize, Type sourceType, Type targetType)
      : OpConversionPattern<stablehlo::ReturnOp>(typeConverter, context),
        concatDimension(concatDimension), convertSignatures(convertSignatures),
        expansionSize(expansionSize), sourceType(sourceType),
        targetType(targetType) {}

  LogicalResult
  matchAndRewrite(stablehlo::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<stablehlo::ReduceOp>(op->getParentOp()))
      return failure();

    Location loc = op.getLoc();
    SmallVector<Value> newOperands;

    bool isFuncReturn = isa<func::FuncOp>(op->getParentOp());

    if (!isFuncReturn) {
      if (isa<stablehlo::WhileOp>(op->getParentOp())) {
        SmallVector<Value> flatOperands;
        if (concatDimension == "tuple") {
          for (auto operand : adaptor.getOperands()) {
            if (isa<TupleType>(operand.getType())) {
              Value hi =
                  extractLimb(operand, 0, rewriter, loc, concatDimension);
              Value lo =
                  extractLimb(operand, 1, rewriter, loc, concatDimension);
              flatOperands.push_back(hi);
              flatOperands.push_back(lo);
            } else {
              flatOperands.push_back(operand);
            }
          }
        } else {
          for (auto operand : adaptor.getOperands()) {
            flatOperands.push_back(operand);
          }
        }
        rewriter.replaceOpWithNewOp<stablehlo::ReturnOp>(op, flatOperands);
        return success();
      }

      for (auto operand : adaptor.getOperands()) {
        if (!getTypeConverter()->isLegal(operand.getType()))
          return failure();
      }
      rewriter.replaceOpWithNewOp<stablehlo::ReturnOp>(op,
                                                       adaptor.getOperands());
      return success();
    }

    for (auto operand : adaptor.getOperands()) {
      if (convertSignatures) {
        if (concatDimension == "tuple" && isa<TupleType>(operand.getType())) {
          newOperands.push_back(
              rewriter.create<stablehlo::GetTupleElementOp>(loc, operand, 0));
          newOperands.push_back(
              rewriter.create<stablehlo::GetTupleElementOp>(loc, operand, 1));
        } else {
          newOperands.push_back(operand);
        }
        continue;
      }

      Value actualOperand = operand;
      if (operand.getDefiningOp()) {
        if (auto castOp =
                dyn_cast<UnrealizedConversionCastOp>(operand.getDefiningOp())) {
          actualOperand = castOp.getOperand(0);
        }
      }

      if (auto tensorType =
              dyn_cast<RankedTensorType>(actualOperand.getType())) {
        Type elType = tensorType.getElementType();
        if (elType == targetType) {
          bool isTuple = concatDimension == "tuple";
          bool isFirst = concatDimension == "first";

          if (expansionSize == 1) {
            auto f64Type =
                RankedTensorType::get(tensorType.getShape(), sourceType);
            Value f64Val = rewriter.create<stablehlo::ConvertOp>(loc, f64Type,
                                                                 actualOperand);
            newOperands.push_back(f64Val);
            continue;
          }

          if (!isTuple && tensorType.getRank() > 0 &&
              tensorType.getShape()[(isFirst ? 0 : tensorType.getRank() - 1)] ==
                  2) {
            Value high =
                extractLimb(actualOperand, 0, rewriter, loc, concatDimension);
            Value low =
                extractLimb(actualOperand, 1, rewriter, loc, concatDimension);

            SmallVector<int64_t> origShape =
                llvm::to_vector(tensorType.getShape());
            origShape.erase(origShape.begin() +
                            (isFirst ? 0 : origShape.size() - 1));
            auto origTensorType = RankedTensorType::get(origShape, targetType);
            high = rewriter.create<stablehlo::ReshapeOp>(loc, origTensorType,
                                                         high);
            low =
                rewriter.create<stablehlo::ReshapeOp>(loc, origTensorType, low);

            auto f64Type = RankedTensorType::get(origShape, sourceType);
            Value high64 =
                rewriter.create<stablehlo::ConvertOp>(loc, f64Type, high);
            Value low64 =
                rewriter.create<stablehlo::ConvertOp>(loc, f64Type, low);

            Value sum = rewriter.create<stablehlo::AddOp>(loc, high64, low64);
            newOperands.push_back(sum);
            continue;
          }
        }
      } else if (auto tupleType = dyn_cast<TupleType>(operand.getType())) {
        if (tupleType.size() == 2) {
          Value high = extractLimb(operand, 0, rewriter, loc, concatDimension);
          Value low = extractLimb(operand, 1, rewriter, loc, concatDimension);

          auto f64Type = RankedTensorType::get(
              cast<RankedTensorType>(high.getType()).getShape(), sourceType);
          Value high64 =
              rewriter.create<stablehlo::ConvertOp>(loc, f64Type, high);
          Value low64 =
              rewriter.create<stablehlo::ConvertOp>(loc, f64Type, low);

          Value sum = rewriter.create<stablehlo::AddOp>(loc, high64, low64);
          newOperands.push_back(sum);
          continue;
        }
      }
      newOperands.push_back(operand);
    }

    rewriter.replaceOpWithNewOp<stablehlo::ReturnOp>(op, newOperands);
    return success();
  }
};

struct ReduceWindowOpConversion
    : public OpConversionPattern<stablehlo::ReduceWindowOp> {
  StringRef concatDimension;

  int expansionSize;

  Type sourceType;
  Type targetType;

  ReduceWindowOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                           StringRef concatDimension, int expansionSize,
                           Type sourceType, Type targetType)
      : OpConversionPattern<stablehlo::ReduceWindowOp>(typeConverter, context),
        concatDimension(concatDimension), expansionSize(expansionSize),
        sourceType(sourceType), targetType(targetType) {}

  LogicalResult
  matchAndRewrite(stablehlo::ReduceWindowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    bool hasPadding = false;
    if (auto paddingAttr = op.getPadding()) {
      for (auto p : paddingAttr->getValues<int64_t>()) {
        if (p > 0)
          hasPadding = true;
      }
    }
    if (expansionSize > 1 && hasPadding)
      return failure();
    bool isTuple = concatDimension == "tuple";

    if (expansionSize == 1) {
      if (adaptor.getOperands().size() < 2)
        return failure();
      Value operand = adaptor.getOperands()[0];
      Value initValue = adaptor.getOperands()[1];

      auto outType = cast<RankedTensorType>(op.getResults()[0].getType());
      auto partType = RankedTensorType::get(outType.getShape(), targetType);

      auto newOp = rewriter.create<stablehlo::ReduceWindowOp>(
          loc, partType, operand, initValue, op.getWindowDimensions(),
          op.getWindowStridesAttr() ? op.getWindowStridesAttr() : nullptr,
          op.getBaseDilationsAttr() ? op.getBaseDilationsAttr() : nullptr,
          op.getWindowDilationsAttr() ? op.getWindowDilationsAttr() : nullptr,
          op.getPaddingAttr() ? op.getPaddingAttr() : nullptr);

      rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                  newOp.getRegion().end());
      if (failed(rewriter.convertRegionTypes(&newOp.getRegion(),
                                             *getTypeConverter())))
        return failure();

      rewriter.replaceOp(op, newOp.getResults());
      return success();
    }

    if (isTuple) {
      return failure();
    }

    Value operand = adaptor.getOperands()[0];
    Value initValue = adaptor.getOperands()[1];

    Value hi_operand = extractLimb(operand, 0, rewriter, loc, concatDimension);
    Value lo_operand = extractLimb(operand, 1, rewriter, loc, concatDimension);
    Value hi_init = extractLimb(initValue, 0, rewriter, loc, concatDimension);
    Value lo_init = extractLimb(initValue, 1, rewriter, loc, concatDimension);

    auto outType = cast<RankedTensorType>(op.getResults()[0].getType());
    auto partType = RankedTensorType::get(outType.getShape(), targetType);

    auto hi_op = rewriter.create<stablehlo::ReduceWindowOp>(
        loc, partType, hi_operand, hi_init, op.getWindowDimensions(),
        op.getWindowStridesAttr() ? op.getWindowStridesAttr() : nullptr,
        op.getBaseDilationsAttr() ? op.getBaseDilationsAttr() : nullptr,
        op.getWindowDilationsAttr() ? op.getWindowDilationsAttr() : nullptr,
        op.getPaddingAttr() ? op.getPaddingAttr() : nullptr);

    rewriter.inlineRegionBefore(op.getRegion(), hi_op.getRegion(),
                                hi_op.getRegion().end());
    if (failed(rewriter.convertRegionTypes(&hi_op.getRegion(),
                                           *getTypeConverter())))
      return failure();

    auto lo_op = rewriter.create<stablehlo::ReduceWindowOp>(
        loc, partType, lo_operand, lo_init, op.getWindowDimensions(),
        op.getWindowStridesAttr() ? op.getWindowStridesAttr() : nullptr,
        op.getBaseDilationsAttr() ? op.getBaseDilationsAttr() : nullptr,
        op.getWindowDilationsAttr() ? op.getWindowDilationsAttr() : nullptr,
        op.getPaddingAttr() ? op.getPaddingAttr() : nullptr);

    rewriter.cloneRegionBefore(hi_op.getRegion(), lo_op.getRegion(),
                               lo_op.getRegion().end());

    Value packed = packLimbs(hi_op.getResult(0), lo_op.getResult(0), rewriter,
                             loc, concatDimension);
    rewriter.replaceOp(op, packed);
    return success();
  }
};

struct BroadcastInDimOpConversion
    : public OpConversionPattern<stablehlo::BroadcastInDimOp> {
  StringRef concatDimension;

  BroadcastInDimOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                             StringRef concatDimension)
      : OpConversionPattern<stablehlo::BroadcastInDimOp>(typeConverter,
                                                         context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::BroadcastInDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool isTuple = concatDimension == "tuple";
    bool isFirst = concatDimension == "first";

    if (isTuple) {
      Value high = extractLimb(adaptor.getOperands()[0], 0, rewriter, loc,
                               concatDimension);
      Value low = extractLimb(adaptor.getOperands()[0], 1, rewriter, loc,
                              concatDimension);

      auto outType =
          cast<TupleType>(getTypeConverter()->convertType(op.getType()));
      auto partType = outType.getType(0);

      auto bcastHigh = rewriter.create<stablehlo::BroadcastInDimOp>(
          loc, partType, high, op.getBroadcastDimensions());
      auto bcastLow = rewriter.create<stablehlo::BroadcastInDimOp>(
          loc, partType, low, op.getBroadcastDimensions());

      Value packed =
          packLimbs(bcastHigh, bcastLow, rewriter, loc, concatDimension);
      rewriter.replaceOp(op, packed);
      return success();
    }

    SmallVector<int64_t> broadcastDims =
        llvm::to_vector(op.getBroadcastDimensions());
    auto outType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
    if (outType == op.getType())
      return failure();

    if (isFirst) {
      SmallVector<int64_t> newBroadcastDims;
      newBroadcastDims.push_back(0);
      for (auto dim : broadcastDims) {
        newBroadcastDims.push_back(dim + 1);
      }
      broadcastDims = std::move(newBroadcastDims);
    } else {
      broadcastDims.push_back(outType.getRank() - 1);
    }

    auto bcastOp = rewriter.create<stablehlo::BroadcastInDimOp>(
        loc, outType, adaptor.getOperands()[0],
        rewriter.getDenseI64ArrayAttr(broadcastDims));

    rewriter.replaceOp(op, bcastOp);
    return success();
  }
};

struct TransposeOpConversion
    : public OpConversionPattern<stablehlo::TransposeOp> {
  StringRef concatDimension;

  TransposeOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                        StringRef concatDimension)
      : OpConversionPattern<stablehlo::TransposeOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool isTuple = concatDimension == "tuple";
    bool isFirst = concatDimension == "first";

    if (isTuple) {
      Value high = extractLimb(adaptor.getOperands()[0], 0, rewriter, loc,
                               concatDimension);
      Value low = extractLimb(adaptor.getOperands()[0], 1, rewriter, loc,
                              concatDimension);

      auto outType =
          cast<TupleType>(getTypeConverter()->convertType(op.getType()));
      auto partType = outType.getType(0);

      auto transHigh = rewriter.create<stablehlo::TransposeOp>(
          loc, partType, high, op.getPermutation());
      auto transLow = rewriter.create<stablehlo::TransposeOp>(
          loc, partType, low, op.getPermutation());

      Value packed =
          packLimbs(transHigh, transLow, rewriter, loc, concatDimension);
      rewriter.replaceOp(op, packed);
      return success();
    }

    SmallVector<int64_t> permutation = llvm::to_vector(op.getPermutation());
    auto outType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

    if (isFirst) {
      SmallVector<int64_t> newPerm;
      newPerm.push_back(0);
      for (auto idx : permutation) {
        newPerm.push_back(idx + 1);
      }
      permutation = std::move(newPerm);
    } else {
      permutation.push_back(outType.getRank() - 1);
    }

    auto transOp = rewriter.create<stablehlo::TransposeOp>(
        loc, outType, adaptor.getOperands()[0],
        rewriter.getDenseI64ArrayAttr(permutation));

    rewriter.replaceOp(op, transOp);
    return success();
  }
};

struct ReshapeOpConversion : public OpConversionPattern<stablehlo::ReshapeOp> {
  StringRef concatDimension;

  ReshapeOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                      StringRef concatDimension)
      : OpConversionPattern<stablehlo::ReshapeOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool isTuple = concatDimension == "tuple";

    if (isTuple) {
      Value high = extractLimb(adaptor.getOperands()[0], 0, rewriter, loc,
                               concatDimension);
      Value low = extractLimb(adaptor.getOperands()[0], 1, rewriter, loc,
                              concatDimension);

      auto outType =
          cast<TupleType>(getTypeConverter()->convertType(op.getType()));
      auto partType = outType.getType(0);

      auto reshapeHigh =
          rewriter.create<stablehlo::ReshapeOp>(loc, partType, high);
      auto reshapeLow =
          rewriter.create<stablehlo::ReshapeOp>(loc, partType, low);

      Value packed =
          packLimbs(reshapeHigh, reshapeLow, rewriter, loc, concatDimension);
      rewriter.replaceOp(op, packed);
      return success();
    }

    auto outType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

    auto reshapeOp = rewriter.create<stablehlo::ReshapeOp>(
        loc, outType, adaptor.getOperands()[0]);

    rewriter.replaceOp(op, reshapeOp);
    return success();
  }
};

struct CompareOpConversion : public OpConversionPattern<stablehlo::CompareOp> {
  StringRef concatDimension;

  CompareOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                      StringRef concatDimension)
      : OpConversionPattern<stablehlo::CompareOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::CompareOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];

    auto resultType = cast<RankedTensorType>(op.getType());
    auto direction = op.getComparisonDirection();

    Type expectedLhsTy =
        getTypeConverter()->convertType(op.getOperands()[0].getType());
    Type expectedRhsTy =
        getTypeConverter()->convertType(op.getOperands()[1].getType());

    if (lhs.getType() != expectedLhsTy) {
      lhs = rewriter.create<UnrealizedConversionCastOp>(loc, expectedLhsTy, lhs)
                .getResult(0);
    }
    if (rhs.getType() != expectedRhsTy) {
      rhs = rewriter.create<UnrealizedConversionCastOp>(loc, expectedRhsTy, rhs)
                .getResult(0);
    }

    bool isTuple = (concatDimension == "tuple");
    if (!isTuple && (direction == stablehlo::ComparisonDirection::EQ ||
                     direction == stablehlo::ComparisonDirection::NE)) {
      Value cmp =
          rewriter.create<stablehlo::CompareOp>(loc, lhs, rhs, direction);
      auto tensorType = cast<RankedTensorType>(lhs.getType());
      int reduceDim =
          (concatDimension == "first") ? 0 : (tensorType.getRank() - 1);

      Value init_val;
      auto boolType = rewriter.getI1Type();
      if (direction == stablehlo::ComparisonDirection::EQ) {
        init_val = rewriter.create<stablehlo::ConstantOp>(
            loc,
            DenseElementsAttr::get(
                RankedTensorType::get(ArrayRef<int64_t>{}, boolType), true));
      } else {
        init_val = rewriter.create<stablehlo::ConstantOp>(
            loc,
            DenseElementsAttr::get(
                RankedTensorType::get(ArrayRef<int64_t>{}, boolType), false));
      }

      auto reduceOp = rewriter.create<stablehlo::ReduceOp>(
          loc, cmp, init_val, rewriter.getDenseI64ArrayAttr({reduceDim}));
      Block *reduceBlock = new Block();
      reduceOp.getBody().push_back(reduceBlock);
      auto tensorBoolType =
          RankedTensorType::get(ArrayRef<int64_t>{}, boolType);
      reduceBlock->addArguments({tensorBoolType, tensorBoolType}, {loc, loc});
      auto blockBuilder = OpBuilder::atBlockBegin(reduceBlock);
      Value reducedRes;
      if (direction == stablehlo::ComparisonDirection::EQ) {
        reducedRes = blockBuilder.create<stablehlo::AndOp>(
            loc, reduceBlock->getArgument(0), reduceBlock->getArgument(1));
      } else {
        reducedRes = blockBuilder.create<stablehlo::OrOp>(
            loc, reduceBlock->getArgument(0), reduceBlock->getArgument(1));
      }
      blockBuilder.create<stablehlo::ReturnOp>(loc, reducedRes);

      Value res = reduceOp.getResults()[0];
      if (res.getType() != resultType) {
        res = rewriter.create<stablehlo::ReshapeOp>(loc, resultType, res);
      }
      rewriter.replaceOp(op, res);
      return success();
    }

    Value lhs_hi = extractLimb(lhs, 0, rewriter, loc, concatDimension);
    Value lhs_lo = extractLimb(lhs, 1, rewriter, loc, concatDimension);
    Value rhs_hi = extractLimb(rhs, 0, rewriter, loc, concatDimension);
    Value rhs_lo = extractLimb(rhs, 1, rewriter, loc, concatDimension);

    Value res;
    if (direction == stablehlo::ComparisonDirection::EQ) {
      Value hi_eq = rewriter.create<stablehlo::CompareOp>(
          loc, lhs_hi, rhs_hi, stablehlo::ComparisonDirection::EQ);
      Value lo_eq = rewriter.create<stablehlo::CompareOp>(
          loc, lhs_lo, rhs_lo, stablehlo::ComparisonDirection::EQ);

      hi_eq = rewriter.create<stablehlo::ReshapeOp>(loc, resultType, hi_eq);
      lo_eq = rewriter.create<stablehlo::ReshapeOp>(loc, resultType, lo_eq);

      res = rewriter.create<stablehlo::AndOp>(loc, hi_eq, lo_eq);
    } else if (direction == stablehlo::ComparisonDirection::NE) {
      Value hi_ne = rewriter.create<stablehlo::CompareOp>(
          loc, lhs_hi, rhs_hi, stablehlo::ComparisonDirection::NE);
      Value lo_ne = rewriter.create<stablehlo::CompareOp>(
          loc, lhs_lo, rhs_lo, stablehlo::ComparisonDirection::NE);

      hi_ne = rewriter.create<stablehlo::ReshapeOp>(loc, resultType, hi_ne);
      lo_ne = rewriter.create<stablehlo::ReshapeOp>(loc, resultType, lo_ne);

      res = rewriter.create<stablehlo::OrOp>(loc, hi_ne, lo_ne);
    } else {
      stablehlo::ComparisonDirection dir_hi;
      stablehlo::ComparisonDirection dir_lo;
      if (direction == stablehlo::ComparisonDirection::GE) {
        dir_hi = stablehlo::ComparisonDirection::GT;
        dir_lo = stablehlo::ComparisonDirection::GE;
      } else if (direction == stablehlo::ComparisonDirection::GT) {
        dir_hi = stablehlo::ComparisonDirection::GT;
        dir_lo = stablehlo::ComparisonDirection::GT;
      } else if (direction == stablehlo::ComparisonDirection::LE) {
        dir_hi = stablehlo::ComparisonDirection::LT;
        dir_lo = stablehlo::ComparisonDirection::LE;
      } else if (direction == stablehlo::ComparisonDirection::LT) {
        dir_hi = stablehlo::ComparisonDirection::LT;
        dir_lo = stablehlo::ComparisonDirection::LT;
      } else {
        return failure();
      }

      Value hi_gt =
          rewriter.create<stablehlo::CompareOp>(loc, lhs_hi, rhs_hi, dir_hi);
      Value hi_eq = rewriter.create<stablehlo::CompareOp>(
          loc, lhs_hi, rhs_hi, stablehlo::ComparisonDirection::EQ);
      Value lo_cond =
          rewriter.create<stablehlo::CompareOp>(loc, lhs_lo, rhs_lo, dir_lo);

      LLVM_DEBUG(llvm::dbgs() << "CompareOpConversion other SelectOp:\n");
      LLVM_DEBUG(llvm::dbgs() << "  hi_eq type: " << hi_eq.getType() << "\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "  lo_cond type: " << lo_cond.getType() << "\n");
      LLVM_DEBUG(llvm::dbgs() << "  hi_gt type: " << hi_gt.getType() << "\n");

      res = rewriter.create<stablehlo::SelectOp>(loc, hi_eq, lo_cond, hi_gt);
    }

    if (res.getType() != resultType) {
      res = rewriter.create<stablehlo::ReshapeOp>(loc, resultType, res);
    }

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct MaxOpConversion : public OpConversionPattern<stablehlo::MaxOp> {
  StringRef concatDimension;

  MaxOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                  StringRef concatDimension)
      : OpConversionPattern<stablehlo::MaxOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::MaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value uncoverted_lhs = op.getOperands()[0];
    Value uncoverted_rhs = op.getOperands()[1];

    Value cmp = rewriter.create<stablehlo::CompareOp>(
        loc, uncoverted_lhs, uncoverted_rhs,
        stablehlo::ComparisonDirection::GT);

    Value select = rewriter.create<stablehlo::SelectOp>(
        loc, cmp, uncoverted_lhs, uncoverted_rhs);

    rewriter.replaceOp(op, select);
    return success();
  }
};

struct DynamicUpdateSliceOpConversion
    : public OpConversionPattern<stablehlo::DynamicUpdateSliceOp> {
  StringRef concatDimension;
  Type targetType;

  DynamicUpdateSliceOpConversion(TypeConverter &typeConverter,
                                 MLIRContext *context,
                                 StringRef concatDimension, Type targetType)
      : OpConversionPattern<stablehlo::DynamicUpdateSliceOp>(typeConverter,
                                                             context),
        concatDimension(concatDimension), targetType(targetType) {}

  LogicalResult
  matchAndRewrite(stablehlo::DynamicUpdateSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value operand = adaptor.getOperands()[0];
    Value update = adaptor.getOperands()[1];
    auto indices = adaptor.getOperands().drop_front(2);

    bool isTuple = concatDimension == "tuple";

    if (!isTuple) {
      Type indexType = indices[0].getType();
      Value zero;
      if (auto tensorType = dyn_cast<RankedTensorType>(indexType)) {
        zero = rewriter.create<stablehlo::ConstantOp>(
            loc, DenseIntElementsAttr::get(tensorType, 0));
      } else {
        zero = rewriter.create<stablehlo::ConstantOp>(
            loc, rewriter.getIntegerAttr(indexType, 0));
      }

      SmallVector<Value> newIndices;
      if (concatDimension == "first") {
        newIndices.push_back(zero);
        for (auto idx : indices)
          newIndices.push_back(idx);
      } else {
        for (auto idx : indices)
          newIndices.push_back(idx);
        newIndices.push_back(zero);
      }

      Value res = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
          loc, operand.getType(), operand, update, newIndices);
      rewriter.replaceOp(op, res);
      return success();
    }

    Value operand_hi = extractLimb(operand, 0, rewriter, loc, concatDimension);
    Value operand_lo = extractLimb(operand, 1, rewriter, loc, concatDimension);
    Value update_hi = extractLimb(update, 0, rewriter, loc, concatDimension);
    Value update_lo = extractLimb(update, 1, rewriter, loc, concatDimension);

    Value hi_N = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
        loc, operand_hi.getType(), operand_hi, update_hi, indices);
    Value lo_N = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
        loc, operand_lo.getType(), operand_lo, update_lo, indices);

    Value packed = packLimbs(hi_N, lo_N, rewriter, loc, concatDimension);
    rewriter.replaceOp(op, packed);
    return success();
  }
};

struct DotGeneralToMulReducePattern
    : public OpRewritePattern<stablehlo::DotGeneralOp> {
  using OpRewritePattern<stablehlo::DotGeneralOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsType = dyn_cast<RankedTensorType>(op.getOperands()[0].getType());
    auto rhsType = dyn_cast<RankedTensorType>(op.getOperands()[1].getType());
    if (!lhsType || !rhsType)
      return failure();

    auto dNums = op.getDotDimensionNumbers();
    auto lhsContracting = dNums.getLhsContractingDimensions();
    auto rhsContracting = dNums.getRhsContractingDimensions();

    bool lhsAllContracted = (lhsContracting.size() == lhsType.getRank());
    bool rhsAllContracted = (rhsContracting.size() == rhsType.getRank());

    if (!lhsAllContracted && !rhsAllContracted)
      return failure(); // Not a reduction-like operation

    Value lhs = op.getOperands()[0];
    Value rhs = op.getOperands()[1];

    Value broadcastedLhs;
    auto dimsToReduce = lhsAllContracted ? rhsContracting : lhsContracting;

    if (lhsAllContracted) {
      broadcastedLhs = rewriter.create<stablehlo::BroadcastInDimOp>(
          op.getLoc(), rhs.getType(), lhs,
          rewriter.getDenseI64ArrayAttr(rhsContracting));
    } else {
      broadcastedLhs = lhs;
      Value broadcastedRhs = rewriter.create<stablehlo::BroadcastInDimOp>(
          op.getLoc(), lhs.getType(), rhs,
          rewriter.getDenseI64ArrayAttr(lhsContracting));
      rhs = broadcastedRhs;
    }

    Value mul = rewriter.create<stablehlo::MulOp>(
        op.getLoc(), broadcastedLhs.getType(), broadcastedLhs, rhs);

    auto elemType = lhsType.getElementType();

    if (dimsToReduce.size() == 1) {
      int64_t reduceDim = dimsToReduce[0];
      auto mulType = cast<RankedTensorType>(mul.getType());
      int64_t reduceDimSize = mulType.getShape()[reduceDim];

      if (reduceDimSize <= 64) {
        Value sum;
        for (int64_t i = 0; i < reduceDimSize; ++i) {
          SmallVector<int64_t> startIndices(mulType.getRank(), 0);
          SmallVector<int64_t> limitIndices(mulType.getShape().begin(),
                                            mulType.getShape().end());
          SmallVector<int64_t> strides(mulType.getRank(), 1);

          startIndices[reduceDim] = i;
          limitIndices[reduceDim] = i + 1;

          Value slice = rewriter.create<stablehlo::SliceOp>(
              op.getLoc(), mul, rewriter.getDenseI64ArrayAttr(startIndices),
              rewriter.getDenseI64ArrayAttr(limitIndices),
              rewriter.getDenseI64ArrayAttr(strides));

          SmallVector<int64_t> newShape;
          for (int64_t d = 0; d < mulType.getRank(); ++d) {
            if (d != reduceDim) {
              newShape.push_back(mulType.getShape()[d]);
            }
          }
          auto reshapedType =
              RankedTensorType::get(newShape, mulType.getElementType());
          Value reshapedSlice = rewriter.create<stablehlo::ReshapeOp>(
              op.getLoc(), reshapedType, slice);

          if (i == 0) {
            sum = reshapedSlice;
          } else {
            sum = rewriter.create<stablehlo::AddOp>(op.getLoc(), reshapedType,
                                                    sum, reshapedSlice);
          }
        }
        rewriter.replaceOp(op, sum);
        return success();
      }
    }

    auto zeroAttr = rewriter.getFloatAttr(elemType, 0.0);
    auto scalarType = RankedTensorType::get({}, elemType);
    Value zero = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), SplatElementsAttr::get(scalarType, zeroAttr));

    auto reduceOp = rewriter.create<stablehlo::ReduceOp>(
        op.getLoc(), op.getType(), mul, zero,
        rewriter.getDenseI64ArrayAttr(dimsToReduce));

    Block *block = new Block();
    reduceOp.getBody().push_back(block);
    block->addArguments({scalarType, scalarType}, {op.getLoc(), op.getLoc()});
    auto b = OpBuilder::atBlockBegin(block);
    Value add = b.create<stablehlo::AddOp>(
        op.getLoc(), scalarType, block->getArgument(0), block->getArgument(1));
    b.create<stablehlo::ReturnOp>(op.getLoc(), add);

    rewriter.replaceOp(op, reduceOp.getResults());
    return success();
  }
};
Value getMaxValue(Value tensor, OpBuilder &builder, Location loc) {
  auto type = cast<RankedTensorType>(tensor.getType());
  auto absOp = builder.create<stablehlo::AbsOp>(loc, tensor);

  auto elemType = type.getElementType();
  auto zeroAttr = builder.getFloatAttr(elemType, 0.0);
  auto scalarType = RankedTensorType::get({}, elemType);
  Value zero = builder.create<stablehlo::ConstantOp>(
      loc, SplatElementsAttr::get(scalarType, zeroAttr));

  SmallVector<int64_t> dims;
  for (int64_t i = 0; i < type.getRank(); ++i)
    dims.push_back(i);

  auto reduceOp = builder.create<stablehlo::ReduceOp>(
      loc, TypeRange{scalarType}, ValueRange{absOp.getResult()},
      ValueRange{zero}, builder.getDenseI64ArrayAttr(dims));

  Block *block = new Block();
  reduceOp.getBody().push_back(block);
  block->addArguments({scalarType, scalarType}, {loc, loc});

  auto b = OpBuilder::atBlockBegin(block);
  Value max = b.create<stablehlo::MaxOp>(loc, block->getArgument(0),
                                         block->getArgument(1));
  b.create<stablehlo::ReturnOp>(loc, max);

  return reduceOp.getResult(0);
}
Value lookupTable(Value index, ArrayRef<double> table, OpBuilder &builder,
                  Location loc, Type tensorType) {
  auto rankedTensorType = cast<RankedTensorType>(tensorType);
  auto floatTy = cast<FloatType>(rankedTensorType.getElementType());

  auto getConst = [&](double val) {
    return builder.create<stablehlo::ConstantOp>(
        loc, SplatElementsAttr::get(rankedTensorType,
                                    builder.getFloatAttr(floatTy, val)));
  };

  auto getIntConst = [&](int val) {
    auto intType = RankedTensorType::get(rankedTensorType.getShape(),
                                         builder.getI32Type());
    return builder.create<stablehlo::ConstantOp>(
        loc, SplatElementsAttr::get(intType, builder.getI32IntegerAttr(val)));
  };

  std::vector<Value> current_level;
  for (double val : table) {
    current_level.push_back(getConst(val));
  }

  for (int bit = 0; bit < 5; ++bit) {
    std::vector<Value> next_level;
    Value bit_mask = getIntConst(1 << bit);
    Value is_set_bits = builder.create<stablehlo::AndOp>(loc, index, bit_mask);
    Value zero = getIntConst(0);
    Value cond = builder.create<stablehlo::CompareOp>(
        loc, is_set_bits, zero, stablehlo::ComparisonDirection::NE);

    for (size_t i = 0; i < current_level.size(); i += 2) {
      Value false_val = current_level[i];
      Value true_val = current_level[i + 1];
      Value selected =
          builder.create<stablehlo::SelectOp>(loc, cond, true_val, false_val);
      next_level.push_back(selected);
    }
    current_level = next_level;
  }

  return current_level[0];
}

struct LogOpConversion : public OpConversionPattern<stablehlo::LogOp> {
  LogOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                  StringRef concatDimension)
      : OpConversionPattern<stablehlo::LogOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  StringRef concatDimension;

  LogicalResult
  matchAndRewrite(stablehlo::LogOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value operand = adaptor.getOperands()[0];

    // Support only expansion-size=2 for now
    // We assume rank 1 or 2 based on tests

    Value x_hi = extractLimb(operand, 0, rewriter, loc, concatDimension);
    Value x_lo = extractLimb(operand, 1, rewriter, loc, concatDimension);

    auto tensorType = cast<RankedTensorType>(x_hi.getType());
    auto floatTy = cast<FloatType>(tensorType.getElementType());

    auto getConst = [&](double val) {
      return rewriter.create<stablehlo::ConstantOp>(
          loc, SplatElementsAttr::get(tensorType,
                                      rewriter.getFloatAttr(floatTy, val)));
    };

    // Tables from MultiFloats.jl
    static const double kCenters[32] = {
        1.015625, 1.046875, 1.078125, 1.109375, 1.140625, 1.171875, 1.203125,
        1.234375, 1.265625, 1.296875, 1.328125, 1.359375, 1.390625, 1.421875,
        1.453125, 1.484375, 1.515625, 1.546875, 1.578125, 1.609375, 1.640625,
        1.671875, 1.703125, 1.734375, 1.765625, 1.796875, 1.828125, 1.859375,
        1.890625, 1.921875, 1.953125, 1.984375};

    static const double kValues[32][2] = {
        {0.022367813, 3.3533545e-10}, {0.06608919, -8.4925476e-11},
        {0.10852446, 1.5795268e-10},  {0.14974712, 1.1508384e-9},
        {0.18982457, -7.365062e-9},   {0.22881868, 5.6795204e-9},
        {0.26678655, -4.820159e-9},   {0.30378073, 1.3638071e-8},
        {0.33985, -5.6030767e-9},     {0.37503943, 2.8744058e-9},
        {0.40939093, 9.776618e-9},    {0.44294348, 1.2257648e-8},
        {0.47573343, 1.772901e-9},    {0.5077946, 2.1592129e-8},
        {0.5391588, -9.997926e-9},    {0.56985563, -2.2066848e-8},
        {0.5999128, 1.9940575e-8},    {0.6293566, -2.6163132e-9},
        {0.65821147, 1.3101526e-8},   {0.68650055, -2.2133188e-8},
        {0.7142455, 1.9485734e-8},    {0.741467, -1.2652808e-8},
        {0.7681843, 2.0539561e-8},    {0.7944159, -2.4820396e-8},
        {0.820179, -2.3180515e-8},    {0.84549004, 1.2549447e-8},
        {0.8703647, -6.006348e-9},    {0.89481777, -6.2194916e-9},
        {0.91886324, 3.7045023e-10},  {0.9425145, 2.6178931e-8},
        {0.9657843, -2.6632469e-8},   {0.9886847, -2.706832e-8}};

    static const double kNarrowCoefs[4][2] = {
        {+0x1.715476p+001, +0x1.4AE0C0p-025},
        {+0x1.EC709Ep-001, -0x1.E2FDB2p-028},
        {+0x1.2776C6p-001, 0.0},
        {+0x1.A6217Cp-002, 0.0}};

    static const double kWideCoefs[5][2] = {
        {+0x1.715476p+001, +0x1.4AE0C0p-025},
        {+0x1.EC709Ep-001, -0x1.E2FE88p-028},
        {+0x1.2776C6p-001, -0x1.DE12AAp-026},
        {+0x1.A61738p-002, 0.0},
        {+0x1.48FE3Ap-002, 0.0}};

    static const double kLn2[2] = {0.6931471824645996, -1.9046542101288585e-9};

    // 1. Extract exponent and significand
    Type intEltTy = rewriter.getIntegerType(floatTy.getWidth());
    auto intType = RankedTensorType::get(tensorType.getShape(), intEltTy);
    Value bits =
        rewriter.create<stablehlo::BitcastConvertOp>(loc, intType, x_hi);

    auto getIntConst = [&](int val) {
      return rewriter.create<stablehlo::ConstantOp>(
          loc, SplatElementsAttr::get(intType,
                                      rewriter.getIntegerAttr(intEltTy, val)));
    };

    int mantissaWidth = floatTy.getFPMantissaWidth() - 1;
    int expBits = floatTy.getWidth() - floatTy.getFPMantissaWidth();
    int bias = (1 << (expBits - 1)) - 1;

    int maskVal = ((1 << expBits) - 1) << mantissaWidth;
    int shiftAmount = mantissaWidth;

    Value mask = getIntConst(maskVal);
    Value biased_bits = rewriter.create<stablehlo::AndOp>(loc, bits, mask);
    Value biased_exp = rewriter.create<stablehlo::ShiftRightLogicalOp>(
        loc, biased_bits, getIntConst(shiftAmount));
    Value exp = rewriter.create<stablehlo::SubtractOp>(loc, biased_exp,
                                                       getIntConst(bias));

    Value biased_neg_exp =
        rewriter.create<stablehlo::SubtractOp>(loc, getIntConst(bias), exp);
    Value scale_bits = rewriter.create<stablehlo::ShiftLeftOp>(
        loc, biased_neg_exp, getIntConst(shiftAmount));
    Value scale = rewriter.create<stablehlo::BitcastConvertOp>(loc, tensorType,
                                                               scale_bits);

    // m = ldexp(x, -e)
    Value m_hi = rewriter.create<stablehlo::MulOp>(loc, x_hi, scale);
    Value m_lo = rewriter.create<stablehlo::MulOp>(loc, x_lo, scale);

    // 2. Table lookup
    Value index_bits = rewriter.create<stablehlo::ShiftRightLogicalOp>(
        loc, bits, getIntConst(mantissaWidth - 5));
    Value index =
        rewriter.create<stablehlo::AndOp>(loc, index_bits, getIntConst(0x1F));

    Value index_i32 = index;
    if (floatTy.getWidth() != 32) {
      auto int32Type =
          RankedTensorType::get(tensorType.getShape(), rewriter.getI32Type());
      index_i32 = rewriter.create<stablehlo::ConvertOp>(loc, int32Type, index);
    }

    Value center_hi =
        lookupTable(index_i32, kCenters, rewriter, loc, tensorType);
    Value center_lo = getConst(0.0); // Since all low parts are zero!

    std::vector<double> values_hi, values_lo;
    for (int i = 0; i < 32; ++i) {
      values_hi.push_back(kValues[i][0]);
      values_lo.push_back(kValues[i][1]);
    }

    Value val_hi = lookupTable(index_i32, values_hi, rewriter, loc, tensorType);
    Value val_lo = lookupTable(index_i32, values_lo, rewriter, loc, tensorType);
    // 3. Polynomial evaluation
    // t_direct = (x - 1) / (x + 1)
    auto [neg_one_hi, neg_one_lo] =
        std::make_pair(getConst(-1.0), getConst(0.0));
    auto [x_minus_one_hi, x_minus_one_lo] =
        multiFloatAdd(x_hi, x_lo, neg_one_hi, neg_one_lo, rewriter, loc);
    auto [x_plus_one_hi, x_plus_one_lo] =
        multiFloatAdd(x_hi, x_lo, getConst(1.0), getConst(0.0), rewriter, loc);
    auto [t_direct_hi, t_direct_lo] =
        multiFloatDiv(x_minus_one_hi, x_minus_one_lo, x_plus_one_hi,
                      x_plus_one_lo, rewriter, loc);

    // t_table = (m - center) / (m + center)
    Value neg_center_hi = rewriter.create<stablehlo::NegOp>(loc, center_hi);
    Value neg_center_lo = rewriter.create<stablehlo::NegOp>(loc, center_lo);
    auto [m_minus_center_hi, m_minus_center_lo] =
        multiFloatAdd(m_hi, m_lo, neg_center_hi, neg_center_lo, rewriter, loc);
    auto [m_plus_center_hi, m_plus_center_lo] =
        multiFloatAdd(m_hi, m_lo, center_hi, center_lo, rewriter, loc);
    auto [t_table_hi, t_table_lo] =
        multiFloatDiv(m_minus_center_hi, m_minus_center_lo, m_plus_center_hi,
                      m_plus_center_lo, rewriter, loc);

    // t^2
    auto [t_direct_sq_hi, t_direct_sq_lo] = multiFloatMul(
        t_direct_hi, t_direct_lo, t_direct_hi, t_direct_lo, rewriter, loc);
    auto [t_table_sq_hi, t_table_sq_lo] = multiFloatMul(
        t_table_hi, t_table_lo, t_table_hi, t_table_lo, rewriter, loc);

    // Evaluate polynomials
    // For narrow (table)
    Value p_table_hi = getConst(kNarrowCoefs[3][0]);
    Value p_table_lo = getConst(kNarrowCoefs[3][1]);

    for (int i = 2; i >= 0; --i) {
      auto [mul_hi, mul_lo] = multiFloatMul(
          p_table_hi, p_table_lo, t_table_sq_hi, t_table_sq_lo, rewriter, loc);
      auto [add_hi, add_lo] =
          multiFloatAdd(mul_hi, mul_lo, getConst(kNarrowCoefs[i][0]),
                        getConst(kNarrowCoefs[i][1]), rewriter, loc);
      p_table_hi = add_hi;
      p_table_lo = add_lo;
    }

    // For wide (direct)
    Value p_direct_hi = getConst(kWideCoefs[4][0]);
    Value p_direct_lo = getConst(kWideCoefs[4][1]);

    for (int i = 3; i >= 0; --i) {
      auto [mul_hi, mul_lo] =
          multiFloatMul(p_direct_hi, p_direct_lo, t_direct_sq_hi,
                        t_direct_sq_lo, rewriter, loc);
      auto [add_hi, add_lo] =
          multiFloatAdd(mul_hi, mul_lo, getConst(kWideCoefs[i][0]),
                        getConst(kWideCoefs[i][1]), rewriter, loc);
      p_direct_hi = add_hi;
      p_direct_lo = add_lo;
    }

    auto [res_direct_hi, res_direct_lo] = multiFloatMul(
        t_direct_hi, t_direct_lo, p_direct_hi, p_direct_lo, rewriter, loc);
    auto [res_table_hi, res_table_lo] = multiFloatMul(
        t_table_hi, t_table_lo, p_table_hi, p_table_lo, rewriter, loc);
    // 4. Combine results
    Value e_f32 = rewriter.create<stablehlo::ConvertOp>(loc, tensorType, exp);
    auto [e_plus_val_hi, e_plus_val_lo] =
        multiFloatAdd(e_f32, getConst(0.0), val_hi, val_lo, rewriter, loc);
    auto [else_hi, else_lo] =
        multiFloatAdd(e_plus_val_hi, e_plus_val_lo, res_table_hi, res_table_lo,
                      rewriter, loc);

    Value direct_lo = getConst(0.9375);
    Value direct_hi = getConst(1.0625);

    Value cond1 = rewriter.create<stablehlo::CompareOp>(
        loc, direct_lo, x_hi, stablehlo::ComparisonDirection::LT);
    Value cond2 = rewriter.create<stablehlo::CompareOp>(
        loc, x_hi, direct_hi, stablehlo::ComparisonDirection::LT);
    Value cond = rewriter.create<stablehlo::AndOp>(loc, cond1, cond2);

    Value log2_hi =
        rewriter.create<stablehlo::SelectOp>(loc, cond, res_direct_hi, else_hi);
    Value log2_lo =
        rewriter.create<stablehlo::SelectOp>(loc, cond, res_direct_lo, else_lo);

    // Multiply by ln(2)
    auto [res_hi, res_lo] = multiFloatMul(log2_hi, log2_lo, getConst(kLn2[0]),
                                          getConst(kLn2[1]), rewriter, loc);

    // Handle special cases
    Value zero = getConst(0.0);
    Value one = getConst(1.0);
    Value nan = getConst(std::numeric_limits<double>::quiet_NaN());
    Value inf = getConst(std::numeric_limits<double>::infinity());
    Value neg_inf = getConst(-std::numeric_limits<double>::infinity());

    // Check x == 0
    Value is_zero = rewriter.create<stablehlo::CompareOp>(
        loc, x_hi, zero, stablehlo::ComparisonDirection::EQ);

    // Check x == 1
    Value is_one_hi = rewriter.create<stablehlo::CompareOp>(
        loc, x_hi, one, stablehlo::ComparisonDirection::EQ);
    Value is_one_lo = rewriter.create<stablehlo::CompareOp>(
        loc, x_lo, zero, stablehlo::ComparisonDirection::EQ);
    Value is_one = rewriter.create<stablehlo::AndOp>(loc, is_one_hi, is_one_lo);

    // Check x < 0
    Value is_neg = rewriter.create<stablehlo::CompareOp>(
        loc, x_hi, zero, stablehlo::ComparisonDirection::LT);

    // Check x == +inf
    Value is_inf = rewriter.create<stablehlo::CompareOp>(
        loc, x_hi, inf, stablehlo::ComparisonDirection::EQ);

    Value packed_res =
        packLimbs(res_hi, res_lo, rewriter, loc, concatDimension);
    Value packed_zero = packLimbs(zero, zero, rewriter, loc, concatDimension);
    Value packed_inf = packLimbs(inf, zero, rewriter, loc, concatDimension);
    Value packed_neg_inf =
        packLimbs(neg_inf, zero, rewriter, loc, concatDimension);
    Value packed_nan = packLimbs(nan, nan, rewriter, loc, concatDimension);

    auto limbType = cast<RankedTensorType>(x_hi.getType());
    int concatDim = (concatDimension == "first") ? 0 : limbType.getRank() - 1;
    SmallVector<int64_t> outShape = llvm::to_vector(limbType.getShape());
    outShape[concatDim] = 2;
    auto i1Type = RankedTensorType::get(outShape, rewriter.getI1Type());

    SmallVector<int64_t> bcastDims;
    for (int i = 0; i < limbType.getRank(); ++i)
      bcastDims.push_back(i);

    auto broadcastCond = [&](Value c) {
      return rewriter.create<stablehlo::BroadcastInDimOp>(loc, i1Type, c,
                                                          bcastDims);
    };

    Value is_one_bcast = broadcastCond(is_one);
    Value is_inf_bcast = broadcastCond(is_inf);
    Value is_zero_bcast = broadcastCond(is_zero);
    Value is_neg_bcast = broadcastCond(is_neg);

    Value final_packed = packed_res;

    // If x == 1 -> 0
    final_packed = rewriter.create<stablehlo::SelectOp>(
        loc, is_one_bcast, packed_zero, final_packed);

    // If x == +inf -> +inf
    final_packed = rewriter.create<stablehlo::SelectOp>(
        loc, is_inf_bcast, packed_inf, final_packed);

    // If x == 0 -> -inf
    final_packed = rewriter.create<stablehlo::SelectOp>(
        loc, is_zero_bcast, packed_neg_inf, final_packed);

    // If x < 0 -> NaN
    final_packed = rewriter.create<stablehlo::SelectOp>(
        loc, is_neg_bcast, packed_nan, final_packed);

    rewriter.replaceOp(op, final_packed);
    return success();
  }
};

struct DotGeneralOpConversion
    : public OpConversionPattern<stablehlo::DotGeneralOp> {
  StringRef concatDimension;

  Type targetType;

  DotGeneralOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                         StringRef concatDimension, Type targetType)
      : OpConversionPattern<stablehlo::DotGeneralOp>(typeConverter, context),
        concatDimension(concatDimension), targetType(targetType) {}

  LogicalResult
  matchAndRewrite(stablehlo::DotGeneralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "DotGeneralOpConversion called\n");
    Location loc = op.getLoc();
    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];

    Type expectedLhsTy =
        getTypeConverter()->convertType(op.getOperands()[0].getType());
    Type expectedRhsTy =
        getTypeConverter()->convertType(op.getOperands()[1].getType());

    if (lhs.getType() != expectedLhsTy) {
      lhs = rewriter.create<UnrealizedConversionCastOp>(loc, expectedLhsTy, lhs)
                .getResult(0);
    }
    if (rhs.getType() != expectedRhsTy) {
      rhs = rewriter.create<UnrealizedConversionCastOp>(loc, expectedRhsTy, rhs)
                .getResult(0);
    }

    Value lhs_hi = extractLimb(lhs, 0, rewriter, loc, concatDimension);
    Value lhs_lo = extractLimb(lhs, 1, rewriter, loc, concatDimension);
    Value rhs_hi = extractLimb(rhs, 0, rewriter, loc, concatDimension);
    Value rhs_lo = extractLimb(rhs, 1, rewriter, loc, concatDimension);

    auto origLhsTy = cast<RankedTensorType>(op.getOperands()[0].getType());
    auto origRhsTy = cast<RankedTensorType>(op.getOperands()[1].getType());

    lhs_hi = rewriter.create<stablehlo::ReshapeOp>(
        loc, RankedTensorType::get(origLhsTy.getShape(), targetType), lhs_hi);
    lhs_lo = rewriter.create<stablehlo::ReshapeOp>(
        loc, RankedTensorType::get(origLhsTy.getShape(), targetType), lhs_lo);
    rhs_hi = rewriter.create<stablehlo::ReshapeOp>(
        loc, RankedTensorType::get(origRhsTy.getShape(), targetType), rhs_hi);
    rhs_lo = rewriter.create<stablehlo::ReshapeOp>(
        loc, RankedTensorType::get(origRhsTy.getShape(), targetType), rhs_lo);

    auto origType = cast<RankedTensorType>(op.getType());
    auto origShape = origType.getShape();
    auto prodType = RankedTensorType::get(origShape, targetType);

    // Re-merge limbs and apply Ozaki splitting on combined value
    Type sourceType = origLhsTy.getElementType();
    auto sourceTensorLhsType =
        RankedTensorType::get(origLhsTy.getShape(), sourceType);
    auto sourceTensorRhsType =
        RankedTensorType::get(origRhsTy.getShape(), sourceType);

    Value lhs_hi_src =
        rewriter.create<stablehlo::ConvertOp>(loc, sourceTensorLhsType, lhs_hi);
    Value lhs_lo_src =
        rewriter.create<stablehlo::ConvertOp>(loc, sourceTensorLhsType, lhs_lo);
    Value lhs_combined = rewriter.create<stablehlo::AddOp>(
        loc, sourceTensorLhsType, lhs_hi_src, lhs_lo_src);

    Value rhs_hi_src =
        rewriter.create<stablehlo::ConvertOp>(loc, sourceTensorRhsType, rhs_hi);
    Value rhs_lo_src =
        rewriter.create<stablehlo::ConvertOp>(loc, sourceTensorRhsType, rhs_lo);
    Value rhs_combined = rewriter.create<stablehlo::AddOp>(
        loc, sourceTensorRhsType, rhs_hi_src, rhs_lo_src);

    auto floatTargetType = cast<FloatType>(targetType);
    unsigned mantissaWidth = floatTargetType.getFPMantissaWidth();

    // Compute N (number of accumulated products)
    auto dimNumbers = op.getDotDimensionNumbers();
    auto lhsContractingDims = dimNumbers.getLhsContractingDimensions();
    auto lhsType = cast<RankedTensorType>(op.getLhs().getType());
    auto lhsShape = lhsType.getShape();
    int64_t N = 1;
    for (auto dim : lhsContractingDims) {
      N *= lhsShape[dim];
    }

    double log2_N = std::log2(static_cast<double>(N));
    int64_t ceil_log2_N = std::ceil(log2_N);
    int64_t splitBits =
        std::floor((static_cast<int64_t>(mantissaWidth) - ceil_log2_N) / 2.0);
    if (splitBits <= 0)
      splitBits = 0; // Fallback to 0 as approved!

    double splitFactor = std::pow(2.0, splitBits);

    Value max_A = getMaxValue(lhs_combined, rewriter, loc);
    Value max_B = getMaxValue(rhs_combined, rewriter, loc);

    auto sourceScalarType = RankedTensorType::get({}, sourceType);
    Value ln_2 = rewriter.create<stablehlo::ConstantOp>(
        loc, SplatElementsAttr::get(
                 sourceScalarType,
                 rewriter.getFloatAttr(sourceType, std::log(2.0))));

    // Compute scale_A
    assert(ln_2.getType() == max_A.getType());
    Value ln_A = rewriter.create<stablehlo::LogOp>(loc, max_A);
    Value log2_A = rewriter.create<stablehlo::DivOp>(loc, ln_A, ln_2);
    Value ceil_log2_A = rewriter.create<stablehlo::CeilOp>(loc, log2_A);
    Value scaled_log2_A =
        rewriter.create<stablehlo::MulOp>(loc, ceil_log2_A, ln_2);
    Value scale_A_scalar =
        rewriter.create<stablehlo::ExpOp>(loc, scaled_log2_A);

    Value scale_A = rewriter.create<stablehlo::BroadcastInDimOp>(
        loc, sourceTensorLhsType, scale_A_scalar,
        rewriter.getDenseI64ArrayAttr({}));

    // Compute scale_B
    Value ln_B = rewriter.create<stablehlo::LogOp>(loc, max_B);
    Value log2_B = rewriter.create<stablehlo::DivOp>(loc, ln_B, ln_2);
    Value ceil_log2_B = rewriter.create<stablehlo::CeilOp>(loc, log2_B);
    Value scaled_log2_B =
        rewriter.create<stablehlo::MulOp>(loc, ceil_log2_B, ln_2);
    Value scale_B_scalar =
        rewriter.create<stablehlo::ExpOp>(loc, scaled_log2_B);

    Value scale_B = rewriter.create<stablehlo::BroadcastInDimOp>(
        loc, sourceTensorRhsType, scale_B_scalar,
        rewriter.getDenseI64ArrayAttr({}));

    Value lhs_norm =
        rewriter.create<stablehlo::DivOp>(loc, lhs_combined, scale_A);
    Value rhs_norm =
        rewriter.create<stablehlo::DivOp>(loc, rhs_combined, scale_B);

    Value scale_const = rewriter.create<stablehlo::ConstantOp>(
        loc,
        SplatElementsAttr::get(sourceTensorLhsType,
                               rewriter.getFloatAttr(sourceType, splitFactor)));
    Value scaled_A =
        rewriter.create<stablehlo::MulOp>(loc, lhs_norm, scale_const);
    Value floored_A = rewriter.create<stablehlo::FloorOp>(loc, scaled_A);
    Value A_hi = floored_A;
    Value A_lo =
        rewriter.create<stablehlo::SubtractOp>(loc, scaled_A, floored_A);

    Value scale_const_B = rewriter.create<stablehlo::ConstantOp>(
        loc,
        SplatElementsAttr::get(sourceTensorRhsType,
                               rewriter.getFloatAttr(sourceType, splitFactor)));
    Value scaled_B =
        rewriter.create<stablehlo::MulOp>(loc, rhs_norm, scale_const_B);
    Value floored_B = rewriter.create<stablehlo::FloorOp>(loc, scaled_B);
    Value B_hi = floored_B;
    Value B_lo =
        rewriter.create<stablehlo::SubtractOp>(loc, scaled_B, floored_B);

    auto targetTensorLhsType =
        RankedTensorType::get(origLhsTy.getShape(), targetType);
    auto targetTensorRhsType =
        RankedTensorType::get(origRhsTy.getShape(), targetType);

    Value lhs_hi_hi =
        rewriter.create<stablehlo::ConvertOp>(loc, targetTensorLhsType, A_hi);
    Value lhs_hi_lo =
        rewriter.create<stablehlo::ConvertOp>(loc, targetTensorLhsType, A_lo);
    Value rhs_hi_hi =
        rewriter.create<stablehlo::ConvertOp>(loc, targetTensorRhsType, B_hi);
    Value rhs_hi_lo =
        rewriter.create<stablehlo::ConvertOp>(loc, targetTensorRhsType, B_lo);

    Value p1 = rewriter.create<stablehlo::DotGeneralOp>(
        loc, prodType, lhs_hi_hi, rhs_hi_hi, op.getDotDimensionNumbers(),
        op.getPrecisionConfigAttr(), op.getAlgorithmAttr());
    Value p2 = rewriter.create<stablehlo::DotGeneralOp>(
        loc, prodType, lhs_hi_hi, rhs_hi_lo, op.getDotDimensionNumbers(),
        op.getPrecisionConfigAttr(), op.getAlgorithmAttr());
    Value p3 = rewriter.create<stablehlo::DotGeneralOp>(
        loc, prodType, lhs_hi_lo, rhs_hi_hi, op.getDotDimensionNumbers(),
        op.getPrecisionConfigAttr(), op.getAlgorithmAttr());
    Value p4 = rewriter.create<stablehlo::DotGeneralOp>(
        loc, prodType, lhs_hi_lo, rhs_hi_lo, op.getDotDimensionNumbers(),
        op.getPrecisionConfigAttr(), op.getAlgorithmAttr());

    // Scale back and sum in sourceType
    auto sourceProdType = RankedTensorType::get(origShape, sourceType);
    Value p1_src =
        rewriter.create<stablehlo::ConvertOp>(loc, sourceProdType, p1);
    Value p2_src =
        rewriter.create<stablehlo::ConvertOp>(loc, sourceProdType, p2);
    Value p3_src =
        rewriter.create<stablehlo::ConvertOp>(loc, sourceProdType, p3);
    Value p4_src =
        rewriter.create<stablehlo::ConvertOp>(loc, sourceProdType, p4);

    Value total_scale =
        rewriter.create<stablehlo::MulOp>(loc, scale_A_scalar, scale_B_scalar);
    Value split_factor_squared = rewriter.create<stablehlo::ConstantOp>(
        loc, SplatElementsAttr::get(
                 sourceScalarType,
                 rewriter.getFloatAttr(sourceType, splitFactor * splitFactor)));
    Value scale_back_scalar = rewriter.create<stablehlo::DivOp>(
        loc, total_scale, split_factor_squared);

    Value scale_back = rewriter.create<stablehlo::BroadcastInDimOp>(
        loc, sourceProdType, scale_back_scalar,
        rewriter.getDenseI64ArrayAttr({}));

    p1_src = rewriter.create<stablehlo::MulOp>(loc, p1_src, scale_back);
    p2_src = rewriter.create<stablehlo::MulOp>(loc, p2_src, scale_back);
    p3_src = rewriter.create<stablehlo::MulOp>(loc, p3_src, scale_back);
    p4_src = rewriter.create<stablehlo::MulOp>(loc, p4_src, scale_back);

    Value sum1 =
        rewriter.create<stablehlo::AddOp>(loc, sourceProdType, p1_src, p2_src);
    Value sum2 =
        rewriter.create<stablehlo::AddOp>(loc, sourceProdType, p3_src, p4_src);
    Value sum =
        rewriter.create<stablehlo::AddOp>(loc, sourceProdType, sum1, sum2);

    int expansionSize = 2; // Hardcoded for now
    Value packed = convertToMultifloat(sum, rewriter, loc, targetType,
                                       concatDimension, expansionSize);
    rewriter.replaceOp(op, packed);

    return success();
  }
};

struct WrapOpConversion : public OpConversionPattern<enzymexla::WrapOp> {
  StringRef concatDimension;

  WrapOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                   StringRef concatDimension)
      : OpConversionPattern<enzymexla::WrapOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(enzymexla::WrapOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getOperands()[0];

    bool isTuple = concatDimension == "tuple";

    Type convertedType = getTypeConverter()->convertType(op.getType());

    if (isTuple && isa<TupleType>(convertedType)) {
      Value input_hi = extractLimb(input, 0, rewriter, loc, concatDimension);
      Value input_lo = extractLimb(input, 1, rewriter, loc, concatDimension);

      auto tupleType = cast<TupleType>(convertedType);
      Type hiType = tupleType.getType(0);
      Type loType = tupleType.getType(1);

      auto hiOp = rewriter.create<enzymexla::WrapOp>(
          loc, TypeRange{hiType}, ValueRange{input_hi}, op->getAttrs());
      auto loOp = rewriter.create<enzymexla::WrapOp>(
          loc, TypeRange{loType}, ValueRange{input_lo}, op->getAttrs());

      Value packed = packLimbs(hiOp.getResult(), loOp.getResult(), rewriter,
                               loc, concatDimension);
      rewriter.replaceOp(op, packed);
      return success();
    }

    int64_t dim = op.getDimension();
    if (concatDimension == "first") {
      dim += 1;
    }

    SmallVector<NamedAttribute, 4> newAttrs;
    for (auto attr : op->getAttrs()) {
      if (attr.getName() == "dimension") {
        newAttrs.push_back(rewriter.getNamedAttr(
            "dimension", rewriter.getI64IntegerAttr(dim)));
      } else {
        newAttrs.push_back(attr);
      }
    }

    auto newOp = rewriter.create<enzymexla::WrapOp>(
        loc, TypeRange{convertedType}, ValueRange{input}, newAttrs);

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct RotateOpConversion : public OpConversionPattern<enzymexla::RotateOp> {
  StringRef concatDimension;

  RotateOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                     StringRef concatDimension)
      : OpConversionPattern<enzymexla::RotateOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(enzymexla::RotateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getOperands()[0];

    bool isTuple = concatDimension == "tuple";

    Type convertedType = getTypeConverter()->convertType(op.getType());

    if (isTuple && isa<TupleType>(convertedType)) {
      Value input_hi = extractLimb(input, 0, rewriter, loc, concatDimension);
      Value input_lo = extractLimb(input, 1, rewriter, loc, concatDimension);

      auto tupleType = cast<TupleType>(convertedType);
      Type hiType = tupleType.getType(0);
      Type loType = tupleType.getType(1);

      auto hiOp = rewriter.create<enzymexla::RotateOp>(
          loc, TypeRange{hiType}, ValueRange{input_hi}, op->getAttrs());
      auto loOp = rewriter.create<enzymexla::RotateOp>(
          loc, TypeRange{loType}, ValueRange{input_lo}, op->getAttrs());

      Value packed = packLimbs(hiOp.getResult(), loOp.getResult(), rewriter,
                               loc, concatDimension);
      rewriter.replaceOp(op, packed);
      return success();
    }

    int32_t dim = op.getDimension();
    if (concatDimension == "first") {
      dim += 1;
    }

    SmallVector<NamedAttribute, 4> newAttrs;
    for (auto attr : op->getAttrs()) {
      if (attr.getName() == "dimension") {
        newAttrs.push_back(rewriter.getNamedAttr(
            "dimension", rewriter.getI32IntegerAttr(dim)));
      } else {
        newAttrs.push_back(attr);
      }
    }

    auto newOp = rewriter.create<enzymexla::RotateOp>(
        loc, TypeRange{convertedType}, ValueRange{input}, newAttrs);

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ExtendOpConversion : public OpConversionPattern<enzymexla::ExtendOp> {
  StringRef concatDimension;

  ExtendOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                     StringRef concatDimension)
      : OpConversionPattern<enzymexla::ExtendOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(enzymexla::ExtendOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getOperands()[0];

    bool isTuple = concatDimension == "tuple";

    Type convertedType = getTypeConverter()->convertType(op.getType());

    if (isTuple && isa<TupleType>(convertedType)) {
      Value input_hi = extractLimb(input, 0, rewriter, loc, concatDimension);
      Value input_lo = extractLimb(input, 1, rewriter, loc, concatDimension);

      auto tupleType = cast<TupleType>(convertedType);
      Type hiType = tupleType.getType(0);
      Type loType = tupleType.getType(1);

      auto hiOp = rewriter.create<enzymexla::ExtendOp>(
          loc, hiType, input_hi, op.getLhsAttr(), op.getRhsAttr(),
          op.getDimensionAttr());
      auto loOp = rewriter.create<enzymexla::ExtendOp>(
          loc, loType, input_lo, op.getLhsAttr(), op.getRhsAttr(),
          op.getDimensionAttr());

      Value packed = packLimbs(hiOp.getResult(), loOp.getResult(), rewriter,
                               loc, concatDimension);
      rewriter.replaceOp(op, packed);
      return success();
    }

    int64_t dim = op.getDimension();
    if (concatDimension == "first") {
      dim += 1;
    }

    auto newOp = rewriter.create<enzymexla::ExtendOp>(
        loc, convertedType, input, op.getLhsAttr(), op.getRhsAttr(),
        rewriter.getI64IntegerAttr(dim));

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct UpdateWithoutCornersOpConversion
    : public OpConversionPattern<enzymexla::UpdateWithoutCornersOp> {
  StringRef concatDimension;

  UpdateWithoutCornersOpConversion(TypeConverter &typeConverter,
                                   MLIRContext *context,
                                   StringRef concatDimension)
      : OpConversionPattern<enzymexla::UpdateWithoutCornersOp>(typeConverter,
                                                               context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(enzymexla::UpdateWithoutCornersOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    bool isTuple = concatDimension == "tuple";

    Type convertedType = getTypeConverter()->convertType(op.getType());

    if (isTuple && isa<TupleType>(convertedType)) {
      Value operand = adaptor.getOperand();
      Value update = adaptor.getUpdate();

      Value operand_hi =
          extractLimb(operand, 0, rewriter, loc, concatDimension);
      Value operand_lo =
          extractLimb(operand, 1, rewriter, loc, concatDimension);
      Value update_hi = extractLimb(update, 0, rewriter, loc, concatDimension);
      Value update_lo = extractLimb(update, 1, rewriter, loc, concatDimension);

      auto tupleType = cast<TupleType>(convertedType);
      Type hiType = tupleType.getType(0);
      Type loType = tupleType.getType(1);

      auto hiOp = rewriter.create<enzymexla::UpdateWithoutCornersOp>(
          loc, TypeRange{hiType}, ValueRange{operand_hi, update_hi},
          op->getAttrs());
      auto loOp = rewriter.create<enzymexla::UpdateWithoutCornersOp>(
          loc, TypeRange{loType}, ValueRange{operand_lo, update_lo},
          op->getAttrs());

      Value packed = packLimbs(hiOp.getResult(), loOp.getResult(), rewriter,
                               loc, concatDimension);
      rewriter.replaceOp(op, packed);
      return success();
    }

    int64_t dimX = op.getDimensionX();
    int64_t dimY = op.getDimensionY();
    if (concatDimension == "first") {
      dimX += 1;
      dimY += 1;
    }

    SmallVector<NamedAttribute> newAttrs;
    for (auto attr : op->getAttrs()) {
      if (attr.getName() == "dimensionX") {
        newAttrs.push_back(rewriter.getNamedAttr(
            "dimensionX", rewriter.getI64IntegerAttr(dimX)));
      } else if (attr.getName() == "dimensionY") {
        newAttrs.push_back(rewriter.getNamedAttr(
            "dimensionY", rewriter.getI64IntegerAttr(dimY)));
      } else {
        newAttrs.push_back(attr);
      }
    }

    auto newOp = rewriter.create<enzymexla::UpdateWithoutCornersOp>(
        loc, TypeRange{convertedType},
        ValueRange{adaptor.getOperand(), adaptor.getUpdate()}, newAttrs);

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConcatenateOpOptimization
    : public OpConversionPattern<stablehlo::ConcatenateOp> {
  using OpConversionPattern<stablehlo::ConcatenateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(stablehlo::ConcatenateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getOperands().empty())
      return failure();

    Operation *firstOp = op.getOperands()[0].getDefiningOp();
    if (!firstOp)
      return failure();

    for (auto operand : op.getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp)
        return failure();
      if (defOp->getName() != firstOp->getName())
        return failure();
      if (!defOp->hasTrait<mlir::OpTrait::Elementwise>())
        return failure();
      if (defOp->getAttrDictionary() != firstOp->getAttrDictionary())
        return failure();
      if (defOp->getNumOperands() != firstOp->getNumOperands())
        return failure();

      for (auto *user : defOp->getUsers()) {
        if (user != op)
          return failure();
      }
    }

    SmallVector<Value> newOperands;
    for (unsigned i = 0; i < firstOp->getNumOperands(); ++i) {
      SmallVector<Value> operandsToConcat;
      for (auto operand : op.getOperands()) {
        operandsToConcat.push_back(operand.getDefiningOp()->getOperand(i));
      }

      auto type = cast<RankedTensorType>(operandsToConcat[0].getType());
      SmallVector<int64_t> newShape = llvm::to_vector(type.getShape());
      for (size_t j = 1; j < operandsToConcat.size(); ++j) {
        newShape[op.getDimension()] +=
            cast<RankedTensorType>(operandsToConcat[j].getType())
                .getShape()[op.getDimension()];
      }
      auto outType = RankedTensorType::get(newShape, type.getElementType());

      auto newConcat = rewriter.create<stablehlo::ConcatenateOp>(
          op.getLoc(), outType, operandsToConcat, op.getDimension());
      newOperands.push_back(newConcat);
    }

    OperationState state(op.getLoc(), firstOp->getName().getStringRef());
    state.addOperands(newOperands);
    state.addAttributes(firstOp->getAttrDictionary().getValue());
    state.addTypes(op.getType());

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct ConcatenateOpConversion
    : public OpConversionPattern<stablehlo::ConcatenateOp> {
  StringRef concatDimension;

  ConcatenateOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                          StringRef concatDimension)
      : OpConversionPattern<stablehlo::ConcatenateOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::ConcatenateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    if (op.getType() == getTypeConverter()->convertType(op.getType()))
      return failure();

    if (concatDimension == "tuple") {
      auto tupleType =
          cast<TupleType>(getTypeConverter()->convertType(op.getType()));
      unsigned numLimbs = tupleType.size();

      SmallVector<SmallVector<Value>> limbsOperands(numLimbs);
      for (auto operand : adaptor.getOperands()) {
        for (unsigned j = 0; j < numLimbs; ++j) {
          limbsOperands[j].push_back(
              extractLimb(operand, j, rewriter, loc, "tuple"));
        }
      }

      SmallVector<Value> concatenatedLimbs;
      for (unsigned j = 0; j < numLimbs; ++j) {
        concatenatedLimbs.push_back(rewriter.create<stablehlo::ConcatenateOp>(
            loc, tupleType.getType(j), limbsOperands[j], op.getDimension()));
      }

      Value packed = packLimbs(concatenatedLimbs, rewriter, loc, "tuple");
      rewriter.replaceOp(op, packed);
      return success();
    }

    auto outType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

    int64_t dim = op.getDimension();
    int64_t mappedDim = dim;
    if (concatDimension == "first") {
      mappedDim = dim + 1;
    } else if (concatDimension == "last") {
      mappedDim = dim;
    }

    Value newOp = rewriter.create<stablehlo::ConcatenateOp>(
        loc, outType, adaptor.getOperands(), mappedDim);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct PadOpConversion : public OpConversionPattern<stablehlo::PadOp> {
  StringRef concatDimension;

  PadOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                  StringRef concatDimension)
      : OpConversionPattern<stablehlo::PadOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::PadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type convType = getTypeConverter()->convertType(op->getResult(0).getType());
    auto outType = dyn_cast<RankedTensorType>(convType);

    if (concatDimension == "tuple") {
      auto tupleType =
          cast<TupleType>(getTypeConverter()->convertType(op.getType()));
      Value hiOperand =
          extractLimb(adaptor.getOperand(), 0, rewriter, loc, "tuple");
      Value loOperand =
          extractLimb(adaptor.getOperand(), 1, rewriter, loc, "tuple");

      Value hiPadValue =
          extractLimb(adaptor.getPaddingValue(), 0, rewriter, loc, "tuple");
      Value loPadValue =
          extractLimb(adaptor.getPaddingValue(), 1, rewriter, loc, "tuple");

      Value hi = rewriter.create<stablehlo::PadOp>(
          loc, tupleType.getType(0), hiOperand, hiPadValue,
          op.getEdgePaddingLow(), op.getEdgePaddingHigh(),
          op.getInteriorPadding());
      Value lo = rewriter.create<stablehlo::PadOp>(
          loc, tupleType.getType(1), loOperand, loPadValue,
          op.getEdgePaddingLow(), op.getEdgePaddingHigh(),
          op.getInteriorPadding());
      Value packed = packLimbs(hi, lo, rewriter, loc, "tuple");
      rewriter.replaceOp(op, packed);
      return success();
    }

    Value padValue = adaptor.getPaddingValue();
    bool isSymmetric = false;
    Value origPadValue = op.getPaddingValue();
    auto origPadConst = origPadValue.getDefiningOp<stablehlo::ConstantOp>();

    if (origPadConst) {
      if (auto elementsAttr =
              dyn_cast<SplatElementsAttr>(origPadConst.getValue())) {
        if (auto floatAttr =
                dyn_cast<FloatAttr>(elementsAttr.getSplatValue<Attribute>())) {
          if (floatAttr.getValueAsDouble() == 0.0) {
            isSymmetric = true;
          }
        }
      }
    }

    if (isSymmetric) {
      auto floatTy =
          cast<FloatType>(cast<RankedTensorType>(outType).getElementType());
      auto zeroAttr = rewriter.getFloatAttr(floatTy, 0.0);
      auto splatAttr =
          SplatElementsAttr::get(RankedTensorType::get({}, floatTy), zeroAttr);
      Value zeroConst = rewriter.create<stablehlo::ConstantOp>(loc, splatAttr);

      SmallVector<int64_t> low = llvm::to_vector(op.getEdgePaddingLow());
      SmallVector<int64_t> high = llvm::to_vector(op.getEdgePaddingHigh());
      SmallVector<int64_t> interior = llvm::to_vector(op.getInteriorPadding());

      if (concatDimension == "first") {
        low.insert(low.begin(), 0);
        high.insert(high.begin(), 0);
        interior.insert(interior.begin(), 0);
      } else if (concatDimension == "last") {
        low.push_back(0);
        high.push_back(0);
        interior.push_back(0);
      }

      Value newOp = rewriter.create<stablehlo::PadOp>(
          loc, adaptor.getOperand(), zeroConst,
          rewriter.getDenseI64ArrayAttr(low),
          rewriter.getDenseI64ArrayAttr(high),
          rewriter.getDenseI64ArrayAttr(interior));
      rewriter.replaceOp(op, newOp);
      return success();
    }

    Value hiOperand =
        extractLimb(adaptor.getOperand(), 0, rewriter, loc, concatDimension);
    Value loOperand =
        extractLimb(adaptor.getOperand(), 1, rewriter, loc, concatDimension);

    Value hiPadValue = extractLimb(padValue, 0, rewriter, loc, concatDimension);
    Value loPadValue = extractLimb(padValue, 1, rewriter, loc, concatDimension);

    // Reshape pad values to rank 0 scalar
    auto scalarType = RankedTensorType::get(
        {}, cast<RankedTensorType>(hiPadValue.getType()).getElementType());
    hiPadValue =
        rewriter.create<stablehlo::ReshapeOp>(loc, scalarType, hiPadValue);
    loPadValue =
        rewriter.create<stablehlo::ReshapeOp>(loc, scalarType, loPadValue);

    SmallVector<int64_t> low = llvm::to_vector(op.getEdgePaddingLow());
    SmallVector<int64_t> high = llvm::to_vector(op.getEdgePaddingHigh());
    SmallVector<int64_t> interior = llvm::to_vector(op.getInteriorPadding());

    if (concatDimension == "first") {
      low.insert(low.begin(), 0);
      high.insert(high.begin(), 0);
      interior.insert(interior.begin(), 0);
    } else if (concatDimension == "last") {
      low.push_back(0);
      high.push_back(0);
      interior.push_back(0);
    }

    Value hi = rewriter.create<stablehlo::PadOp>(
        loc, hiOperand, hiPadValue, rewriter.getDenseI64ArrayAttr(low),
        rewriter.getDenseI64ArrayAttr(high),
        rewriter.getDenseI64ArrayAttr(interior));
    Value lo = rewriter.create<stablehlo::PadOp>(
        loc, loOperand, loPadValue, rewriter.getDenseI64ArrayAttr(low),
        rewriter.getDenseI64ArrayAttr(high),
        rewriter.getDenseI64ArrayAttr(interior));

    Value packed = packLimbs(hi, lo, rewriter, loc, concatDimension);
    rewriter.replaceOp(op, packed);
    return success();
  }
};

struct LowerReduceWindowOp
    : public OpRewritePattern<stablehlo::ReduceWindowOp> {
  Type sourceType;

  LowerReduceWindowOp(MLIRContext *context, Type sourceType)
      : OpRewritePattern<stablehlo::ReduceWindowOp>(context),
        sourceType(sourceType) {}

  LogicalResult matchAndRewrite(stablehlo::ReduceWindowOp reduceOp,
                                PatternRewriter &rewriter) const override {
    TypedValue<RankedTensorType> input =
        cast<TypedValue<RankedTensorType>>(reduceOp.getInputs()[0]);
    auto inputType = input.getType();
    auto outputType =
        cast<RankedTensorType>(reduceOp.getResults()[0].getType());

    if (inputType.getElementType() != sourceType) {
      return failure();
    }

    if (reduceOp->getNumResults() != 1) {
      return failure();
    }

    Value initValues = reduceOp.getInitValues()[0];
    if (!matchPattern(initValues, m_Zero()) &&
        !matchPattern(initValues, m_AnyZeroFloat())) {
      return failure();
    }

    Region &region = reduceOp.getBody();
    if (region.getBlocks().size() != 1)
      return failure();
    Block &block = region.front();
    if (block.getNumArguments() != 2)
      return failure();
    if (block.getOperations().size() != 2)
      return failure();
    auto addOp = dyn_cast<stablehlo::AddOp>(&block.front());
    if (!addOp)
      return failure();
    auto returnOp = dyn_cast<stablehlo::ReturnOp>(block.getOperations().back());
    if (!returnOp || returnOp.getOperands().front() != addOp.getResult())
      return failure();

    auto strides = reduceOp.getWindowStrides();
    if (strides) {
      for (auto s : *strides) {
        if (s != 1)
          return failure();
      }
    }

    auto dims = reduceOp.getWindowDimensions();
    int reduceDim = -1;
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims[i] > 1) {
        if (reduceDim != -1)
          return failure();
        reduceDim = i;
      }
    }
    if (reduceDim == -1)
      return failure();

    auto windowDilations = reduceOp.getWindowDilations();
    if (windowDilations) {
      for (auto &&[i, d] : llvm::enumerate(*windowDilations)) {
        if (i == reduceDim)
          continue;
        if (d != 1)
          return failure();
      }
    }

    auto baseDilations = reduceOp.getBaseDilations();
    if (baseDilations) {
      for (auto d : *baseDilations) {
        if (d != 1)
          return failure();
      }
    }

    int64_t windowSize = dims[reduceDim];

    Location loc = reduceOp.getLoc();

    size_t newRank = inputType.getRank();
    Value paddedInput = input;

    if (auto paddingAttr = reduceOp.getPadding()) {
      auto paddingType = paddingAttr->getType();
      if (paddingType.getRank() != 2 || paddingType.getDimSize(1) != 2)
        return failure();

      SmallVector<int64_t> lowPadding, highPadding;
      auto vals = paddingAttr->getValues<int64_t>();
      auto it = vals.begin();
      for (size_t i = 0; i < dims.size(); ++i) {
        int64_t low = *it++;
        int64_t high = *it++;
        lowPadding.push_back(low);
        highPadding.push_back(high);
      }

      SmallVector<int64_t> interiorPadding(newRank, 0);
      paddedInput = stablehlo::PadOp::create(
          rewriter, loc, input, initValues,
          rewriter.getDenseI64ArrayAttr(lowPadding),
          rewriter.getDenseI64ArrayAttr(highPadding),
          rewriter.getDenseI64ArrayAttr(interiorPadding));
    }

    Value result = nullptr;

    SmallVector<int64_t> startOffsets(newRank, 0);
    SmallVector<int64_t> sliceStrides(newRank, 1);
    SmallVector<int64_t> sliceLimits = llvm::to_vector(outputType.getShape());

    int64_t dilation = 1;
    if (windowDilations) {
      dilation = (*windowDilations)[reduceDim];
    }

    for (int k = 0; k < windowSize; ++k) {
      startOffsets[reduceDim] = k * dilation;
      sliceLimits[reduceDim] = k * dilation + outputType.getDimSize(reduceDim);

      auto sliceOp = stablehlo::SliceOp::create(
          rewriter, loc, paddedInput, startOffsets, sliceLimits, sliceStrides);

      if (result) {
        result = stablehlo::AddOp::create(rewriter, loc, result, sliceOp);
      } else {
        result = sliceOp;
      }
    }

    rewriter.replaceOp(reduceOp, result);
    return success();
  }
};

template <typename OpTy> struct IsResultTypeLegal {
  const TypeConverter &typeConverter;
  IsResultTypeLegal(const TypeConverter &tc) : typeConverter(tc) {}

  bool operator()(OpTy op) const { return typeConverter.isLegal(op.getType()); }
};

template <> struct IsResultTypeLegal<stablehlo::ReduceWindowOp> {
  const TypeConverter &typeConverter;
  IsResultTypeLegal(const TypeConverter &tc) : typeConverter(tc) {}

  bool operator()(stablehlo::ReduceWindowOp op) const {
    if (op.getResults().empty())
      return true;
    if (!typeConverter.isLegal(op.getResults()[0].getType()))
      return false;
    if (op.getInputs().empty())
      return true;
    auto inputType = cast<RankedTensorType>(op.getInputs()[0].getType());
    if (inputType.getRank() != op.getWindowDimensions().size())
      return false;
    return true;
  }
};

template <> struct IsResultTypeLegal<stablehlo::ReduceOp> {
  const TypeConverter &typeConverter;
  IsResultTypeLegal(const TypeConverter &tc) : typeConverter(tc) {}

  bool operator()(stablehlo::ReduceOp op) const {
    if (op.getResults().empty())
      return true;
    return typeConverter.isLegal(op.getResults()[0].getType());
  }
};

template <typename OpTy> struct IsResultOrOperandTypeLegal {
  const TypeConverter &typeConverter;
  IsResultOrOperandTypeLegal(const TypeConverter &tc) : typeConverter(tc) {}

  bool operator()(OpTy op) const {
    if (std::is_same<OpTy, stablehlo::DynamicUpdateSliceOp>::value) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Checking DynamicUpdateSliceOp legality for result: "
                 << op.getType() << "\n");
    }
    if (!this->typeConverter.isLegal(op.getType())) {
      if (std::is_same<OpTy, stablehlo::DynamicUpdateSliceOp>::value) {
        LLVM_DEBUG(llvm::dbgs() << "DynamicUpdateSliceOp result is illegal\n");
      }
      return false;
    }
    if (op->getNumOperands() > 0 &&
        !this->typeConverter.isLegal(op->getOperand(0).getType())) {
      if (std::is_same<OpTy, stablehlo::DynamicUpdateSliceOp>::value) {
        LLVM_DEBUG(llvm::dbgs()
                   << "DynamicUpdateSliceOp operand 0 is illegal\n");
      }
      return false;
    }
    if (std::is_same<OpTy, stablehlo::DynamicUpdateSliceOp>::value) {
      LLVM_DEBUG(llvm::dbgs() << "DynamicUpdateSliceOp is legal!\n");
    }
    return true;
  }
};

struct MultiFloatConversionPass
    : public enzyme::impl::MultiFloatConversionPassBase<
          MultiFloatConversionPass> {
  using MultiFloatConversionPassBase::MultiFloatConversionPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto op = getOperation();

    Type srcTy = getFloatTypeFromString(sourceType, context);
    Type tgtTy = getFloatTypeFromString(targetType, context);

    if (!srcTy || !tgtTy) {
      op->emitError() << "Invalid source or target type specified.";
      signalPassFailure();
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "sourceType: " << sourceType << "\n");
    LLVM_DEBUG(llvm::dbgs() << "targetType: " << targetType << "\n");
    LLVM_DEBUG(llvm::dbgs() << "concatDimension: " << concatDimension << "\n");
    LLVM_DEBUG(llvm::dbgs() << "expansionSize: " << expansionSize << "\n");

    if (concatDimension != "first" && concatDimension != "last" &&
        concatDimension != "tuple") {
      op->emitError() << "Invalid concat-dimension specified: "
                      << concatDimension;
      signalPassFailure();
      return;
    }

    ConversionTarget target(*context);
    target.addLegalDialect<stablehlo::StablehloDialect>();
    target.addLegalDialect<mlir::enzymexla::EnzymeXLADialect>();
    target.addLegalDialect<arith::ArithDialect>();

    TypeConverter typeConverter;

    bool isTuple = concatDimension == "tuple";
    bool isFirst = concatDimension == "first";

    typeConverter.addConversion([&](Type type) -> std::optional<Type> {
      if (type == srcTy) {
        if (expansionSize == 1)
          return tgtTy;
        if (isTuple) {
          return TupleType::get(context, {tgtTy, tgtTy});
        }
        return RankedTensorType::get({2}, tgtTy);
      }
      if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
        if (tensorTy.getElementType() == srcTy) {
          if (expansionSize == 1)
            return RankedTensorType::get(tensorTy.getShape(), tgtTy);
          if (isTuple) {
            auto partTy = RankedTensorType::get(tensorTy.getShape(), tgtTy);
            return TupleType::get(context, {partTy, partTy});
          }
          SmallVector<int64_t> newShape;
          if (isFirst) {
            newShape.push_back(2);
            for (auto dim : tensorTy.getShape()) {
              newShape.push_back(dim);
            }
          } else {
            for (auto dim : tensorTy.getShape()) {
              newShape.push_back(dim);
            }
            newShape.push_back(2);
          }
          auto resultType = RankedTensorType::get(newShape, tgtTy);
          LLVM_DEBUG(llvm::dbgs() << "Tensor type converted: " << type << " -> "
                                  << resultType << "\n");
          return resultType;
        }
      }
      return type;
    });

    typeConverter.addSourceMaterialization([](OpBuilder &builder, Type type,
                                              ValueRange inputs,
                                              Location loc) -> Value {
      if (inputs.size() == 1)
        return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
            .getResult(0);
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    });
    typeConverter.addTargetMaterialization([](OpBuilder &builder, Type type,
                                              ValueRange inputs,
                                              Location loc) -> Value {
      if (inputs.size() == 1)
        return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
            .getResult(0);
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    });

    if (convertSignatures) {
      op->walk([&](func::FuncOp func) {
        if (func->hasAttr("enzyme.no_multifloat"))
          return;
        SmallVector<Type> oldArgTypes;
        for (auto arg : func.getArguments()) {
          oldArgTypes.push_back(arg.getType());
        }

        SmallVector<Type> newArgTypes;
        for (auto ty : oldArgTypes) {
          Type newTy = typeConverter.convertType(ty);
          LLVM_DEBUG(llvm::dbgs() << "Converting arg type: " << ty << " -> "
                                  << newTy << "\n");
          newArgTypes.push_back(newTy);
        }

        SmallVector<Type> oldResTypes;
        for (auto ty : func.getFunctionType().getResults()) {
          oldResTypes.push_back(ty);
        }

        SmallVector<Type> newResTypes;
        for (auto ty : oldResTypes) {
          newResTypes.push_back(typeConverter.convertType(ty));
        }

        auto newFuncType = FunctionType::get(context, newArgTypes, newResTypes);
        func.setType(newFuncType);

        auto updateShardingAttr = [&](Attribute attr) -> Attribute {
          auto sharding = dyn_cast_or_null<sdy::TensorShardingAttr>(attr);
          if (!sharding)
            return attr;
          SmallVector<sdy::DimensionShardingAttr> newDimShardings;
          auto emptyDim = sdy::DimensionShardingAttr::get(
              context, ArrayRef<sdy::AxisRefAttr>{}, /*is_closed=*/false);
          if (isFirst) {
            newDimShardings.push_back(emptyDim);
            for (auto ds : sharding.getDimShardings())
              newDimShardings.push_back(ds);
          } else {
            for (auto ds : sharding.getDimShardings())
              newDimShardings.push_back(ds);
            newDimShardings.push_back(emptyDim);
          }
          return sdy::TensorShardingAttr::get(
              context, sharding.getMeshOrRef(), newDimShardings,
              sharding.getReplicatedAxes(), sharding.getUnreducedAxes());
        };

        for (unsigned i = 0; i < oldArgTypes.size(); ++i) {
          if (oldArgTypes[i] != newArgTypes[i]) {
            if (auto attr = func.getArgAttr(i, "sdy.sharding"))
              func.setArgAttr(i, "sdy.sharding", updateShardingAttr(attr));
          }
        }
        for (unsigned i = 0; i < oldResTypes.size(); ++i) {
          if (oldResTypes[i] != newResTypes[i]) {
            if (auto attr = func.getResultAttr(i, "sdy.sharding"))
              func.setResultAttr(i, "sdy.sharding", updateShardingAttr(attr));
          }
        }

        if (!func.empty()) {
          OpBuilder builder(&func.front(), func.front().begin());
          auto &block = func.front();
          for (unsigned i = 0; i < block.getNumArguments(); ++i) {
            Type oldType = oldArgTypes[i];
            Type newType = newArgTypes[i];
            if (oldType != newType) {
              block.getArgument(i).setType(newType);
              auto cast = builder.create<UnrealizedConversionCastOp>(
                  func.getLoc(), oldType, block.getArgument(i));
              block.getArgument(i).replaceAllUsesExcept(cast.getResult(0),
                                                        cast);
            }
          }
        }

        // Update return ops to keep IR valid for verifier
        func.walk([&](func::ReturnOp returnOp) {
          OpBuilder builder(returnOp);
          SmallVector<Value> newReturnOperands;
          for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
            Type oldType = returnOp.getOperand(i).getType();
            Type newType = newResTypes[i];
            if (oldType != newType) {
              auto cast = builder.create<UnrealizedConversionCastOp>(
                  returnOp.getLoc(), newType, returnOp.getOperand(i));
              newReturnOperands.push_back(cast.getResult(0));
            } else {
              newReturnOperands.push_back(returnOp.getOperand(i));
            }
          }
          returnOp.getOperation()->setOperands(newReturnOperands);
        });
      });
    }

    target.addLegalOp<UnrealizedConversionCastOp>();

    IsResultTypeLegal<stablehlo::AddOp> addLegal(typeConverter);
    IsResultTypeLegal<stablehlo::SubtractOp> subLegal(typeConverter);
    IsResultTypeLegal<stablehlo::MulOp> mulLegal(typeConverter);
    IsResultTypeLegal<stablehlo::DivOp> divLegal(typeConverter);
    IsResultTypeLegal<stablehlo::SelectOp> selectLegal(typeConverter);
    IsResultTypeLegal<stablehlo::ReverseOp> reverseLegal(typeConverter);
    IsResultTypeLegal<stablehlo::AbsOp> absLegal(typeConverter);
    IsResultTypeLegal<stablehlo::SqrtOp> sqrtLegal(typeConverter);
    IsResultTypeLegal<stablehlo::PadOp> padLegal(typeConverter);
    IsResultTypeLegal<stablehlo::ReduceWindowOp> reduceWindowLegal(
        typeConverter);
    IsResultTypeLegal<stablehlo::ReduceOp> reduceLegal(typeConverter);

    IsResultOrOperandTypeLegal<stablehlo::SliceOp> sliceLegal(typeConverter);
    IsResultOrOperandTypeLegal<stablehlo::BroadcastInDimOp> broadcastLegal(
        typeConverter);
    IsResultOrOperandTypeLegal<stablehlo::TransposeOp> transposeLegal(
        typeConverter);
    IsResultOrOperandTypeLegal<stablehlo::ReshapeOp> reshapeLegal(
        typeConverter);
    IsResultOrOperandTypeLegal<stablehlo::CompareOp> compareLegal(
        typeConverter);
    IsResultOrOperandTypeLegal<stablehlo::FloorOp> floorLegal(typeConverter);
    IsResultOrOperandTypeLegal<stablehlo::CeilOp> ceilLegal(typeConverter);
    IsResultOrOperandTypeLegal<stablehlo::ExpOp> expLegal(typeConverter);
    IsResultOrOperandTypeLegal<stablehlo::LogOp> logLegal(typeConverter);
    IsResultOrOperandTypeLegal<stablehlo::MaxOp> maxLegal(typeConverter);
    IsResultOrOperandTypeLegal<stablehlo::MinOp> minLegal(typeConverter);
    IsResultOrOperandTypeLegal<stablehlo::SineOp> sineLegal(typeConverter);
    IsResultOrOperandTypeLegal<stablehlo::DotGeneralOp> dotGeneralLegal(
        typeConverter);
    IsResultOrOperandTypeLegal<enzymexla::RotateOp> rotateLegal(typeConverter);
    IsResultOrOperandTypeLegal<enzymexla::WrapOp> wrapLegal(typeConverter);
    IsResultOrOperandTypeLegal<enzymexla::ExtendOp> extendLegal(typeConverter);
    IsResultOrOperandTypeLegal<enzymexla::UpdateWithoutCornersOp>
        updateWithoutCornersLegal(typeConverter);
    IsResultOrOperandTypeLegal<stablehlo::NegOp> negLegal(typeConverter);
    IsResultOrOperandTypeLegal<stablehlo::DynamicUpdateSliceOp>
        dynamicUpdateSliceLegal(typeConverter);

    target.addDynamicallyLegalOp<stablehlo::ConstantOp>(
        [&](stablehlo::ConstantOp op) {
          return typeConverter.isLegal(op.getType());
        });

    target.addDynamicallyLegalOp<stablehlo::ConcatenateOp>(
        [&](stablehlo::ConcatenateOp op) {
          if (!typeConverter.isLegal(op.getResult().getType()))
            return false;
          if (op.getOperands().size() != 2)
            return true;
          Operation *lhsOp = op.getOperands()[0].getDefiningOp();
          Operation *rhsOp = op.getOperands()[1].getDefiningOp();
          if (!lhsOp || !rhsOp)
            return true;
          if (lhsOp->getName() != rhsOp->getName())
            return true;
          if (!lhsOp->hasTrait<mlir::OpTrait::Elementwise>())
            return true;
          if (lhsOp->getAttrDictionary() != rhsOp->getAttrDictionary())
            return true;
          for (auto operand : op.getOperands()) {
            for (auto *user : operand.getDefiningOp()->getUsers()) {
              if (user != op)
                return true;
            }
          }
          return false;
        });
    target.addDynamicallyLegalOp<stablehlo::SliceOp>(sliceLegal);
    target.addDynamicallyLegalOp<stablehlo::BroadcastInDimOp>(broadcastLegal);
    target.addDynamicallyLegalOp<stablehlo::TransposeOp>(transposeLegal);
    target.addDynamicallyLegalOp<stablehlo::ReshapeOp>(reshapeLegal);
    target.addDynamicallyLegalOp<stablehlo::CompareOp>(compareLegal);
    target.addDynamicallyLegalOp<stablehlo::FloorOp>(floorLegal);
    target.addDynamicallyLegalOp<stablehlo::CeilOp>(ceilLegal);
    target.addDynamicallyLegalOp<stablehlo::ExpOp>(expLegal);
    target.addDynamicallyLegalOp<stablehlo::LogOp>(logLegal);
    target.addDynamicallyLegalOp<stablehlo::SineOp>(sineLegal);
    target.addDynamicallyLegalOp<stablehlo::DotGeneralOp>(dotGeneralLegal);
    target.addDynamicallyLegalOp<enzymexla::RotateOp>(rotateLegal);
    target.addDynamicallyLegalOp<enzymexla::WrapOp>(wrapLegal);
    target.addDynamicallyLegalOp<enzymexla::ExtendOp>(extendLegal);
    target.addDynamicallyLegalOp<enzymexla::UpdateWithoutCornersOp>(
        updateWithoutCornersLegal);
    target.addDynamicallyLegalOp<stablehlo::NegOp>(negLegal);
    target.addDynamicallyLegalOp<stablehlo::DynamicUpdateSliceOp>(
        dynamicUpdateSliceLegal);
    target.addDynamicallyLegalOp<stablehlo::MaxOp>(maxLegal);
    target.addDynamicallyLegalOp<stablehlo::MinOp>(minLegal);
    target.addDynamicallyLegalOp<stablehlo::AddOp>([&](stablehlo::AddOp op) {
      if (isa<stablehlo::ReduceOp>(op->getParentOp()))
        return true;
      return addLegal(op);
    });
    target.addDynamicallyLegalOp<stablehlo::SubtractOp>(subLegal);
    target.addDynamicallyLegalOp<stablehlo::MulOp>(mulLegal);
    target.addDynamicallyLegalOp<stablehlo::DivOp>(divLegal);
    target.addDynamicallyLegalOp<stablehlo::SelectOp>(selectLegal);
    target.addDynamicallyLegalOp<stablehlo::ConvertOp>(
        [&](stablehlo::ConvertOp op) {
          auto outTensorType =
              dyn_cast<RankedTensorType>(op.getResult().getType());
          if (!outTensorType)
            return true;
          auto inTensorType =
              dyn_cast<RankedTensorType>(op.getOperand().getType());
          if (!inTensorType)
            return true;
          Type outElType = outTensorType.getElementType();
          Type inElType = inTensorType.getElementType();

          if (outElType == srcTy && inElType != srcTy)
            return false;
          if (inElType == srcTy && outElType != srcTy)
            return false;
          if (outElType == srcTy && inElType == srcTy &&
              op.getOperand().getType() == op.getResult().getType())
            return false;
          return true;
        });
    target.addDynamicallyLegalOp<stablehlo::ReverseOp>(reverseLegal);
    target.addDynamicallyLegalOp<stablehlo::AbsOp>(absLegal);
    target.addDynamicallyLegalOp<stablehlo::SqrtOp>(sqrtLegal);
    target.addDynamicallyLegalOp<stablehlo::PadOp>(padLegal);
    target.addDynamicallyLegalOp<stablehlo::WhileOp>(
        [&](stablehlo::WhileOp op) {
          return typeConverter.isLegal(op.getOperandTypes()) &&
                 typeConverter.isLegal(op.getResultTypes());
        });
    target.addDynamicallyLegalOp<stablehlo::ReturnOp>(
        [&](stablehlo::ReturnOp op) {
          if (isa<stablehlo::ReduceOp>(op->getParentOp()))
            return true;

          if (!convertSignatures && isa<func::FuncOp>(op->getParentOp())) {
            OpBuilder b(op.getContext());
            for (auto type : op.getOperandTypes()) {
              if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
                if (tensorType.getElementType() == tgtTy) {
                  return false;
                }
              }
            }
          }
          return typeConverter.isLegal(op.getOperandTypes());
        });
    target.addDynamicallyLegalOp<stablehlo::ReduceWindowOp>(reduceWindowLegal);
    target.addDynamicallyLegalOp<stablehlo::ReduceOp>(reduceLegal);

    if (expansionSize >= 2) {
      SmallVector<func::FuncOp> funcsToConvert;
      op->walk([&](func::FuncOp func) {
        if (!func->hasAttr("enzyme.no_multifloat")) {
          funcsToConvert.push_back(func);
        }
      });

      GreedyRewriteConfig config;
      for (auto func : funcsToConvert) {
        RewritePatternSet patterns(context);
        patterns.add<LowerReduceWindowOp>(context, srcTy);
        if (dotGeneralToReduce)
          patterns.add<DotGeneralToMulReducePattern>(context);
        if (failed(applyPatternsGreedily(func, std::move(patterns), config))) {
          signalPassFailure();
          return;
        }
      }
    }
    RewritePatternSet patterns(context);
    patterns.add<ConstantOpConversion>(typeConverter, context, concatDimension,
                                       expansionSize, srcTy, tgtTy);
    if (expansionSize == 1) {
      patterns.add<GenericOpConversion<stablehlo::AddOp>>(typeConverter,
                                                          context);
      patterns.add<GenericOpConversion<stablehlo::SubtractOp>>(typeConverter,
                                                               context);
      patterns.add<GenericOpConversion<stablehlo::MulOp>>(typeConverter,
                                                          context);
      patterns.add<GenericOpConversion<stablehlo::DivOp>>(typeConverter,
                                                          context);
      patterns.add<GenericOpConversion<stablehlo::SelectOp>>(typeConverter,
                                                             context);
      patterns.add<GenericOpConversion<stablehlo::CompareOp>>(typeConverter,
                                                              context);
      patterns.add<GenericOpConversion<stablehlo::ReverseOp>>(typeConverter,
                                                              context);
      patterns.add<GenericOpConversion<stablehlo::AbsOp>>(typeConverter,
                                                          context);
      patterns.add<GenericOpConversion<stablehlo::ExpOp>>(typeConverter,
                                                          context);
      patterns.add<GenericOpConversion<stablehlo::FloorOp>>(typeConverter,
                                                            context);
      patterns.add<GenericOpConversion<stablehlo::SqrtOp>>(typeConverter,
                                                           context);
      patterns.add<GenericOpConversion<stablehlo::SliceOp>>(typeConverter,
                                                            context);
      patterns.add<GenericOpConversion<stablehlo::BroadcastInDimOp>>(
          typeConverter, context);
      patterns.add<GenericOpConversion<stablehlo::TransposeOp>>(typeConverter,
                                                                context);
      patterns.add<GenericOpConversion<stablehlo::ReshapeOp>>(typeConverter,
                                                              context);
      patterns.add<GenericOpConversion<stablehlo::DotGeneralOp>>(typeConverter,
                                                                 context);
      patterns.add<GenericOpConversion<stablehlo::PadOp>>(typeConverter,
                                                          context);
      patterns.add<ReduceWindowOpConversion>(
          typeConverter, context, concatDimension, expansionSize, srcTy, tgtTy);
      patterns.add<WhileOpConversion>(typeConverter, context, concatDimension);
      patterns.add<ReturnOpConversion>(typeConverter, context, concatDimension,
                                       convertSignatures, expansionSize, srcTy,
                                       tgtTy);
      patterns.add<GenericOpConversion<stablehlo::ConcatenateOp>>(typeConverter,
                                                                  context);
      patterns.add<GenericOpConversion<stablehlo::ConvertOp>>(typeConverter,
                                                              context);
      patterns.add<GenericOpConversion<enzymexla::RotateOp>>(typeConverter,
                                                             context);
      patterns.add<GenericOpConversion<enzymexla::WrapOp>>(typeConverter,
                                                           context);
      patterns.add<GenericOpConversion<enzymexla::ExtendOp>>(typeConverter,
                                                             context);
      patterns.add<GenericOpConversion<enzymexla::UpdateWithoutCornersOp>>(
          typeConverter, context);
      patterns.add<GenericOpConversion<stablehlo::SineOp>>(typeConverter,
                                                           context);
      patterns.add<GenericOpConversion<stablehlo::NegOp>>(typeConverter,
                                                          context);
      patterns.add<GenericOpConversion<stablehlo::DynamicUpdateSliceOp>>(
          typeConverter, context);
    } else if (expansionSize == 2) {
      patterns.add<AddOpConversion>(typeConverter, context, srcTy,
                                    concatDimension);
      patterns.add<SubOpConversion>(typeConverter, context, concatDimension);
      patterns.add<MulOpConversion>(typeConverter, context, concatDimension);
      patterns.add<ReduceOpConversion>(typeConverter, context, concatDimension,
                                       preciseReduce, srcTy, tgtTy);
      patterns.add<DivOpConversion>(typeConverter, context, concatDimension,
                                    divSubsteps);
      patterns.add<SelectOpConversion>(typeConverter, context, concatDimension);
      patterns.add<MaxOpConversion>(typeConverter, context, concatDimension);
      patterns.add<ReverseOpConversion>(typeConverter, context,
                                        concatDimension);
      patterns.add<AbsOpConversion>(typeConverter, context, concatDimension);
      patterns.add<FloorOpConversion>(typeConverter, context, concatDimension);
      patterns.add<CeilOpConversion>(typeConverter, context, concatDimension);
      patterns.add<ExpOpConversion>(typeConverter, context, concatDimension);
      patterns.add<LogOpConversion>(typeConverter, context, concatDimension);
      patterns.add<SqrtOpConversion>(typeConverter, context, concatDimension);
      patterns.add<SliceOpConversion>(typeConverter, context, concatDimension);
      patterns.add<BroadcastInDimOpConversion>(typeConverter, context,
                                               concatDimension);
      patterns.add<TransposeOpConversion>(typeConverter, context,
                                          concatDimension);
      patterns.add<ReshapeOpConversion>(typeConverter, context,
                                        concatDimension);
      patterns.add<DotGeneralOpConversion>(typeConverter, context,
                                           concatDimension, tgtTy);
      patterns.add<PadOpConversion>(typeConverter, context, concatDimension);
      patterns.add<WhileOpConversion>(typeConverter, context, concatDimension);
      patterns.add<ReturnOpConversion>(typeConverter, context, concatDimension,
                                       convertSignatures, expansionSize, srcTy,
                                       tgtTy);
      patterns.add<ReduceWindowOpConversion>(
          typeConverter, context, concatDimension, expansionSize, srcTy, tgtTy);
      patterns.add<ConcatenateOpOptimization>(typeConverter, context, 2);
      patterns.add<ConcatenateOpConversion>(typeConverter, context,
                                            concatDimension);
      patterns.add<ConvertOpConversion>(typeConverter, context, concatDimension,
                                        srcTy, tgtTy, expansionSize);
      patterns.add<CompareOpConversion>(typeConverter, context,
                                        concatDimension);
      if (concatDimension == "tuple") {
        patterns.add<NegOpConversion>(typeConverter, context, concatDimension);
      } else {
        patterns.add<GenericOpConversion<stablehlo::NegOp>>(typeConverter,
                                                            context);
      }
      patterns.add<DynamicUpdateSliceOpConversion>(typeConverter, context,
                                                   concatDimension, tgtTy);
      patterns.add<RotateOpConversion>(typeConverter, context, concatDimension);
      patterns.add<UpdateWithoutCornersOpConversion>(typeConverter, context,
                                                     concatDimension);
      patterns.add<WrapOpConversion>(typeConverter, context, concatDimension);
      patterns.add<ExtendOpConversion>(typeConverter, context, concatDimension);
      patterns.add<SineOpConversion>(typeConverter, context, concatDimension);
    } else {
      op->emitError() << "Unsupported expansion size specified: "
                      << (int)expansionSize;
      signalPassFailure();
      return;
    }

    SmallVector<func::FuncOp> funcsToConvert;
    op->walk([&](func::FuncOp func) {
      if (!func->hasAttr("enzyme.no_multifloat")) {
        funcsToConvert.push_back(func);
      }
    });

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    for (auto func : funcsToConvert) {
      if (failed(applyPartialConversion(func, target, frozenPatterns))) {
        signalPassFailure();
        return;
      }
    }

    if (expansionSize > 1 && !convertSignatures) {
      SmallVector<func::FuncOp> funcs;
      op->walk([&](func::FuncOp f) {
        if (f->getParentOp() == op && !f->hasAttr("enzyme.no_multifloat")) {
          funcs.push_back(f);
        }
      });
      for (auto func : funcs) {
        OpBuilder b(func.getContext());
        b.setInsertionPointToStart(&func.getBody().front());
        for (auto arg : func.getArguments()) {
          if (auto tensorType = dyn_cast<RankedTensorType>(arg.getType())) {
            if (tensorType.getElementType() == srcTy) {
              Location loc = func.getLoc();
              bool needed = false;
              for (auto user : arg.getUsers()) {
                if (isa<UnrealizedConversionCastOp>(user)) {
                  needed = true;
                  break;
                }
              }
              if (!needed)
                continue;

              Value converted = convertToMultifloat(
                  arg, b, loc, tgtTy, concatDimension, expansionSize);
              SmallVector<Operation *> users(arg.getUsers().begin(),
                                             arg.getUsers().end());
              for (auto user : users) {
                if (auto cast = dyn_cast<UnrealizedConversionCastOp>(user)) {
                  cast.getResult(0).replaceAllUsesWith(converted);
                }
              }
            }
          }
        }
        for (auto &blk : func.getBody()) {
          SetVector<UnrealizedConversionCastOp> toErase;
          for (auto &op : blk.getOperations()) {
            if (auto returnOp = dyn_cast<func::ReturnOp>(&op)) {
              OpBuilder b_ret(returnOp);
              SmallVector<Value> newOperands;
              Value zero = nullptr;
              bool changed = false;
              for (auto operand : returnOp.getOperands()) {
                if (auto castOp =
                        operand.getDefiningOp<UnrealizedConversionCastOp>()) {
                  Location loc = returnOp.getLoc();
                  Value packedVal = castOp.getOperand(0);
                  Value converted = convertFromMultifloat(
                      packedVal, b_ret, loc, srcTy, concatDimension, zero);
                  newOperands.push_back(converted);
                  changed = true;
                } else {
                  newOperands.push_back(operand);
                }
              }
              if (changed) {
                returnOp.getOperation()->setOperands(newOperands);
              }
            }
            for (auto op : toErase) {
              if (op.use_empty()) {
                op->erase();
              }
            }
          }
        }
      }
    }

    llvm::SetVector<UnrealizedConversionCastOp> worklist;
    op->walk([&](UnrealizedConversionCastOp castOp) {
      if (castOp.use_empty()) {
        worklist.insert(castOp);
      }
    });

    while (!worklist.empty()) {
      UnrealizedConversionCastOp castOp = worklist.pop_back_val();
      SmallVector<Value> operands(castOp.getOperands().begin(),
                                  castOp.getOperands().end());
      castOp.erase();

      for (Value operand : operands) {
        if (auto defOp = operand.getDefiningOp<UnrealizedConversionCastOp>()) {
          if (defOp.use_empty()) {
            worklist.insert(defOp);
          }
        }
      }
    }
  }
};

} // namespace
