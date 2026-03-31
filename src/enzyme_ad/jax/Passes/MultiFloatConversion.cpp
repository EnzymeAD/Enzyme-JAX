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
  if (floatTy.getWidth() == 64)
    return {11, 53};
  if (floatTy.getWidth() == 32)
    return {8, 24};
  if (floatTy.getWidth() == 16)
    return floatTy.isF16() ? std::make_pair(5, 11) : std::make_pair(8, 8);
  if (floatTy.getWidth() == 8)
    return isa<Float8E4M3FNType>(floatTy) ? std::make_pair(4, 4)
                                          : std::make_pair(5, 3);
  return {0, 0};
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
  Value err2 = builder.create<stablehlo::AddOp>(loc, err1, p2);
  Value err3 = builder.create<stablehlo::AddOp>(loc, err2, p3);
  Value err4 = builder.create<stablehlo::AddOp>(loc, err3, p4);

  return {p, err4};
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

    auto outType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

    if (expansionSize == 1) {
      if (elementsAttr.isSplat()) {
        auto val = elementsAttr.getSplatValue<APFloat>();
        bool losesInfo;
        val.convert(cast<FloatType>(targetType).getFloatSemantics(),
                    APFloat::rmNearestTiesToEven, &losesInfo);
        auto newAttr = SplatElementsAttr::get(outType, val);
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, newAttr);
        return success();
      } else {
        SmallVector<Attribute> convertedAttrs;
        for (auto val : elementsAttr.getValues<APFloat>()) {
          bool losesInfo;
          APFloat newVal = val;
          newVal.convert(cast<FloatType>(targetType).getFloatSemantics(),
                         APFloat::rmNearestTiesToEven, &losesInfo);
          convertedAttrs.push_back(
              rewriter.getFloatAttr(targetType, newVal.convertToDouble()));
        }
        auto newAttr = DenseElementsAttr::get(outType, convertedAttrs);
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, newAttr);
        return success();
      }
    } else if (expansionSize == 2) {
      if (concatDimension == "tuple") {
        auto tupleType =
            cast<TupleType>(getTypeConverter()->convertType(op.getType()));
        auto limbType = cast<RankedTensorType>(tupleType.getType(0));

        if (elementsAttr.isSplat()) {
          auto val = elementsAttr.getSplatValue<APFloat>();
          APFloat hiAP = val;
          bool losesInfo;
          hiAP.convert(cast<FloatType>(targetType).getFloatSemantics(),
                       APFloat::rmNearestTiesToEven, &losesInfo);

          APFloat hiSource = hiAP;
          hiSource.convert(cast<FloatType>(sourceType).getFloatSemantics(),
                           APFloat::rmNearestTiesToEven, &losesInfo);

          APFloat loSource = val;
          loSource.subtract(hiSource, APFloat::rmNearestTiesToEven);

          APFloat loAP = loSource;
          loAP.convert(cast<FloatType>(targetType).getFloatSemantics(),
                       APFloat::rmNearestTiesToEven, &losesInfo);

          auto hiAttr = SplatElementsAttr::get(limbType, hiAP);
          auto loAttr = SplatElementsAttr::get(limbType, loAP);

          Value hiConst = rewriter.create<stablehlo::ConstantOp>(loc, hiAttr);
          Value loConst = rewriter.create<stablehlo::ConstantOp>(loc, loAttr);

          rewriter.replaceOpWithNewOp<stablehlo::TupleOp>(
              op, ValueRange{hiConst, loConst});
          return success();
        } else {
          SmallVector<Attribute> hiAttrs;
          SmallVector<Attribute> loAttrs;
          for (auto val : elementsAttr.getValues<APFloat>()) {
            APFloat hiAP = val;
            bool losesInfo;
            hiAP.convert(cast<FloatType>(targetType).getFloatSemantics(),
                         APFloat::rmNearestTiesToEven, &losesInfo);

            APFloat hiSource = hiAP;
            hiSource.convert(cast<FloatType>(sourceType).getFloatSemantics(),
                             APFloat::rmNearestTiesToEven, &losesInfo);

            APFloat loSource = val;
            loSource.subtract(hiSource, APFloat::rmNearestTiesToEven);

            APFloat loAP = loSource;
            loAP.convert(cast<FloatType>(targetType).getFloatSemantics(),
                         APFloat::rmNearestTiesToEven, &losesInfo);

            hiAttrs.push_back(rewriter.getFloatAttr(targetType, hiAP));
            loAttrs.push_back(rewriter.getFloatAttr(targetType, loAP));
          }
          auto hiAttr = DenseElementsAttr::get(limbType, hiAttrs);
          auto loAttr = DenseElementsAttr::get(limbType, loAttrs);

          Value hiConst = rewriter.create<stablehlo::ConstantOp>(loc, hiAttr);
          Value loConst = rewriter.create<stablehlo::ConstantOp>(loc, loAttr);

          rewriter.replaceOpWithNewOp<stablehlo::TupleOp>(
              op, ValueRange{hiConst, loConst});
          return success();
        }
      } else if (concatDimension == "first" || concatDimension == "last") {
        if (elementsAttr.isSplat()) {
          auto val = elementsAttr.getSplatValue<APFloat>();
          APFloat hiAP = val;
          bool losesInfo;
          hiAP.convert(cast<FloatType>(targetType).getFloatSemantics(),
                       APFloat::rmNearestTiesToEven, &losesInfo);

          APFloat hiSource = hiAP;
          hiSource.convert(cast<FloatType>(sourceType).getFloatSemantics(),
                           APFloat::rmNearestTiesToEven, &losesInfo);

          APFloat loSource = val;
          loSource.subtract(hiSource, APFloat::rmNearestTiesToEven);

          APFloat loAP = loSource;
          loAP.convert(cast<FloatType>(targetType).getFloatSemantics(),
                       APFloat::rmNearestTiesToEven, &losesInfo);

          SmallVector<int64_t> limbShape;
          if (concatDimension == "first") {
            limbShape.push_back(1);
            for (auto dim : elementsAttr.getType().getShape())
              limbShape.push_back(dim);
          } else {
            for (auto dim : elementsAttr.getType().getShape())
              limbShape.push_back(dim);
            limbShape.push_back(1);
          }
          auto limbType = RankedTensorType::get(limbShape, targetType);
          auto hiAttr = SplatElementsAttr::get(limbType, hiAP);
          auto loAttr = SplatElementsAttr::get(limbType, loAP);

          Value hiConst = rewriter.create<stablehlo::ConstantOp>(loc, hiAttr);
          Value loConst = rewriter.create<stablehlo::ConstantOp>(loc, loAttr);

          Value packed =
              packLimbs({hiConst, loConst}, rewriter, loc, concatDimension);
          rewriter.replaceOp(op, packed);
          return success();
        } else if (concatDimension == "first") {
          SmallVector<Attribute> vals;
          for (auto val : elementsAttr.getValues<APFloat>()) {
            double dval = val.convertToDouble();
            float hi = (float)dval;
            vals.push_back(rewriter.getFloatAttr(targetType, hi));
          }
          for (auto val : elementsAttr.getValues<APFloat>()) {
            double dval = val.convertToDouble();
            float hi = (float)dval;
            float lo = (float)(dval - hi);
            vals.push_back(rewriter.getFloatAttr(targetType, lo));
          }
          auto newAttr = DenseElementsAttr::get(outType, vals);
          rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, newAttr);
          return success();
        } else {
          return failure();
        }
      }
    }
    return failure();
  }
};

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

    auto [a, b] = twoSum(x1, y1, rewriter, loc);
    auto [c, d] = twoSum(x2, y2, rewriter, loc);
    auto [new_a, new_c] = fastTwoSum(a, c, rewriter, loc);
    Value b2 = rewriter.create<stablehlo::AddOp>(loc, b, d);
    Value b3 = rewriter.create<stablehlo::AddOp>(loc, b2, new_c);
    auto [final_a, final_b] = fastTwoSum(new_a, b3, rewriter, loc);

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

    auto [p00, e00] = twoProdDekker(x1, y1, rewriter, loc);
    Value p01 = rewriter.create<stablehlo::MulOp>(loc, x1, y2);
    Value p10 = rewriter.create<stablehlo::MulOp>(loc, x2, y1);
    Value p01_p10 = rewriter.create<stablehlo::AddOp>(loc, p01, p10);
    Value e00_new = rewriter.create<stablehlo::AddOp>(loc, e00, p01_p10);
    auto [final_p, final_e] = fastTwoSum(p00, e00_new, rewriter, loc);

    Value packed = packLimbs(final_p, final_e, rewriter, loc, concatDimension);
    rewriter.replaceOp(op, packed);
    return success();
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
                  StringRef concatDimension)
      : OpConversionPattern<stablehlo::DivOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  StringRef concatDimension;

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

    Value z_hi = rewriter.create<stablehlo::DivOp>(loc, x1, y1);

    auto [p_hi, p_l] = twoProdDekker(z_hi, y1, rewriter, loc);

    Value d1 = rewriter.create<stablehlo::SubtractOp>(loc, x1, p_hi);
    Value d2 = rewriter.create<stablehlo::SubtractOp>(loc, d1, p_l);
    Value d3 = rewriter.create<stablehlo::AddOp>(loc, d2, x2);
    Value q = rewriter.create<stablehlo::MulOp>(loc, z_hi, y2);
    Value r = rewriter.create<stablehlo::SubtractOp>(loc, d3, q);

    Value z_lo = rewriter.create<stablehlo::DivOp>(loc, r, y1);

    auto [final_h, final_l] = fastTwoSum(z_hi, z_lo, rewriter, loc);

    Value packed = packLimbs(final_h, final_l, rewriter, loc, concatDimension);
    rewriter.replaceOp(op, packed);
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

    Value hi = extractLimb(adaptor.getOperands()[0], 0, rewriter, loc,
                           concatDimension);
    Value lo = extractLimb(adaptor.getOperands()[0], 1, rewriter, loc,
                           concatDimension);

    auto tensorType = cast<RankedTensorType>(hi.getType());
    auto floatTy = cast<FloatType>(tensorType.getElementType());

    auto zeroAttr = rewriter.getFloatAttr(floatTy, 0.0);
    auto splatAttr = SplatElementsAttr::get(tensorType, zeroAttr);
    Value zero = rewriter.create<stablehlo::ConstantOp>(loc, splatAttr);

    Value isPositive = rewriter.create<stablehlo::CompareOp>(
        loc, hi, zero, stablehlo::ComparisonDirection::GE);

    Value neg_hi = rewriter.create<stablehlo::NegOp>(loc, hi);
    Value neg_lo = rewriter.create<stablehlo::NegOp>(loc, lo);

    Value res_hi =
        rewriter.create<stablehlo::SelectOp>(loc, isPositive, hi, neg_hi);
    Value res_lo =
        rewriter.create<stablehlo::SelectOp>(loc, isPositive, lo, neg_lo);

    Value packed = packLimbs(res_hi, res_lo, rewriter, loc, concatDimension);
    rewriter.replaceOp(op, packed);
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

    Value z_hi = rewriter.create<stablehlo::SqrtOp>(loc, x_hi);
    Value p = rewriter.create<stablehlo::MulOp>(loc, z_hi, z_hi);
    Value d = rewriter.create<stablehlo::SubtractOp>(loc, x_hi, p);
    Value r = rewriter.create<stablehlo::AddOp>(loc, d, x_lo);

    auto twoAttr = rewriter.getFloatAttr(floatTy, 2.0);
    auto splatTwo = SplatElementsAttr::get(tensorType, twoAttr);
    Value two = rewriter.create<stablehlo::ConstantOp>(loc, splatTwo);

    Value two_z_hi = rewriter.create<stablehlo::MulOp>(loc, two, z_hi);
    Value z_lo = rewriter.create<stablehlo::DivOp>(loc, r, two_z_hi);

    auto [final_h, final_l] = fastTwoSum(z_hi, z_lo, rewriter, loc);

    Value packed = packLimbs(final_h, final_l, rewriter, loc, concatDimension);
    rewriter.replaceOp(op, packed);
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
    Location loc = op.getLoc();
    SmallVector<Value> newOperands;

    bool isFuncReturn = isa<func::FuncOp>(op->getParentOp());

    if (!isFuncReturn) {
      if (isa<stablehlo::WhileOp>(op->getParentOp())) {
        SmallVector<Value> flatOperands;
        if (concatDimension == "tuple") {
          for (auto operand : adaptor.getOperands()) {
            Value hi = extractLimb(operand, 0, rewriter, loc, concatDimension);
            Value lo = extractLimb(operand, 1, rewriter, loc, concatDimension);
            flatOperands.push_back(hi);
            flatOperands.push_back(lo);
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
          cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
      auto partType = cast<TupleType>(outType).getType(0);

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
          cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
      auto partType = cast<TupleType>(outType).getType(0);

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
          cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
      auto partType = cast<TupleType>(outType).getType(0);

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

    Value hi_hi = rewriter.create<stablehlo::DotGeneralOp>(
        loc, prodType, lhs_hi, rhs_hi, op.getDotDimensionNumbers(),
        op.getPrecisionConfigAttr(), op.getAlgorithmAttr());
    Value hi_lo = rewriter.create<stablehlo::DotGeneralOp>(
        loc, prodType, lhs_hi, rhs_lo, op.getDotDimensionNumbers(),
        op.getPrecisionConfigAttr(), op.getAlgorithmAttr());
    Value lo_hi = rewriter.create<stablehlo::DotGeneralOp>(
        loc, prodType, lhs_lo, rhs_hi, op.getDotDimensionNumbers(),
        op.getPrecisionConfigAttr(), op.getAlgorithmAttr());
    Value lo_lo = rewriter.create<stablehlo::DotGeneralOp>(
        loc, prodType, lhs_lo, rhs_lo, op.getDotDimensionNumbers(),
        op.getPrecisionConfigAttr(), op.getAlgorithmAttr());

    auto prodTensorTy = cast<RankedTensorType>(prodType);
    Value L = rewriter.create<stablehlo::AddOp>(loc, hi_lo, lo_hi);
    L = rewriter.create<stablehlo::AddOp>(loc, L, lo_lo);

    if (concatDimension != "tuple") {
      if (prodTensorTy.getRank() == 0) {
        Type rank1Ty = RankedTensorType::get({1}, targetType);
        hi_hi = rewriter.create<stablehlo::ReshapeOp>(loc, rank1Ty, hi_hi);
        L = rewriter.create<stablehlo::ReshapeOp>(loc, rank1Ty, L);
      } else {
        SmallVector<int64_t> packedShape;
        if (concatDimension == "first") {
          packedShape.push_back(1);
          for (auto s : origShape)
            packedShape.push_back(s);
        } else {
          for (auto s : origShape)
            packedShape.push_back(s);
          packedShape.push_back(1);
        }
        Type packedTy = RankedTensorType::get(packedShape, targetType);
        hi_hi = rewriter.create<stablehlo::ReshapeOp>(loc, packedTy, hi_hi);
        L = rewriter.create<stablehlo::ReshapeOp>(loc, packedTy, L);
      }
    }

    Value packed = packLimbs(hi_hi, L, rewriter, loc, concatDimension);
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
    auto outType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));

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
    : public OpConversionPattern<stablehlo::ReduceWindowOp> {
  Type sourceType;
  StringRef concatDimension;

  LowerReduceWindowOp(TypeConverter &typeConverter, MLIRContext *context,
                      Type sourceType, StringRef concatDimension)
      : OpConversionPattern<stablehlo::ReduceWindowOp>(typeConverter, context),
        sourceType(sourceType), concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::ReduceWindowOp reduceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInputs()[0];
    LLVM_DEBUG(llvm::dbgs() << "adaptor.getInputs().size(): "
                            << adaptor.getInputs().size() << "\n");
    for (size_t i = 0; i < adaptor.getInputs().size(); ++i) {
      LLVM_DEBUG(llvm::dbgs() << "  input " << i << " type: "
                              << adaptor.getInputs()[i].getType() << "\n");
    }
    auto tensorType = cast<RankedTensorType>(input.getType());

    int origRank = reduceOp.getWindowDimensions().size();
    int newRank = tensorType.getRank();
    bool hasPacking = (newRank == origRank + 1);
    int packingSize = hasPacking ? tensorType.getShape()[0] : 1;

    if (!hasPacking && tensorType.getElementType() != sourceType) {
      return failure();
    }

    int rankDiff = newRank - origRank;

    bool isFirst = concatDimension == "first";
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

    int64_t windowSize = dims[reduceDim];

    auto paddingAttr = reduceOp.getPadding();
    if (!paddingAttr)
      return failure();

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

    if (rankDiff > 0) {
      if (isFirst) {
        lowPadding.insert(lowPadding.begin(), 0);
        highPadding.insert(highPadding.begin(), 0);
      } else {
        lowPadding.push_back(0);
        highPadding.push_back(0);
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "input type: " << input.getType() << "\n");

    Location loc = reduceOp.getLoc();

    bool needsPadding = false;
    for (auto p : lowPadding)
      if (p > 0)
        needsPadding = true;
    for (auto p : highPadding)
      if (p > 0)
        needsPadding = true;

    if (!needsPadding)
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "lowPadding: ");
    LLVM_DEBUG(for (auto p : lowPadding) llvm::dbgs() << p << " ");
    LLVM_DEBUG(llvm::dbgs() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "highPadding: ");
    LLVM_DEBUG(for (auto p : highPadding) llvm::dbgs() << p << " ");
    LLVM_DEBUG(llvm::dbgs() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "needsPadding: " << needsPadding << "\n");

    Value paddedInput = input;
    if (needsPadding) {
      Value initValues = adaptor.getInitValues()[0];
      Value initValue =
          extractLimb(initValues, 0, rewriter, loc, concatDimension);

      auto scalarType = RankedTensorType::get(
          {}, cast<RankedTensorType>(initValue.getType()).getElementType());
      initValue =
          rewriter.create<stablehlo::ReshapeOp>(loc, scalarType, initValue);

      SmallVector<int64_t> interiorPadding(newRank, 0);
      paddedInput = rewriter.create<stablehlo::PadOp>(
          loc, input, initValue, rewriter.getDenseI64ArrayAttr(lowPadding),
          rewriter.getDenseI64ArrayAttr(highPadding),
          rewriter.getDenseI64ArrayAttr(interiorPadding));
    }

    auto outType =
        getTypeConverter()->convertType(reduceOp.getResult(0).getType());
    LLVM_DEBUG(llvm::dbgs() << "Converted type of ReduceWindowOp result: "
                            << outType << "\n");
    auto outTensorType = cast<RankedTensorType>(outType);
    auto outShape = outTensorType.getShape();
    int outRank = outTensorType.getRank();

    int actualReduceDim = reduceDim + (hasPacking ? 1 : 0);

    LLVM_DEBUG(llvm::dbgs() << "hasPacking: " << hasPacking << "\n");
    LLVM_DEBUG(llvm::dbgs() << "packingSize: " << packingSize << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "tensorType rank: " << tensorType.getRank() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "dims size: " << dims.size() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "tensorType: " << tensorType << "\n");

    bool outHasPacking = (outRank == dims.size() + 1);
    int actualOutReduceDim = reduceDim + (outHasPacking ? 1 : 0);
    int outSize = outShape[actualOutReduceDim];

    SmallVector<SmallVector<Value>> rollingResultsPerLimb(packingSize);

    for (int k = 0; k < outSize; ++k) {
      SmallVector<int64_t> sliceShape;
      SmallVector<int64_t> startOffsets(newRank, 0);
      SmallVector<int64_t> endOffsets;
      for (int d = 0; d < newRank; ++d) {
        if (hasPacking && d == 0) {
          startOffsets[d] = 0;
          endOffsets.push_back(1);
        } else if (d == actualReduceDim) {
          startOffsets[d] = k;
          endOffsets.push_back(k + windowSize);
        } else {
          int outD = d;
          if (hasPacking && outRank < newRank) {
            if (d > actualReduceDim)
              outD = d - 1;
          } else if (!hasPacking && outRank > newRank) {
            if (d > actualReduceDim)
              outD = d + 1;
          }
          startOffsets[d] = 0;
          endOffsets.push_back(outShape[outD]);
        }
      }
      for (size_t d = 0; d < newRank; ++d) {
        sliceShape.push_back(endOffsets[d] - startOffsets[d]);
      }

      SmallVector<Type> reduceResultTypes;
      SmallVector<int64_t> resShape;
      for (int d = 0; d < newRank; ++d) {
        if (d != actualReduceDim)
          resShape.push_back(sliceShape[d]);
      }
      Type reduceResultType = RankedTensorType::get(
          resShape,
          cast<RankedTensorType>(paddedInput.getType()).getElementType());
      for (int p = 0; p < packingSize; ++p)
        reduceResultTypes.push_back(reduceResultType);

      SmallVector<Value> operands;
      for (int p = 0; p < packingSize; ++p) {
        SmallVector<int64_t> pStart = startOffsets;
        SmallVector<int64_t> pEnd = endOffsets;
        if (hasPacking) {
          pStart[0] = p;
          pEnd[0] = p + 1;
        }
        auto sliceOp = rewriter.create<stablehlo::SliceOp>(
            loc,
            RankedTensorType::get(
                sliceShape,
                cast<RankedTensorType>(paddedInput.getType()).getElementType()),
            paddedInput, rewriter.getDenseI64ArrayAttr(pStart),
            rewriter.getDenseI64ArrayAttr(pEnd),
            rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(newRank, 1)));
        operands.push_back(sliceOp);
      }

      SmallVector<Value> splitInitValues;
      if (packingSize == 2 && adaptor.getInitValues().size() == 1) {
        Value initVal = adaptor.getInitValues()[0];
        Value hiInit = extractLimb(initVal, 0, rewriter, loc, concatDimension);
        Value loInit = extractLimb(initVal, 1, rewriter, loc, concatDimension);

        auto hiType = cast<RankedTensorType>(hiInit.getType());
        auto loType = cast<RankedTensorType>(loInit.getType());

        if (hiType.getRank() == 1 && hiType.getShape()[0] == 1) {
          hiInit = rewriter.create<stablehlo::ReshapeOp>(
              loc, RankedTensorType::get({}, hiType.getElementType()), hiInit);
        }
        if (loType.getRank() == 1 && loType.getShape()[0] == 1) {
          loInit = rewriter.create<stablehlo::ReshapeOp>(
              loc, RankedTensorType::get({}, loType.getElementType()), loInit);
        }
        splitInitValues.push_back(hiInit);
        splitInitValues.push_back(loInit);
      } else {
        for (auto val : adaptor.getInitValues()) {
          if (auto tt = dyn_cast<RankedTensorType>(val.getType())) {
            if (tt.getRank() == 1 && tt.getShape()[0] == 1) {
              val = rewriter.create<stablehlo::ReshapeOp>(
                  loc, RankedTensorType::get({}, tt.getElementType()), val);
            }
          }
          splitInitValues.push_back(val);
        }
      }

      auto newReduceOp = rewriter.create<stablehlo::ReduceOp>(
          loc, TypeRange(reduceResultTypes), operands, splitInitValues,
          ArrayRef<int64_t>{actualReduceDim});
      LLVM_DEBUG(llvm::dbgs() << "ReduceOp created successfully\n");

      LLVM_DEBUG(llvm::dbgs() << "Rewriting region...\n");
      // Create a new block with updated signature
      Region &origRegion = reduceOp.getRegion();
      Block &origBlock = origRegion.front();

      Region &newRegion = newReduceOp.getRegion();
      Type argType =
          splitInitValues[0]
              .getType(); // Use split init value type for region arguments
      LLVM_DEBUG(llvm::dbgs() << "Arg type for new block: " << argType << "\n");
      SmallVector<Type> argTypes;
      SmallVector<Location> argLocs;
      for (int p = 0; p < packingSize; ++p) {
        argTypes.push_back(argType);
        argLocs.push_back(loc);
      }
      for (int p = 0; p < packingSize; ++p) {
        argTypes.push_back(argType);
        argLocs.push_back(loc);
      }

      Block *newBlock;
      if (newReduceOp.getRegion().empty()) {
        LLVM_DEBUG(llvm::dbgs() << "newRegion is empty, creating new block\n");
        newBlock = rewriter.createBlock(&newRegion, {}, argTypes, argLocs);
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "newRegion is NOT empty, obtaining existing block\n");
        newBlock = &newRegion.front();
        LLVM_DEBUG(llvm::dbgs()
                   << "  Existing block has "
                   << newBlock->getOperations().size() << " ops\n");
        LLVM_DEBUG(llvm::dbgs()
                   << "  Existing block has " << newBlock->getNumArguments()
                   << " arguments\n");
        if (!newBlock->empty()) {
          while (!newBlock->empty()) {
            newBlock->front().erase();
          }
          LLVM_DEBUG(llvm::dbgs() << "  Cleared existing block ops\n");
        }
        if (newBlock->getNumArguments() == 0) {
          LLVM_DEBUG(llvm::dbgs() << "  Adding arguments to existing block\n");
          for (size_t i = 0; i < argTypes.size(); ++i) {
            newBlock->addArgument(argTypes[i], argLocs[i]);
          }
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "  Existing block ALREADY had arguments:\n");
          for (unsigned i = 0; i < newBlock->getNumArguments(); ++i) {
            LLVM_DEBUG(llvm::dbgs()
                       << "    arg " << i << " type: "
                       << newBlock->getArgument(i).getType() << "\n");
          }
        }
      }

      rewriter.setInsertionPointToStart(newBlock);

      // Reconstruct combined inputs for the original block ops
      SmallVector<Value> combinedArgs;
      for (int i = 0; i < 2; ++i) { // 2 operands in original region (lhs, rhs)
        LLVM_DEBUG(llvm::dbgs() << "Processing operand " << i << "\n");
        if (packingSize == 1) {
          Value high = newBlock->getArgument(i);
          combinedArgs.push_back(high);
        } else if (packingSize == 2) {
          Value high = newBlock->getArgument(i * 2);
          Value low = newBlock->getArgument(i * 2 + 1);

          if (cast<RankedTensorType>(high.getType()).getRank() == 0) {
            high = rewriter.create<stablehlo::ReshapeOp>(
                loc,
                RankedTensorType::get(
                    {1},
                    cast<RankedTensorType>(high.getType()).getElementType()),
                high);
            low = rewriter.create<stablehlo::ReshapeOp>(
                loc,
                RankedTensorType::get(
                    {1},
                    cast<RankedTensorType>(low.getType()).getElementType()),
                low);
          }

          Value combined;
          if (isFirst) {
            combined = rewriter.create<stablehlo::ConcatenateOp>(
                loc,
                RankedTensorType::get(
                    {2},
                    cast<RankedTensorType>(high.getType()).getElementType()),
                ValueRange{high, low}, 0);
          } else {
            combined = rewriter.create<stablehlo::ConcatenateOp>(
                loc,
                RankedTensorType::get(
                    {2},
                    cast<RankedTensorType>(high.getType()).getElementType()),
                ValueRange{low, high}, 0);
          }
          combinedArgs.push_back(combined);
        }
      }

      LLVM_DEBUG(llvm::dbgs() << "Cloning ops using IRMapping...\n");
      IRMapping mapper;
      for (Operation &op : origBlock) {
        for (Value operand : op.getOperands()) {
          Value rootValue = operand;
          while (auto castOp =
                     dyn_cast_or_null<mlir::UnrealizedConversionCastOp>(
                         rootValue.getDefiningOp())) {
            rootValue = castOp.getOperand(0);
          }
          LLVM_DEBUG(llvm::dbgs()
                     << "Operand: " << operand
                     << " traces back to rootValue: " << rootValue << "\n");
          if (rootValue.getDefiningOp()) {
            LLVM_DEBUG(llvm::dbgs() << "  Defining op: "
                                    << *rootValue.getDefiningOp() << "\n");
          } else {
            LLVM_DEBUG(llvm::dbgs()
                       << "  No defining op (must be block arg)\n");
          }
          if (auto blockArg = dyn_cast<BlockArgument>(rootValue)) {
            LLVM_DEBUG(llvm::dbgs()
                       << "  Is block arg, owner block: " << blockArg.getOwner()
                       << ", origBlock: " << &origBlock << "\n");
            bool shouldMap = false;
            if (blockArg.getOwner() == &origBlock) {
              shouldMap = true;
            } else if (!blockArg.getOwner()->getParentOp()) {
              shouldMap = true; // Detached block, likely old reduction block
            }
            if (shouldMap) {
              unsigned idx = blockArg.getArgNumber();
              mapper.map(operand, combinedArgs[idx]);
              LLVM_DEBUG(llvm::dbgs() << "  Mapped operand to combinedArgs["
                                      << idx << "]\n");
            }
          }
        }
      }

      rewriter.setInsertionPointToEnd(newBlock);
      Value actualRetVal = nullptr;
      LLVM_DEBUG(llvm::dbgs() << "Mapper contents:\n");
      for (size_t i = 0; i < origBlock.getNumArguments(); ++i) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  " << origBlock.getArgument(i) << " -> "
                   << mapper.lookupOrDefault(origBlock.getArgument(i)) << "\n");
      }

      for (Operation &op : origBlock) {
        LLVM_DEBUG(llvm::dbgs() << "Orig op before cloning: " << op << "\n");
        if (isa<stablehlo::ReturnOp>(&op)) {
          actualRetVal = mapper.lookupOrDefault(op.getOperand(0));
          continue;
        }
        OperationState state(op.getLoc(), op.getName().getStringRef());
        SmallVector<Value> mappedOperands;
        for (Value operand : op.getOperands()) {
          mappedOperands.push_back(mapper.lookupOrDefault(operand));
        }
        state.addOperands(mappedOperands);
        state.addAttributes(op.getAttrs());
        for (Type resultType : op.getResultTypes()) {
          if (packingSize == 1 && resultType.isF64()) {
            state.addTypes(rewriter.getF32Type());
          } else if (packingSize == 1 && isa<RankedTensorType>(resultType) &&
                     cast<RankedTensorType>(resultType)
                         .getElementType()
                         .isF64()) {
            auto tt = cast<RankedTensorType>(resultType);
            state.addTypes(
                RankedTensorType::get(tt.getShape(), rewriter.getF32Type()));
          } else {
            Type convertedType = getTypeConverter()->convertType(resultType);
            if (convertedType) {
              state.addTypes(convertedType);
            } else {
              state.addTypes(resultType);
            }
          }
        }
        Operation *clone = rewriter.create(state);

        LLVM_DEBUG(llvm::dbgs() << "Cloning op: " << op.getName() << "\n");
#ifndef NDEBUG
        for (Value operand : op.getOperands()) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  Orig operand type: " << operand.getType() << "\n");
        }
        for (Value operand : clone->getOperands()) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  Cloned operand type: " << operand.getType() << "\n");
        }
#endif
        for (size_t i = 0; i < op.getNumResults(); ++i) {
          mapper.map(op.getResult(i), clone->getResult(i));
        }
      }

      if (!actualRetVal) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "Error: Could not find return value from original region\n");
        return failure();
      }

      // Now create the ACTUAL ReturnOp based on actualRetVal
      if (packingSize == 1) {
        rewriter.create<stablehlo::ReturnOp>(loc, actualRetVal);
      } else if (packingSize == 2) {
        Value highRet =
            extractLimb(actualRetVal, 0, rewriter, loc, concatDimension);
        Value lowRet =
            extractLimb(actualRetVal, 1, rewriter, loc, concatDimension);

        auto highType = cast<RankedTensorType>(highRet.getType());
        auto lowType = cast<RankedTensorType>(lowRet.getType());

        if (highType.getRank() == 1 && highType.getShape()[0] == 1) {
          highRet = rewriter.create<stablehlo::ReshapeOp>(
              loc, RankedTensorType::get({}, highType.getElementType()),
              highRet);
        }
        if (lowType.getRank() == 1 && lowType.getShape()[0] == 1) {
          lowRet = rewriter.create<stablehlo::ReshapeOp>(
              loc, RankedTensorType::get({}, lowType.getElementType()), lowRet);
        }

        LLVM_DEBUG(llvm::dbgs() << "Creating tablehlo::ReturnOp for "
                                   "packingSize == 2 with 2 values\n");
        rewriter.create<stablehlo::ReturnOp>(loc, ValueRange{highRet, lowRet});
        LLVM_DEBUG(llvm::dbgs() << "tablehlo::ReturnOp created successfully\n");
      }

      LLVM_DEBUG(llvm::dbgs() << "Region rewritten using IRMapping cloning\n");

      rewriter.setInsertionPointAfter(newReduceOp);

      for (int p = 0; p < packingSize; ++p) {
        Value res = newReduceOp.getResult(p);
        SmallVector<int64_t> reshapedShape;
        for (int d = 0; d < newRank; ++d) {
          if (d == actualReduceDim) {
            reshapedShape.push_back(1);
          } else {
            int resD = d;
            if (d > actualReduceDim)
              resD = d - 1;
            reshapedShape.push_back(
                cast<RankedTensorType>(res.getType()).getShape()[resD]);
          }
        }
        Value reshapedRes = rewriter.create<stablehlo::ReshapeOp>(
            loc,
            RankedTensorType::get(
                reshapedShape,
                cast<RankedTensorType>(res.getType()).getElementType()),
            res);
        rollingResultsPerLimb[p].push_back(reshapedRes);
      }
    }

    SmallVector<Value> packingResults;
    for (int p = 0; p < packingSize; ++p) {
      Value limbResult;
      if (rollingResultsPerLimb[p].size() == 1) {
        limbResult = rollingResultsPerLimb[p][0];
      } else {
        SmallVector<int64_t> concatShape = llvm::to_vector(
            cast<RankedTensorType>(rollingResultsPerLimb[p][0].getType())
                .getShape());
        concatShape[actualReduceDim] = outSize;
        limbResult = rewriter.create<stablehlo::ConcatenateOp>(
            loc,
            RankedTensorType::get(
                concatShape,
                cast<RankedTensorType>(rollingResultsPerLimb[p][0].getType())
                    .getElementType()),
            rollingResultsPerLimb[p], actualReduceDim);
      }
      packingResults.push_back(limbResult);
    }

    Value finalResult;
    if (packingResults.size() == 1) {
      finalResult = packingResults[0];
    } else {
      if (hasPacking) {
        SmallVector<int64_t> concatShape = llvm::to_vector(
            mlir::cast<RankedTensorType>(packingResults[0].getType())
                .getShape());
        concatShape[0] = packingSize;

        finalResult = rewriter.create<stablehlo::ConcatenateOp>(
            loc,
            RankedTensorType::get(concatShape, mlir::cast<RankedTensorType>(
                                                   packingResults[0].getType())
                                                   .getElementType()),
            packingResults, 0);
      } else {
        SmallVector<Value> reshapedPackingResults;
        for (Value res : packingResults) {
          SmallVector<int64_t> reshapedShape = {1};
          for (int64_t s :
               mlir::cast<RankedTensorType>(res.getType()).getShape()) {
            reshapedShape.push_back(s);
          }
          reshapedPackingResults.push_back(
              rewriter.create<stablehlo::ReshapeOp>(
                  loc,
                  RankedTensorType::get(
                      reshapedShape, mlir::cast<RankedTensorType>(res.getType())
                                         .getElementType()),
                  res));
        }

        SmallVector<int64_t> concatShape = {packingSize};
        for (int64_t s :
             mlir::cast<RankedTensorType>(packingResults[0].getType())
                 .getShape()) {
          concatShape.push_back(s);
        }

        finalResult = rewriter.create<stablehlo::ConcatenateOp>(
            loc,
            RankedTensorType::get(concatShape, mlir::cast<RankedTensorType>(
                                                   packingResults[0].getType())
                                                   .getElementType()),
            reshapedPackingResults, 0);
      }
    }

    if (outRank < newRank) {
      finalResult =
          rewriter.create<stablehlo::ReshapeOp>(loc, outType, finalResult);
    }

    rewriter.replaceOp(reduceOp, finalResult);
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

    IsResultOrOperandTypeLegal<stablehlo::SliceOp> sliceLegal(typeConverter);
    IsResultOrOperandTypeLegal<stablehlo::BroadcastInDimOp> broadcastLegal(
        typeConverter);
    IsResultOrOperandTypeLegal<stablehlo::TransposeOp> transposeLegal(
        typeConverter);
    IsResultOrOperandTypeLegal<stablehlo::ReshapeOp> reshapeLegal(
        typeConverter);
    IsResultOrOperandTypeLegal<stablehlo::CompareOp> compareLegal(
        typeConverter);
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
          return false;
        });
    target.addDynamicallyLegalOp<stablehlo::SliceOp>(sliceLegal);
    target.addDynamicallyLegalOp<stablehlo::BroadcastInDimOp>(broadcastLegal);
    target.addDynamicallyLegalOp<stablehlo::TransposeOp>(transposeLegal);
    target.addDynamicallyLegalOp<stablehlo::ReshapeOp>(reshapeLegal);
    target.addDynamicallyLegalOp<stablehlo::CompareOp>(compareLegal);
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
    target.addDynamicallyLegalOp<stablehlo::AddOp>(addLegal);
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

    RewritePatternSet patterns(context);
    patterns.add<ConstantOpConversion>(typeConverter, context, concatDimension,
                                       expansionSize, srcTy, tgtTy);
    if (expansionSize >= 2) {
      patterns.add<LowerReduceWindowOp>(typeConverter, context, srcTy,
                                        concatDimension);
    }
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
      patterns.add<DivOpConversion>(typeConverter, context, concatDimension);
      patterns.add<SelectOpConversion>(typeConverter, context, concatDimension);
      patterns.add<ReverseOpConversion>(typeConverter, context,
                                        concatDimension);
      patterns.add<AbsOpConversion>(typeConverter, context, concatDimension);
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
      patterns.add<GenericOpConversion<stablehlo::NegOp>>(typeConverter,
                                                          context);
      patterns.add<DynamicUpdateSliceOpConversion>(typeConverter, context,
                                                   concatDimension, tgtTy);
      patterns.add<RotateOpConversion>(typeConverter, context, concatDimension);
      patterns.add<UpdateWithoutCornersOpConversion>(typeConverter, context,
                                                     concatDimension);
      patterns.add<WrapOpConversion>(typeConverter, context, concatDimension);
      patterns.add<ExtendOpConversion>(typeConverter, context, concatDimension);
      patterns.add<GenericOpConversion<stablehlo::SineOp>>(typeConverter,
                                                           context);
    } else {
      op->emitError() << "Unsupported expansion size specified: "
                      << (int)expansionSize;
      signalPassFailure();
      return;
    }

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
    }

    if (expansionSize > 1 && !convertSignatures) {
      SmallVector<func::FuncOp> funcs;
      op->walk([&](func::FuncOp f) {
        if (f->getParentOp() == op) {
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
