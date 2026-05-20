//===- LowerComplexOperations.cpp - Lower Complex Operations Pass
//-----------===//
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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/ConversionUtils.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "lower-complex-operations"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERCOMPLEXOPERATIONSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {

bool isComplexType(Type ty) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(ty)) {
    return isa<ComplexType>(tensorTy.getElementType());
  }
  return isa<ComplexType>(ty);
}

bool operatesOnOriginalArgument(Operation *op) {
  for (auto operand : op->getOperands()) {
    if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
      if (blockArg.getOwner()->isEntryBlock()) {
        if (isa<func::FuncOp>(blockArg.getOwner()->getParentOp()))
          return true;
      }
    }
  }
  return false;
}

bool isReturned(Operation *op) {
  for (auto result : op->getResults()) {
    for (auto &use : result.getUses()) {
      if (isa<func::ReturnOp>(use.getOwner()))
        return true;
    }
  }
  return false;
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
      if (auto cast =
              operand.template getDefiningOp<UnrealizedConversionCastOp>()) {
        if (this->getTypeConverter()->isLegal(cast.getOperand(0).getType())) {
          operand = cast.getOperand(0);
        }
      }
      operands.push_back(operand);
    }

    OperationState state(op.getLoc(), OpTy::getOperationName());
    state.addOperands(operands);
    state.addAttributes(op->getAttrs());
    state.addTypes(
        this->getTypeConverter()->convertType(op->getResult(0).getType()));

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct RealOpConversion : public OpConversionPattern<stablehlo::RealOp> {
  StringRef concatDimension;

  RealOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                   StringRef concatDimension)
      : OpConversionPattern<stablehlo::RealOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::RealOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value convertedOperand = adaptor.getOperands()[0];
    Value realWithDim =
        extractLimb(convertedOperand, 0, rewriter, loc, concatDimension);

    auto type = cast<RankedTensorType>(realWithDim.getType());
    SmallVector<int64_t> targetShape;
    bool isFirst = concatDimension == "first";
    if (isFirst) {
      for (size_t i = 1; i < type.getRank(); ++i)
        targetShape.push_back(type.getShape()[i]);
    } else {
      for (size_t i = 0; i < type.getRank() - 1; ++i)
        targetShape.push_back(type.getShape()[i]);
    }

    Value real = rewriter.create<stablehlo::ReshapeOp>(
        loc, RankedTensorType::get(targetShape, type.getElementType()),
        realWithDim);

    rewriter.replaceOp(op, real);
    return success();
  }
};

struct ImagOpConversion : public OpConversionPattern<stablehlo::ImagOp> {
  StringRef concatDimension;

  ImagOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                   StringRef concatDimension)
      : OpConversionPattern<stablehlo::ImagOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::ImagOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value convertedOperand = adaptor.getOperands()[0];
    Value imagWithDim =
        extractLimb(convertedOperand, 1, rewriter, loc, concatDimension);

    auto type = cast<RankedTensorType>(imagWithDim.getType());
    SmallVector<int64_t> targetShape;
    bool isFirst = concatDimension == "first";
    if (isFirst) {
      for (size_t i = 1; i < type.getRank(); ++i)
        targetShape.push_back(type.getShape()[i]);
    } else {
      for (size_t i = 0; i < type.getRank() - 1; ++i)
        targetShape.push_back(type.getShape()[i]);
    }

    Value imag = rewriter.create<stablehlo::ReshapeOp>(
        loc, RankedTensorType::get(targetShape, type.getElementType()),
        imagWithDim);

    rewriter.replaceOp(op, imag);
    return success();
  }
};

struct ComplexOpConversion : public OpConversionPattern<stablehlo::ComplexOp> {
  StringRef concatDimension;

  ComplexOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                      StringRef concatDimension)
      : OpConversionPattern<stablehlo::ComplexOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::ComplexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value real = adaptor.getOperands()[0];
    Value imag = adaptor.getOperands()[1];

    auto type = cast<RankedTensorType>(real.getType());
    bool isFirst = concatDimension == "first";
    SmallVector<int64_t> expandedShape;
    if (isFirst) {
      expandedShape.push_back(1);
      for (auto dim : type.getShape())
        expandedShape.push_back(dim);
    } else {
      for (auto dim : type.getShape())
        expandedShape.push_back(dim);
      expandedShape.push_back(1);
    }

    auto expandedType =
        RankedTensorType::get(expandedShape, type.getElementType());
    Value expandedReal =
        rewriter.create<stablehlo::ReshapeOp>(loc, expandedType, real);
    Value expandedImag =
        rewriter.create<stablehlo::ReshapeOp>(loc, expandedType, imag);

    Value packed =
        packLimbs({expandedReal, expandedImag}, rewriter, loc, concatDimension);
    rewriter.replaceOp(op, packed);
    return success();
  }
};

struct MulOpConversion : public OpConversionPattern<stablehlo::MulOp> {
  StringRef concatDimension;

  MulOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                  StringRef concatDimension)
      : OpConversionPattern<stablehlo::MulOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];

    Value a = extractLimb(lhs, 0, rewriter, loc, concatDimension);
    Value b = extractLimb(lhs, 1, rewriter, loc, concatDimension);
    Value c = extractLimb(rhs, 0, rewriter, loc, concatDimension);
    Value d = extractLimb(rhs, 1, rewriter, loc, concatDimension);

    Value ac = rewriter.create<stablehlo::MulOp>(loc, a, c);
    Value bd = rewriter.create<stablehlo::MulOp>(loc, b, d);
    Value ad = rewriter.create<stablehlo::MulOp>(loc, a, d);
    Value bc = rewriter.create<stablehlo::MulOp>(loc, b, c);

    Value real = rewriter.create<stablehlo::SubtractOp>(loc, ac, bd);
    Value imag = rewriter.create<stablehlo::AddOp>(loc, ad, bc);

    Value packed = packLimbs(real, imag, rewriter, loc, concatDimension);
    rewriter.replaceOp(op, packed);
    return success();
  }
};

struct DivOpConversion : public OpConversionPattern<stablehlo::DivOp> {
  StringRef concatDimension;

  DivOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                  StringRef concatDimension)
      : OpConversionPattern<stablehlo::DivOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];

    Value a = extractLimb(lhs, 0, rewriter, loc, concatDimension);
    Value b = extractLimb(lhs, 1, rewriter, loc, concatDimension);
    Value c = extractLimb(rhs, 0, rewriter, loc, concatDimension);
    Value d = extractLimb(rhs, 1, rewriter, loc, concatDimension);

    Value ac = rewriter.create<stablehlo::MulOp>(loc, a, c);
    Value bd = rewriter.create<stablehlo::MulOp>(loc, b, d);
    Value bc = rewriter.create<stablehlo::MulOp>(loc, b, c);
    Value ad = rewriter.create<stablehlo::MulOp>(loc, a, d);

    Value c2 = rewriter.create<stablehlo::MulOp>(loc, c, c);
    Value d2 = rewriter.create<stablehlo::MulOp>(loc, d, d);

    Value denom = rewriter.create<stablehlo::AddOp>(loc, c2, d2);

    Value real_num = rewriter.create<stablehlo::AddOp>(loc, ac, bd);
    Value imag_num = rewriter.create<stablehlo::SubtractOp>(loc, bc, ad);

    Value real = rewriter.create<stablehlo::DivOp>(loc, real_num, denom);
    Value imag = rewriter.create<stablehlo::DivOp>(loc, imag_num, denom);

    Value packed = packLimbs(real, imag, rewriter, loc, concatDimension);
    rewriter.replaceOp(op, packed);
    return success();
  }
};

struct AddOpConversion : public OpConversionPattern<stablehlo::AddOp> {
  StringRef concatDimension;

  AddOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                  StringRef concatDimension)
      : OpConversionPattern<stablehlo::AddOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];

    Value a = extractLimb(lhs, 0, rewriter, loc, concatDimension);
    Value b = extractLimb(lhs, 1, rewriter, loc, concatDimension);
    Value c = extractLimb(rhs, 0, rewriter, loc, concatDimension);
    Value d = extractLimb(rhs, 1, rewriter, loc, concatDimension);

    Value real = rewriter.create<stablehlo::AddOp>(loc, a, c);
    Value imag = rewriter.create<stablehlo::AddOp>(loc, b, d);

    Value packed = packLimbs(real, imag, rewriter, loc, concatDimension);
    rewriter.replaceOp(op, packed);
    return success();
  }
};

struct SubOpConversion : public OpConversionPattern<stablehlo::SubtractOp> {
  StringRef concatDimension;

  SubOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                  StringRef concatDimension)
      : OpConversionPattern<stablehlo::SubtractOp>(typeConverter, context),
        concatDimension(concatDimension) {}

  LogicalResult
  matchAndRewrite(stablehlo::SubtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];

    Value a = extractLimb(lhs, 0, rewriter, loc, concatDimension);
    Value b = extractLimb(lhs, 1, rewriter, loc, concatDimension);
    Value c = extractLimb(rhs, 0, rewriter, loc, concatDimension);
    Value d = extractLimb(rhs, 1, rewriter, loc, concatDimension);

    Value real = rewriter.create<stablehlo::SubtractOp>(loc, a, c);
    Value imag = rewriter.create<stablehlo::SubtractOp>(loc, b, d);

    Value packed = packLimbs(real, imag, rewriter, loc, concatDimension);
    rewriter.replaceOp(op, packed);
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
    bool isFirst = concatDimension == "first";

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
    bool isFirst = concatDimension == "first";

    SmallVector<int64_t> permutation = llvm::to_vector(op.getPermutation());
    SmallVector<int64_t> newPermutation;

    if (isFirst) {
      newPermutation.push_back(0);
      for (auto p : permutation) {
        newPermutation.push_back(p + 1);
      }
    } else {
      for (auto p : permutation) {
        newPermutation.push_back(p);
      }
      newPermutation.push_back(permutation.size());
    }

    auto outType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

    auto transposeOp = rewriter.create<stablehlo::TransposeOp>(
        loc, outType, adaptor.getOperands()[0],
        rewriter.getDenseI64ArrayAttr(newPermutation));

    rewriter.replaceOp(op, transposeOp);
    return success();
  }
};

struct LowerComplexOperationsPass
    : public enzyme::impl::LowerComplexOperationsPassBase<
          LowerComplexOperationsPass> {
  using LowerComplexOperationsPassBase::LowerComplexOperationsPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto op = getOperation();

    LLVM_DEBUG(llvm::dbgs() << "Running LowerComplexOperationsPass\n");

    ConversionTarget target(*context);
    target.addLegalDialect<stablehlo::StablehloDialect>();
    target.addLegalDialect<mlir::enzymexla::EnzymeXLADialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    TypeConverter typeConverter;

    typeConverter.addConversion([&](Type type) -> std::optional<Type> {
      if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
        if (auto complexTy = dyn_cast<ComplexType>(tensorTy.getElementType())) {
          Type elTy = complexTy.getElementType();
          SmallVector<int64_t> newShape;
          if (concatDimension == "first") {
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
          return RankedTensorType::get(newShape, elTy);
        }
      }
      return type;
    });

    typeConverter.addSourceMaterialization([](OpBuilder &builder, Type type,
                                              ValueRange inputs,
                                              Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    });
    typeConverter.addTargetMaterialization([](OpBuilder &builder, Type type,
                                              ValueRange inputs,
                                              Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    });

    auto isLegalOp = [&](Operation *op) {
      bool involvesComplex = false;
      for (auto ty : op->getOperandTypes()) {
        if (isComplexType(ty))
          involvesComplex = true;
      }
      for (auto ty : op->getResultTypes()) {
        if (isComplexType(ty))
          involvesComplex = true;
      }
      if (!involvesComplex)
        return true;

      if (auto realOp = dyn_cast<stablehlo::RealOp>(op)) {
        if (auto blockArg = dyn_cast<BlockArgument>(realOp.getOperand())) {
          if (blockArg.getOwner()->isEntryBlock() &&
              isa<func::FuncOp>(blockArg.getOwner()->getParentOp()))
            return true;
        }
      }
      if (auto imagOp = dyn_cast<stablehlo::ImagOp>(op)) {
        if (auto blockArg = dyn_cast<BlockArgument>(imagOp.getOperand())) {
          if (blockArg.getOwner()->isEntryBlock() &&
              isa<func::FuncOp>(blockArg.getOwner()->getParentOp()))
            return true;
        }
      }
      if (auto complexOp = dyn_cast<stablehlo::ComplexOp>(op)) {
        for (auto result : complexOp->getResults()) {
          for (auto &use : result.getUses()) {
            if (isa<func::ReturnOp>(use.getOwner()))
              return true;
          }
        }
      }

      return false;
    };

    target.addDynamicallyLegalOp<stablehlo::AddOp>(isLegalOp);
    target.addDynamicallyLegalOp<stablehlo::SubtractOp>(isLegalOp);
    target.addDynamicallyLegalOp<stablehlo::MulOp>(isLegalOp);
    target.addDynamicallyLegalOp<stablehlo::DivOp>(isLegalOp);
    target.addDynamicallyLegalOp<stablehlo::RealOp>(isLegalOp);
    target.addDynamicallyLegalOp<stablehlo::ImagOp>(isLegalOp);
    target.addDynamicallyLegalOp<stablehlo::ComplexOp>(isLegalOp);
    target.addDynamicallyLegalOp<stablehlo::SliceOp>(isLegalOp);
    target.addDynamicallyLegalOp<stablehlo::TransposeOp>(isLegalOp);

    RewritePatternSet patterns(context);
    patterns.add<RealOpConversion>(typeConverter, context, concatDimension);
    patterns.add<ImagOpConversion>(typeConverter, context, concatDimension);
    patterns.add<ComplexOpConversion>(typeConverter, context, concatDimension);
    patterns.add<AddOpConversion>(typeConverter, context, concatDimension);
    patterns.add<SubOpConversion>(typeConverter, context, concatDimension);
    patterns.add<MulOpConversion>(typeConverter, context, concatDimension);
    patterns.add<DivOpConversion>(typeConverter, context, concatDimension);
    patterns.add<SliceOpConversion>(typeConverter, context, concatDimension);
    patterns.add<TransposeOpConversion>(typeConverter, context,
                                        concatDimension);

    SmallVector<func::FuncOp> funcsToConvert;
    op->walk([&](func::FuncOp func) { funcsToConvert.push_back(func); });

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    for (auto func : funcsToConvert) {
      if (failed(applyPartialConversion(func, target, frozenPatterns))) {
        signalPassFailure();
        return;
      }
    }

    // Handle boundaries
    for (auto func : funcsToConvert) {
      OpBuilder b(func.getContext());
      if (func.getBody().empty())
        continue;
      b.setInsertionPointToStart(&func.getBody().front());
      for (auto arg : func.getArguments()) {
        if (isComplexType(arg.getType())) {
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

          Value real = b.create<stablehlo::RealOp>(loc, arg);
          Value imag = b.create<stablehlo::ImagOp>(loc, arg);
          Value converted = packLimbs(real, imag, b, loc, concatDimension);

          SmallVector<Operation *> users(arg.getUsers().begin(),
                                         arg.getUsers().end());
          for (auto user : users) {
            if (auto cast = dyn_cast<UnrealizedConversionCastOp>(user)) {
              cast.getResult(0).replaceAllUsesWith(converted);
            }
          }
        }
      }

      func.walk([&](func::ReturnOp returnOp) {
        OpBuilder b_ret(returnOp);
        SmallVector<Value> newOperands;
        bool changed = false;
        for (auto operand : returnOp.getOperands()) {
          if (auto castOp =
                  operand.getDefiningOp<UnrealizedConversionCastOp>()) {
            Location loc = returnOp.getLoc();
            Value packedVal = castOp.getOperand(0);
            Value realWithDim =
                extractLimb(packedVal, 0, b_ret, loc, concatDimension);
            Value imagWithDim =
                extractLimb(packedVal, 1, b_ret, loc, concatDimension);

            auto type = cast<RankedTensorType>(realWithDim.getType());
            SmallVector<int64_t> targetShape;
            bool isFirst = concatDimension == "first";
            if (isFirst) {
              for (size_t i = 1; i < type.getRank(); ++i)
                targetShape.push_back(type.getShape()[i]);
            } else {
              for (size_t i = 0; i < type.getRank() - 1; ++i)
                targetShape.push_back(type.getShape()[i]);
            }

            Value real = b_ret.create<stablehlo::ReshapeOp>(
                loc, RankedTensorType::get(targetShape, type.getElementType()),
                realWithDim);
            Value imag = b_ret.create<stablehlo::ReshapeOp>(
                loc, RankedTensorType::get(targetShape, type.getElementType()),
                imagWithDim);

            Value converted =
                b_ret.create<stablehlo::ComplexOp>(loc, real, imag);
            newOperands.push_back(converted);
            changed = true;
          } else {
            newOperands.push_back(operand);
          }
        }
        if (changed) {
          returnOp.getOperation()->setOperands(newOperands);
        }
      });
    }
  }
};

} // namespace
