//===- ArithRaising.cpp - Raise to Arith dialect --------------------------- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file implements a pass to raise operations to arith dialect.
//
//===---------------------------------------------------------------------===//

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_ARITHRAISINGPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {

// isnan %a -> and(not(is_finite %a), not(is_inf %a))
//
// Emitted inline instead of creating a math::IsNaNOp for RaiseIsNaN to pick up:
// the pass is a single walk, so ops created by a pattern are never revisited.
static Value emitIsNaN(OpBuilder &builder, Location loc, Value val) {
  Value isFinite = stablehlo::IsFiniteOp::create(builder, loc, val);
  Value isNotFinite = stablehlo::NotOp::create(builder, loc, isFinite);

  Value isNotInf = stablehlo::NotOp::create(
      builder, loc, chlo::IsInfOp::create(builder, loc, val));

  return stablehlo::AndOp::create(builder, loc, isNotFinite, isNotInf);
}

template <typename SrcOp, typename StableHLOOp, typename MHLOOp>
struct RaiseBinary : public OpRewritePattern<SrcOp> {
  RaiseBinary(bool useStablehlo, MLIRContext *context,
              PatternBenefit benefit = 1)
      : OpRewritePattern<SrcOp>(context, benefit), useStablehlo(useStablehlo) {}

  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<RankedTensorType>(op->getResult(0).getType()))
      return failure();

    if (useStablehlo)
      rewriter.replaceOpWithNewOp<StableHLOOp>(op, op->getOperand(0),
                                               op->getOperand(1));
    else
      rewriter.replaceOpWithNewOp<MHLOOp>(op, op->getOperand(0),
                                          op->getOperand(1));
    return success();
  }

  bool useStablehlo;
};

template <typename SrcOp, typename StableHLOOp, typename MHLOOp>
struct RaiseUnary : public OpRewritePattern<SrcOp> {
  RaiseUnary(bool useStablehlo, MLIRContext *context,
             PatternBenefit benefit = 1)
      : OpRewritePattern<SrcOp>(context, benefit), useStablehlo(useStablehlo) {}

  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<RankedTensorType>(op.getType()))
      return failure();

    if (useStablehlo)
      rewriter.replaceOpWithNewOp<StableHLOOp>(op, op->getOperand(0));
    else
      rewriter.replaceOpWithNewOp<MHLOOp>(op, op->getOperand(0));
    return success();
  }

  bool useStablehlo;
};

// Cast-like arith ops whose raised form is just a stablehlo.convert to the
// result type.
template <typename SrcOp>
struct RaiseToConvert : public OpRewritePattern<SrcOp> {
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    auto ty = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!ty)
      return failure();

    // stablehlo.convert reads i1 as boolean (true -> 1), but a sign
    // extension of i1 means true -> -1: negate the boolean's conversion.
    if (std::is_same_v<SrcOp, arith::ExtSIOp> &&
        cast<RankedTensorType>(op.getIn().getType())
            .getElementType()
            .isInteger(1)) {
      Value conv =
          stablehlo::ConvertOp::create(rewriter, op.getLoc(), ty, op.getIn());
      rewriter.replaceOpWithNewOp<stablehlo::NegOp>(op, conv);
      return success();
    }

    rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, ty, op.getIn());
    return success();
  }
};

struct RaiseBitcast : public OpRewritePattern<arith::BitcastOp> {
  using OpRewritePattern<arith::BitcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::BitcastOp op,
                                PatternRewriter &rewriter) const override {
    auto ty = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!ty)
      return failure();

    size_t outSize =
        cast<AutoDiffTypeInterface>(ty.getElementType()).getApproxSize();
    size_t inSize =
        cast<AutoDiffTypeInterface>(
            cast<RankedTensorType>(op.getOperand().getType()).getElementType())
            .getApproxSize();

    Value res;
    if (outSize == inSize) {
      res = stablehlo::BitcastConvertOp::create(rewriter, op.getLoc(), ty,
                                                op.getIn());
    } else if (outSize < inSize) {
      SmallVector<int64_t> dims2 = llvm::to_vector(ty.getShape());
      auto oidx = dims2.size();
      dims2.push_back(inSize / outSize);
      if (oidx != 0 && dims2[oidx - 1] != ShapedType::kDynamic) {
        dims2[oidx - 1] /= inSize / outSize;
      }
      res = stablehlo::BitcastConvertOp::create(
          rewriter, op.getLoc(),
          RankedTensorType::get(dims2, ty.getElementType()), op.getIn());
      bool anyDynamic = false;
      for (auto idx : dims2) {
        if (idx == ShapedType::kDynamic) {
          anyDynamic = true;
          break;
        }
      }
      if (anyDynamic) {
        SmallVector<Value> vals;
        for (size_t i = 0; i < ty.getShape().size(); i++) {
          auto val = stablehlo::GetDimensionSizeOp::create(
              rewriter, op.getLoc(), op.getIn(), i);
          Value vval = val;
          if (i == ty.getShape().size() - 1) {
            auto cst = stablehlo::ConstantOp::create(
                rewriter, op.getLoc(), val.getType(),
                cast<ElementsAttr>(makeAttr(val.getType(), inSize / outSize)));
            vval = stablehlo::MulOp::create(rewriter, op.getLoc(), vval, cst);
          }
          vval = stablehlo::ReshapeOp::create(
              rewriter, op.getLoc(),
              RankedTensorType::get({1}, val.getType().getElementType()), vval);
          vals.push_back(vval);
        }

        auto idxs =
            stablehlo::ConcatenateOp::create(rewriter, op.getLoc(), vals, 0);
        res = stablehlo::DynamicReshapeOp::create(rewriter, op.getLoc(), ty,
                                                  res, idxs);
      } else {
        res = stablehlo::ReshapeOp::create(rewriter, op.getLoc(), ty, res);
      }
    } else {
      SmallVector<int64_t> dims2 = llvm::to_vector(ty.getShape());
      auto oidx = dims2.size();
      dims2.push_back(outSize / inSize);
      if (oidx != 0 && dims2[oidx - 1] != ShapedType::kDynamic) {
        dims2[oidx - 1] /= outSize / inSize;
      }
      res = stablehlo::ReshapeOp::create(
          rewriter, op.getLoc(),
          RankedTensorType::get(
              dims2, cast<RankedTensorType>(op.getOperand().getType())
                         .getElementType()),
          op.getIn());
      res = stablehlo::BitcastConvertOp::create(rewriter, op.getLoc(), ty, res);
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

// stablehlo has no fused multiply-add, so the separate mul+add is the required
// lowering for strict fma and the permitted one for fmuladd alike.
template <typename SrcOp> struct RaiseMulAdd : public OpRewritePattern<SrcOp> {
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp fma,
                                PatternRewriter &rewriter) const override {
    if (!isa<RankedTensorType>(fma.getResult().getType()))
      return failure();

    auto res = stablehlo::MulOp::create(rewriter, fma.getLoc(),
                                        fma.getOperand(0), fma.getOperand(1));
    rewriter.replaceOpWithNewOp<stablehlo::AddOp>(fma, res, fma.getOperand(2));
    return success();
  }
};

struct RaiseCopySign : public OpRewritePattern<math::CopySignOp> {
  using OpRewritePattern<math::CopySignOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(math::CopySignOp copySignOp,
                                PatternRewriter &rewriter) const override {
    auto ty = dyn_cast<RankedTensorType>(copySignOp.getResult().getType());
    if (!ty)
      return failure();

    // The copysign returns a value with the magnitude of the first operand
    // and the sign of the second operand.
    auto loc = copySignOp.getLoc();
    Value val = copySignOp.getLhs();
    Value sign = copySignOp.getRhs();
    Attribute constAttr = FloatAttr::get(ty.getElementType(), 0);
    Value zero = stablehlo::ConstantOp::create(
        rewriter, loc, ty, SplatElementsAttr::get(ty, constAttr));
    Value signPositive = stablehlo::CompareOp::create(
        rewriter, loc, sign, zero, stablehlo::ComparisonDirection::GE);
    Value valPositive = stablehlo::CompareOp::create(
        rewriter, loc, val, zero, stablehlo::ComparisonDirection::GE);
    Value notSameSign =
        stablehlo::XorOp::create(rewriter, loc, signPositive, valPositive);
    Value negVal = stablehlo::NegOp::create(rewriter, loc, val);

    rewriter.replaceOpWithNewOp<stablehlo::SelectOp>(copySignOp, notSameSign,
                                                     negVal, val);
    return success();
  }
};

struct RaiseTruncOp : public OpRewritePattern<math::TruncOp> {
  using OpRewritePattern<math::TruncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(math::TruncOp truncOp,
                                PatternRewriter &rewriter) const override {
    // trunc(x) rounds towards zero: select(x >= 0, floor(x), ceil(x)).
    auto ty = dyn_cast<RankedTensorType>(truncOp.getResult().getType());
    if (!ty)
      return failure();

    auto loc = truncOp.getLoc();
    Value val = truncOp.getOperand();
    Attribute constAttr = FloatAttr::get(ty.getElementType(), 0);
    Value zero = stablehlo::ConstantOp::create(
        rewriter, loc, ty, SplatElementsAttr::get(ty, constAttr));
    Value isNonNegative = stablehlo::CompareOp::create(
        rewriter, loc, val, zero, stablehlo::ComparisonDirection::GE);
    Value flr = stablehlo::FloorOp::create(rewriter, loc, val);
    Value cl = stablehlo::CeilOp::create(rewriter, loc, val);

    rewriter.replaceOpWithNewOp<stablehlo::SelectOp>(truncOp, isNonNegative,
                                                     flr, cl);
    return success();
  }
};

struct RaiseAtan : public OpRewritePattern<math::AtanOp> {
  using OpRewritePattern<math::AtanOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(math::AtanOp atanOp,
                                PatternRewriter &rewriter) const override {
    // atan %a -> atan2(%a, 1.0)
    auto ty = dyn_cast<RankedTensorType>(atanOp.getResult().getType());
    if (!ty)
      return failure();

    Attribute oneAttr0;
    if (isa<IntegerType>(ty.getElementType()))
      oneAttr0 = rewriter.getIntegerAttr(ty.getElementType(), 1);
    else if (isa<FloatType>(ty.getElementType()))
      oneAttr0 = rewriter.getFloatAttr(ty.getElementType(), 1);
    else if (auto CT = dyn_cast<ComplexType>(ty.getElementType()))
      oneAttr0 = complex::NumberAttr::get(CT, 1, 0);
    else
      return failure();

    auto oneAttr = DenseElementsAttr::get(ty, oneAttr0);

    Value one =
        stablehlo::ConstantOp::create(rewriter, atanOp.getLoc(), oneAttr);
    rewriter.replaceOpWithNewOp<stablehlo::Atan2Op>(atanOp, atanOp.getOperand(),
                                                    one);
    return success();
  }
};

struct RaiseMaxNumF : public OpRewritePattern<arith::MaxNumFOp> {
  using OpRewritePattern<arith::MaxNumFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MaxNumFOp maxOp,
                                PatternRewriter &rewriter) const override {
    // maxnumf %a,%b -> select(isnan(%a), %b, max(%a, %b))
    if (!isa<RankedTensorType>(maxOp.getResult().getType()))
      return failure();

    Value isLhsNaN = emitIsNaN(rewriter, maxOp.getLoc(), maxOp.getLhs());
    Value max = stablehlo::MaxOp::create(rewriter, maxOp.getLoc(),
                                         maxOp.getLhs(), maxOp.getRhs());
    rewriter.replaceOpWithNewOp<stablehlo::SelectOp>(maxOp, isLhsNaN,
                                                     maxOp.getRhs(), max);
    return success();
  }
};

struct RaiseMinNumF : public OpRewritePattern<arith::MinNumFOp> {
  using OpRewritePattern<arith::MinNumFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MinNumFOp minOp,
                                PatternRewriter &rewriter) const override {
    // minnumf %a,%b -> select(isnan(%a), %b, min(%a, %b))
    if (!isa<RankedTensorType>(minOp.getResult().getType()))
      return failure();

    Value isLhsNaN = emitIsNaN(rewriter, minOp.getLoc(), minOp.getLhs());
    Value min = stablehlo::MinOp::create(rewriter, minOp.getLoc(),
                                         minOp.getLhs(), minOp.getRhs());
    rewriter.replaceOpWithNewOp<stablehlo::SelectOp>(minOp, isLhsNaN,
                                                     minOp.getRhs(), min);
    return success();
  }
};

struct RaiseIsNaN : public OpRewritePattern<math::IsNaNOp> {
  using OpRewritePattern<math::IsNaNOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(math::IsNaNOp nanOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<RankedTensorType>(nanOp.getResult().getType()))
      return failure();

    rewriter.replaceOp(nanOp,
                       emitIsNaN(rewriter, nanOp.getLoc(), nanOp.getOperand()));
    return success();
  }
};

struct RaiseConj : public OpRewritePattern<complex::ConjOp> {
  using OpRewritePattern<complex::ConjOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(complex::ConjOp conjOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<RankedTensorType>(conjOp->getResultTypes()[0]))
      return failure();

    rewriter.replaceOpWithNewOp<chlo::ConjOp>(conjOp, conjOp->getOperand(0));
    return success();
  }
};

struct RaiseConstant : public OpRewritePattern<arith::ConstantOp> {
  using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp constOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<RankedTensorType>(constOp.getType()))
      return failure();

    auto valueAttr = constOp.getValueAttr();
    if (!isa<ElementsAttr>(valueAttr))
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(constOp, valueAttr);
    return success();
  }
};

struct RaiseFPToSI : public OpRewritePattern<arith::FPToSIOp> {
  using OpRewritePattern<arith::FPToSIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::FPToSIOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<RankedTensorType>(op->getResultTypes()[0]))
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(
        op, op->getOperand(0),
        cast<RankedTensorType>(op->getResult(0).getType()).getElementType());
    return success();
  }
};

struct RaiseSIToFP : public OpRewritePattern<arith::SIToFPOp> {
  using OpRewritePattern<arith::SIToFPOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SIToFPOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<RankedTensorType>(op->getResultTypes()[0]))
      return failure();

    Value newOp = stablehlo::ConvertOp::create(
        rewriter, op.getLoc(), op->getOperand(0),
        cast<RankedTensorType>(op->getResult(0).getType()).getElementType());
    if (cast<RankedTensorType>(op.getOperand().getType())
            .getElementType()
            .isInteger(1)) {
      newOp = stablehlo::NegOp::create(rewriter, op.getLoc(), newOp);
    }
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct RaiseUIToFP : public OpRewritePattern<arith::UIToFPOp> {
  using OpRewritePattern<arith::UIToFPOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::UIToFPOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<RankedTensorType>(op->getResultTypes()[0]))
      return failure();
    if (!cast<RankedTensorType>(op.getOperand().getType())
             .getElementType()
             .isInteger(1))
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(
        op, op->getOperand(0),
        cast<RankedTensorType>(op->getResult(0).getType()).getElementType());
    return success();
  }
};

struct RaiseBroadcast : public OpRewritePattern<enzyme::BroadcastOp> {
  RaiseBroadcast(bool useStablehlo, MLIRContext *context,
                 PatternBenefit benefit = 1)
      : OpRewritePattern<enzyme::BroadcastOp>(context, benefit),
        useStablehlo(useStablehlo) {}

  LogicalResult matchAndRewrite(enzyme::BroadcastOp broadcastOp,
                                PatternRewriter &rewriter) const override {
    assert(useStablehlo);
    SmallVector<int64_t> broadcastDims;
    auto shape = cast<TensorType>(broadcastOp.getInput().getType()).getShape();
    broadcastDims.reserve(shape.size());
    for (auto en : llvm::enumerate(shape)) {
      // original dimensions end up one further because the batch dimension
      // is prepended:
      broadcastDims.push_back(en.index() + 1);
    }
    rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
        broadcastOp, broadcastOp.getType(), broadcastOp.getInput(),
        rewriter.getDenseI64ArrayAttr(broadcastDims));
    return success();
  }

  bool useStablehlo;
};

struct RaiseSelect : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp selectOp,
                                PatternRewriter &rewriter) const override {
    if (llvm::any_of(selectOp->getOperandTypes(),
                     [](Type ty) { return !isa<RankedTensorType>(ty); }))
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::SelectOp>(
        selectOp, selectOp.getType(), selectOp.getCondition(),
        selectOp.getTrueValue(), selectOp.getFalseValue());
    return success();
  }
};

struct RaiseCmpI : public OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern<arith::CmpIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp cmpOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<TensorType>(cmpOp.getType()))
      return failure();

    auto operandType =
        dyn_cast<RankedTensorType>(cmpOp.getOperand(0).getType());
    if (!operandType)
      return failure();

    auto predicate = cmpOp.getPredicate();
    // Booleans (i1) and unsigned integers lower to PRED/unsigned HLO types,
    // which require an UNSIGNED comparison type regardless of the predicate.
    auto elemType = operandType.getElementType();
    bool unsignedPredicate = isUnsignedPredicate(predicate);
    stablehlo::ComparisonType compType =
        (unsignedPredicate || elemType.isUnsignedInteger() ||
         elemType.isInteger(1))
            ? stablehlo::ComparisonType::UNSIGNED
            : stablehlo::ComparisonType::SIGNED;

    stablehlo::ComparisonDirection direction;
    switch (predicate) {
    case arith::CmpIPredicate::eq:
      direction = stablehlo::ComparisonDirection::EQ;
      break;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      direction = stablehlo::ComparisonDirection::GT;
      break;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      direction = stablehlo::ComparisonDirection::GE;
      break;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      direction = stablehlo::ComparisonDirection::LT;
      break;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      direction = stablehlo::ComparisonDirection::LE;
      break;
    case arith::CmpIPredicate::ne:
      direction = stablehlo::ComparisonDirection::NE;
      break;
    default:
      return failure();
    }

    Value lhs = cmpOp.getOperand(0);
    Value rhs = cmpOp.getOperand(1);
    if (unsignedPredicate &&
        (elemType.isSignlessInteger() || elemType.isSignedInteger())) {
      auto unsignedElemType = IntegerType::get(rewriter.getContext(),
                                               elemType.getIntOrFloatBitWidth(),
                                               IntegerType::Unsigned);
      auto unsignedType =
          RankedTensorType::get(operandType.getShape(), unsignedElemType);
      lhs = stablehlo::ConvertOp::create(rewriter, cmpOp.getLoc(), unsignedType,
                                         lhs);
      rhs = stablehlo::ConvertOp::create(rewriter, cmpOp.getLoc(), unsignedType,
                                         rhs);
    }
    rewriter.replaceOpWithNewOp<stablehlo::CompareOp>(cmpOp, lhs, rhs,
                                                      direction, compType);
    return success();
  }
};

struct RaiseCmpF : public OpRewritePattern<arith::CmpFOp> {
  RaiseCmpF(bool useStablehlo, MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<arith::CmpFOp>(context, benefit),
        useStablehlo(useStablehlo) {}

  LogicalResult matchAndRewrite(arith::CmpFOp cmpOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<TensorType>(cmpOp.getType()))
      return failure();

    // TODO: check fast math flags?
    if (useStablehlo) {
      stablehlo::ComparisonDirection direction;
      switch (cmpOp.getPredicate()) {
      case arith::CmpFPredicate::UEQ:
      case arith::CmpFPredicate::OEQ:
        direction = stablehlo::ComparisonDirection::EQ;
        break;
      case arith::CmpFPredicate::UGT:
      case arith::CmpFPredicate::OGT:
        direction = stablehlo::ComparisonDirection::GT;
        break;
      case arith::CmpFPredicate::UGE:
      case arith::CmpFPredicate::OGE:
        direction = stablehlo::ComparisonDirection::GE;
        break;
      case arith::CmpFPredicate::ULT:
      case arith::CmpFPredicate::OLT:
        direction = stablehlo::ComparisonDirection::LT;
        break;
      case arith::CmpFPredicate::ULE:
      case arith::CmpFPredicate::OLE:
        direction = stablehlo::ComparisonDirection::LE;
        break;
      case arith::CmpFPredicate::UNE:
      case arith::CmpFPredicate::ONE:
        direction = stablehlo::ComparisonDirection::NE;
        break;
      default:
        return failure();
      }
      rewriter.replaceOpWithNewOp<stablehlo::CompareOp>(
          cmpOp, cmpOp->getOperand(0), cmpOp->getOperand(1), direction,
          stablehlo::ComparisonType::FLOAT);
    } else {
      mhlo::ComparisonDirection direction;
      switch (cmpOp.getPredicate()) {
      case arith::CmpFPredicate::OEQ:
        direction = mhlo::ComparisonDirection::EQ;
        break;
      case arith::CmpFPredicate::OGT:
        direction = mhlo::ComparisonDirection::GT;
        break;
      case arith::CmpFPredicate::OGE:
        direction = mhlo::ComparisonDirection::GE;
        break;
      case arith::CmpFPredicate::OLT:
        direction = mhlo::ComparisonDirection::LT;
        break;
      case arith::CmpFPredicate::OLE:
        direction = mhlo::ComparisonDirection::LE;
        break;
      case arith::CmpFPredicate::ONE:
        direction = mhlo::ComparisonDirection::NE;
        break;
      default:
        return failure();
      }
      rewriter.replaceOpWithNewOp<mhlo::CompareOp>(
          cmpOp, cmpOp->getOperand(0), cmpOp->getOperand(1), direction);
    }
    return success();
  }

  bool useStablehlo;
};

struct ArithRaisingPass
    : public enzyme::impl::ArithRaisingPassBase<ArithRaisingPass> {
  using ArithRaisingPassBase::ArithRaisingPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    auto *context = patterns.getContext();

    // Ops with both a stablehlo and an mhlo raising.
    // clang-format off
    patterns.add<
        RaiseBinary<arith::AddFOp,     stablehlo::AddOp,       mhlo::AddOp>,
        RaiseBinary<arith::AddIOp,     stablehlo::AddOp,       mhlo::AddOp>,
        RaiseBinary<arith::SubFOp,     stablehlo::SubtractOp,  mhlo::SubtractOp>,
        RaiseBinary<arith::SubIOp,     stablehlo::SubtractOp,  mhlo::SubtractOp>,
        RaiseBinary<arith::MulFOp,     stablehlo::MulOp,       mhlo::MulOp>,
        RaiseBinary<arith::MulIOp,     stablehlo::MulOp,       mhlo::MulOp>,
        RaiseBinary<arith::DivFOp,     stablehlo::DivOp,       mhlo::DivOp>,
        RaiseBinary<arith::DivSIOp,    stablehlo::DivOp,       mhlo::DivOp>,
        RaiseBinary<arith::DivUIOp,    stablehlo::DivOp,       mhlo::DivOp>,
        RaiseBinary<arith::MaximumFOp, stablehlo::MaxOp,       mhlo::MaxOp>,
        RaiseBinary<arith::MaxSIOp,    stablehlo::MaxOp,       mhlo::MaxOp>,
        RaiseBinary<arith::MaxUIOp,    stablehlo::MaxOp,       mhlo::MaxOp>,
        RaiseBinary<arith::MinimumFOp, stablehlo::MinOp,       mhlo::MinOp>,
        RaiseBinary<arith::MinSIOp,    stablehlo::MinOp,       mhlo::MinOp>,
        RaiseBinary<arith::MinUIOp,    stablehlo::MinOp,       mhlo::MinOp>,
        RaiseBinary<complex::AddOp,    stablehlo::AddOp,       mhlo::AddOp>,
        RaiseBinary<arith::AndIOp,     stablehlo::AndOp,       mhlo::AndOp>,
        RaiseBinary<arith::OrIOp,      stablehlo::OrOp,        mhlo::OrOp>,
        RaiseBinary<arith::XOrIOp,     stablehlo::XorOp,       mhlo::XorOp>,
        RaiseBinary<math::PowFOp,      stablehlo::PowOp,       mhlo::PowOp>,
        RaiseBinary<arith::RemFOp,     stablehlo::RemOp,       mhlo::RemOp>,
        RaiseBinary<arith::RemSIOp,    stablehlo::RemOp,       mhlo::RemOp>,
        RaiseBinary<arith::RemUIOp,    stablehlo::RemOp,       mhlo::RemOp>,
        RaiseBinary<arith::ShLIOp,     stablehlo::ShiftLeftOp, mhlo::ShiftLeftOp>,
        RaiseBinary<arith::ShRSIOp,    stablehlo::ShiftRightArithmeticOp, mhlo::ShiftRightArithmeticOp>,
        RaiseBinary<arith::ShRUIOp,    stablehlo::ShiftRightLogicalOp, mhlo::ShiftRightLogicalOp>,
        RaiseBinary<math::Atan2Op,     stablehlo::Atan2Op,     mhlo::Atan2Op>,

        RaiseUnary<math::SinOp,         stablehlo::SineOp,     mhlo::SineOp>,
        RaiseUnary<math::CosOp,         stablehlo::CosineOp,   mhlo::CosineOp>,
        RaiseUnary<math::LogOp,         stablehlo::LogOp,      mhlo::LogOp>,
        RaiseUnary<math::Log1pOp,       stablehlo::Log1pOp,    stablehlo::Log1pOp>,
        RaiseUnary<math::ExpOp,         stablehlo::ExpOp,      mhlo::ExpOp>,
        RaiseUnary<math::ExpM1Op,       stablehlo::Expm1Op,    stablehlo::Expm1Op>,
        RaiseUnary<math::TanhOp,        stablehlo::TanhOp,     stablehlo::TanhOp>,
        RaiseUnary<math::SqrtOp,        stablehlo::SqrtOp,     mhlo::SqrtOp>,
        RaiseUnary<math::RsqrtOp,       stablehlo::RsqrtOp,    mhlo::RsqrtOp>,
        RaiseUnary<math::CbrtOp,        stablehlo::CbrtOp,     mhlo::CbrtOp>,
        RaiseUnary<math::CountLeadingZerosOp, stablehlo::ClzOp, mhlo::ClzOp>,
        RaiseUnary<math::CtPopOp,       stablehlo::PopulationCountOp, mhlo::PopulationCountOp>,
        RaiseUnary<math::AbsFOp,        stablehlo::AbsOp,      mhlo::AbsOp>,
        RaiseUnary<math::AbsIOp,        stablehlo::AbsOp,      mhlo::AbsOp>,
        RaiseUnary<math::IsFiniteOp,    stablehlo::IsFiniteOp, mhlo::IsFiniteOp>,
        RaiseUnary<math::CeilOp,        stablehlo::CeilOp,     mhlo::CeilOp>,
        RaiseUnary<math::FloorOp,       stablehlo::FloorOp,    mhlo::FloorOp>,
        RaiseUnary<math::RoundEvenOp,   stablehlo::RoundNearestEvenOp, mhlo::RoundNearestEvenOp>,
        RaiseUnary<math::RoundOp,       stablehlo::RoundOp,    mhlo::RoundOp>,
        RaiseUnary<math::ErfOp,         chlo::ErfOp,           chlo::ErfOp>,
        RaiseUnary<arith::NegFOp,       stablehlo::NegOp,      mhlo::NegOp>,
        RaiseUnary<enzymexla::LGammaOp, chlo::LgammaOp,        chlo::LgammaOp>,

        RaiseCmpF,
        RaiseBroadcast>(use_stablehlo, context);
    // clang-format on

    // Ops always raised to chlo, regardless of the option.
    patterns.add<RaiseConj>(context);

    // Ops that only have a stablehlo raising.
    if (use_stablehlo)
      patterns
          .add<RaiseBitcast, RaiseToConvert<arith::ConvertFOp>,
               RaiseToConvert<arith::TruncFOp>, RaiseToConvert<arith::ExtFOp>,
               // TODO: either SI or UI is wrong
               RaiseToConvert<arith::ExtUIOp>, RaiseToConvert<arith::ExtSIOp>,
               RaiseToConvert<arith::TruncIOp>, RaiseMulAdd<math::FmaOp>,
               RaiseMulAdd<enzymexla::FMulAddOp>, RaiseCopySign, RaiseTruncOp,
               RaiseAtan, RaiseMaxNumF, RaiseMinNumF, RaiseIsNaN, RaiseConstant,
               RaiseFPToSI, RaiseSIToFP, RaiseUIToFP, RaiseSelect, RaiseCmpI>(
              context);

    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // end anonymous namespace
