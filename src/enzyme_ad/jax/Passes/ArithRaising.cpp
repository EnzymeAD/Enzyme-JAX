//===- ArithRaising.cpp - Raise to Arith dialect --------------------------- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file implements a pass to raise operations to arith dialect.
//===---------------------------------------------------------------------===//

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_ARITHRAISINGPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {
struct ArithRaisingPass
    : public enzyme::impl::ArithRaisingPassBase<ArithRaisingPass> {
  using ArithRaisingPassBase::ArithRaisingPassBase;

  void runOnOperation() override {
    auto op = getOperation();

#define RAISE_BINARY(BinaryOp, StableHLOOp, MHLOOp)                            \
  op->walk([=](BinaryOp origOp) {                                              \
    if (!isa<RankedTensorType>(origOp->getResult(0).getType()))                \
      return;                                                                  \
    OpBuilder builder(origOp);                                                 \
    Value newOp;                                                               \
    if (use_stablehlo)                                                         \
      newOp =                                                                  \
          StableHLOOp::create(builder, origOp.getLoc(), origOp->getOperand(0), \
                              origOp->getOperand(1))                           \
              .getResult();                                                    \
    else                                                                       \
      newOp = MHLOOp::create(builder, origOp.getLoc(), origOp->getOperand(0),  \
                             origOp->getOperand(1))                            \
                  .getResult();                                                \
    origOp.replaceAllUsesWith(newOp);                                          \
    origOp.erase();                                                            \
  });

    RAISE_BINARY(arith::AddFOp, stablehlo::AddOp, mhlo::AddOp);
    RAISE_BINARY(arith::AddIOp, stablehlo::AddOp, mhlo::AddOp);
    RAISE_BINARY(arith::SubFOp, stablehlo::SubtractOp, mhlo::SubtractOp);
    RAISE_BINARY(arith::SubIOp, stablehlo::SubtractOp, mhlo::SubtractOp);
    RAISE_BINARY(arith::MulFOp, stablehlo::MulOp, mhlo::MulOp);
    RAISE_BINARY(arith::MulIOp, stablehlo::MulOp, mhlo::MulOp);
    RAISE_BINARY(arith::DivFOp, stablehlo::DivOp, mhlo::DivOp);
    RAISE_BINARY(arith::DivSIOp, stablehlo::DivOp, mhlo::DivOp);
    RAISE_BINARY(arith::DivUIOp, stablehlo::DivOp, mhlo::DivOp);
    RAISE_BINARY(arith::MaximumFOp, stablehlo::MaxOp, mhlo::MaxOp);
    RAISE_BINARY(arith::MaxSIOp, stablehlo::MaxOp, mhlo::MaxOp);
    RAISE_BINARY(arith::MaxUIOp, stablehlo::MaxOp, mhlo::MaxOp);
    RAISE_BINARY(arith::MinimumFOp, stablehlo::MinOp, mhlo::MinOp);
    RAISE_BINARY(arith::MinSIOp, stablehlo::MinOp, mhlo::MinOp);
    RAISE_BINARY(arith::MinUIOp, stablehlo::MinOp, mhlo::MinOp);
    RAISE_BINARY(arith::ShLIOp, stablehlo::ShiftLeftOp, mhlo::ShiftLeftOp);
    RAISE_BINARY(arith::ShRSIOp, stablehlo::ShiftRightArithmeticOp,
                 mhlo::ShiftRightArithmeticOp);
    RAISE_BINARY(arith::ShRUIOp, stablehlo::ShiftRightLogicalOp,
                 mhlo::ShiftRightLogicalOp);
    RAISE_BINARY(complex::AddOp, stablehlo::AddOp, mhlo::AddOp);
    RAISE_BINARY(arith::AndIOp, stablehlo::AndOp, mhlo::AndOp);
    RAISE_BINARY(arith::OrIOp, stablehlo::OrOp, mhlo::OrOp);
    RAISE_BINARY(arith::XOrIOp, stablehlo::XorOp, mhlo::XorOp);
    RAISE_BINARY(math::PowFOp, stablehlo::PowOp, mhlo::PowOp);
    RAISE_BINARY(arith::RemFOp, stablehlo::RemOp, mhlo::RemOp);
    RAISE_BINARY(arith::RemUIOp, stablehlo::RemOp, mhlo::RemOp);

#undef RAISE_BINARY

#define RAISE_UNARY(InputOp, StableHLOOp, MHLOOp)                              \
  op->walk([=](InputOp inpOp) {                                                \
    if (!isa<RankedTensorType>(inpOp.getType()))                               \
      return;                                                                  \
    OpBuilder builder(inpOp);                                                  \
    Value newAddOp;                                                            \
    if (use_stablehlo)                                                         \
      newAddOp =                                                               \
          StableHLOOp::create(builder, inpOp.getLoc(), inpOp->getOperand(0));  \
    else                                                                       \
      newAddOp =                                                               \
          MHLOOp::create(builder, inpOp.getLoc(), inpOp->getOperand(0));       \
    inpOp.replaceAllUsesWith(newAddOp);                                        \
    inpOp.erase();                                                             \
  });

    RAISE_UNARY(math::SinOp, stablehlo::SineOp, mhlo::SineOp);
    RAISE_UNARY(math::CosOp, stablehlo::CosineOp, mhlo::CosineOp);
    RAISE_UNARY(math::LogOp, stablehlo::LogOp, mhlo::LogOp);
    RAISE_UNARY(math::ExpOp, stablehlo::ExpOp, mhlo::ExpOp);
    RAISE_UNARY(math::SqrtOp, stablehlo::SqrtOp, mhlo::SqrtOp);
    RAISE_UNARY(math::RsqrtOp, stablehlo::RsqrtOp, mhlo::RsqrtOp);
    RAISE_UNARY(math::CbrtOp, stablehlo::CbrtOp, mhlo::CbrtOp);
    RAISE_UNARY(math::AbsFOp, stablehlo::AbsOp, mhlo::AbsOp);
    RAISE_UNARY(math::IsFiniteOp, stablehlo::IsFiniteOp, mhlo::IsFiniteOp);
    RAISE_UNARY(arith::NegFOp, stablehlo::NegOp, mhlo::NegOp);

#undef RAISE_UNARY

    op->walk([=](arith::BitcastOp op) {
      auto ty = dyn_cast<RankedTensorType>(op.getResult().getType());
      if (!use_stablehlo || !ty)
        return;

      size_t outSize =
          cast<AutoDiffTypeInterface>(ty.getElementType()).getApproxSize();
      size_t inSize = cast<AutoDiffTypeInterface>(
                          cast<RankedTensorType>(op.getOperand().getType())
                              .getElementType())
                          .getApproxSize();

      OpBuilder builder(op);
      Value res;
      if (outSize == inSize) {
        res = stablehlo::BitcastConvertOp::create(builder, op.getLoc(), ty,
                                                  op.getIn());
      } else if (outSize < inSize) {
        SmallVector<int64_t> dims2 = llvm::to_vector(ty.getShape());
        auto oidx = dims2.size();
        dims2.push_back(inSize / outSize);
        if (oidx != 0 && dims2[oidx - 1] != ShapedType::kDynamic) {
          dims2[oidx - 1] /= inSize / outSize;
        }
        res = stablehlo::BitcastConvertOp::create(
            builder, op.getLoc(),
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
                builder, op.getLoc(), op.getIn(), i);
            Value vval = val;
            if (i == ty.getShape().size() - 1) {
              auto cst = arith::ConstantOp::create(
                  builder, op.getLoc(), val.getType(),
                  cast<ElementsAttr>(
                      makeAttr(val.getType(), inSize / outSize)));
              vval = stablehlo::DivOp::create(builder, op.getLoc(), vval, cst);
            }
            vval = stablehlo::ReshapeOp::create(
                builder, op.getLoc(),
                RankedTensorType::get({1}, val.getType().getElementType()),
                vval);
            vals.push_back(vval);
          }

          auto idxs =
              stablehlo::ConcatenateOp::create(builder, op.getLoc(), vals, 0);
          res = stablehlo::DynamicReshapeOp::create(builder, op.getLoc(), ty,
                                                    res, idxs);
        } else {
          res = stablehlo::ReshapeOp::create(builder, op.getLoc(), ty, res);
        }
      } else {
        SmallVector<int64_t> dims2 = llvm::to_vector(ty.getShape());
        auto oidx = dims2.size();
        dims2.push_back(outSize / inSize);
        if (oidx != 0 && dims2[oidx - 1] != ShapedType::kDynamic) {
          dims2[oidx - 1] /= outSize / inSize;
        }
        res = stablehlo::ReshapeOp::create(
            builder, op.getLoc(),
            RankedTensorType::get(
                dims2, cast<RankedTensorType>(op.getOperand().getType())
                           .getElementType()),
            op.getIn());
        res =
            stablehlo::BitcastConvertOp::create(builder, op.getLoc(), ty, res);
      }
      op.replaceAllUsesWith(res);
      op.erase();
    });
    op->walk([=](arith::TruncFOp truncOp) {
      auto ty = dyn_cast<RankedTensorType>(truncOp.getResult().getType());
      if (!use_stablehlo || !ty)
        return;

      OpBuilder builder(truncOp);
      auto res = stablehlo::ConvertOp::create(builder, truncOp.getLoc(), ty,
                                              truncOp.getIn());
      truncOp.replaceAllUsesWith(res.getResult());
      truncOp.erase();
    });
    op->walk([=](arith::ExtFOp truncOp) {
      auto ty = dyn_cast<RankedTensorType>(truncOp.getResult().getType());
      if (!use_stablehlo || !ty)
        return;

      OpBuilder builder(truncOp);
      auto res = stablehlo::ConvertOp::create(builder, truncOp.getLoc(), ty,
                                              truncOp.getIn());
      truncOp.replaceAllUsesWith(res.getResult());
      truncOp.erase();
    });
    op->walk([=](math::FmaOp fma) {
      auto ty = dyn_cast<RankedTensorType>(fma.getResult().getType());
      if (!use_stablehlo || !ty)
        return;

      OpBuilder builder(fma);
      auto res = stablehlo::MulOp::create(builder, fma.getLoc(),
                                          fma.getOperand(0), fma.getOperand(1));
      auto res2 = stablehlo::AddOp::create(builder, fma.getLoc(), res,
                                           fma.getOperand(2));
      fma.replaceAllUsesWith(res2.getResult());
      fma.erase();
    });

    op->walk([=](math::CopySignOp copySignOp) {
      auto ty = dyn_cast<RankedTensorType>(copySignOp.getResult().getType());
      if (!use_stablehlo || !ty)
        return;

      // The copysign returns a value with the magnitude of the first operand
      // and the sign of the second operand.
      OpBuilder builder(copySignOp);
      auto loc = copySignOp.getLoc();
      Value val = copySignOp.getLhs();
      Value sign = copySignOp.getRhs();
      Attribute constAttr = FloatAttr::get(ty.getElementType(), 0);
      Value zero = stablehlo::ConstantOp::create(
          builder, loc, ty, SplatElementsAttr::get(ty, constAttr));
      Value signPositive = stablehlo::CompareOp::create(
          builder, loc, sign, zero, stablehlo::ComparisonDirection::GE);
      Value valPositive = stablehlo::CompareOp::create(
          builder, loc, val, zero, stablehlo::ComparisonDirection::GE);
      Value notSameSign =
          stablehlo::XorOp::create(builder, loc, signPositive, valPositive);
      Value negVal = stablehlo::NegOp::create(builder, loc, val);
      Value res =
          stablehlo::SelectOp::create(builder, loc, notSameSign, negVal, val);

      copySignOp.replaceAllUsesWith(res);
      copySignOp.erase();
    });

    op->walk([=](math::AtanOp atanOp) {
      // atan %a -> atan2(%a, 1.0)
      auto ty = dyn_cast<RankedTensorType>(atanOp.getResult().getType());
      if (!use_stablehlo || !ty)
        return;

      OpBuilder builder(atanOp);

      Attribute oneAttr0;
      if (isa<IntegerType>(ty.getElementType()))
        oneAttr0 = builder.getIntegerAttr(ty.getElementType(), 1);
      else if (isa<FloatType>(ty.getElementType()))
        oneAttr0 = builder.getFloatAttr(ty.getElementType(), 1);
      else if (auto CT = dyn_cast<ComplexType>(ty.getElementType()))
        oneAttr0 = complex::NumberAttr::get(CT, 1, 0);
      else
        return;

      DenseElementsAttr oneAttr;
      if (auto complexAttr = dyn_cast<complex::NumberAttr>(oneAttr0))
        oneAttr = DenseElementsAttr::get(ty, oneAttr0);
      else
        oneAttr = DenseElementsAttr::get(ty, oneAttr0);

      Value one =
          stablehlo::ConstantOp::create(builder, atanOp.getLoc(), oneAttr);
      Value res = stablehlo::Atan2Op::create(builder, atanOp.getLoc(),
                                             atanOp.getOperand(), one);
      atanOp.replaceAllUsesWith(res);
      atanOp.erase();
    });

    op->walk([=](arith::MaxNumFOp maxOp) {
      // maxnumf %a,%b -> select(isnan(%a), %b, max(%a, %b))
      if (!use_stablehlo || !isa<RankedTensorType>(maxOp.getResult().getType()))
        return;

      OpBuilder builder(maxOp);
      Value isLhsNaN =
          math::IsNaNOp::create(builder, maxOp.getLoc(), maxOp.getLhs());
      Value max = stablehlo::MaxOp::create(builder, maxOp.getLoc(),
                                           maxOp.getLhs(), maxOp.getRhs());
      Value res = stablehlo::SelectOp::create(builder, maxOp.getLoc(), isLhsNaN,
                                              maxOp.getRhs(), max);
      maxOp.replaceAllUsesWith(res);
      maxOp.erase();
    });
    op->walk([=](arith::MinNumFOp minOp) {
      // maxnumf %a,%b -> select(isnan(%a), %b, min(%a, %b))
      if (!use_stablehlo || !isa<RankedTensorType>(minOp.getResult().getType()))
        return;

      OpBuilder builder(minOp);
      Value isLhsNaN =
          math::IsNaNOp::create(builder, minOp.getLoc(), minOp.getLhs());
      Value min = stablehlo::MinOp::create(builder, minOp.getLoc(),
                                           minOp.getLhs(), minOp.getRhs());
      Value res = stablehlo::SelectOp::create(builder, minOp.getLoc(), isLhsNaN,
                                              minOp.getRhs(), min);
      minOp.replaceAllUsesWith(res);
      minOp.erase();
    });
    op->walk([=](math::IsNaNOp nanOp) {
      if (!use_stablehlo || !isa<RankedTensorType>(nanOp.getResult().getType()))
        return;

      OpBuilder builder(nanOp);

      Value isFinite = stablehlo::IsFiniteOp::create(builder, nanOp.getLoc(),
                                                     nanOp.getOperand());
      Value isNotFinite =
          stablehlo::NotOp::create(builder, nanOp.getLoc(), isFinite);

      Value isNotInf = stablehlo::NotOp::create(
          builder, nanOp.getLoc(),
          chlo::IsInfOp::create(builder, nanOp.getLoc(), nanOp.getOperand()));

      Value isNaN = stablehlo::AndOp::create(builder, nanOp.getLoc(),
                                             isNotFinite, isNotInf);

      nanOp.replaceAllUsesWith(isNaN);
      nanOp.erase();
    });
    op->walk([=](complex::ConjOp addOp) {
      if (!isa<RankedTensorType>(addOp->getResultTypes()[0]))
        return;
      OpBuilder builder(addOp);
      Value newAddOp;
      newAddOp =
          chlo::ConjOp::create(builder, addOp.getLoc(), addOp->getOperand(0));
      addOp.replaceAllUsesWith(newAddOp);
      addOp.erase();
    });
    op->walk([=](arith::ConstantOp constOp) {
      if (!use_stablehlo || !isa<RankedTensorType>(constOp.getType()))
        return;

      auto valueAttr = constOp.getValueAttr();
      if (!isa<ElementsAttr>(valueAttr))
        return;

      OpBuilder builder(constOp);
      Value newConstOp =
          stablehlo::ConstantOp::create(builder, constOp.getLoc(), valueAttr);
      constOp.replaceAllUsesWith(newConstOp);
      constOp.erase();
    });
    op->walk([=](arith::FPToSIOp addOp) {
      if (!use_stablehlo || !isa<RankedTensorType>(addOp->getResultTypes()[0]))
        return;
      OpBuilder builder(addOp);
      Value newAddOp;
      newAddOp = stablehlo::ConvertOp::create(
          builder, addOp.getLoc(), addOp->getOperand(0),
          cast<RankedTensorType>(addOp->getResult(0).getType())
              .getElementType());
      addOp.replaceAllUsesWith(newAddOp);
      addOp.erase();
    });
    op->walk([=](arith::SIToFPOp addOp) {
      if (!use_stablehlo || !isa<RankedTensorType>(addOp->getResultTypes()[0]))
        return;
      OpBuilder builder(addOp);
      Value newAddOp;
      newAddOp = stablehlo::ConvertOp::create(
          builder, addOp.getLoc(), addOp->getOperand(0),
          cast<RankedTensorType>(addOp->getResult(0).getType())
              .getElementType());
      if (cast<RankedTensorType>(addOp.getOperand().getType())
              .getElementType()
              .isInteger(1)) {
        newAddOp = stablehlo::NegOp::create(builder, addOp.getLoc(), newAddOp);
      }
      addOp.replaceAllUsesWith(newAddOp);
      addOp.erase();
    });
    op->walk([=](arith::UIToFPOp addOp) {
      if (!use_stablehlo || !isa<RankedTensorType>(addOp->getResultTypes()[0]))
        return;
      if (!cast<RankedTensorType>(addOp.getOperand().getType())
               .getElementType()
               .isInteger(1)) {
        return;
      }
      OpBuilder builder(addOp);
      Value newAddOp;
      newAddOp = stablehlo::ConvertOp::create(
          builder, addOp.getLoc(), addOp->getOperand(0),
          cast<RankedTensorType>(addOp->getResult(0).getType())
              .getElementType());
      addOp.replaceAllUsesWith(newAddOp);
      addOp.erase();
    });
    op->walk([=](arith::ExtUIOp addOp) {
      if (!use_stablehlo || !isa<RankedTensorType>(addOp->getResultTypes()[0]))
        return;
      if (!cast<RankedTensorType>(addOp.getOperand().getType())
               .getElementType()
               .isInteger(1))
        return;
      OpBuilder builder(addOp);
      Value newAddOp;
      newAddOp = stablehlo::ConvertOp::create(
          builder, addOp.getLoc(), addOp->getOperand(0),
          cast<RankedTensorType>(addOp->getResult(0).getType())
              .getElementType());
      addOp.replaceAllUsesWith(newAddOp);
      addOp.erase();
    });
    op->walk([=](enzyme::BroadcastOp broadcastOp) {
      OpBuilder builder(broadcastOp);
      Value newBroadcastOp;
      assert(use_stablehlo);
      SmallVector<int64_t> broadcastDims;
      auto shape =
          cast<TensorType>(broadcastOp.getInput().getType()).getShape();
      broadcastDims.reserve(shape.size());
      for (auto en : llvm::enumerate(shape)) {
        // original dimensions end up one further because the batch dimension
        // is prepended:
        broadcastDims.push_back(en.index() + 1);
      }
      newBroadcastOp = stablehlo::BroadcastInDimOp::create(
          builder, broadcastOp.getLoc(), broadcastOp.getType(),
          broadcastOp.getInput(), builder.getDenseI64ArrayAttr(broadcastDims));
      broadcastOp.replaceAllUsesWith(newBroadcastOp);
      broadcastOp.erase();
    });
    op->walk([=](arith::SelectOp selectOp) {
      if (!use_stablehlo ||
          llvm::any_of(selectOp->getOperandTypes(),
                       [](Type ty) { return !isa<RankedTensorType>(ty); }))
        return;

      OpBuilder builder(selectOp);
      auto newOp = stablehlo::SelectOp::create(
          builder, selectOp.getLoc(), selectOp.getType(),
          selectOp.getCondition(), selectOp.getTrueValue(),
          selectOp.getFalseValue());
      selectOp.replaceAllUsesWith(newOp.getResult());
      selectOp.erase();
    });
    op->walk([=](arith::CmpIOp cmpOp) {
      if (!isa<TensorType>(cmpOp.getType()))
        return;

      OpBuilder builder(cmpOp);

      Value newCmpOp;
      if (use_stablehlo) {
        stablehlo::ComparisonType compType = stablehlo::ComparisonType::SIGNED;
        auto predicate = cmpOp.getPredicate();
        if (predicate == arith::CmpIPredicate::ugt ||
            predicate == arith::CmpIPredicate::uge ||
            predicate == arith::CmpIPredicate::ult ||
            predicate == arith::CmpIPredicate::ule)
          compType = stablehlo::ComparisonType::UNSIGNED;

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
          return;
        }
        newCmpOp = stablehlo::CompareOp::create(
            builder, cmpOp->getLoc(), cmpOp->getOperand(0),
            cmpOp->getOperand(1), direction, compType);
      } else {
        return;
      }
      cmpOp.replaceAllUsesWith(newCmpOp);
      cmpOp.erase();
    });
    op->walk([=](arith::CmpFOp cmpOp) {
      if (!isa<TensorType>(cmpOp.getType()))
        return;

      // TODO: check fast math flags?
      OpBuilder builder(cmpOp);

      Value newCmpOp;
      if (use_stablehlo) {
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
          return;
        }
        newCmpOp = stablehlo::CompareOp::create(
            builder, cmpOp->getLoc(), cmpOp->getOperand(0),
            cmpOp->getOperand(1), direction, stablehlo::ComparisonType::FLOAT);
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
          return;
        }
        newCmpOp = mhlo::CompareOp::create(builder, cmpOp->getLoc(),
                                           cmpOp->getOperand(0),
                                           cmpOp->getOperand(1), direction);
      }
      cmpOp.replaceAllUsesWith(newCmpOp);
      cmpOp.erase();
    });
  }
};

} // end anonymous namespace
