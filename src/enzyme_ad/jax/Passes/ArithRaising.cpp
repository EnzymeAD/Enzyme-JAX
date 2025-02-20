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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
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
  op->walk([=](BinaryOp addOp) {                                               \
    if (!addOp.getType().isa<RankedTensorType>())                              \
      return;                                                                  \
    OpBuilder builder(addOp);                                                  \
    Value newAddOp;                                                            \
    if (use_stablehlo)                                                         \
      newAddOp = builder.create<StableHLOOp>(                                  \
          addOp.getLoc(), addOp->getOperand(0), addOp->getOperand(1));         \
    else                                                                       \
      newAddOp = builder.create<MHLOOp>(addOp.getLoc(), addOp->getOperand(0),  \
                                        addOp->getOperand(1));                 \
    addOp.replaceAllUsesWith(newAddOp);                                        \
    addOp.erase();                                                             \
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

#undef RAISE_BINARY

#define RAISE_UNARY(InputOp, StableHLOOp, MHLOOp)                              \
  op->walk([=](InputOp inpOp) {                                                \
    if (!inpOp.getType().isa<RankedTensorType>())                              \
      return;                                                                  \
    OpBuilder builder(inpOp);                                                  \
    Value newAddOp;                                                            \
    if (use_stablehlo)                                                         \
      newAddOp =                                                               \
          builder.create<StableHLOOp>(inpOp.getLoc(), inpOp->getOperand(0));   \
    else                                                                       \
      newAddOp = builder.create<MHLOOp>(inpOp.getLoc(), inpOp->getOperand(0)); \
    inpOp.replaceAllUsesWith(newAddOp);                                        \
    inpOp.erase();                                                             \
  });

    RAISE_UNARY(math::SinOp, stablehlo::SineOp, mhlo::SineOp);
    RAISE_UNARY(math::CosOp, stablehlo::CosineOp, mhlo::CosineOp);
    RAISE_UNARY(math::LogOp, stablehlo::LogOp, mhlo::LogOp);
    RAISE_UNARY(math::ExpOp, stablehlo::ExpOp, mhlo::ExpOp);
    RAISE_UNARY(math::SqrtOp, stablehlo::SqrtOp, mhlo::SqrtOp);
    RAISE_UNARY(math::RsqrtOp, stablehlo::RsqrtOp, mhlo::RsqrtOp);
    RAISE_UNARY(math::AbsFOp, stablehlo::AbsOp, mhlo::AbsOp);
    RAISE_UNARY(arith::NegFOp, stablehlo::NegOp, mhlo::NegOp);

#undef RAISE_UNARY

    op->walk([=](complex::ConjOp addOp) {
      if (!addOp->getResultTypes()[0].isa<RankedTensorType>())
        return;
      OpBuilder builder(addOp);
      Value newAddOp;
      newAddOp =
          builder.create<chlo::ConjOp>(addOp.getLoc(), addOp->getOperand(0));
      addOp.replaceAllUsesWith(newAddOp);
      addOp.erase();
    });
    op->walk([=](arith::ConstantOp constOp) {
      if (!constOp.getType().isa<RankedTensorType>())
        return;
      auto CT = constOp.getType();
      if (isa<TensorType>(CT)) {
        OpBuilder builder(constOp);
        Value newConstOp = builder.create<stablehlo::ConstantOp>(
            constOp.getLoc(), constOp.getValueAttr());
        constOp.replaceAllUsesWith(newConstOp);
        constOp.erase();
      }
    });
    op->walk([=](enzyme::BroadcastOp broadcastOp) {
      OpBuilder builder(broadcastOp);
      Value newBroadcastOp;
      assert(use_stablehlo);
      SmallVector<int64_t> broadcastDims;
      auto shape =
          broadcastOp.getInput().getType().cast<TensorType>().getShape();
      broadcastDims.reserve(shape.size());
      for (auto en : llvm::enumerate(shape)) {
        // original dimensions end up one further because the batch dimension
        // is prepended:
        broadcastDims.push_back(en.index() + 1);
      }
      newBroadcastOp = builder.create<stablehlo::BroadcastInDimOp>(
          broadcastOp.getLoc(), broadcastOp.getType(), broadcastOp.getInput(),
          builder.getDenseI64ArrayAttr(broadcastDims));
      broadcastOp.replaceAllUsesWith(newBroadcastOp);
      broadcastOp.erase();
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
        case arith::CmpFPredicate::OEQ:
          direction = stablehlo::ComparisonDirection::EQ;
          break;
        case arith::CmpFPredicate::OGT:
          direction = stablehlo::ComparisonDirection::GT;
          break;
        case arith::CmpFPredicate::OGE:
          direction = stablehlo::ComparisonDirection::GE;
          break;
        case arith::CmpFPredicate::OLT:
          direction = stablehlo::ComparisonDirection::LT;
          break;
        case arith::CmpFPredicate::OLE:
          direction = stablehlo::ComparisonDirection::LE;
          break;
        case arith::CmpFPredicate::ONE:
          direction = stablehlo::ComparisonDirection::NE;
          break;
        default:
          return;
        }
        newCmpOp = builder.create<stablehlo::CompareOp>(
            cmpOp->getLoc(), cmpOp->getOperand(0), cmpOp->getOperand(1),
            direction, stablehlo::ComparisonType::FLOAT);
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
        newCmpOp = builder.create<mhlo::CompareOp>(
            cmpOp->getLoc(), cmpOp->getOperand(0), cmpOp->getOperand(1),
            direction);
      }
      cmpOp.replaceAllUsesWith(newCmpOp);
      cmpOp.erase();
    });
  }
}; // namespace

} // end anonymous namespace
