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

    op->walk([=](arith::AddFOp addOp) {
      if (!addOp.getType().isa<RankedTensorType>())
        return;
      OpBuilder builder(addOp);
      Value newAddOp;
      if (use_stablehlo)
        newAddOp = builder.create<stablehlo::AddOp>(
            addOp.getLoc(), addOp->getOperand(0), addOp->getOperand(1));
      else
        newAddOp = builder.create<mhlo::AddOp>(
            addOp.getLoc(), addOp->getOperand(0), addOp->getOperand(1));
      addOp.replaceAllUsesWith(newAddOp);
      addOp.erase();
    });
    op->walk([=](complex::AddOp addOp) {
      if (!addOp->getResultTypes()[0].isa<RankedTensorType>())
        return;
      OpBuilder builder(addOp);
      Value newAddOp;
      if (use_stablehlo)
        newAddOp = builder.create<stablehlo::AddOp>(
            addOp.getLoc(), addOp->getOperand(0), addOp->getOperand(1));
      else
        newAddOp = builder.create<mhlo::AddOp>(
            addOp.getLoc(), addOp->getOperand(0), addOp->getOperand(1));
      addOp.replaceAllUsesWith(newAddOp);
      addOp.erase();
    });
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
    op->walk([=](arith::AddIOp addOp) {
      if (!addOp.getType().isa<RankedTensorType>())
        return;
      OpBuilder builder(addOp);
      Value newAddOp;
      if (use_stablehlo)
        newAddOp = builder.create<stablehlo::AddOp>(
            addOp.getLoc(), addOp->getOperand(0), addOp->getOperand(1));
      else
        newAddOp = builder.create<mhlo::AddOp>(
            addOp.getLoc(), addOp->getOperand(0), addOp->getOperand(1));
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
  }
};

} // end anonymous namespace
