//===- EnzymeWrapPass.cpp - Replace calls with their derivatives ------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to create wrapper functions which differentiate
// ops.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "src/enzyme_ad/jax/Passes/PassDetails.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

namespace {
struct ArithRaisingPass : public ArithRaisingPassBase<ArithRaisingPass> {

  void runOnOperation() override {
    auto op = getOperation();

    op->walk([=](arith::AddFOp addOp) {
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
    op->walk([=](arith::AddIOp addOp) {
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
  }
};

} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createArithRaisingPass() {
  return std::make_unique<ArithRaisingPass>();
}
} // namespace enzyme
} // namespace mlir
