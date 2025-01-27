//===- ControlFlowToSCF.h - ControlFlow to SCF -------------*- C++ ------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Define conversions from the ControlFlow dialect to the SCF dialect.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"

#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/CFGToSCF.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_ENZYMELIFTCONTROLFLOWTOSCFPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

struct EnzymeLiftControlFlowToSCF
    : public enzyme::impl::EnzymeLiftControlFlowToSCFPassBase<
          EnzymeLiftControlFlowToSCF> {

  using EnzymeLiftControlFlowToSCFPassBase::EnzymeLiftControlFlowToSCFPassBase;

  void runOnOperation() override {
    ControlFlowToSCFTransformation transformation;

    bool changed = false;
    Operation *op = getOperation();
    WalkResult result = op->walk([&](Region *region) {
      if (region->empty())
        return WalkResult::advance();

      Operation *regionParent = region->getParentOp();
      auto &domInfo = regionParent != op
                          ? getChildAnalysis<DominanceInfo>(regionParent)
                          : getAnalysis<DominanceInfo>();

      auto visitor = [&](Operation *innerOp) -> WalkResult {
        for (Region &reg : innerOp->getRegions()) {
          FailureOr<bool> changedFunc =
              transformCFGToSCF(reg, transformation, domInfo);
          if (failed(changedFunc))
            return WalkResult::interrupt();

          changed |= *changedFunc;
        }
        return WalkResult::advance();
      };

      if (region->walk<WalkOrder::PostOrder>(visitor).wasInterrupted())
        return WalkResult::interrupt();

      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();

    if (!changed)
      markAllAnalysesPreserved();
  }
};
} // namespace
