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

static void lowerIndexSwitchToIfChain(scf::IndexSwitchOp switchOp,
                                      IRRewriter &rewriter) {
  Location loc = switchOp.getLoc();
  int numCases = switchOp.getNumCases();
  TypeRange resultTypes = switchOp.getResultTypes();

  if (numCases == 0) {
    rewriter.setInsertionPoint(switchOp);
    Block &defaultBlock = switchOp.getDefaultBlock();
    Operation *yield = defaultBlock.getTerminator();
    SmallVector<Value> yieldOperands(yield->getOperands());
    rewriter.inlineBlockBefore(&defaultBlock, switchOp);
    rewriter.eraseOp(yield);
    rewriter.replaceOp(switchOp, yieldOperands);
    return;
  }

  rewriter.setInsertionPoint(switchOp);
  ArrayRef<int64_t> cases = switchOp.getCases();
  Value arg = switchOp.getArg();

  // Create the outermost if for case 0
  Value caseVal = arith::ConstantIndexOp::create(rewriter, loc, cases[0]);
  Value cmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                    arg, caseVal);
  scf::IfOp outerIf = scf::IfOp::create(rewriter, loc, resultTypes, cmp,
                                        /*withElseRegion=*/true);

  // Merge case 0 into the then block
  if (outerIf.thenBlock()->mightHaveTerminator())
    rewriter.eraseOp(outerIf.thenBlock()->getTerminator());
  if (outerIf.elseBlock()->mightHaveTerminator())
    rewriter.eraseOp(outerIf.elseBlock()->getTerminator());

  Block &caseBlock0 = switchOp.getCaseBlock(0);
  rewriter.mergeBlocks(&caseBlock0, outerIf.thenBlock());

  Block *prevElseBlock = outerIf.elseBlock();

  // Create nested ifs for cases 1..N-1
  for (int i = 1; i < numCases; i++) {
    rewriter.setInsertionPointToStart(prevElseBlock);

    Value caseVal = arith::ConstantIndexOp::create(rewriter, loc, cases[i]);
    Value cmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                      arg, caseVal);
    scf::IfOp nestedIf = scf::IfOp::create(rewriter, loc, resultTypes, cmp,
                                           /*withElseRegion=*/true);

    // Yield nested if results from the previous else block
    scf::YieldOp::create(rewriter, loc, nestedIf.getResults());

    // Merge case i into the then block
    if (nestedIf.thenBlock()->mightHaveTerminator())
      rewriter.eraseOp(nestedIf.thenBlock()->getTerminator());
    if (nestedIf.elseBlock()->mightHaveTerminator())
      rewriter.eraseOp(nestedIf.elseBlock()->getTerminator());

    Block &caseBlockI = switchOp.getCaseBlock(i);
    rewriter.mergeBlocks(&caseBlockI, nestedIf.thenBlock());

    prevElseBlock = nestedIf.elseBlock();
  }

  // Merge default block into the last else
  Block &defaultBlock = switchOp.getDefaultBlock();
  rewriter.mergeBlocks(&defaultBlock, prevElseBlock);

  rewriter.replaceOp(switchOp, outerIf.getResults());
}

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

    if (rewrite_index_switch) {
      SmallVector<scf::IndexSwitchOp> switchOps;
      op->walk(
          [&](scf::IndexSwitchOp switchOp) { switchOps.push_back(switchOp); });
      if (!switchOps.empty()) {
        IRRewriter rewriter(&getContext());
        for (auto switchOp : switchOps)
          lowerIndexSwitchToIfChain(switchOp, rewriter);
        changed = true;
      }
    }

    if (!changed)
      markAllAnalysesPreserved();
  }
};
} // namespace
