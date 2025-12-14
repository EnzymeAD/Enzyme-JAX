//===- PrintPass.cpp - Print the MLIR module                     ------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to print the MLIR module
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_SCFPARALLELSERIALIZATION
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {

struct ParallelSerialization : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp parallelOp,
                                PatternRewriter &rewriter) const {
    Location loc = parallelOp.getLoc();
    auto reductionOp =
        dyn_cast<scf::ReduceOp>(parallelOp.getBody()->getTerminator());
    if (!reductionOp) {
      return failure();
    }

    // For a parallel loop, we essentially need to create an n-dimensional loop
    // nest. We do this by translating to scf.for ops and have those lowered in
    // a further rewrite. If a parallel loop contains reductions (and thus
    // returns values), forward the initial values for the reductions down the
    // loop hierarchy and bubble up the results by modifying the "yield"
    // terminator.
    SmallVector<Value, 4> iterArgs =
        llvm::to_vector<4>(parallelOp.getInitVals());
    SmallVector<Value, 4> ivs;
    ivs.reserve(parallelOp.getNumLoops());
    bool first = true;
    SmallVector<Value, 4> loopResults(iterArgs);
    for (auto [iv, lower, upper, step] :
         llvm::zip(parallelOp.getInductionVars(), parallelOp.getLowerBound(),
                   parallelOp.getUpperBound(), parallelOp.getStep())) {
      scf::ForOp forOp =
          scf::ForOp::create(rewriter, loc, lower, upper, step, iterArgs);
      ivs.push_back(forOp.getInductionVar());
      auto iterRange = forOp.getRegionIterArgs();
      iterArgs.assign(iterRange.begin(), iterRange.end());

      if (first) {
        // Store the results of the outermost loop that will be used to replace
        // the results of the parallel loop when it is fully rewritten.
        loopResults.assign(forOp.result_begin(), forOp.result_end());
        first = false;
      } else if (!forOp.getResults().empty()) {
        // A loop is constructed with an empty "yield" terminator if there are
        // no results.
        rewriter.setInsertionPointToEnd(rewriter.getInsertionBlock());
        scf::YieldOp::create(rewriter, loc, forOp.getResults());
      }

      rewriter.setInsertionPointToStart(forOp.getBody());
    }

    // First, merge reduction blocks into the main region.
    SmallVector<Value> yieldOperands;
    yieldOperands.reserve(parallelOp.getNumResults());
    for (int64_t i = 0, e = parallelOp.getNumResults(); i < e; ++i) {
      Block &reductionBody = reductionOp.getReductions()[i].front();
      Value arg = iterArgs[yieldOperands.size()];
      yieldOperands.push_back(
          cast<scf::ReduceReturnOp>(reductionBody.getTerminator()).getResult());
      rewriter.eraseOp(reductionBody.getTerminator());
      rewriter.inlineBlockBefore(&reductionBody, reductionOp,
                                 {arg, reductionOp.getOperands()[i]});
    }
    rewriter.eraseOp(reductionOp);

    // Then merge the loop body without the terminator.
    Block *newBody = rewriter.getInsertionBlock();
    if (newBody->empty())
      rewriter.mergeBlocks(parallelOp.getBody(), newBody, ivs);
    else
      rewriter.inlineBlockBefore(parallelOp.getBody(), newBody->getTerminator(),
                                 ivs);

    // Finally, create the terminator if required (for loops with no results, it
    // has been already created in loop construction).
    if (!yieldOperands.empty()) {
      rewriter.setInsertionPointToEnd(rewriter.getInsertionBlock());
      scf::YieldOp::create(rewriter, loc, yieldOperands);
    }

    rewriter.replaceOp(parallelOp, loopResults);

    return success();
  }
};
struct SCFParallelSerializationPass
    : public enzyme::impl::SCFParallelSerializationBase<
          SCFParallelSerializationPass> {
  using SCFParallelSerializationBase::SCFParallelSerializationBase;

  void runOnOperation() override {
    auto m = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.insert<ParallelSerialization>(&getContext());
    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns), config))) {
      signalPassFailure();
      return;
    }
  }
};

} // end anonymous namespace
