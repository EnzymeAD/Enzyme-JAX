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

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

#include "src/enzyme_ad/jax/Passes/PassDetails.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Ops.h"
#include "stablehlo/transforms/Passes.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

namespace {

struct WhileUnroll : public OpRewritePattern<mlir::stablehlo::WhileOp> {
  using OpRewritePattern<mlir::stablehlo::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::WhileOp op,
                                PatternRewriter &rewriter) const final {

    auto &condBlk = op.getCond().front();
    if (condBlk.getOperations().size() != 2)
      return failure();
    auto condTerm = cast<stablehlo::ReturnOp>(&condBlk.back());
    auto condV = condTerm->getOperand(0);
    auto cond = condV.getDefiningOp<stablehlo::CompareOp>();
    if (!cond)
      return failure();

    auto induct = cond.getOperand(0).dyn_cast<BlockArgument>();
    if (!induct)
      return failure();
    if (induct.getOwner() != &condBlk)
      return failure();

    if (cond.getComparisonDirection() != stablehlo::ComparisonDirection::LT)
      return failure();

    DenseIntElementsAttr limit;
    if (!matchPattern(cond.getOperand(1), m_Constant(&limit)))
      return failure();

    DenseIntElementsAttr start;
    if (!matchPattern(op.getOperands()[induct.getArgNumber()],
                      m_Constant(&start)))
      return failure();

    auto bodyTerm = cast<stablehlo::ReturnOp>(&op.getBody().front().back());
    auto incV = bodyTerm->getOperand(induct.getArgNumber());
    auto inc = incV.getDefiningOp<stablehlo::AddOp>();
    if (!inc)
      return failure();

    auto loopBodyBlock = &op.getBody().front();

    auto incba = inc.getOperand(0).dyn_cast<BlockArgument>();

    if (!incba)
      return failure();

    if (incba.getOwner() != loopBodyBlock)
      return failure();

    if (incba.getArgNumber() != induct.getArgNumber())
      return failure();

    DenseIntElementsAttr step;
    if (!matchPattern(inc.getOperand(1), m_Constant(&step)))
      return failure();

    if (!(*step.begin()).isOne())
      return failure();

    auto iters = (*limit.begin()) - (*start.begin());

    IRMapping operandMap;

    SmallVector<Value> results(op.getOperands().begin(),
                               op.getOperands().end());

    for (size_t iter = 0; iter < iters.getSExtValue(); iter++) {
      operandMap.map(loopBodyBlock->getArguments(), results);

      Block::iterator srcBlockEnd = std::prev(loopBodyBlock->end(), 2);
      for (auto it = loopBodyBlock->begin(); it != std::next(srcBlockEnd);
           it++) {
        rewriter.clone(*it, operandMap);
      }

      results.clear();
      for (auto r : bodyTerm->getOperands()) {
        results.push_back(operandMap.lookupOrDefault(r));
      }
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct EnzymeHLOUnrollPass
    : public EnzymeHLOUnrollPassBase<EnzymeHLOUnrollPass> {

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);
    patterns.add<WhileUnroll>(context);
    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createEnzymeHLOUnrollPass() {
  return std::make_unique<EnzymeHLOUnrollPass>();
}
} // namespace enzyme
} // namespace mlir
