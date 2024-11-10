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

#include "src/enzyme_ad/jax/Implementations/WhileLoopInfo.h"

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

namespace {

struct WhileUnroll : public OpRewritePattern<mlir::stablehlo::WhileOp> {
  using OpRewritePattern<mlir::stablehlo::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::WhileOp op,
                                PatternRewriter &rewriter) const final {

    WhileLoopInfo info(op);
    if (info.computeInfo().failed() || !info.isConstant())
      return failure();

    auto bodyTerm = cast<stablehlo::ReturnOp>(&op.getBody().front().back());
    auto loopBodyBlock = &op.getBody().front();

    auto constantIters = info.getConstantNumIters();
    if (!constantIters.has_value())
      return failure();

    auto iters = constantIters.value();

    SmallVector<Value> results(op.getOperands().begin(),
                               op.getOperands().end());

    for (size_t iter = 0; iter < iters; iter++) {
      IRMapping operandMap;
      operandMap.map(loopBodyBlock->getArguments(), results);

      for (auto &it : loopBodyBlock->without_terminator()) {
        rewriter.clone(it, operandMap);
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
