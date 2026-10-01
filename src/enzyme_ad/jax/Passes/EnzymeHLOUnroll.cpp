//===- EnzymeHLOUnroll.cpp - Unroll stablehlo.while loops -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to unroll stablehlo.while ops with known number
// of iterations.
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/EnzymeHLOUnroll.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

#include "src/enzyme_ad/jax/CheckedRewrite.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
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

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_ENZYMEHLOUNROLLPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

LogicalResult unrollWhileOp(mlir::stablehlo::WhileOp op, RewriterBase &rewriter,
                            int64_t maxNumIterations,
                            int64_t maxOperationThreshold,
                            SmallVectorImpl<Value> *replacements) {

  // Unrolling a checkpoint segment loop makes every iteration of the segment
  // live at once, which is what the checkpointing was paying recompute to
  // avoid. See markCheckpointSegmentLoop.
  if (isOrContainsCheckpointSegmentLoop(op))
    return failure();

  WhileLoopInfo info(op);
  if (info.computeInfo().failed() || !info.isConstant())
    return failure();

  auto bodyTerm = cast<stablehlo::ReturnOp>(&op.getBody().front().back());
  auto loopBodyBlock = &op.getBody().front();

  auto iters = info.getConstantNumIters();
  if (maxNumIterations != -1 && iters > maxNumIterations)
    return failure();
  // The unrolled copies hand each yielded value straight to the next copy's
  // uses and to the loop's results: a value shaped less precisely than the
  // result (a body may yield a dynamically shaped value for a static carried
  // type) would then reach uses that rely on the static shape.
  for (auto [y, r] : llvm::zip(bodyTerm.getOperands(), op->getResultTypes())) {
    auto yt = dyn_cast<RankedTensorType>(y.getType());
    auto rt = dyn_cast<RankedTensorType>(r);
    if (!yt || !rt || yt.getRank() != rt.getRank())
      continue;
    for (auto [yd, rd] : llvm::zip(yt.getShape(), rt.getShape()))
      if (ShapedType::isDynamic(yd) && !ShapedType::isDynamic(rd))
        return failure();
  }

  // A body over the threshold still unrolls when the copies together stay
  // within the same budget as the threshold allows at the full iteration
  // limit: a few iterations of a long body cost no more than many of a
  // short one.
  if (iters > 1 && maxOperationThreshold > -1 &&
      std::distance(loopBodyBlock->begin(), loopBodyBlock->end()) >
          maxOperationThreshold) {
    // every op the copies would carry, those of nested regions included
    int64_t ops = 0;
    op.getBody().walk([&](Operation *) { ++ops; });
    if (maxNumIterations == -1 ||
        iters * ops > maxNumIterations * maxOperationThreshold)
      return failure();
  }

  SmallVector<Value> results(op.getOperands().begin(), op.getOperands().end());

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

  if (replacements)
    *replacements = results;

  rewriter.replaceOp(op, results);
  return success();
}

LogicalResult
WhileUnroll::matchAndRewriteImpl(mlir::stablehlo::WhileOp op,
                                 PatternRewriter &rewriter) const {
  return unrollWhileOp(op, rewriter, maxNumIterations, maxOperationThreshold);
}

struct EnzymeHLOUnrollPass
    : public enzyme::impl::EnzymeHLOUnrollPassBase<EnzymeHLOUnrollPass> {
  using EnzymeHLOUnrollPassBase::EnzymeHLOUnrollPassBase;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);
    patterns.add<WhileUnroll>(maxNumIterations, maxOperationThreshold, context);
    GreedyRewriteConfig config;
    config.enableFolding();
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};
