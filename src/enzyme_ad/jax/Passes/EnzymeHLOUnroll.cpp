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
#include "mlir/Interfaces/SideEffectInterfaces.h"
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
                            SmallVectorImpl<Value> *replacements,
                            int64_t maxBoundedIterations, bool boundedGuardIf) {

  // Unrolling a checkpoint segment loop makes every iteration of the segment
  // live at once, which is what the checkpointing was paying recompute to
  // avoid. See markCheckpointSegmentLoop.
  if (isOrContainsCheckpointSegmentLoop(op))
    return failure();

  WhileLoopInfo info(op);
  if (info.computeInfo().failed() || !info.isValid())
    return failure();

  auto bodyTerm = cast<stablehlo::ReturnOp>(&op.getBody().front().back());
  auto loopBodyBlock = &op.getBody().front();

  // The trip count, or when the limit is not a constant but bounded (the
  // raiser's enzymexla.bounds on it) the most and the fewest trips it allows:
  // the loop then unrolls to the most, and every copy past the fewest runs
  // under the loop's own condition: in a stablehlo.if, or run regardless
  // with its results kept by a select only where the condition holds. Once
  // the condition fails the state stays, so the copies after that are the
  // loop's exit, and the bound says the condition fails by the last.
  int64_t iters, certain;
  bool bounded = !info.isConstant();
  if (!bounded) {
    iters = certain = info.getConstantNumIters();
  } else {
    if (!info.isConstantStart() || !info.isConstantStep() ||
        *info.getConstantStep() <= 0)
      return failure();
    auto limitTy = dyn_cast<RankedTensorType>(info.getLimit().getType());
    if (!limitTy || !isa<IntegerType>(limitTy.getElementType()))
      return failure();
    auto bounds =
        getBoundsFromIR(info.getLimit(), limitTy.getElementTypeBitWidth());
    if (!bounds)
      return failure();
    int64_t start = *info.getConstantStart(), step = *info.getConstantStep();
    int64_t lo = bounds->first.getSExtValue(),
            hi = bounds->second.getSExtValue();
    iters = hi > start ? (hi - start + step - 1) / step : 0;
    certain = lo > start ? (lo - start + step - 1) / step : 0;
    // selected, the copies run whether or not the loop would: nothing in
    // them may have an effect
    if (!boundedGuardIf) {
      for (Operation &it : loopBodyBlock->without_terminator())
        if (!isMemoryEffectFree(&it))
          return failure();
      for (Operation &it : op.getCond().front().without_terminator())
        if (!isMemoryEffectFree(&it))
          return failure();
    }
  }
  int64_t limit = bounded && maxBoundedIterations != -1 ? maxBoundedIterations
                                                        : maxNumIterations;
  if (limit != -1 && iters > limit)
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

  Block &condBlock = op.getCond().front();
  auto condTerm = cast<stablehlo::ReturnOp>(condBlock.getTerminator());
  SmallVector<Value> results(op.getOperands().begin(), op.getOperands().end());

  for (int64_t iter = 0; iter < iters; iter++) {
    Value guard;
    if (iter >= certain) {
      IRMapping condMap;
      condMap.map(condBlock.getArguments(), results);
      for (auto &it : condBlock.without_terminator())
        rewriter.clone(it, condMap);
      guard = condMap.lookupOrDefault(condTerm->getOperand(0));
    }
    IRMapping operandMap;
    operandMap.map(loopBodyBlock->getArguments(), results);

    if (guard && boundedGuardIf) {
      // the copy in the then branch, the state so far in the else branch
      SmallVector<Type> types(op.getResultTypes().begin(),
                              op.getResultTypes().end());
      auto ifOp = stablehlo::IfOp::create(rewriter, op.getLoc(), types, guard);
      {
        OpBuilder::InsertionGuard g(rewriter);
        Block *then = rewriter.createBlock(&ifOp.getTrueBranch());
        rewriter.setInsertionPointToEnd(then);
        for (auto &it : loopBodyBlock->without_terminator())
          rewriter.clone(it, operandMap);
        SmallVector<Value> yields;
        for (Value r : bodyTerm->getOperands())
          yields.push_back(operandMap.lookupOrDefault(r));
        stablehlo::ReturnOp::create(rewriter, op.getLoc(), yields);
        Block *otherwise = rewriter.createBlock(&ifOp.getFalseBranch());
        rewriter.setInsertionPointToEnd(otherwise);
        stablehlo::ReturnOp::create(rewriter, op.getLoc(), results);
      }
      results.assign(ifOp.getResults().begin(), ifOp.getResults().end());
      continue;
    }

    for (auto &it : loopBodyBlock->without_terminator()) {
      rewriter.clone(it, operandMap);
    }

    SmallVector<Value> next;
    for (auto [r, prev] : llvm::zip(bodyTerm->getOperands(), results)) {
      Value v = operandMap.lookupOrDefault(r);
      if (guard) {
        auto ty = cast<RankedTensorType>(v.getType());
        Value g = stablehlo::BroadcastInDimOp::create(
            rewriter, op.getLoc(), ty.clone(rewriter.getI1Type()), guard,
            rewriter.getDenseI64ArrayAttr({}));
        v = stablehlo::SelectOp::create(rewriter, op.getLoc(), g, v, prev);
      }
      next.push_back(v);
    }
    results = std::move(next);
  }

  if (replacements)
    *replacements = results;

  rewriter.replaceOp(op, results);
  return success();
}

LogicalResult
WhileUnroll::matchAndRewriteImpl(mlir::stablehlo::WhileOp op,
                                 PatternRewriter &rewriter) const {
  return unrollWhileOp(op, rewriter, maxNumIterations, maxOperationThreshold,
                       nullptr, maxBoundedIterations, boundedGuardIf);
}

struct EnzymeHLOUnrollPass
    : public enzyme::impl::EnzymeHLOUnrollPassBase<EnzymeHLOUnrollPass> {
  using EnzymeHLOUnrollPassBase::EnzymeHLOUnrollPassBase;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);
    patterns.add<WhileUnroll>(maxNumIterations, maxOperationThreshold, context,
                              1, maxBoundedIterations, boundedGuardIf);
    GreedyRewriteConfig config;
    config.enableFolding();
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};
