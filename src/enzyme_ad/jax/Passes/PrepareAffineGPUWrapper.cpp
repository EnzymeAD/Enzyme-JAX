//===- PrepareAffineGPUWrapper.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reshape the affine.parallel nested in a gpu_wrapper so that its dimensions
// are exactly the wrapper's six grid and block bounds again.
//
// kernel-call-to-gpu-wrapper emits a 6-d affine.parallel over the launch
// bounds, but the affine optimizations that run afterwards drop unit
// dimensions and split induction variables (e.g. a 1-d launch of 300x256
// becomes a 50x16x16x3x2 loop). convert-parallel-to-gpu1 can then no longer
// recover which loop dimensions belong to the grid and which to the block.
// This pass groups the loop dimensions by the launch bound whose extent they
// multiply to, and re-linearizes each group into one induction variable whose
// upper bound is the wrapper's bound value itself.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#define DEBUG_TYPE "prepare-affine-gpu-wrapper"

namespace mlir::enzyme {
#define GEN_PASS_DEF_PREPAREAFFINEGPUWRAPPERPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace mlir::enzyme

using namespace mlir;

namespace {

/// Whether the dimensions of `par` already are the launch bounds `bounds`, in
/// order, with the bound values themselves as upper bounds.
bool isAlignedToLaunch(affine::AffineParallelOp par, ValueRange bounds) {
  if (par.getNumDims() != bounds.size() || par.hasMinMaxBounds())
    return false;
  if (!llvm::all_of(par.getSteps(), [](int64_t s) { return s == 1; }))
    return false;
  for (AffineExpr lb : par.getLowerBoundsMap().getResults()) {
    auto cst = dyn_cast<AffineConstantExpr>(lb);
    if (!cst || cst.getValue() != 0)
      return false;
  }
  AffineMap ubMap = par.getUpperBoundsMap();
  auto operands = par.getUpperBoundsOperands();
  for (auto [expr, bound] : llvm::zip_equal(ubMap.getResults(), bounds)) {
    Value operand;
    if (auto sym = dyn_cast<AffineSymbolExpr>(expr))
      operand = operands[ubMap.getNumDims() + sym.getPosition()];
    else if (auto dim = dyn_cast<AffineDimExpr>(expr))
      operand = operands[dim.getPosition()];
    if (operand != bound)
      return false;
  }
  return true;
}

/// Assign every loop dimension of extent > 1 to a launch bound so that the
/// extents assigned to each bound multiply to it. `remaining[g]` is the part
/// of bound `g` not yet covered.
bool assignDims(ArrayRef<int64_t> extents, unsigned dim,
                MutableArrayRef<int64_t> remaining,
                MutableArrayRef<int> groupOf) {
  if (dim == extents.size())
    return llvm::all_of(remaining, [](int64_t r) { return r == 1; });
  if (extents[dim] == 1) {
    groupOf[dim] = -1;
    return assignDims(extents, dim + 1, remaining, groupOf);
  }
  for (unsigned g = 0; g < remaining.size(); g++) {
    if (remaining[g] % extents[dim] != 0)
      continue;
    remaining[g] /= extents[dim];
    groupOf[dim] = g;
    if (assignDims(extents, dim + 1, remaining, groupOf))
      return true;
    remaining[g] *= extents[dim];
  }
  return false;
}

struct AlignAffineParallelToWrapperBounds
    : public OpRewritePattern<enzymexla::GPUWrapperOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::GPUWrapperOp wrapper,
                                PatternRewriter &rewriter) const override {
    ValueRange bounds = wrapper.getBlockDims();
    if (bounds.size() != 6)
      return rewriter.notifyMatchFailure(wrapper, "expected 6 launch bounds");

    SmallVector<int64_t, 6> launch;
    for (Value bound : bounds) {
      std::optional<int64_t> cst = getConstantIntValue(bound);
      if (!cst || *cst <= 0)
        return rewriter.notifyMatchFailure(wrapper, "non-constant launch");
      launch.push_back(*cst);
    }

    affine::AffineParallelOp par = nullptr;
    for (Operation &op : wrapper.getBody()->without_terminator()) {
      if (auto p = dyn_cast<affine::AffineParallelOp>(op)) {
        if (par)
          return rewriter.notifyMatchFailure(wrapper, "multiple parallel ops");
        par = p;
      }
    }
    if (!par)
      return rewriter.notifyMatchFailure(wrapper, "no affine.parallel");
    if (isAlignedToLaunch(par, bounds))
      return rewriter.notifyMatchFailure(wrapper, "already aligned");

    // Regrouping the iteration space is only a bijection on independent
    // iterations; a barrier synchronizes exactly the original block's threads.
    if (par->walk([](enzymexla::BarrierOp) { return WalkResult::interrupt(); })
            .wasInterrupted())
      return rewriter.notifyMatchFailure(wrapper, "contains a barrier");

    if (par.hasMinMaxBounds() ||
        !llvm::all_of(par.getSteps(), [](int64_t s) { return s == 1; }))
      return rewriter.notifyMatchFailure(wrapper, "non-unit steps or min/max");
    for (AffineExpr lb : par.getLowerBoundsMap().getResults()) {
      auto cst = dyn_cast<AffineConstantExpr>(lb);
      if (!cst || cst.getValue() != 0)
        return rewriter.notifyMatchFailure(wrapper, "non-zero lower bound");
    }
    std::optional<SmallVector<int64_t, 8>> extents = par.getConstantRanges();
    if (!extents)
      return rewriter.notifyMatchFailure(wrapper, "non-constant loop bounds");

    SmallVector<int64_t, 6> remaining(launch);
    SmallVector<int> groupOf(extents->size(), -1);
    if (!assignDims(*extents, 0, remaining, groupOf))
      return rewriter.notifyMatchFailure(
          wrapper, "loop extents do not factor into the launch bounds");

    MLIRContext *ctx = rewriter.getContext();
    Location loc = par.getLoc();

    SmallVector<arith::AtomicRMWKind> reductions;
    for (Attribute attr : par.getReductions())
      reductions.push_back(cast<arith::AtomicRMWKindAttr>(attr).getValue());

    SmallVector<AffineMap> lbMaps(6, AffineMap::getConstantMap(0, ctx));
    SmallVector<AffineMap> ubMaps;
    for (unsigned i = 0; i < 6; i++)
      ubMaps.push_back(AffineMap::get(0, 6, getAffineSymbolExpr(i, ctx)));
    SmallVector<int64_t> steps(6, 1);

    rewriter.setInsertionPoint(par);
    auto newPar = affine::AffineParallelOp::create(
        rewriter, loc, par.getResultTypes(), reductions, lbMaps, ValueRange(),
        ubMaps, bounds, steps);
    Block *newBody = newPar.getBody();
    if (!newBody->empty())
      rewriter.eraseOp(newBody->getTerminator());

    // Delinearize each new induction variable into the loop dimensions of its
    // group, row-major in the original dimension order (the last dimension of
    // a group varies fastest).
    rewriter.setInsertionPointToStart(newBody);
    SmallVector<Value> replacements(extents->size());
    for (unsigned g = 0; g < 6; g++) {
      SmallVector<unsigned> dims;
      for (unsigned d = 0; d < extents->size(); d++)
        if (groupOf[d] == (int)g)
          dims.push_back(d);
      Value iv = newBody->getArgument(g);
      if (dims.size() == 1) {
        replacements[dims[0]] = iv;
        continue;
      }
      int64_t stride = launch[g];
      for (auto [k, d] : llvm::enumerate(dims)) {
        int64_t extent = (*extents)[d];
        stride /= extent;
        AffineExpr expr = getAffineDimExpr(0, ctx);
        if (stride != 1)
          expr = expr.floorDiv(stride);
        if (k != 0)
          expr = expr % extent;
        replacements[d] = affine::AffineApplyOp::create(
            rewriter, loc, AffineMap::get(1, 0, expr), ValueRange(iv));
      }
    }
    for (unsigned d = 0; d < extents->size(); d++)
      if (groupOf[d] == -1)
        replacements[d] = arith::ConstantIndexOp::create(rewriter, loc, 0);

    rewriter.mergeBlocks(par.getBody(), newBody, replacements);
    rewriter.replaceOp(par, newPar.getResults());
    return success();
  }
};

struct PrepareAffineGPUWrapperPass
    : public mlir::enzyme::impl::PrepareAffineGPUWrapperPassBase<
          PrepareAffineGPUWrapperPass> {
  using PrepareAffineGPUWrapperPassBase::PrepareAffineGPUWrapperPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<AlignAffineParallelToWrapperBounds>(&getContext());
    // Not the greedy driver: folding affine.parallel would turn the symbolic
    // launch-bound upper bounds back into constants, and
    // convert-parallel-to-gpu1 matches the bounds by value.
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace
