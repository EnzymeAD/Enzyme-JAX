//===- LowerAlignedAffineAccesses.cpp - keep alignment through lowering ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// llvm-to-affine-access records the alignment of the access it raised as an
// attribute on the affine.load/affine.store it made. Upstream lower-affine
// rebuilds those as memref accesses without carrying attributes over, so the
// recorded alignment is lost and the final llvm access is emitted at the
// element type's ABI alignment -- which may promise more than the pointer
// holds. clang says `load i128, align 8` for the 16-byte member swap in MFEM's
// Mesh::Swap, at a struct offset that is 8 mod 16; upgraded to align 16, the
// backend's movaps traps.
//
// Run before lower-affine, this lowers exactly the accesses that carry an
// alignment attribute, keeping their attributes. The affine.apply it leaves
// behind, and every access without the attribute, lower as before.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERALIGNEDAFFINEACCESSESPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

// The map's results become one affine.apply per index, which lower-affine
// knows how to take apart; the access itself becomes the memref op the
// attributes can ride on.
static SmallVector<Value> expandMap(PatternRewriter &rewriter, Location loc,
                                    AffineMap map, ValueRange operands) {
  SmallVector<Value> indices;
  for (unsigned i = 0, e = map.getNumResults(); i < e; ++i)
    indices.push_back(affine::AffineApplyOp::create(
        rewriter, loc, map.getSubMap({i}), operands));
  return indices;
}

struct LowerAlignedAffineLoad : public OpRewritePattern<affine::AffineLoadOp> {
  using OpRewritePattern<affine::AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineLoadOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getAlignment())
      return failure();
    auto indices = expandMap(rewriter, op.getLoc(), op.getAffineMap(),
                             op.getMapOperands());
    auto newLoad =
        memref::LoadOp::create(rewriter, op.getLoc(), op.getMemRef(), indices);
    // Both ops own their alignment; carry it across, then the remaining
    // discardable attributes.
    newLoad.setAlignmentAttr(op.getAlignmentAttr());
    for (NamedAttribute attr : op->getDiscardableAttrs())
      newLoad->setAttr(attr.getName(), attr.getValue());
    rewriter.replaceOp(op, newLoad);
    return success();
  }
};

struct LowerAlignedAffineStore
    : public OpRewritePattern<affine::AffineStoreOp> {
  using OpRewritePattern<affine::AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineStoreOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getAlignment())
      return failure();
    auto indices = expandMap(rewriter, op.getLoc(), op.getAffineMap(),
                             op.getMapOperands());
    auto newStore = memref::StoreOp::create(
        rewriter, op.getLoc(), op.getValueToStore(), op.getMemRef(), indices);
    // See LowerAlignedAffineLoad.
    newStore.setAlignmentAttr(op.getAlignmentAttr());
    for (NamedAttribute attr : op->getDiscardableAttrs())
      newStore->setAttr(attr.getName(), attr.getValue());
    rewriter.replaceOp(op, newStore);
    return success();
  }
};

struct LowerAlignedAffineAccessesPass
    : public enzyme::impl::LowerAlignedAffineAccessesPassBase<
          LowerAlignedAffineAccessesPass> {
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<LowerAlignedAffineLoad, LowerAlignedAffineStore>(
        &getContext());
    GreedyRewriteConfig config;
    // Merging identical blocks threads the differing values through new
    // block arguments on every predecessor's terminator; an llvm.invoke only
    // takes LLVM types there. Nothing here needs the merge. TODO(#2815):
    // aggressive is fine once block merging asks the terminator
    // (llvm/llvm-project#215036).
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Normal);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace
