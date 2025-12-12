//===- EnzymeBatchToStableHLOPass.cpp  ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to print the MLIR module
//===----------------------------------------------------------------------===//

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_ENZYMEBATCHTOSTABLEHLOPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;
namespace {

struct ExtractOpLowering : public OpRewritePattern<enzyme::ExtractOp> {
  using OpRewritePattern<enzyme::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzyme::ExtractOp op,
                                PatternRewriter &rewriter) const override {

    auto inTy = op.getInput().getType();
    auto outTy = op.getOutput().getType();
    auto outRankTy = dyn_cast<RankedTensorType>(outTy);
    // stablehlo always has tensor type
    auto inRankTy = dyn_cast<RankedTensorType>(inTy);
    auto ndims = inRankTy.getRank(); // is atleast 1

    if (ndims < 1)
      return failure();

    // static slice
    SmallVector<int64_t> start_indices;
    start_indices.push_back(op.getIndex());
    for (int i = 1; i < ndims; ++i) {
      start_indices.push_back(0);
    }
    SmallVector<int64_t> limit_indices;
    limit_indices.push_back(op.getIndex() + 1);
    limit_indices.append(outRankTy.getShape().begin(),
                         outRankTy.getShape().end());
    SmallVector<int64_t> strides(ndims, 1);

    Value slicedOut =
        stablehlo::SliceOp::create(rewriter, op->getLoc(), op.getInput(),
                                   start_indices, limit_indices, strides);
    // reshape slicedOut to our final Op
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, outTy, slicedOut);
    return success();
  }
};

struct ConcatOpLowering : public OpRewritePattern<enzyme::ConcatOp> {
  using OpRewritePattern<enzyme::ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzyme::ConcatOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = op.getInputs();
    if (inputs.empty())
      return failure();

    // stablehlo always has tensor type
    // reshape each input to 1xinput_rank and concatenate on dim=0

    SmallVector<Value> expandedInputs;
    for (Value in : inputs) {
      auto inRankTy = cast<RankedTensorType>(in.getType());
      auto inShape = inRankTy.getShape();
      SmallVector<int64_t> newInShape = {1};
      newInShape.append(inShape.begin(), inShape.end());
      auto newInTy = inRankTy.clone(newInShape);
      Value newInput =
          stablehlo::ReshapeOp::create(rewriter, op->getLoc(), newInTy, in);
      expandedInputs.push_back(newInput);
    }

    // concatenate on dim=0
    rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(
        op, op->getResultTypes(), expandedInputs, /*dim=*/0);
    return success();
  }
};

struct EnzymeBatchToStableHLOPass
    : public enzyme::impl::EnzymeBatchToStableHLOPassBase<
          EnzymeBatchToStableHLOPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConcatOpLowering, ExtractOpLowering>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }

    // Verify that all illegal ops have been lowered
    auto walkResult = getOperation()->walk([&](Operation *op) {
      if (isa<enzyme::ConcatOp, enzyme::ExtractOp>(op)) {
        op->emitError("Failed to lower enzyme batch operation");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
      signalPassFailure();
    }
  };
};
} // namespace
