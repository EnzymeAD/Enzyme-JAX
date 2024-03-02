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
#include "stablehlo/transforms/Passes.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"


#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

namespace {


struct SliceSimplification final : OpRewritePattern<mlir::stablehlo::SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type) return failure();
	for (auto sz : type.getShape())
	  if (sz < 0) return failure();
	if (op.getOperand().getType() == type) {
	  rewriter.replaceOp(op, op.getOperand());
	  return success();
	}
	return failure();
  }
};


struct AddSimplify : public OpRewritePattern<mlir::stablehlo::AddOp> {
  using OpRewritePattern<mlir::stablehlo::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::AddOp op,
                                PatternRewriter &rewriter) const final {

    if (matchPattern(op.getLhs(), m_AnyZeroFloat())) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }

    if (matchPattern(op.getRhs(), m_AnyZeroFloat())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    return failure();
  }
};

struct SubSimplify : public OpRewritePattern<mlir::stablehlo::SubtractOp> {
  using OpRewritePattern<mlir::stablehlo::SubtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SubtractOp op,
                                PatternRewriter &rewriter) const final {

    if (matchPattern(op.getRhs(), m_AnyZeroFloat())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    if (matchPattern(op.getLhs(), m_AnyZeroFloat())) {
      rewriter.replaceOpWithNewOp<stablehlo::NegOp>(op, op.getRhs());
      return success();
    }

    return failure();
  }
};

struct EnzymeHLOOptPass : public EnzymeHLOOptPassBase<EnzymeHLOOptPass> {

  void runOnOperation() override {
    auto context = getOperation()->getContext();
	RewritePatternSet patterns(context);
	patterns.add<SliceSimplification, AddSimplify, SubSimplify>(context);
    mlir::stablehlo::populateStablehloCanonicalizationPatterns(context, &patterns);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createEnzymeHLOOptPass() {
  return std::make_unique<EnzymeHLOOptPass>();
}
} // namespace enzyme
} // namespace mlir
