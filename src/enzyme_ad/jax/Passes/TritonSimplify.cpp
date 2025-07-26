#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>

#define DEBUG_TYPE "enzymexla-triton-simplify"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_TRITONSIMPLIFY
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::triton;

struct ChainedSplatOpMerge : public OpRewritePattern<triton::SplatOp> {
  using OpRewritePattern<triton::SplatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::SplatOp op,
                                PatternRewriter &rewriter) const override {
    auto operandSplatOp = op.getOperand().getDefiningOp<triton::SplatOp>();
    if (!operandSplatOp)
      return failure();

    auto operandRank =
        cast<RankedTensorType>(operandSplatOp.getType()).getRank();
    if (operandRank != 0)
      return failure();

    auto originalScalar = operandSplatOp.getOperand();
    // Check that the original input is actually a scalar (not a tensor)
    if (isa<TensorType>(originalScalar.getType()))
      return failure();

    // Replace the chained splats with a single splat from scalar to final
    // tensor type
    rewriter.replaceOpWithNewOp<triton::SplatOp>(op, op.getType(),
                                                 originalScalar);

    return success();
  }
};

struct BroadcastReshapeToSplat : public OpRewritePattern<triton::BroadcastOp> {
  using OpRewritePattern<triton::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getOperand();
    auto inputType = cast<RankedTensorType>(input.getType());

    auto reshapeOp = input.getDefiningOp<triton::ReshapeOp>();
    if (!reshapeOp)
      return failure();

    auto inputNumElements = inputType.getNumElements();
    if (inputNumElements != 1)
      return failure();

    auto reshapedInputType =
        cast<RankedTensorType>(reshapeOp.getOperand().getType());
    auto reshapedInputRank = reshapedInputType.getRank();
    if (reshapedInputRank != 0 && reshapedInputRank != 1)
      return failure();

    rewriter.replaceOpWithNewOp<triton::SplatOp>(op, op.getType(),
                                                 reshapeOp.getOperand());
    return success();
  }
};

struct TritonSimplifyPass
    : public enzyme::impl::TritonSimplifyBase<TritonSimplifyPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<ChainedSplatOpMerge, BroadcastReshapeToSplat>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
