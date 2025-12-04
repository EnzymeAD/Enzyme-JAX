#pragma once

#include "mlir/IR/PatternMatch.h"
#include "src/enzyme_ad/jax/CheckedRewrite.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace mlir::enzyme;

struct WhileUnroll
    : public CheckedOpRewritePattern<stablehlo::WhileOp, WhileUnroll> {
  using CheckedOpRewritePattern<stablehlo::WhileOp,
                                WhileUnroll>::CheckedOpRewritePattern;

  int64_t maxNumIterations = -1;

  WhileUnroll(int64_t maxNumIterations, MLIRContext *ctx,
              PatternBenefit benefit = 1)
      : CheckedOpRewritePattern<stablehlo::WhileOp, WhileUnroll>(ctx, benefit),
        maxNumIterations(maxNumIterations) {}

  LogicalResult matchAndRewriteImpl(stablehlo::WhileOp op,
                                    PatternRewriter &rewriter) const;
};
