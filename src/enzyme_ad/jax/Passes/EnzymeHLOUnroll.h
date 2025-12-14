#pragma once

#include "mlir/IR/PatternMatch.h"
#include "src/enzyme_ad/jax/CheckedRewrite.h"
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"
#endif
#include "stablehlo/dialect/StablehloOps.h"
#ifdef __clang__
#pragma clang diagnostic pop
#else
#pragma GCC diagnostic pop
#endif

using namespace mlir;
using namespace mlir::enzyme;

struct WhileUnroll
    : public CheckedOpRewritePattern<stablehlo::WhileOp, WhileUnroll> {
  using CheckedOpRewritePattern<stablehlo::WhileOp,
                                WhileUnroll>::CheckedOpRewritePattern;

  int64_t maxNumIterations = -1;
  int64_t maxOperationThreshold = -1;

  WhileUnroll(int64_t maxNumIterations, int64_t maxOperationThreshold,
              MLIRContext *ctx, PatternBenefit benefit = 1)
      : CheckedOpRewritePattern<stablehlo::WhileOp, WhileUnroll>(ctx, benefit),
        maxNumIterations(maxNumIterations),
        maxOperationThreshold(maxOperationThreshold) {}

  LogicalResult matchAndRewriteImpl(stablehlo::WhileOp op,
                                    PatternRewriter &rewriter) const;
};
