#include "mhlo/IR/hlo_ops.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>

#define DEBUG_TYPE "lower-factorization"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERFACTORIZATIONPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

struct LUFactorizationOpLowering
    : public OpRewritePattern<enzymexla::LUFactorizationOp> {

  std::string backend;
  LUFactorizationOpLowering(std::string backend, MLIRContext *context,
                            PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend) {}

  LogicalResult matchAndRewrite(enzymexla::LUFactorizationOp op,
                                PatternRewriter &rewriter) const override {
    if (backend == "cpu") {
      return rewriter.notifyMatchFailure(
          op, "CPU backend lowering not yet implemented.");
    } else if (backend == "cuda") {
      return rewriter.notifyMatchFailure(
          op, "CUDA backend lowering not yet implemented.");
    } else if (backend == "tpu") {
      return rewriter.notifyMatchFailure(
          op, "TPU backend lowering not yet implemented.");
    } else {
      return rewriter.notifyMatchFailure(op, "Unknown backend " + backend);
    }
  }
};

struct LowerFactorizationPass
    : public enzyme::impl::LowerFactorizationPassBase<LowerFactorizationPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<LUFactorizationOpLowering>(backend, context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
