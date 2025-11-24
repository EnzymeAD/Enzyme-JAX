#pragma once

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "stablehlo/dialect/StablehloOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "lower-enzymexla-blas"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEXLABLASPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzymexla;
using namespace mlir::stablehlo;

struct SyrkOpLowering : public OpRewritePattern<enzymexla::SyrkOp> {
  using OpRewritePattern<enzymexla::SyrkOp>::OpRewritePattern;

  SyrkOpLowering(std::string backend, int64_t blasIntWidth,
                 MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend),
        blasIntWidth(blasIntWidth) {};

  LogicalResult matchAndRewrite(enzymexla::SyrkOp op,
                                PatternRewriter &rewriter) const override {
    // if (backend == "cpu")
    //   return matchAndRewriteCPU(op, rewriter);

    return matchAndRewriteFallback(op, rewriter);
  }

  // LogicalResult matchAndRewriteCPU(enzymexla::SyrkOp op,
  //                                  PatternRewriter &rewriter) const {
  //   return success();
  // }

  LogicalResult matchAndRewriteFallback(enzymexla::SyrkOp op,
                                        PatternRewriter &rewriter) const {
    // fallback to emitting a stablehlo.dot_general that computes:
    //   alpha * A * A^T + beta * C
    return success();
  }

private:
  std::string backend;
  int64_t blasIntWidth;
};

struct LowerEnzymeXLABLASPass
    : public enzyme::impl::LowerEnzymeXLABLASPassBase<LowerEnzymeXLABLASPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<SyrkOpLowering>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
