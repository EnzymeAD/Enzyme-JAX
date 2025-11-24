#pragma once

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Passes/EnzymeBatchPass.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "stablehlo/dialect/StablehloOps.h"

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
        blasIntWidth(blasIntWidth){};

  LogicalResult matchAndRewrite(enzymexla::SyrkOp op,
                                PatternRewriter &rewriter) const override {
    // if (backend == "cpu")
    //   return matchAndRewriteCPU(op, rewriter);

    return matchAndRewriteFallback(op, rewriter);
  }

  // TODO: cpu lowering
  // LogicalResult matchAndRewriteCPU(enzymexla::SyrkOp op,
  //                                  PatternRewriter &rewriter) const {
  //   return success();
  // }

  // TODO: gpu lowering after we register the cublas functions via XLA FFI

  LogicalResult matchAndRewriteFallback(enzymexla::SyrkOp op,
                                        PatternRewriter &rewriter) const {
    auto nBatchDims = cast<RankedTensorType>(op.getA().getType()).getRank() - 2;
    SmallVector<int64_t> batchDims(nBatchDims, 0);
    std::iota(batchDims.begin(), batchDims.end(), 0);

    // fallback to emitting a stablehlo.dot_general that computes:
    //   alpha * A * A^T + beta * C
    //   alpha * A^T * A + beta * C
    stablehlo::DotDimensionNumbersAttr dotDims;
    if (op.getTranspose() == enzymexla::LapackTranspose::none) {
      dotDims = stablehlo::DotDimensionNumbersAttr::get(
          op.getContext(), batchDims, batchDims, {nBatchDims + 1},
          {nBatchDims + 1});
    } else if (op.getTranspose() == enzymexla::LapackTranspose::transpose) {
      dotDims = stablehlo::DotDimensionNumbersAttr::get(
          op.getContext(), batchDims, batchDims, {nBatchDims}, {nBatchDims});
    } else {
      llvm_unreachable("unsupported transpose attribute");
    }

    auto AAT = rewriter.create<stablehlo::DotGeneralOp>(
        op.getLoc(), cast<RankedTensorType>(op.getC().getType()), op.getA(),
        op.getA(), dotDims, nullptr, nullptr);

    auto alpha = rewriter.create<stablehlo::BroadcastInDimOp>(
        op.getLoc(), cast<RankedTensorType>(AAT.getType()), op.getAlpha(),
        rewriter.getDenseI64ArrayAttr({}));

    auto lhs = rewriter.create<stablehlo::MulOp>(op.getLoc(), alpha, AAT);

    auto beta = rewriter.create<stablehlo::BroadcastInDimOp>(
        op.getLoc(), cast<RankedTensorType>(op.getC().getType()), op.getBeta(),
        rewriter.getDenseI64ArrayAttr({}));

    auto rhs = rewriter.create<stablehlo::MulOp>(op.getLoc(), beta, op.getC());

    rewriter.replaceOpWithNewOp<stablehlo::AddOp>(op, lhs, rhs);
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
