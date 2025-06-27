#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "mhlo/IR/hlo_ops.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "lower-enzymexla-ml"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEXLAMLPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

struct LowerReluOpToStablehlo : public OpRewritePattern<enzymexla::ReluOp> {
  using OpRewritePattern<enzymexla::ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::ReluOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::MaxOp>(
        op, op.getOperand(),
        rewriter.create<stablehlo::ConstantOp>(
            op.getLoc(), cast<ElementsAttr>(makeAttr(op.getType(), 0))));
    return success();
  }
};

struct LowerEnzymeXLAMLPass
    : public enzyme::impl::LowerEnzymeXLAMLPassBase<LowerEnzymeXLAMLPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<LowerReluOpToStablehlo>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
