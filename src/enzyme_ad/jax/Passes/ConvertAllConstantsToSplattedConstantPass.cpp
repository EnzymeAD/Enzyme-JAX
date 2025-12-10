#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "src/enzyme_ad/jax/Utils.h"

using namespace mlir;
using namespace mlir::stablehlo;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_CONVERTALLCONSTANTSTOSPLATTEDCONSTANTPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

struct ConvertSHLOConstantsToSplat
    : public OpRewritePattern<stablehlo::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto attr = cast<DenseElementsAttr>(op.getValue());
    if (attr.isSplat()) // already splatted
      return failure();

    auto elementIt = attr.getValues<Attribute>().begin();
    Attribute firstVal = *elementIt;

    auto splatAttr = SplatElementsAttr::get(op.getType(), firstVal);
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, splatAttr);
    return success();
  }
};

namespace {

struct ConvertAllConstantsToSplattedConstantPass
    : public enzyme::impl::ConvertAllConstantsToSplattedConstantPassBase<
          ConvertAllConstantsToSplattedConstantPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertSHLOConstantsToSplat>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};

} // namespace
