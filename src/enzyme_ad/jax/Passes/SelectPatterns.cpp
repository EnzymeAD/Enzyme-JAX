#include "src/enzyme_ad/jax/Passes/SelectPatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

template <typename SelectOpTy>
struct SelectExtractElementToExtractElementSelect
    : public OpRewritePattern<LLVM::ExtractElementOp> {
  using OpRewritePattern<LLVM::ExtractElementOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::ExtractElementOp op,
                                PatternRewriter &rewriter) const override {
    auto selectOp = op.getVector().template getDefiningOp<SelectOpTy>();
    if (!selectOp)
      return failure();

    // Get select operands and extract position
    auto cond = selectOp.getCondition();
    auto a = selectOp.getTrueValue();
    auto b = selectOp.getFalseValue();
    auto idx = op.getPosition();

    // Create new extract operations
    auto aExtract =
        LLVM::ExtractElementOp::create(rewriter, op.getLoc(), a, idx);
    auto bExtract =
        LLVM::ExtractElementOp::create(rewriter, op.getLoc(), b, idx);

    // Create new select with same condition and operands
    auto newSelect = SelectOpTy::create(rewriter, selectOp.getLoc(),
                                        op.getType(), cond, aExtract, bExtract);

    // Replace old extract with new select
    rewriter.replaceOp(op, newSelect);

    return success();
  }
};

template <typename SelectOpTy>
struct SelectExtractValueToExtractValueSelect
    : public OpRewritePattern<LLVM::ExtractValueOp> {
  using OpRewritePattern<LLVM::ExtractValueOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::ExtractValueOp op,
                                PatternRewriter &rewriter) const override {
    auto selectOp = op.getContainer().template getDefiningOp<SelectOpTy>();
    if (!selectOp)
      return failure();

    // Get select operands and extract position
    auto cond = selectOp.getCondition();
    auto a = selectOp.getTrueValue();
    auto b = selectOp.getFalseValue();
    auto idx = op.getPosition();

    // Create new extract operations
    auto aExtract = LLVM::ExtractValueOp::create(rewriter, op.getLoc(), a, idx);
    auto bExtract = LLVM::ExtractValueOp::create(rewriter, op.getLoc(), b, idx);

    // Create new select with same condition and operands
    auto newSelect = SelectOpTy::create(rewriter, selectOp.getLoc(),
                                        op.getType(), cond, aExtract, bExtract);

    // Replace old extract with new select
    rewriter.replaceOp(op, newSelect);

    return success();
  }
};

} // namespace

void mlir::enzyme::populateSelectExtractPatterns(RewritePatternSet &patterns) {
  patterns.add<SelectExtractElementToExtractElementSelect<LLVM::SelectOp>,
               SelectExtractElementToExtractElementSelect<arith::SelectOp>,
               SelectExtractValueToExtractValueSelect<LLVM::SelectOp>,
               SelectExtractValueToExtractValueSelect<arith::SelectOp>>(
      patterns.getContext());
}
