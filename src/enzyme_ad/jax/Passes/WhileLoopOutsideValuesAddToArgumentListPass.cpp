#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_WHILELOOPOUTSIDEVALUESADDTOARGUMENTLISTPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::stablehlo;

namespace {

static bool definedOutside(Value v, Operation *op) {
  return !op->isAncestor(v.getParentBlock()->getParentOp());
}

struct SHLOWhileOpUpdateArgumentListPattern final
    : public OpRewritePattern<stablehlo::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    // Collect values used inside cond/body that are defined outside the WhileOp
    SmallVector<Value, 4> extraValues;
    SmallPtrSet<Value, 8> seen;

    auto collectExternal = [&](Region &region) {
      region.walk([&](Operation *op) {
        for (OpOperand &operand : op->getOpOperands()) {
          Value v = operand.get();
          if (!v)
            continue;
          if (definedOutside(v, whileOp) && !seen.contains(v)) {
            seen.insert(v);
            extraValues.push_back(v);
          }
        }
      });
    };

    collectExternal(whileOp.getCond());
    collectExternal(whileOp.getBody());

    if (extraValues.empty())
      return failure();

    // Build new operand list = existing operands + external values
    SmallVector<Value, 8> newOperands(whileOp.getOperands().begin(),
                                      whileOp.getOperands().end());
    for (Value v : extraValues)
      newOperands.push_back(v);

    SmallVector<Type, 8> newResultTypes;
    newResultTypes.reserve(newOperands.size());
    for (Value v : newOperands)
      newResultTypes.push_back(v.getType());

    auto newWhile = stablehlo::WhileOp::create(rewriter, whileOp.getLoc(),
                                               newResultTypes, newOperands);

    rewriter.inlineRegionBefore(whileOp.getCond(), newWhile.getCond(),
                                newWhile.getCond().end());
    rewriter.inlineRegionBefore(whileOp.getBody(), newWhile.getBody(),
                                newWhile.getBody().end());

    // Append block arguments for the extra values
    Block &condBlock = newWhile.getCond().front();
    Block &bodyBlock = newWhile.getBody().front();

    unsigned origArgCount = whileOp.getNumOperands();
    SmallVector<BlockArgument, 8> addedCondArgs, addedBodyArgs;
    addedCondArgs.reserve(extraValues.size());
    addedBodyArgs.reserve(extraValues.size());
    for (Value v : extraValues) {
      addedCondArgs.push_back(condBlock.addArgument(v.getType(), v.getLoc()));
      addedBodyArgs.push_back(bodyBlock.addArgument(v.getType(), v.getLoc()));
    }

    // Remap uses of external values inside the regions to the new block args
    auto remapRegionUses = [&](Region &region, ArrayRef<Value> externals,
                               ArrayRef<BlockArgument> args) {
      region.walk([&](Operation *op) {
        for (OpOperand &operand : op->getOpOperands()) {
          Value v = operand.get();
          for (auto [ext, arg] : llvm::zip(externals, args)) {
            if (v == ext) {
              operand.set(arg);
              break;
            }
          }
        }
      });
    };

    remapRegionUses(newWhile.getCond(), extraValues, addedCondArgs);
    remapRegionUses(newWhile.getBody(), extraValues, addedBodyArgs);

    Operation *terminator = bodyBlock.getTerminator();
    if (!terminator) {
      return rewriter.notifyMatchFailure(whileOp, "missing body terminator");
    }

    auto retOp = dyn_cast<stablehlo::ReturnOp>(terminator);
    assert(retOp && "expected stablehlo::ReturnOp");

    SmallVector<Value, 8> newRetVals(retOp.getOperands().begin(),
                                     retOp.getOperands().end());
    for (BlockArgument arg : addedBodyArgs)
      newRetVals.push_back(arg);

    rewriter.setInsertionPoint(terminator);
    rewriter.replaceOpWithNewOp<stablehlo::ReturnOp>(terminator, newRetVals);

    for (unsigned i = 0; i < origArgCount; ++i)
      rewriter.replaceAllUsesWith(whileOp.getResult(i), newWhile.getResult(i));
    rewriter.eraseOp(whileOp);
    return success();
  }
};

struct WhileLoopOutsideValuesAddToArgumentListPass
    : public enzyme::impl::WhileLoopOutsideValuesAddToArgumentListPassBase<
          WhileLoopOutsideValuesAddToArgumentListPass> {
  using WhileLoopOutsideValuesAddToArgumentListPassBase::
      WhileLoopOutsideValuesAddToArgumentListPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SHLOWhileOpUpdateArgumentListPattern>(patterns.getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace
