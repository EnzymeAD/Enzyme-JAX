#include "mhlo/IR/hlo_ops.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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

#define DEBUG_TYPE "shard-p2p"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_SHARDPOINTTOPOINT
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
struct SliceConcatSimplify : public OpRewritePattern<stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumOperands() <= 1) {
      return failure();
    }

    SmallVector<stablehlo::SliceOp> sliceOps;
    for (Value operand : op.getOperands()) {
      auto sliceOp = operand.getDefiningOp<stablehlo::SliceOp>();
      if (!sliceOp)
        return failure();

      if (!sliceOp->hasOneUse())
        return failure();

      sliceOps.push_back(sliceOp);
    }

    return failure();
  }
};
struct ShardPointToPointPass
    : public enzyme::impl::ShardPointToPointBase<ShardPointToPointPass> {
  using Base::Base;
  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<SliceConcatSimplify>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
