#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_DROPIDENTITYPARTITIONINGANCHORSPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// AnchorPartitioning's input/output types are always structurally identical
// (AllTypesMatch), so every instance is a semantic no-op: it exists only to
// keep an axis binding discoverable for earlier passes (chiefly
// InlineDeviceLocalAxesPass). By the time this pass runs (after
// distributed-lower-kernels), that metadata has already been fully consumed,
// so every instance can simply be dropped to create simpler SSA chains.
struct DropAnchorPartitioning : public OpRewritePattern<AnchorPartitioningOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AnchorPartitioningOp anchorOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(anchorOp, anchorOp.getInput());
    return success();
  }
};

// A Cast whose input/output types already happen to be equal is doing no
// real local/global shape change. Earlier in the pipeline such a cast may
// still be a legitimate DeviceLocalAxis growth point for
// InlineDeviceLocalAxesPass, so this can't be a general canonicalizer; by
// the time this pass runs that growth has already happened. Folds to an
// AnchorPartitioningOp -- rather than a direct replace -- purely so
// DropAnchorPartitioning above finishes the job in the same greedy fixed
// point instead of duplicating that replacement here.
template <typename CastOp>
struct FoldTrivialCast : public OpRewritePattern<CastOp> {
  using OpRewritePattern<CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CastOp castOp,
                                PatternRewriter &rewriter) const override {
    if (castOp.getInput().getType() != castOp.getOutput().getType()) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<AnchorPartitioningOp>(
        castOp, castOp.getInput(), castOp.getPartitioningAxes());
    return success();
  }
};

struct DropIdentityPartitioningAnchorsPass
    : public impl::DropIdentityPartitioningAnchorsPassBase<
          DropIdentityPartitioningAnchorsPass> {
  using DropIdentityPartitioningAnchorsPassBase::
      DropIdentityPartitioningAnchorsPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    AnchorPartitioningOp::getCanonicalizationPatterns(patterns, context);
    DistributedCastGlobalToLocalOp::getCanonicalizationPatterns(patterns,
                                                                context);
    DistributedCastLocalToGlobalOp::getCanonicalizationPatterns(patterns,
                                                                context);
    patterns.add<DropAnchorPartitioning,
                 FoldTrivialCast<DistributedCastGlobalToLocalOp>,
                 FoldTrivialCast<DistributedCastLocalToGlobalOp>>(context);

    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
