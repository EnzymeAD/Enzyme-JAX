#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_DISTRIBUTEDTOHLOPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

bool hasReductionAxis(distributed::DistributedCollectiveOp op) {
  assert(false && "hasReductionAxis not implemented");
  return false;
}

struct DistributedCollectiveToStablehloPattern
    : public OpRewritePattern<distributed::DistributedCollectiveOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(distributed::DistributedCollectiveOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Lower distributed.collective to one or more stablehlo
    // communication ops (all_reduce/all_gather/all_to_all/etc.).
    // Our collectives are more general than stableHLOs. It is
    // possible to decompose them into sequences of stableHLO-like
    // collectives and reshapes/slices. Here, we will assert
    // that it does match some stableHLO collective, and rely on
    // prior conditioning passes to do any transformation into
    // compliant forms.

    // Cases:
    // 1. all-reduce
    // 2. reduce-scatter
    // 3. collective-permute
    // 4. all-gather
    // 5. all-to-all
    // 6. collective-broadcast

    // Case 1: all-reduce
    // Happens when all non-reduction axes have an identity map, all
    // reduction axes are spatial axis, and all reduction axis have a
    // replicate LHS.

    // Case 2: reduce-scatter
    // Happens when all reduction axes are spatial axes,
    // exactly 1 tensor axis maps to the product of the spatial axes,
    // and all other tensor axes have an identy map.
    // Note: identity map here must account for the differing tensor
    // types on the LHS and RHS. Essentially, aside from the split dimension,
    // we are looking for same rank, extent, and stride.

    // Case 3: collective-permute
    // Happens with no reduction axes and all tensor dimensions are identity
    // mapped.

    // Case 4: all-gather.
    // No reduction axes. Map contains some replicate --> SpaceDim(i),
    // and those same SpaceDim(i) map to a contiguous in-order subdimension
    // of the output tensor (the contatenation dim). All other dimensions
    // identity map (modulo tensor type).

    // Case 5: all-to-all
    // No reduction axes, one subaxis moving from space to tensor, one subaxis
    // moving from tensor to space.

    // Case 6: collective-broadcast: not handled here, belongs more with
    // point-to-point communication.
    (void)op;
    (void)rewriter;
    return failure();
  }
};

static void
populateDistributedCollectiveToStablehloPatterns(RewritePatternSet &patterns) {
  patterns.add<DistributedCollectiveToStablehloPattern>(patterns.getContext());
}

struct DistributedToHloPass
    : public impl::DistributedToHloPassBase<DistributedToHloPass> {
  using DistributedToHloPassBase::DistributedToHloPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateDistributedCollectiveToStablehloPatterns(patterns);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
