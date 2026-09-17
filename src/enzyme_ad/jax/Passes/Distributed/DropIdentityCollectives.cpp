#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_DROPIDENTITYCOLLECTIVESPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// A collective performs no real communication when it does no reduction,
// its input and output meshes are the exact same factors in the exact same
// order, its mapping is a fully identity factor-to-factor correspondence,
// and its declared output type is literally its own input type. Callers
// must have already put `op`'s mapping into maximal one-to-one form (see
// CanonicalizeAxisMapsPass) before calling this, since predGroupPairIsIdentity
// checks per-pair identity, not per-factor.
bool isIdentityCollective(DistributedCollectiveOp op) {
  if (!op.getReductionGroups().empty()) {
    return false;
  }

  auto inputMeshFactors = axis::getProductProvenanceFactors(op.getInputMesh());
  auto outputMeshFactors =
      axis::getProductProvenanceFactors(op.getOutputMesh());
  if (failed(inputMeshFactors) || failed(outputMeshFactors)) {
    return false;
  }
  if (!axis::areFactorListsStructurallyEqual(*inputMeshFactors,
                                             *outputMeshFactors)) {
    return false;
  }

  auto mapOp = op.getMapping().getDefiningOp<axis::AxisMapOp>();
  if (!mapOp) {
    return false;
  }
  if (!llvm::all_of(mapOp.getTypedMappingPairs(),
                    axis::predGroupPairIsIdentity())) {
    return false;
  }

  return op.getOutputType() == op.getInputObject().getType();
}

// Removes a distributed.Collective that performs no real communication,
// replacing every distributed.Await use of its result directly with its own
// input_object. No AnchorPartitioning is needed at that edge: type equality
// is already guaranteed by the identity condition, and DistributedAwait
// never itself implemented PartitioningAnchorOpInterface, so any binding a
// downstream consumer needs already lives on a real AnchorPartitioning sitting
// after the await -- which simply gets repointed to input_object, one hop
// shorter.
struct DropIdentityCollective
    : public OpRewritePattern<DistributedCollectiveOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DistributedCollectiveOp collectiveOp,
                                PatternRewriter &rewriter) const override {
    if (!isIdentityCollective(collectiveOp)) {
      return failure();
    }

    for (Operation *user : collectiveOp->getUsers()) {
      if (!isa<DistributedAwait>(user)) {
        return failure();
      }
    }

    Value inputObject = collectiveOp.getInputObject();
    for (Operation *user :
         llvm::make_early_inc_range(collectiveOp->getUsers())) {
      rewriter.replaceOp(cast<DistributedAwait>(user), inputObject);
    }
    rewriter.eraseOp(collectiveOp);
    return success();
  }
};

struct DropIdentityCollectivesPass
    : public impl::DropIdentityCollectivesPassBase<
          DropIdentityCollectivesPass> {
  using DropIdentityCollectivesPassBase::DropIdentityCollectivesPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    // axis.product's own canonicalizer drops extent-1 factors (see
    // DropUnitFactorsFromProduct in Dialect/Axis/Ops.cpp): applying it here,
    // ahead of the canonicalize-axis-maps sub-pass below, ensures no
    // collective's mesh/mapping operand still mixes a unit factor in with
    // real ones by the time axis::split_divisible runs -- that mix is what
    // makes its internal splitting choke.
    RewritePatternSet dropUnitFactorPatterns(context);
    axis::AxisProductOp::getCanonicalizationPatterns(dropUnitFactorPatterns,
                                                     context);
    if (failed(
            applyPatternsGreedily(module, std::move(dropUnitFactorPatterns)))) {
      signalPassFailure();
      return;
    }

    PassManager pm(&getContext());
    pm.addPass(createCanonicalizeAxisMapsPass());
    if (failed(pm.run(module))) {
      module.emitError() << "canonicalize-axis-maps sub-pass failed";
      return signalPassFailure();
    }

    RewritePatternSet patterns(context);
    patterns.add<DropIdentityCollective>(context);
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
