//===- CanonicalizeParallel.cpp - Per-op parallel canonicalization --------===//
//
// The raising pipeline canonicalizes the module many times, and the stock
// canonicalizer walks it as one serial unit: one greedy driver seeds a
// worklist with every operation in the module, and each driver iteration
// re-simplifies every region in it. On a translation unit whose module holds
// thousands of functions that is the pipeline's dominant cost.
//
// This pass canonicalizes each top-level region-holding op as its own unit,
// in parallel. Confinement is as important as the parallelism: region
// simplification -- unreachable-block elimination and the region liveness
// fixpoint -- runs per driver iteration over the scope the driver was given,
// so giving each op its own driver keeps one op's churn from re-walking
// every other op's regions. Regionless children -- globals and declarations
// -- are canonicalized together as one cheap batch.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Threading.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_CANONICALIZEPARALLELPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

struct CanonicalizeParallelPass
    : public enzyme::impl::CanonicalizeParallelPassBase<
          CanonicalizeParallelPass> {
  using CanonicalizeParallelPassBase::CanonicalizeParallelPassBase;

  void runOnOperation() override {
    Operation *root = getOperation();
    MLIRContext *ctx = &getContext();

    // The same pattern collection the canonicalizer performs.
    RewritePatternSet owningPatterns(ctx);
    for (auto *dialect : ctx->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : ctx->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, ctx);
    FrozenRewritePatternSet patterns(std::move(owningPatterns));

    GreedyRewriteConfig config;
    config.enableFolding();
    config.enableConstantCSE();
    // The canonicalizer's default: no identical-block merging. Merging adds
    // successor operands for the values the blocks differed in, and e.g.
    // llvm.invoke cannot carry an index-typed successor operand.
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Normal);

    // Convergence failure below matches the plain canonicalizer's behavior
    // of leaving the IR in its best-effort state.

    // Anchored on a function (the Enzyme postpasses run this over a single
    // freshly differentiated function) -- or on anything else that is not a
    // single-block symbol table -- the anchor itself is the unit.
    if (isa<FunctionOpInterface>(root) ||
        !root->hasTrait<OpTrait::SymbolTable>() || !root->getNumRegions() ||
        !root->getRegion(0).hasOneBlock()) {
      (void)applyPatternsGreedily(root, patterns, config);
      return;
    }

    SmallVector<Operation *> targets;
    SmallVector<Operation *> loose;
    for (Operation &op : root->getRegion(0).front()) {
      if (op.getNumRegions() != 0 &&
          llvm::any_of(op.getRegions(), [](Region &r) { return !r.empty(); }))
        targets.push_back(&op);
      else
        loose.push_back(&op);
    }

    if (!loose.empty()) {
      GreedyRewriteConfig looseConfig = config;
      looseConfig.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
      (void)applyOpPatternsGreedily(loose, patterns, looseConfig);
    }

    if (parallel) {
      parallelForEach(ctx, targets, [&](Operation *op) {
        (void)applyPatternsGreedily(op, patterns, config);
      });
    } else {
      for (Operation *op : targets)
        (void)applyPatternsGreedily(op, patterns, config);
    }
  }
};

} // namespace
