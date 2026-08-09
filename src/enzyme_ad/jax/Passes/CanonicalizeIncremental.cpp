//===- CanonicalizeIncremental.cpp - Skip-unchanged canonicalization ------===//
//
// The raising pipeline canonicalizes the module many times, and most of the
// module does not change between two runs: a test TU carries hundreds of
// exception-handling functions the raising never touches, yet every
// module-scope canonicalize re-walks them -- pattern lookups, folds, and
// region simplification over their blocks -- to conclude nothing.
//
// This pass canonicalizes each top-level region-holding op separately and
// remembers, in a process-side table, a fingerprint of what the op looked
// like when it was last left canonical. An op whose fingerprint still
// matches is skipped whole. The fingerprint comparison itself is the
// correctness guard: a stale table entry (the op was erased and the
// allocation reused) can only cause a skip when the content is identical,
// which is exactly when skipping is right. Ops are processed in parallel;
// per-op canonicalization also confines region simplification to that op.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include <mutex>
#include <unordered_map>

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_CANONICALIZEINCREMENTALPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

struct FingerprintTable {
  std::mutex lock;
  std::unordered_map<Operation *, OperationFingerPrint> stamps;

  static FingerprintTable &get() {
    static FingerprintTable table;
    return table;
  }
};

struct CanonicalizeIncrementalPass
    : public enzyme::impl::CanonicalizeIncrementalPassBase<
          CanonicalizeIncrementalPass> {
  using CanonicalizeIncrementalPassBase::CanonicalizeIncrementalPassBase;

  void runOnOperation() override {
    Operation *module = getOperation();
    MLIRContext *ctx = &getContext();

    // The same pattern collection the canonicalizer performs.
    RewritePatternSet owningPatterns(ctx);
    for (auto *dialect : ctx->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : ctx->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, ctx);
    FrozenRewritePatternSet patterns(std::move(owningPatterns));

    SmallVector<Operation *> targets;
    for (Operation &op : module->getRegion(0).front())
      if (op.getNumRegions() != 0 &&
          llvm::any_of(op.getRegions(), [](Region &r) { return !r.empty(); }))
        targets.push_back(&op);

    FingerprintTable &table = FingerprintTable::get();
    GreedyRewriteConfig config;
    config.enableFolding();
    config.enableConstantCSE();
    // The canonicalizer's default: no identical-block merging. Merging adds
    // successor operands for the values the blocks differed in, and e.g.
    // llvm.invoke cannot carry an index-typed successor operand.
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Normal);

    parallelForEach(ctx, targets, [&](Operation *fn) {
      OperationFingerPrint pre(fn);
      {
        std::lock_guard<std::mutex> guard(table.lock);
        auto it = table.stamps.find(fn);
        if (it != table.stamps.end() && it->second == pre)
          return;
      }
      // Convergence failure here matches the plain canonicalizer's behavior
      // of leaving the IR in its best-effort state.
      (void)applyPatternsGreedily(fn, patterns, config);
      OperationFingerPrint post(fn);
      std::lock_guard<std::mutex> guard(table.lock);
      table.stamps.insert_or_assign(fn, post);
    });
  }
};

} // namespace
