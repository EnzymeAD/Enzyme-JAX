//===- CanonicalizeIncremental.cpp - Skip-unchanged canonicalization ------===//
//
// The raising pipeline canonicalizes the module many times, and most of the
// module does not change between two runs: a test TU carries hundreds of
// exception-handling functions the raising never touches, yet every
// module-scope canonicalize re-walks them -- pattern lookups, folds, and
// region simplification over their blocks -- to conclude nothing.
//
// This pass canonicalizes each top-level region-holding op separately and
// stamps the op with a fingerprint of what it looked like when it was last
// left canonical. An op whose fingerprint still matches is skipped whole.
// The stamp lives in an attribute on the op itself, so it dies with the op:
// no table survives an erased operation to be found again by whatever gets
// allocated at the same address later. Ops are processed in parallel;
// per-op canonicalization also confines region simplification to that op.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "llvm/Support/SHA1.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_CANONICALIZEINCREMENTALPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

// A content stamp in the spirit of OperationFingerPrint (which offers no way
// to read its bytes back, and so cannot be stored in an attribute): pointers
// of everything uniqued -- attribute dictionaries, types, values, blocks --
// plus the properties hash, over the whole nested walk. Pointer identity is
// what makes the stamp cheap, and it errs only toward re-canonicalizing:
// an unchanged op keeps every internal pointer alive and unchanged, while a
// recreated op computes a fresh stamp against an attribute it was not born
// with. The root's own discardable dictionary is hashed entry-by-entry so
// the stamp attribute itself stays out of the stamp.
static StringAttr fingerprint(Operation *root, StringAttr skip) {
  llvm::SHA1 hasher;
  auto addPtr = [&](const void *p) {
    hasher.update(
        ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(&p), sizeof(p)));
  };
  auto addHash = [&](llvm::hash_code h) {
    size_t v = h;
    hasher.update(
        ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(&v), sizeof(v)));
  };
  root->walk([&](Operation *op) {
    addPtr(op);
    addPtr(op->getName().getAsOpaquePointer());
    if (op == root) {
      for (NamedAttribute attr : op->getDiscardableAttrs()) {
        if (attr.getName() == skip)
          continue;
        addPtr(attr.getName().getAsOpaquePointer());
        addPtr(attr.getValue().getAsOpaquePointer());
      }
      addHash(op->hashProperties());
    } else {
      addPtr(op->getRawDictionaryAttrs().getAsOpaquePointer());
      addHash(op->hashProperties());
    }
    for (Type type : op->getResultTypes())
      addPtr(type.getAsOpaquePointer());
    for (Value operand : op->getOperands())
      addPtr(operand.getAsOpaquePointer());
    for (Block *successor : op->getSuccessors())
      addPtr(successor);
    for (Region &region : op->getRegions())
      for (Block &block : region) {
        addPtr(&block);
        for (BlockArgument arg : block.getArguments())
          addPtr(arg.getAsOpaquePointer());
      }
  });
  auto digest = hasher.final();
  return StringAttr::get(
      root->getContext(),
      StringRef(reinterpret_cast<const char *>(digest.data()), digest.size()));
}

struct CanonicalizeIncrementalPass
    : public enzyme::impl::CanonicalizeIncrementalPassBase<
          CanonicalizeIncrementalPass> {
  using CanonicalizeIncrementalPassBase::CanonicalizeIncrementalPassBase;

  Statistic numSkipped{this, "skipped",
                       "Ops whose stamp matched and were not re-walked"};
  Statistic numCanonicalized{this, "canonicalized",
                             "Ops canonicalized and freshly stamped"};

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

    StringAttr stampName = StringAttr::get(ctx, "enzymexla.canonical_fp");
    GreedyRewriteConfig config;
    config.enableFolding();
    config.enableConstantCSE();
    // The canonicalizer's default: no identical-block merging. Merging adds
    // successor operands for the values the blocks differed in, and e.g.
    // llvm.invoke cannot carry an index-typed successor operand.
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Normal);

    parallelForEach(ctx, targets, [&](Operation *fn) {
      auto previous = fn->getAttrOfType<StringAttr>(stampName);
      if (previous && previous == fingerprint(fn, stampName)) {
        ++numSkipped;
        return;
      }
      // Convergence failure here matches the plain canonicalizer's behavior
      // of leaving the IR in its best-effort state.
      (void)applyPatternsGreedily(fn, patterns, config);
      fn->setAttr(stampName, fingerprint(fn, stampName));
      ++numCanonicalized;
    });
  }
};

} // namespace
