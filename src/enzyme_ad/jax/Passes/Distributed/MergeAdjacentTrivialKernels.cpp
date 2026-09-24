#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_MERGEADJACENTTRIVIALKERNELSPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// A CastGlobalToLocal fed directly by a block argument has no dependency on
// anything else in the block, so it's always safe (the op is Pure) to slide
// it earlier past whatever currently precedes it. Doing so clears casts that
// otherwise sit between two kernels for no real reason other than emission
// order, exposing them to MergeAdjacentTrivialKernels. Moves one hop at a
// time so the greedy driver naturally settles once every such cast has
// bubbled past every non-cast predecessor -- stopping short of swapping two
// bubbled casts against each other avoids oscillating forever.
struct HoistBlockArgCastUp
    : public OpRewritePattern<DistributedCastGlobalToLocalOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DistributedCastGlobalToLocalOp castOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<BlockArgument>(castOp.getInput())) {
      return failure();
    }
    Operation *prev = castOp->getPrevNode();
    if (!prev) {
      return failure();
    }
    if (auto prevCast = dyn_cast<DistributedCastGlobalToLocalOp>(prev)) {
      if (isa<BlockArgument>(prevCast.getInput())) {
        return failure();
      }
    }
    rewriter.moveOpBefore(castOp, prev);
    return success();
  }
};

// Symmetric to HoistBlockArgCastUp: a CastLocalToGlobal whose only use is the
// block's terminator has nothing downstream depending on where it sits, so
// it's safe to slide later past whatever currently follows it, clearing it
// out from between kernels the same way.
struct SinkYieldOnlyCastDown
    : public OpRewritePattern<DistributedCastLocalToGlobalOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DistributedCastLocalToGlobalOp castOp,
                                PatternRewriter &rewriter) const override {
    if (!castOp.getOutput().hasOneUse()) {
      return failure();
    }
    Operation *user = *castOp.getOutput().getUsers().begin();
    auto yieldOp = dyn_cast<DistributedYieldOp>(user);
    if (!yieldOp || yieldOp->getBlock() != castOp->getBlock()) {
      return failure();
    }
    Operation *next = castOp->getNextNode();
    if (next == yieldOp) {
      return failure();
    }
    if (auto nextCast = dyn_cast<DistributedCastLocalToGlobalOp>(next)) {
      if (nextCast.getOutput().hasOneUse() &&
          *nextCast.getOutput().getUsers().begin() == yieldOp) {
        return failure();
      }
    }
    rewriter.moveOpAfter(castOp, next);
    return success();
  }
};

// Matches a CastLocalToGlobal/CastGlobalToLocal (or reverse) pair sitting
// immediately after `afterFirst`, immediately followed by the kernel that
// consumes the second cast's result, when the round trip's net effect on
// the tensor is a plain identity: same rank, same static shape, same
// element type going in and coming back out, even though the two casts
// name different logical axes. Requiring each hop to be the block's actual
// next op, and each cast's own result to have exactly the one use that
// continues the chain, rules out anything else in the block observing an
// intermediate value this pattern is about to erase.
//
// A same-shape round trip like this can only be relabeling which logical
// axis explains a given local shard, not moving data across devices -- a
// kernel body only ever sees its own local shard, so if that shard's shape
// is unchanged end to end, the value flowing into the second kernel really
// is the same value the first kernel produced. Whether two different axes
// happen to agree on shape like this is exactly the kind of thing this
// pass's usual merge condition (both kernels already directly adjacent,
// established by isTriviallyLocalKernel plus FoldCastRoundTrip already
// having applied) can't tell from either cast's own local vantage point --
// this pattern is a narrower, special-cased fold on top of that, scoped to
// this one boundary shape rather than a general instruction-scheduling
// merge.
template <typename FirstCastOp, typename SecondCastOp>
static bool matchShapeIdentityCastBoundary(Operation *afterFirst,
                                           DistributedKernelOp &second,
                                           Operation *&cast1, Operation *&cast2,
                                           Value &bypassValue) {
  auto firstCast = dyn_cast<FirstCastOp>(afterFirst);
  if (!firstCast || !firstCast.getOutput().hasOneUse()) {
    return false;
  }
  auto secondCast =
      dyn_cast<SecondCastOp>(*firstCast.getOutput().getUsers().begin());
  if (!secondCast || secondCast->getPrevNode() != firstCast ||
      !secondCast.getOutput().hasOneUse()) {
    return false;
  }
  if (firstCast.getInput().getType() != secondCast.getOutput().getType()) {
    return false;
  }
  auto kernel =
      dyn_cast<DistributedKernelOp>(*secondCast.getOutput().getUsers().begin());
  if (!kernel || kernel->getPrevNode() != secondCast) {
    return false;
  }
  second = kernel;
  cast1 = firstCast;
  cast2 = secondCast;
  bypassValue = firstCast.getInput();
  return true;
}

// Merges two physically-adjacent trivial kernels into one. `next` is always
// `first`'s immediate successor in the block, so nothing needs to be
// reordered -- the merged kernel simply occupies their combined position.
// "Adjacent" also covers the two kernels being joined directly by a
// shape-identity cast round trip (see matchShapeIdentityCastBoundary):
// once that's confirmed identity, the round trip is spliced out along with
// the two kernel bodies coming together.
struct MergeAdjacentTrivialKernels
    : public OpRewritePattern<DistributedKernelOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DistributedKernelOp first,
                                PatternRewriter &rewriter) const override {
    if (!isTriviallyLocalKernel(first)) {
      return failure();
    }
    Operation *next = first->getNextNode();
    if (!next) {
      return failure();
    }
    DistributedKernelOp second;
    Operation *castOp1 = nullptr, *castOp2 = nullptr;
    Value bypassValue;
    if (auto direct = dyn_cast<DistributedKernelOp>(next)) {
      second = direct;
    } else if (!matchShapeIdentityCastBoundary<DistributedCastLocalToGlobalOp,
                                               DistributedCastGlobalToLocalOp>(
                   next, second, castOp1, castOp2, bypassValue) &&
               !matchShapeIdentityCastBoundary<DistributedCastGlobalToLocalOp,
                                               DistributedCastLocalToGlobalOp>(
                   next, second, castOp1, castOp2, bypassValue)) {
      return failure();
    }
    if (!second || !isTriviallyLocalKernel(second)) {
      return failure();
    }

    MLIRContext *ctx = rewriter.getContext();

    // Every operand of `second` that is one of `first`'s results is a direct
    // SSA dependency between the pair: it should be wired to the shared
    // internal value once both bodies live in one block, not carried through
    // as a new external operand of the merged kernel.
    llvm::DenseMap<Value, int64_t> firstResultIndex;
    for (auto [idx, result] : llvm::enumerate(first.getResults())) {
      firstResultIndex[result] = idx;
    }

    struct SecondArgSource {
      bool fromFirstResult;
      int64_t index; // index into first's results, or into mergedOperands.
    };

    // When a shape-identity cast boundary sits between the two kernels,
    // `second`'s operand naming that boundary's own tail value is really
    // naming whatever fed the boundary's head -- substitute that through
    // before doing the same first-result-or-external classification as the
    // no-boundary case.
    Value castChainOutput = castOp2 ? castOp2->getResult(0) : Value();

    SmallVector<Value> mergedOperands(first.getArguments().begin(),
                                      first.getArguments().end());
    SmallVector<SecondArgSource> secondArgSources;
    secondArgSources.reserve(second.getArguments().size());
    for (Value operand : second.getArguments()) {
      Value effective = (castChainOutput && operand == castChainOutput)
                            ? bypassValue
                            : operand;
      auto it = firstResultIndex.find(effective);
      if (it != firstResultIndex.end()) {
        secondArgSources.push_back({true, it->second});
        continue;
      }
      secondArgSources.push_back(
          {false, static_cast<int64_t>(mergedOperands.size())});
      mergedOperands.push_back(effective);
    }

    SmallVector<Type> mergedResultTypes(first.getResultTypes().begin(),
                                        first.getResultTypes().end());
    mergedResultTypes.append(second.getResultTypes().begin(),
                             second.getResultTypes().end());

    SmallVector<IndexedTensorShardingAttr> argumentShardings;
    argumentShardings.reserve(mergedOperands.size());
    for (Value operand : mergedOperands) {
      argumentShardings.push_back(
          buildEmptyShardingForType(ctx, operand.getType()));
    }
    SmallVector<IndexedTensorShardingAttr> outputShardings;
    outputShardings.reserve(mergedResultTypes.size());
    for (Type type : mergedResultTypes) {
      outputShardings.push_back(buildEmptyShardingForType(ctx, type));
    }

    rewriter.setInsertionPoint(first);
    auto merged = rewriter.create<DistributedKernelOp>(
        first.getLoc(), TypeRange(mergedResultTypes),
        ValueRange(mergedOperands), ValueRange(),
        IndexedTensorShardingPerValueAttr::get(ctx, argumentShardings),
        IndexedTensorShardingPerValueAttr::get(ctx, outputShardings));

    Block *mergedBlock = new Block();
    merged.getBody().push_back(mergedBlock);
    for (Value operand : mergedOperands) {
      mergedBlock->addArgument(operand.getType(), first.getLoc());
    }

    // inlineBlockBefore splices a block's ops into another block and
    // replaces its block-args with the given values in one step, including
    // within its terminator -- so the yielded values come out already
    // rewritten in terms of mergedBlock once each half is spliced in.
    Block &firstBlock = first.getBody().front();
    rewriter.inlineBlockBefore(
        &firstBlock, mergedBlock, mergedBlock->end(),
        mergedBlock->getArguments().take_front(firstBlock.getNumArguments()));
    auto firstYield = cast<DistributedYieldOp>(mergedBlock->getTerminator());
    SmallVector<Value> firstYieldOperands(firstYield.getReturns());
    rewriter.eraseOp(firstYield);

    Block &secondBlock = second.getBody().front();
    SmallVector<Value> secondArgValues;
    secondArgValues.reserve(secondArgSources.size());
    for (const SecondArgSource &source : secondArgSources) {
      secondArgValues.push_back(source.fromFirstResult
                                    ? firstYieldOperands[source.index]
                                    : mergedBlock->getArgument(source.index));
    }
    rewriter.inlineBlockBefore(&secondBlock, mergedBlock, mergedBlock->end(),
                               secondArgValues);
    auto secondYield = cast<DistributedYieldOp>(mergedBlock->getTerminator());

    SmallVector<Value> mergedYieldOperands = firstYieldOperands;
    llvm::append_range(mergedYieldOperands, secondYield.getReturns());
    rewriter.setInsertionPoint(secondYield);
    rewriter.replaceOpWithNewOp<DistributedYieldOp>(secondYield,
                                                    mergedYieldOperands);

    int64_t firstNumResults = static_cast<int64_t>(first.getResults().size());
    for (auto [oldResult, newResult] :
         llvm::zip(first.getResults(),
                   merged.getResults().take_front(firstNumResults))) {
      rewriter.replaceAllUsesWith(oldResult, newResult);
    }
    for (auto [oldResult, newResult] :
         llvm::zip(second.getResults(),
                   merged.getResults().drop_front(firstNumResults))) {
      rewriter.replaceAllUsesWith(oldResult, newResult);
    }

    // `second` is erased first so the cast pair's uses actually go to zero
    // before each cast is erased in turn; `first`'s uses were already
    // retargeted to `merged` above, including any use from castOp1 itself.
    rewriter.eraseOp(second);
    if (castOp2) {
      rewriter.eraseOp(castOp2);
      rewriter.eraseOp(castOp1);
    }
    rewriter.eraseOp(first);
    return success();
  }
};

struct MergeAdjacentTrivialKernelsPass
    : public impl::MergeAdjacentTrivialKernelsPassBase<
          MergeAdjacentTrivialKernelsPass> {
  using MergeAdjacentTrivialKernelsPassBase::
      MergeAdjacentTrivialKernelsPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<HoistBlockArgCastUp, SinkYieldOnlyCastDown,
                 MergeAdjacentTrivialKernels>(context);

    // Hoisting and sinking move a cast one op per rewrite, so the number of
    // iterations needed grows with the distance to travel; a fixed cap would
    // make the pass fail on long function bodies.
    GreedyRewriteConfig config;
    config.setMaxIterations(GreedyRewriteConfig::kNoLimit);
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
