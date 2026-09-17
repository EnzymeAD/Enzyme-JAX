#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_MERGEADJACENTTRIVIALKERNELSPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// Merges two physically-adjacent trivial kernels into one. `next` is always
// `first`'s immediate successor in the block, so nothing needs to be
// reordered -- the merged kernel simply occupies their combined position.
struct MergeAdjacentTrivialKernels
    : public OpRewritePattern<DistributedKernelOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DistributedKernelOp first,
                                PatternRewriter &rewriter) const override {
    if (!isTriviallyLocalKernel(first)) {
      return failure();
    }
    auto second = dyn_cast_or_null<DistributedKernelOp>(first->getNextNode());
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

    SmallVector<Value> mergedOperands(first.getArguments().begin(),
                                      first.getArguments().end());
    SmallVector<SecondArgSource> secondArgSources;
    secondArgSources.reserve(second.getArguments().size());
    for (Value operand : second.getArguments()) {
      auto it = firstResultIndex.find(operand);
      if (it != firstResultIndex.end()) {
        secondArgSources.push_back({true, it->second});
        continue;
      }
      secondArgSources.push_back(
          {false, static_cast<int64_t>(mergedOperands.size())});
      mergedOperands.push_back(operand);
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

    rewriter.eraseOp(second);
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
    patterns.add<MergeAdjacentTrivialKernels>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
