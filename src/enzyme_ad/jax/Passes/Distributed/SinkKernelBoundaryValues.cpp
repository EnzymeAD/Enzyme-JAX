#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_SINKKERNELBOUNDARYVALUESPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// Walks `value`'s producer chain within `producer`'s body, deciding whether
// it can be rebuilt inside `consumer`'s body without adding any new
// cross-kernel dependency.
//
// - A leaf that is one of `producer`'s own block arguments corresponds to an
//   external value (producer's operand at that index). If consumer already
//   takes that value as an operand, it is reused. Otherwise the value can be
//   appended to consumer's operands, but only when it is not another
//   kernel's result: a function-level value (an argument cast, a constant)
//   is available everywhere and adds no scheduling edge, whereas a kernel
//   result would make consumer wait on a kernel it never depended on. Since
//   this extends consumer's inputs, the caller only accepts such a chain
//   when it is moved rather than copied (see `isPrivateToConsumer`).
// - A leaf whose defining op sits outside producer's body needs no such
//   check. A DistributedKernelOp body need not be isolated from above (the
//   same reason buildKernelBodyModule re-closes over implicit captures when
//   dispatching a kernel), so anything defined in an enclosing scope already
//   dominates, and is visible to, every sibling kernel including consumer;
//   greedy folding routinely leaves an identical constant in exactly this
//   state, commoned up just above the kernels that use it.
//
// `chain` collects the ops to clone in dependency order (each op's operands
// already visited before the op itself), for replaying with a single
// IRMapping. `leafMap` collects the producer-body-block-argument to
// consumer-body-block-argument substitutions the clone needs; a leaf whose
// external value consumer does not yet take is recorded in `leafMap` with a
// null value and in `newOperands` (external value -> producer block args it
// feeds) for the caller to append once the whole chain is accepted.
static bool
collectSinkableChain(DistributedKernelOp producer, DistributedKernelOp consumer,
                     Value value, llvm::SetVector<Operation *> &chain,
                     DenseMap<Value, Value> &leafMap,
                     llvm::MapVector<Value, SmallVector<Value>> &newOperands) {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (blockArg.getOwner() != &producer.getBody().front())
      return true; // Not producer's own block arg: an outer-scope value.
    if (leafMap.count(blockArg))
      return true;
    Value external = producer.getArguments()[blockArg.getArgNumber()];
    ValueRange consumerArgs = consumer.getArguments();
    auto it = llvm::find(consumerArgs, external);
    if (it == consumerArgs.end()) {
      if (external.getDefiningOp<DistributedKernelOp>())
        return false;
      leafMap[blockArg] = Value();
      newOperands[external].push_back(blockArg);
      return true;
    }
    size_t consumerIdx = std::distance(consumerArgs.begin(), it);
    leafMap[blockArg] = consumer.getBody().front().getArgument(consumerIdx);
    return true;
  }

  Operation *def = value.getDefiningOp();
  if (def->getParentRegion() != &producer.getBody())
    return true; // Already outside producer's body, so already shared.
  if (chain.contains(def))
    return true;
  if (def->getNumRegions() != 0)
    return false;
  bool isConstant = def->hasTrait<OpTrait::ConstantLike>();
  if (!isConstant) {
    if (!isMemoryEffectFree(def))
      return false;
    for (Value operand : def->getOperands())
      if (!collectSinkableChain(producer, consumer, operand, chain, leafMap,
                                newOperands))
        return false;
  }
  chain.insert(def);
  return true;
}

// True if `chain` (rooted at the yielded value `root`) computes something
// only `root`'s single consumer needs, so sinking it relocates the work
// instead of duplicating it into every kernel that reads a shared value.
static bool isPrivateToConsumer(Value operand, Value root,
                                const llvm::SetVector<Operation *> &chain) {
  if (!operand.hasOneUse())
    return false;
  for (Operation *op : chain)
    for (Operation *user : op->getUsers())
      if (!chain.contains(user) &&
          !(isa<DistributedYieldOp>(user) && op == root.getDefiningOp()))
        return false;
  return true;
}

// For every operand of `kernel` produced by another DistributedKernelOp,
// tries to rebuild the yielded value's producer chain directly inside
// `kernel`'s own body (see collectSinkableChain), so the value no longer has
// to cross the kernel boundary as an ordinary operand. Leaves the operand
// list and the producer's results untouched: the block argument simply loses
// its last use, and the existing dead-result/dead-operand canonicalizations
// on DistributedKernelOp clean up the now-unused pair.
struct SinkKernelBoundaryValues : public OpRewritePattern<DistributedKernelOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DistributedKernelOp kernel,
                                PatternRewriter &rewriter) const override {
    Block &body = kernel.getBody().front();
    bool sunkAny = false;
    // Sinking appends operands, which invalidates a live view of the list.
    SmallVector<Value> originalOperands(kernel.getArguments());
    for (auto [idx, operand] : llvm::enumerate(originalOperands)) {
      auto producer = operand.getDefiningOp<DistributedKernelOp>();
      if (!producer || producer == kernel)
        continue;
      BlockArgument blockArg = body.getArgument(idx);
      if (blockArg.use_empty())
        continue;

      auto yieldOp =
          cast<DistributedYieldOp>(producer.getBody().front().getTerminator());
      Value yielded =
          yieldOp.getReturns()[cast<OpResult>(operand).getResultNumber()];

      llvm::SetVector<Operation *> chain;
      DenseMap<Value, Value> leafMap;
      llvm::MapVector<Value, SmallVector<Value>> newOperands;
      if (!collectSinkableChain(producer, kernel, yielded, chain, leafMap,
                                newOperands))
        continue;
      if (!newOperands.empty() && !isPrivateToConsumer(operand, yielded, chain))
        continue;

      if (!newOperands.empty()) {
        MLIRContext *ctx = rewriter.getContext();
        SmallVector<IndexedTensorShardingAttr> shardings(
            kernel.getArgumentShardings().getShardings());
        rewriter.modifyOpInPlace(kernel, [&] {
          for (auto &[external, producerArgs] : newOperands) {
            kernel.getArgumentsMutable().append(external);
            shardings.push_back(
                buildEmptyShardingForType(ctx, external.getType()));
            BlockArgument arg = body.addArgument(external.getType(),
                                                 kernel.getLoc());
            for (Value producerArg : producerArgs)
              leafMap[producerArg] = arg;
          }
          kernel.setArgumentShardingsAttr(
              IndexedTensorShardingPerValueAttr::get(ctx, shardings));
        });
      }

      IRMapping mapping;
      for (auto &entry : leafMap)
        mapping.map(entry.first, entry.second);
      rewriter.setInsertionPointToStart(&body);
      for (Operation *op : chain)
        rewriter.clone(*op, mapping);
      Value sunk = mapping.lookupOrDefault(yielded);
      rewriter.replaceAllUsesWith(blockArg, sunk);
      sunkAny = true;
    }
    return success(sunkAny);
  }
};

struct SinkKernelBoundaryValuesPass
    : public impl::SinkKernelBoundaryValuesPassBase<
          SinkKernelBoundaryValuesPass> {
  using SinkKernelBoundaryValuesPassBase::SinkKernelBoundaryValuesPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<SinkKernelBoundaryValues>(context);
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
