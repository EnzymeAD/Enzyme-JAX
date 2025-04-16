#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_SPLITHUGEBLOCKSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

static void splitLargeBlock(RewriterBase &rewriter, Block *block,
                            uint64_t maxNumOperations) {
  do {
    Block::iterator it = block->begin();
    for (uint64_t i = 0; i < maxNumOperations; ++i) {
      if (it == block->end())
        return;
      it = std::next(it);
    }
    Block *current = block;
    block = rewriter.splitBlock(block, it);
    rewriter.setInsertionPointToEnd(current);
    rewriter.create<cf::BranchOp>(rewriter.getUnknownLoc(), block);
  } while (true);
}

struct SplitHugeBlocksPass
    : public enzyme::impl::SplitHugeBlocksPassBase<SplitHugeBlocksPass> {
  using SplitHugeBlocksPassBase::SplitHugeBlocksPassBase;

  void runOnOperation() override {
    if (max_num_operations == -1)
      return;
    auto context = getOperation()->getContext();
    IRRewriter rewriter(context);
    SmallVector<Block *> originalBlocks = llvm::map_to_vector(
        getOperation().getFunctionBody(), [](Block &b) { return &b; });
    for (Block *block : originalBlocks) {
      splitLargeBlock(rewriter, block, max_num_operations);
    }
  }
};
