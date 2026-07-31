#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "src/enzyme_ad/jax/Passes/Distributed/PartialOrder.h"
#include "src/enzyme_ad/jax/Passes/Distributed/ShardyLogicalAxisAnalysis.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_SHARDYTODISTRIBUTEDHIGHERLEVELPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

static void
dumpLogicalAxesForMainBlock(Block *mainBlock,
                            ShardyLogicalAxisAnalysis &axisAnalysis) {
  llvm::errs()
      << "[ShardyToDistributedHigherLevel] logical axes for main block\n";
  for (Operation &op : mainBlock->getOperations()) {
    auto partitioningAxes = axisAnalysis.getPartitioningAxes(&op);

    llvm::errs() << "  op: " << op << "\n";
    llvm::errs() << "    partitioning axes: ";
    for (const auto &axis : partitioningAxes) {
      llvm::errs() << "[";
      for (const auto &symbol : axis) {
        llvm::errs() << "a" << symbol.getId() << ":" << symbol.getExtent()
                     << " ";
      }
      llvm::errs() << "] ";
    }
    llvm::errs() << "\n";
  }
}

struct ShardyToDistributedHigherLevelPass
    : public impl::ShardyToDistributedHigherLevelPassBase<
          ShardyToDistributedHigherLevelPass> {
  using ShardyToDistributedHigherLevelPassBase::
      ShardyToDistributedHigherLevelPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    func::FuncOp mainFunc;
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (func.getName() == "main") {
        mainFunc = func;
        break;
      }
    }

    if (!mainFunc) {
      emitWarning(module.getLoc()) << "no main function found; skipping pass";
      return;
    }

    Region &body = mainFunc.getBody();
    if (body.empty()) {
      emitError(mainFunc.getLoc()) << "main function has no body";
      signalPassFailure();
      return;
    }

    Block *mainBlock = nullptr;
    if (body.getBlocks().size() != 1) {
      emitError(mainFunc.getLoc())
          << "main function must have exactly one block";
      signalPassFailure();
      return;
    }

    mainBlock = &body.front();
    if (!mainBlock) {
      emitError(mainFunc.getLoc()) << "main function has no entry block";
      signalPassFailure();
      return;
    }

    ShardyLogicalAxisAnalysis axisAnalysis(mainFunc);

    if (dumpLogicalAxes) {
      dumpLogicalAxesForMainBlock(mainBlock, axisAnalysis);
    }

    SSABlockPartialOrderAnalysis partialOrderAnalysis(mainBlock);
    auto &partialOrder = partialOrderAnalysis.getPartialOrder();
    (void)partialOrder;
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
