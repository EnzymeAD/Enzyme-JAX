#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Dialect/TritonExt/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/enzyme_ad/jax/Utils.h"

#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "lower-triton"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERTRITONPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzymexla;
using namespace mlir::enzymexla::triton_ext;

LogicalResult lowerTritonKernelToKernelCall(ModuleOp mod,
                                            triton_ext::TritonCallOp op) {
  SymbolTableCollection symbolTable;
  symbolTable.getSymbolTable(mod);
  auto funcOp = symbolTable.lookupNearestSymbolFrom(op, op.getFnAttr());
  if (!funcOp) {
    op->emitError("Failed to find function '") << op.getFn() << "' in module";
    return failure();
  }

  auto wrappedMod = funcOp->getParentOfType<ModuleOp>();
  if (!wrappedMod) {
    op->emitError("Failed to find parent built-in module.");
    return failure();
  }

  if (!wrappedMod->hasAttr("ttg.shared")) {
    op->emitError("No ttg.shared attribute found. Triton Passes must be run "
                  "before invoking lower-triton pass.");
    return failure();
  }

  auto ttModOP = wrappedMod->getParentOfType<triton_ext::TritonModuleOp>();
  if (!ttModOP) {
    op->emitError("No `triton_ext.module` found!");
    return failure();
  }
  ttModOP.setVisibility(SymbolTable::Visibility::Private);

  OpBuilder builder(op);

  auto sharedMemSizeAttr = wrappedMod->getAttrOfType<IntegerAttr>("ttg.shared");
  auto sharedMemSize = sharedMemSizeAttr.getValue().getZExtValue();
  auto shmemOpType = op.getGridx().getType();
  auto shmemOp = stablehlo::ConstantOp::create(
      builder, op.getLoc(), shmemOpType,
      cast<ElementsAttr>(makeAttr(shmemOpType, sharedMemSize)));

  auto kernelCallOp = enzymexla::KernelCallOp::create(
      builder, op.getLoc(), op.getResultTypes(), op.getFn(), op.getGridx(),
      op.getGridy(), op.getGridz(), op.getBlockx(), op.getBlocky(),
      op.getBlockz(), shmemOp, op.getClusterx(), op.getClustery(),
      op.getClusterz(), op.getInputs(), op.getBackendConfigAttr(),
      op.getOperandLayoutsAttr(), op.getResultLayoutsAttr(),
      op.getArgAttrsAttr(), op.getResAttrsAttr(),
      op.getOutputOperandAliasesAttr(), op.getXlaSideEffectFreeAttr());
  op.replaceAllUsesWith(kernelCallOp);
  op.erase();
  return success();
}

struct LowerTritonPass
    : public mlir::enzyme::impl::LowerTritonPassBase<LowerTritonPass> {
  using Base::Base;

  void runOnOperation() override {
    auto modOp = getOperation();

    modOp->walk([&](triton_ext::TritonCallOp op) {
      if (failed(lowerTritonKernelToKernelCall(modOp, op))) {
        signalPassFailure();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }
};
