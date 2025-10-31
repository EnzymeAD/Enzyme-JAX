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
using namespace mlir::enzyme::triton_ext;

struct TritonCallOpLowering
    : public OpRewritePattern<triton_ext::TritonCallOp> {
  using OpRewritePattern<triton_ext::TritonCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton_ext::TritonCallOp op,
                                PatternRewriter &rewriter) const override {
    auto modOp = op->getParentOfType<ModuleOp>();
    if (!modOp)
      return rewriter.notifyMatchFailure(op, "No module found");

    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(modOp);

    auto fn = symbolTable.lookupNearestSymbolFrom(op, op.getFnAttr());
    if (!fn)
      return rewriter.notifyMatchFailure(op, "No function found");

    auto wrappedMod = fn->getParentOfType<ModuleOp>();
    if (!wrappedMod)
      return rewriter.notifyMatchFailure(op, "No module found");

    bool hasSharedAttr = wrappedMod->hasAttr("ttg.shared");
    if (!hasSharedAttr) {
      op->emitError("No ttg.shared attribute found. Triton Passes must be run "
                    "before invoking lower-triton pass.");
    }

    auto ttModOP = wrappedMod->getParentOfType<triton_ext::TritonModuleOp>();
    if (!ttModOP) {
      op->emitWarning("No `triton_ext.module` found.");
    }
    ttModOP.setVisibility(SymbolTable::Visibility::Private);

    auto intAttr = wrappedMod->getAttrOfType<mlir::IntegerAttr>("ttg.shared");
    int64_t shared = intAttr.getInt();

    auto shmemOpType = op.getGridx().getType();
    auto shmemOp = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), shmemOpType,
        cast<ElementsAttr>(makeAttr(shmemOpType, shared)));

    SmallVector<Value> inputs;
    for (auto input : op.getInputs()) {
      inputs.push_back(input);
    }
    // XXX: Fix this. This is supposed to be the shared memory stack
    auto shmemStack = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), shmemOpType,
        cast<ElementsAttr>(makeAttr(shmemOpType, 32)));
    inputs.push_back(shmemStack);
    // XXX: I am not sure what is this argument for...
    inputs.push_back(shmemStack);

    rewriter.replaceOpWithNewOp<enzymexla::KernelCallOp>(
        op, op.getResultTypes(), op.getFn(), op.getGridx(), op.getGridy(),
        op.getGridz(), op.getBlockx(), op.getBlocky(), op.getBlockz(), shmemOp,
        op.getClusterx(), op.getClustery(), op.getClusterz(), inputs,
        op.getBackendConfigAttr(), op.getOperandLayoutsAttr(),
        op.getResultLayoutsAttr(), op.getArgAttrsAttr(), op.getResAttrsAttr(),
        op.getOutputOperandAliasesAttr(), op.getXlaSideEffectFreeAttr());
    return success();
  }
};

struct LowerTritonPass
    : public mlir::enzyme::impl::LowerTritonPassBase<LowerTritonPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<TritonCallOpLowering>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
