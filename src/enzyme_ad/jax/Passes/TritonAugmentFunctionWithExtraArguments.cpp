#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "src/enzyme_ad/jax/Dialect/TritonExt/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/enzyme_ad/jax/Utils.h"

#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "triton-augment-function-with-extra-arguments"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_TRITONAUGMENTFUNCTIONWITHEXTRAARGUMENTSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzymexla;
using namespace mlir::enzymexla::triton_ext;

// See for description on the extra arguments
// https://github.com/triton-lang/triton/blob/6ac622c57152ce88edd058f11997b5c5e18d096b/lib/Conversion/TritonGPUToLLVM/FuncOpToLLVM.cpp#L12-L25

LogicalResult
augmentTritonCallOpWithExtraArguments(ModuleOp mod,
                                      triton_ext::TritonCallOp op) {
  SymbolTableCollection symbolTable;
  symbolTable.getSymbolTable(mod);
  auto funcOp = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
      mod, op.getFnAttr());
  if (!funcOp) {
    op->emitError("Failed to find function '") << op.getFn() << "' in module";
    return failure();
  }

  auto fnKind = funcOp->getName().getStringRef();
  if (fnKind != "llvm.func") {
    op->emitError("augmentTritonCallOpWithExtraArguments: expected '")
        << op.getFn() << "' to be a llvm.func, got: " << fnKind << ". This "
        << "means that the pass is being called before tt.func is being "
           "lowered to llvm.func";
    return failure();
  }

  if (funcOp.getNumArguments() == op.getInputs().size()) {
    return success(); // already augmented
  }

  // See NOTE: [Additional Function Arguments] in triton-lang/triton
  if (!mlir::triton::isKernel(funcOp)) {
    op->emitError("not a kernel function");
    return failure();
  }

  bool hasProfileScratchMemory =
      funcOp.getNumArguments() ==
      op.getInputs().size() + 2; // to support compatibility with old kernels

  if (funcOp.getNumArguments() !=
      op.getInputs().size() + 1 + hasProfileScratchMemory) {
    op->emitError("Expected ")
        << (funcOp.getNumArguments() - 1 - hasProfileScratchMemory)
        << " arguments, got " << op.getInputs().size();
    return failure();
  }

  auto newInputs = llvm::to_vector(op.getInputs());

  // global scratch memory
  uint64_t gsmNBytes = 0, gsmAlign = 0;
  if (auto gsm = funcOp->getAttrOfType<IntegerAttr>(
          "ttg.global_scratch_memory_size")) {
    gsmNBytes = gsm.getValue().getZExtValue();
  }
  if (auto smalign = funcOp->getAttrOfType<IntegerAttr>(
          "ttg.global_scratch_memory_alignment")) {
    gsmAlign = smalign.getValue().getZExtValue();
  }

  OpBuilder builder(op);

  auto gsmTy = RankedTensorType::get({static_cast<int64_t>(gsmNBytes)},
                                     builder.getIntegerType(8));
  auto gsm = triton_ext::ScratchMemoryOp::create(
      builder, op.getLoc(), gsmTy, builder.getI64IntegerAttr(gsmAlign));
  newInputs.push_back(gsm);

  // profile scratch memory
  if (hasProfileScratchMemory) {
    uint64_t psmNBytes = 0, psmAlign = 1;
    if (auto psm = funcOp->getAttrOfType<IntegerAttr>(
            "ttg.profile_scratch_memory_size")) {
      psmNBytes = psm.getValue().getZExtValue();
    }
    if (auto psmalign = funcOp->getAttrOfType<IntegerAttr>(
            "ttg.profile_scratch_memory_alignment")) {
      psmAlign = psmalign.getValue().getZExtValue();
    }

    auto psmTy = RankedTensorType::get({static_cast<int64_t>(psmNBytes)},
                                       builder.getIntegerType(8));
    auto psm = triton_ext::ScratchMemoryOp::create(
        builder, op.getLoc(), psmTy, builder.getI64IntegerAttr(psmAlign));
    newInputs.push_back(psm);
  }

  auto newCallOp = triton_ext::TritonCallOp::create(
      builder, op.getLoc(), op.getResultTypes(), op.getFn(), op.getGridx(),
      op.getGridy(), op.getGridz(), op.getBlockx(), op.getBlocky(),
      op.getBlockz(), op.getClusterx(), op.getClustery(), op.getClusterz(),
      newInputs, op.getBackendConfigAttr(), op.getOperandLayoutsAttr(),
      /*resultLayouts*/ nullptr, op.getArgAttrsAttr(), op.getResAttrsAttr(),
      op.getOutputOperandAliasesAttr(), op.getXlaSideEffectFreeAttr());
  op.replaceAllUsesWith(newCallOp);
  op.erase();
  return success();
}

struct TritonAugmentFunctionWithExtraArgumentsPass
    : public mlir::enzyme::impl::
          TritonAugmentFunctionWithExtraArgumentsPassBase<
              TritonAugmentFunctionWithExtraArgumentsPass> {
  using Base::Base;

  void runOnOperation() override {
    auto modOp = getOperation();

    modOp->walk([&](triton_ext::TritonCallOp op) -> WalkResult {
      if (failed(augmentTritonCallOpWithExtraArguments(modOp, op))) {
        signalPassFailure();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }
};
