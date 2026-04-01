#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "triton/Conversion/TritonToTritonGPU/Passes.h"

#define DEBUG_TYPE "convert-triton-to-triton-gpu-preserving-module-attributes"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_CONVERTTRITONTOTRITONGPUPRESERVINGMODULEATTRIBUTESPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

struct ConvertTritonToTritonGPUPreservingModuleAttributesPass
    : public mlir::enzyme::impl::
          ConvertTritonToTritonGPUPreservingModuleAttributesPassBase<
              ConvertTritonToTritonGPUPreservingModuleAttributesPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    int32_t numWarps = 4, threadsPerWarp = 32, numCtas = 1;
    bool enableSourceRemat = false;

    if (mod->hasAttr("enzymexla.ttg.num-ctas")) {
      numCtas =
          mod->getAttrOfType<IntegerAttr>("enzymexla.ttg.num-ctas").getInt();
    }

    if (mod->hasAttr("enzymexla.ttg.num-warps")) {
      numWarps =
          mod->getAttrOfType<IntegerAttr>("enzymexla.ttg.num-warps").getInt();
    }

    if (mod->hasAttr("enzymexla.ttg.threads-per-warp")) {
      threadsPerWarp =
          mod->getAttrOfType<IntegerAttr>("enzymexla.ttg.threads-per-warp")
              .getInt();
    }

    if (mod->hasAttr("enzymexla.ttg.enable-source-remat")) {
      enableSourceRemat = true;
    }

    OpPassManager pm;
    pm.addPass(triton::createConvertTritonToTritonGPU(
        {target, numWarps, threadsPerWarp, numCtas, enableSourceRemat}));
    if (failed(runPipeline(pm, mod))) {
      mod->emitError() << "failed to run triton passes";
      signalPassFailure();
      return;
    }

    return;
  }
};
