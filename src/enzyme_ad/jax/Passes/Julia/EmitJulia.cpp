#include "src/enzyme_ad/jax/Passes/Julia/Passes.h"

#include <system_error>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Target/Julia/TranslateToJulia.h"

namespace mlir::enzymexla::julia {

#define GEN_PASS_DEF_EMITJULIAPASS
#include "src/enzyme_ad/jax/Passes/Julia/Passes.h.inc"

namespace {

struct EmitJuliaPass : public impl::EmitJuliaPassBase<EmitJuliaPass> {
  using Base::Base;

  void runOnOperation() override {
    if (output.empty()) {
      if (failed(translateToJulia(getOperation(), llvm::outs())))
        signalPassFailure();
      return;
    }
    std::error_code ec;
    llvm::raw_fd_ostream outFile(output, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      getOperation().emitError("failed to open output file '")
          << output << "': " << ec.message();
      signalPassFailure();
      return;
    }
    if (failed(translateToJulia(getOperation(), outFile)))
      signalPassFailure();
  }
};

} // namespace

void buildStableHLOToJuliaPipeline(
    OpPassManager &pm, const StableHLOToJuliaPipelineOptions &options) {
  pm.addPass(mlir::enzyme::createLowerEnzymeXLAMathPass());
  if (options.runEnzymeHloOpt) {
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    pm.addPass(mlir::enzyme::createEnzymeHLOOptPass());
  }
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  LowerStableHLOToJuliaPassOptions lowerOpts;
  lowerOpts.reverse_dims = options.reverseDims;
  lowerOpts.prefer_blas = options.preferBlas;
  pm.addPass(createLowerStableHLOToJuliaPass(lowerOpts));
  pm.addPass(mlir::createCSEPass());

  if (options.fuseOps) {
    JuliaFuseOpsPassOptions fuseOpts;
    fuseOpts.fuse_horizontal = true;
    fuseOpts.fuse_matmul_epilogue = true;
    pm.addPass(createJuliaFuseOpsPass(fuseOpts));
    pm.addPass(mlir::createCSEPass());
  }

  if (options.bufferize) {
    JuliaBufferizePassOptions bufOpts;
    bufOpts.reuse_buffers = options.reuseBuffers;
    bufOpts.hoist_loop_allocs = true;
    bufOpts.forward_concat_slices = true;
    pm.addPass(createJuliaBufferizePass(bufOpts));
  }
}

void registerStableHLOToJuliaPipeline() {
  PassPipelineRegistration<StableHLOToJuliaPipelineOptions>(
      "shlo-to-julia-pipeline",
      "Optimize and lower StableHLO/CHLO/EnzymeXLA to bufferized Julia dialect",
      buildStableHLOToJuliaPipeline);
}

} // namespace mlir::enzymexla::julia
