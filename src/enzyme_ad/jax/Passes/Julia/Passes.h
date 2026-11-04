#ifndef ENZYMEXLA_PASSES_JULIA_PASSES_H
#define ENZYMEXLA_PASSES_JULIA_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "src/enzyme_ad/jax/Dialect/Julia/Dialect.h"

namespace mlir::enzymexla::julia {

#define GEN_PASS_DECL
#include "src/enzyme_ad/jax/Passes/Julia/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "src/enzyme_ad/jax/Passes/Julia/Passes.h.inc"

struct StableHLOToJuliaPipelineOptions
    : public PassPipelineOptions<StableHLOToJuliaPipelineOptions> {
  Option<bool> reverseDims{
      *this, "reverse-dims",
      llvm::cl::desc("Reverse tensor dimensions to column-major Julia layout"),
      llvm::cl::init(true)};
  Option<bool> runEnzymeHloOpt{
      *this, "enzyme-hlo-opt",
      llvm::cl::desc("Run EnzymeXLA HLO algebraic/loop optimizations before "
                     "lowering to Julia"),
      llvm::cl::init(true)};
  Option<bool> preferBlas{
      *this, "prefer-blas",
      llvm::cl::desc("Lower compatible contractions and factorizations to "
                     "Julia LinearAlgebra calls"),
      llvm::cl::init(true)};
  Option<bool> fuseOps{
      *this, "fuse",
      llvm::cl::desc("Run `julia-fuse-ops` kernel & epilogue fusion"),
      llvm::cl::init(true)};
  Option<bool> bufferize{
      *this, "bufferize",
      llvm::cl::desc("Run `julia-bufferize` destination-passing & buffer-reuse "
                     "optimization"),
      llvm::cl::init(true)};
  Option<bool> reuseBuffers{
      *this, "reuse-buffers",
      llvm::cl::desc("Enable dead-operand in-place reuse and linear-scan "
                     "buffer coloring"),
      llvm::cl::init(true)};
};

/// Populate `pm` with the full StableHLO/CHLO/EnzymeXLA -> optimized bufferized
/// Julia dialect pipeline.
void buildStableHLOToJuliaPipeline(
    OpPassManager &pm, const StableHLOToJuliaPipelineOptions &options);

/// Register `--shlo-to-julia-pipeline` with MLIR's pass pipeline registry.
void registerStableHLOToJuliaPipeline();

} // namespace mlir::enzymexla::julia

#endif // ENZYMEXLA_PASSES_JULIA_PASSES_H
