#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "xla/backends/gpu/codegen/triton/compilation_pipeline.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"

#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Dialect/TritonExt/Dialect.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

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

void collectTritonKernels(SmallVectorImpl<ModuleOp> &tritonKernels,
                          SymbolTableCollection &symbolTable,
                          triton_ext::TritonCallOp op) {
  auto funcOp = symbolTable.lookupNearestSymbolFrom(op, op.getFnAttr());
  if (!funcOp) {
    op->emitError() << "Failed to find function '" << op.getFn() << "' in "
                    << "module";
    return;
  }

  auto wrappedMod = funcOp->getParentOfType<ModuleOp>();
  if (!wrappedMod) {
    op->emitError() << "Failed to find parent built-in module.";
    return;
  }

  auto ttModOP = wrappedMod->getParentOfType<triton_ext::TritonModuleOp>();
  if (!ttModOP) {
    op->emitError() << "No `triton_ext.module` found!";
    return;
  }

  tritonKernels.push_back(wrappedMod);
  return;
}

struct LowerTritonPass
    : public mlir::enzyme::impl::LowerTritonPassBase<LowerTritonPass> {
  using Base::Base;

  void runOnOperation() override {
    auto modOp = getOperation();

    stream_executor::GpuComputeCapability gpuCC;
    if (backend == "cuda") {
      auto cudaCC =
          stream_executor::CudaComputeCapability::FromString(computeCapability);
      if (!cudaCC.ok()) {
        modOp->emitError("Unsupported cuda compute capability: ")
            << cudaCC.status().ToString();
        return;
      }
      gpuCC = stream_executor::GpuComputeCapability(cudaCC.value());
    } else if (backend == "rocm") {
      auto rocmCC = stream_executor::RocmComputeCapability(computeCapability);
      gpuCC = stream_executor::GpuComputeCapability(rocmCC);
    } else {
      modOp->emitError("Unsupported backend: ") << backend;
      return;
    }

    SmallVector<ModuleOp> tritonKernels;
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(modOp);
    modOp->walk([&](triton_ext::TritonCallOp op) {
      collectTritonKernels(tritonKernels, symbolTable, op);
    });

    OpPassManager pm;

    // TODO: bool rewrite_int4, bool allow_tma, int num_stages
    xla::gpu::CreateTritonXlaPipeline(&pm, gpuCC, false, true, 1);

    mlir::triton::nvidia_gpu::ClusterInfo out_cluster_info;
    // TODO: int num_warps, int num_ctas, int num_stages
    xla::gpu::CreateTritonPipeline(&pm, gpuCC, 4, 1, 1, out_cluster_info);

    for (auto tritonMod : tritonKernels) {
      if (failed(runPipeline(pm, tritonMod))) {
        tritonMod->emitError(
            "Failed to lower Triton kernel to TritonGPU kernel");
        signalPassFailure();
        return;
      }
    }
  }
};
