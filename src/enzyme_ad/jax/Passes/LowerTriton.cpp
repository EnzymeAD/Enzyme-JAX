#include "src/enzyme_ad/jax/Passes/Passes.h"

#include <filesystem>

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

#include "xla/pjrt/triton.h"

#include "src/enzyme_ad/jax/Utils.h"

#include "llvm/ADT/DenseMap.h"
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

void collectTritonKernels(
    DenseMap<triton_ext::TritonCallOp, ModuleOp> &tritonKernels,
    SymbolTableCollection &symbolTable, triton_ext::TritonCallOp op) {
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

  tritonKernels[op] = wrappedMod;
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

    DenseMap<triton_ext::TritonCallOp, ModuleOp> tritonKernels;
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(modOp);
    modOp->walk([&](triton_ext::TritonCallOp op) {
      collectTritonKernels(tritonKernels, symbolTable, op);
    });

    SmallVector<mlir::triton::nvidia_gpu::ClusterInfo> clusterInfos;

    OpBuilder builder(modOp);

    bool anyFailed = false;
    for (auto [ttCallOp, innerMod] : tritonKernels) {
      int32_t numWarps = 4;
      if (innerMod->hasAttrOfType<IntegerAttr>("enzymexla.num_warps")) {
        numWarps = innerMod->getAttrOfType<IntegerAttr>("enzymexla.num_warps")
                       .getInt();
      }
      int32_t numCtas = 1;
      if (innerMod->hasAttrOfType<IntegerAttr>("enzymexla.num_ctas")) {
        numCtas =
            innerMod->getAttrOfType<IntegerAttr>("enzymexla.num_ctas").getInt();
      }
      int32_t numStages = 3;
      if (innerMod->hasAttrOfType<IntegerAttr>("enzymexla.num_stages")) {
        numStages = innerMod->getAttrOfType<IntegerAttr>("enzymexla.num_stages")
                        .getInt();
      }

      OpPassManager pm;

      xla::gpu::CreateTritonXlaPipeline(&pm, gpuCC, true, true, numStages);
      mlir::triton::nvidia_gpu::ClusterInfo clusterInfo;
      xla::gpu::CreateTritonPipeline(&pm, gpuCC, numWarps, numCtas, numStages,
                                     clusterInfo);
      clusterInfos.push_back(clusterInfo);

      if (failed(runPipeline(pm, innerMod))) {
        innerMod->emitError(
            "Failed to lower Triton kernel to TritonGPU kernel");
        anyFailed = true;
        continue;
      }

      int32_t threadsPerWarp = 32;
      if (innerMod->hasAttrOfType<IntegerAttr>("ttg.threads_per_warp")) {
        threadsPerWarp =
            innerMod->getAttrOfType<IntegerAttr>("ttg.threads_per_warp")
                .getInt();
      }

      builder.setInsertionPoint(ttCallOp);

      auto sharedMemSizeAttr =
          innerMod->getAttrOfType<IntegerAttr>("ttg.shared");
      auto sharedMemSize = sharedMemSizeAttr.getInt();
      auto shmemOpType = ttCallOp.getGridx().getType();
      auto shmemOp = stablehlo::ConstantOp::create(
          builder, ttCallOp.getLoc(), shmemOpType,
          cast<ElementsAttr>(makeAttr(shmemOpType, sharedMemSize)));

      auto blockX = stablehlo::ConstantOp::create(
          builder, ttCallOp.getLoc(), shmemOpType,
          cast<ElementsAttr>(makeAttr(shmemOpType, threadsPerWarp * numWarps)));
      auto blockYZ = stablehlo::ConstantOp::create(
          builder, ttCallOp.getLoc(), shmemOpType,
          cast<ElementsAttr>(makeAttr(shmemOpType, 1)));

      SmallVector<mlir::Value> newInputs(ttCallOp.getInputs().begin(),
                                         ttCallOp.getInputs().end());
      // we don't use the next 2 inputs
      auto scratchSpace = stablehlo::ConstantOp::create(
          builder, ttCallOp.getLoc(),
          RankedTensorType::get({}, builder.getI8Type()),
          cast<ElementsAttr>(
              makeAttr(RankedTensorType::get({}, builder.getI8Type()), 0)));
      newInputs.push_back(scratchSpace);
      newInputs.push_back(scratchSpace);

      auto kernelCallOp = enzymexla::KernelCallOp::create(
          builder, ttCallOp.getLoc(), ttCallOp.getResultTypes(),
          ttCallOp.getFn(), ttCallOp.getGridx(), ttCallOp.getGridy(),
          ttCallOp.getGridz(), blockX, blockYZ, blockYZ, shmemOp,
          ttCallOp.getClusterx(), ttCallOp.getClustery(),
          ttCallOp.getClusterz(), newInputs, ttCallOp.getBackendConfigAttr(),
          ttCallOp.getOperandLayoutsAttr(), ttCallOp.getResultLayoutsAttr(),
          ttCallOp.getArgAttrsAttr(), ttCallOp.getResAttrsAttr(),
          ttCallOp.getOutputOperandAliasesAttr(),
          ttCallOp.getXlaSideEffectFreeAttr());
      ttCallOp.replaceAllUsesWith(kernelCallOp);
      ttCallOp.erase();
    }

    if (anyFailed) {
      signalPassFailure();
      return;
    }
  }
};
