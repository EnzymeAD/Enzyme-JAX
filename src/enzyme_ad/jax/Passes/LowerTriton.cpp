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
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"

#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"

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

namespace cuda {

namespace fs = std::filesystem;

absl::StatusOr<std::unique_ptr<llvm::TargetMachine>>
CreateTargetMachine(llvm::Module *module, absl::string_view arch_name,
                    bool enable_fp_fusion, absl::string_view features) {
  // Based on createTargetMachine() in triton/python/src/llvm.cc
  std::string error;
  const auto *target =
      llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);
  if (target == nullptr) {
    return absl::InternalError(
        absl::StrFormat("Failed to lookup LLVM target based on triple %s: %s",
                        module->getTargetTriple().str(), error));
  }
  llvm::TargetOptions opt;
  if (enable_fp_fusion) {
    opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  }
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  opt.TrapUnreachable = true;
  opt.MCOptions.AsmVerbose = true;
  opt.MCOptions.PreserveAsmComments = true;
  return std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
      module->getTargetTriple(), arch_name, features, opt, llvm::Reloc::PIC_,
      std::nullopt, llvm::CodeGenOptLevel::Aggressive));
}

absl::Status LinkLibdevice(llvm::Module *module, std::string libdevice_dir) {
  auto libdevice_path = (fs::path(libdevice_dir) / "libdevice.10.bc").string();

  llvm::LLVMContext &ctx = module->getContext();
  llvm::SMDiagnostic err;
  std::unique_ptr<llvm::Module> libdevice_module =
      llvm::parseIRFile(libdevice_path, err, ctx);
  if (!libdevice_module) {
    return absl::InternalError(
        absl::StrFormat("Failed to parse libdevice IR file at %s: %s",
                        libdevice_path, err.getMessage()));
  }

  llvm::Linker linker(*module);
  if (linker.linkInModule(std::move(libdevice_module),
                          llvm::Linker::Flags::LinkOnlyNeeded)) {
    return absl::InternalError("Failed to link libdevice");
  }

  return absl::OkStatus();
}

absl::StatusOr<std::string> LLVMToPTX(mlir::ModuleOp module,
                                      absl::string_view arch_name,
                                      std::string libdevice_dir) {
  // Based on translateLLVMIRToASM() in triton/python/src/llvm.cc
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  module.getContext()->appendDialectRegistry(registry);

  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    return absl::InternalError("Failed to emit LLVM IR");
  }

  auto cc = absl::StrReplaceAll(arch_name, {{".", ""}}); // "8.0" -> "80"
  auto proc = absl::StrCat("sm_", cc, cc == "90" ? "a" : "");
  // We cap the ISA at 8.4 to align with Triton.
  // See get_features() in triton/third_party/nvidia/backend/compiler.py.
  auto features = cc >= "84" ? "+ptx84" : "+ptx" + cc;
  llvmModule->setTargetTriple(llvm::Triple("nvptx64-nvidia-cuda"));
  static absl::once_flag init_target_once;
  absl::call_once(init_target_once, []() {
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
  });

  auto machineOrStatus =
      CreateTargetMachine(llvmModule.get(), proc,
                          /*enable_fp_fusion=*/false, features);
  if (!machineOrStatus.ok()) {
    return machineOrStatus.status();
  }
  auto machine = std::move(machineOrStatus.value());

  llvmModule->setDataLayout(machine->createDataLayout());

  auto needsLibdevice =
      llvm::any_of(llvmModule->functions(), [](const auto &f) {
        return !f.isIntrinsic() && f.isDeclaration() &&
               f.getName().starts_with("__nv_");
      });
  if (needsLibdevice) {
    auto linkStatus = LinkLibdevice(llvmModule.get(), libdevice_dir);
    if (!linkStatus.ok()) {
      return linkStatus;
    }
  }

  auto transformer = mlir::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/machine.get());
  if (auto error = transformer(llvmModule.get()); error) {
    return absl::InternalError("Failed to optimize LLVM IR");
  }

  std::string result;
  {
    llvm::raw_string_ostream stream(result);
    llvm::buffer_ostream bstream(stream);
    llvm::legacy::PassManager pm;
    machine->addPassesToEmitFile(pm, bstream, nullptr,
                                 llvm::CodeGenFileType::AssemblyFile,
                                 /*DisableVerify=*/false);
    if (!pm.run(*llvmModule)) {
      return absl::InternalError("Failed to compile LLVM IR to PTX");
    }
  }
  return result;
}

} // namespace cuda

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

      // remove divisibility attributes from the module before lowering to PTX
      // auto funcOpInterface = dyn_cast<FunctionOpInterface>(
      //     symbolTable.lookupNearestSymbolFrom(ttCallOp,
      //     ttCallOp.getFnAttr()));

      // if (!funcOpInterface) {
      //   innerMod->emitError("Failed to find function '") << ttCallOp.getFn()
      //   <<
      //                       "' in module";
      //   anyFailed = true;
      //   continue;
      // }

      // mlir::StringAttr divAttrName =
      // builder.getStringAttr("tt.divisibility"); for (size_t i = 0; i <
      // ttCallOp.getInputs().size(); ++i) {
      //   funcOpInterface.removeArgAttr(i, divAttrName);
      // }

      // auto ptxOrError =
      //     cuda::LLVMToPTX(innerMod, computeCapability, libdeviceDir);
      // if (!ptxOrError.ok()) {
      //   innerMod->emitError(ptxOrError.status().message());
      //   anyFailed = true;
      //   continue;
      // }

      // auto ptx = ptxOrError.value();
      // llvm::errs() << "Compilation result: " << ptx << "\n";

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
