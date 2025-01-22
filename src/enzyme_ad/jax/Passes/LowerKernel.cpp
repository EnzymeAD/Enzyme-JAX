//===- PrintPass.cpp - Print the MLIR module                     ------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to print the MLIR module
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "src/enzyme_ad/jax/Passes/PassDetails.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Passes/EnzymeHLOPatterns.h"
#include "src/enzyme_ad/jax/Passes/PassDetails.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/RegisterEHFrames.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Target/LLVMIR/Export.h"

#define DEBUG_TYPE "lower-kernel"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::gpu;
using namespace enzyme;
using namespace mlir::enzymexla;
using namespace enzymexla;

using namespace stablehlo;

namespace {

void buildCommonPassPipeline(
    OpPassManager &pm, const mlir::gpu::GPUToNVVMPipelineOptions &options) {
  pm.addPass(createConvertNVGPUToNVVMPass());
  pm.addPass(createGpuKernelOutliningPass());
  pm.addPass(createConvertVectorToSCFPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createConvertNVVMToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(memref::createExpandStridedMetadataPass());

  GpuNVVMAttachTargetOptions nvvmTargetOptions;
  nvvmTargetOptions.triple = options.cubinTriple;
  nvvmTargetOptions.chip = options.cubinChip;
  nvvmTargetOptions.features = options.cubinFeatures;
  nvvmTargetOptions.optLevel = options.optLevel;
  pm.addPass(createGpuNVVMAttachTarget(nvvmTargetOptions));
  pm.addPass(createLowerAffinePass());
  pm.addPass(createArithToLLVMConversionPass());
  ConvertIndexToLLVMPassOptions convertIndexToLLVMPassOpt;
  convertIndexToLLVMPassOpt.indexBitwidth = options.indexBitWidth;
  pm.addPass(createConvertIndexToLLVMPass(convertIndexToLLVMPassOpt));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

//===----------------------------------------------------------------------===//
// GPUModule-specific stuff.
//===----------------------------------------------------------------------===//
void buildGpuPassPipeline(OpPassManager &pm,
                          const mlir::gpu::GPUToNVVMPipelineOptions &options) {
  pm.addNestedPass<gpu::GPUModuleOp>(createStripDebugInfoPass());
  ConvertGpuOpsToNVVMOpsOptions opt;
  opt.useBarePtrCallConv = options.kernelUseBarePtrCallConv;
  opt.indexBitwidth = options.indexBitWidth;
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertGpuOpsToNVVMOps(opt));
  pm.addNestedPass<gpu::GPUModuleOp>(createCanonicalizerPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createReconcileUnrealizedCastsPass());
}

//===----------------------------------------------------------------------===//
// Host Post-GPU pipeline
//===----------------------------------------------------------------------===//
void buildHostPostPipeline(OpPassManager &pm,
                           const mlir::gpu::GPUToNVVMPipelineOptions &options,
                           std::string toolkitPath,
                           llvm::SmallVectorImpl<std::string> &linkFiles) {
  GpuToLLVMConversionPassOptions opt;
  opt.hostBarePtrCallConv = options.hostUseBarePtrCallConv;
  opt.kernelBarePtrCallConv = options.kernelUseBarePtrCallConv;
  pm.addPass(createGpuToLLVMConversionPass(opt));

  GpuModuleToBinaryPassOptions gpuModuleToBinaryPassOptions;
  gpuModuleToBinaryPassOptions.compilationTarget = options.cubinFormat;
  gpuModuleToBinaryPassOptions.toolkitPath = toolkitPath;
  gpuModuleToBinaryPassOptions.linkFiles.append(linkFiles);
  pm.addPass(createGpuModuleToBinaryPass(gpuModuleToBinaryPassOptions));
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

void buildLowerToNVVMPassPipeline(
    OpPassManager &pm, const GPUToNVVMPipelineOptions &options,
    std::string toolkitPath, llvm::SmallVectorImpl<std::string> &linkFiles) {
  // Common pipelines
  buildCommonPassPipeline(pm, options);

  // GPUModule-specific stuff
  buildGpuPassPipeline(pm, options);

  // Host post-GPUModule-specific stuff
  buildHostPostPipeline(pm, options, toolkitPath, linkFiles);
}

} // namespace

struct CallInfo {
  void (*run)(void *, void *, void **);
  void *(*init)();
};

llvm::StringMap<CallInfo> kernels;
llvm::sys::SmartRWMutex<true> kernel_mutex;
std::unique_ptr<llvm::orc::LLJIT> JIT = nullptr;

CallInfo CompileHostModule(std::string &key, mlir::ModuleOp modOp,
                           bool run_init, size_t *cuLaunchPtr) {
  if (!JIT) {
    auto tJIT =
        llvm::orc::LLJITBuilder()
            .setLinkProcessSymbolsByDefault(true)
            .setObjectLinkingLayerCreator(
                [](llvm::orc::ExecutionSession &ES, const llvm::Triple &OLL)
                    -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
                  auto obj = std::make_unique<
                      llvm::orc::RTDyldObjectLinkingLayer>(ES, []() {
                    return std::make_unique<llvm::SectionMemoryManager>();
                  });
                  if (getenv("ENABLE_GDBLISTENER")) {
                    auto list =
                        llvm::JITEventListener::createGDBRegistrationListener();
                    obj->registerJITEventListener(*list);
                  }
                  return obj;
                })
            .create();
    if (!tJIT) {
      llvm::errs() << " jit creating error: " << tJIT.takeError() << "\n";
      return {};
    }
    JIT = std::move(tJIT.get());
    assert(JIT);
    auto GlobalPrefix = JIT->getDataLayout().getGlobalPrefix();

    llvm::orc::DynamicLibrarySearchGenerator::SymbolPredicate Pred;

    auto ProcessSymsGenerator =
        llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            GlobalPrefix, Pred);

    if (!ProcessSymsGenerator) {
      llvm::errs() << " failure creating symbol generator: "
                   << ProcessSymsGenerator.takeError() << "\n";
      return {};
    }

    JIT->getMainJITDylib().addGenerator(std::move(ProcessSymsGenerator.get()));
  }

  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext);
  auto llvmModule = translateModuleToLLVMIR(modOp, *ctx);
  if (!llvmModule) {
    llvm::errs() << "modOp: " << *modOp << "\n";
    llvm::errs() << "could not convert to LLVM IR\n";
    return {};
  }
  llvmModule->setDataLayout(JIT->getDataLayout());
  llvmModule->setTargetTriple(JIT->getTargetTriple().getTriple());

  auto LibA =
      JIT->createJITDylib("enzymecudadl_" + std::to_string(kernels.size()));
  if (auto Err = JIT->addIRModule(
          LibA.get(),
          llvm::orc::ThreadSafeModule(std::move(llvmModule), std::move(ctx)))) {
    llvm::errs() << " addIRModuleError " << Err << "\n";
    return {};
  }

  if (cuLaunchPtr[0] == 0) {
    // Look up the JIT'd code entry point.
    auto LaunchSym = JIT->lookup(LibA.get(), "cuLaunchKernel");
    if (!LaunchSym) {
      llvm::errs() << " lookupError[cuLaunchKernel] " << LaunchSym.takeError()
                   << "\n";
      return {};
    }
    *cuLaunchPtr = (size_t)LaunchSym->getValue();
  }

  auto NVSym = JIT->lookup(LibA.get(), "nv_func_init");
  if (!NVSym) {
    llvm::errs() << " lookupError " << NVSym.takeError() << "\n";
    return {};
  }

  auto nvptr = (void *)NVSym->getValue();

  auto Entry = JIT->lookup(LibA.get(), "entry");
  if (!Entry) {
    llvm::errs() << " lookupError " << Entry.takeError() << "\n";
    return {};
  }

  auto ptr = (void *)Entry->getValue();

  return CallInfo{(void (*)(void *, void *, void **))ptr, (void *(*)())nvptr};
}

gpu::ObjectAttr getSelectedObject(gpu::BinaryOp op) {
  ArrayRef<Attribute> objects = op.getObjectsAttr().getValue();

  // Obtain the index of the object to select.
  int64_t index = -1;
  if (Attribute target =
          cast<gpu::SelectObjectAttr>(op.getOffloadingHandlerAttr())
              .getTarget()) {
    // If the target attribute is a number it is the index. Otherwise compare
    // the attribute to every target inside the object array to find the index.
    if (auto indexAttr = mlir::dyn_cast<IntegerAttr>(target)) {
      index = indexAttr.getInt();
    } else {
      for (auto [i, attr] : llvm::enumerate(objects)) {
        auto obj = mlir::dyn_cast<gpu::ObjectAttr>(attr);
        if (obj.getTarget() == target) {
          index = i;
        }
      }
    }
  } else {
    // If the target attribute is null then it's selecting the first object in
    // the object array.
    index = 0;
  }

  if (index < 0 || index >= static_cast<int64_t>(objects.size())) {
    op->emitError("the requested target object couldn't be found");
    return nullptr;
  }
  return mlir::dyn_cast<gpu::ObjectAttr>(objects[index]);
}

CallInfo CompileKernel(SymbolTableCollection &symbolTable, mlir::Location loc,
                       FunctionOpInterface op, bool jit, size_t gridx,
                       size_t gridy, size_t gridz, size_t blockx, size_t blocky,
                       size_t blockz, size_t shmem, std::string toolkitPath,
                       llvm::SmallVectorImpl<std::string> &linkFiles,
                       int indexBitWidth, std::string cubinChip,
                       std::string cubinFeatures, size_t cuLaunchKernelPtr,
                       size_t cuModuleLoadDataPtr,
                       size_t cuModuleGetFunctionPtr, bool compileLaunch,
                       bool run_init, enzymexla::KernelCallOp kernelCallOp,
                       bool debug, size_t cuResultHandlerPtr,
                       size_t cuStreamSynchronizePtr, std::string cubinFormat,
                       int cuOptLevel, std::string cubinTriple) {

  OpBuilder builder(op);

  auto ptrty = LLVM::LLVMPointerType::get(builder.getContext());
  mlir::Type intys[] = {ptrty, ptrty, ptrty};
  FunctionType calleeType = builder.getFunctionType(intys, {});
  mlir::Type intys2[] = {ptrty, ptrty};
  FunctionType printType = builder.getFunctionType(intys2, {});

  FunctionType gpuTy0 = dyn_cast<FunctionType>(op.getFunctionType());
  if (!gpuTy0) {
    if (auto lty = dyn_cast<LLVM::LLVMFunctionType>(op.getFunctionType())) {
      gpuTy0 = builder.getFunctionType(lty.getParams(), {});
    } else {
      op.emitError(
          "Require target operand to have functiontype or llvmfunctiontype");
      return {};
    }
  }
  SmallVector<Type, 1> newParams;
  for (Type p : gpuTy0.getInputs()) {
    if (auto AT = dyn_cast<LLVM::LLVMArrayType>(p)) {
      p = AT.getElementType();
    }
    newParams.push_back(p);
  }
  FunctionType gpuTy = builder.getFunctionType(newParams, {});

  static size_t id = 0;
  id++;
  auto submod = builder.create<ModuleOp>(loc, "offload" + std::to_string(id));
  submod->setAttr("gpu.container_module", builder.getUnitAttr());
  builder.setInsertionPointToStart(&submod.getBodyRegion().front());

  auto gpumod = builder.create<gpu::GPUModuleOp>(loc, "gpumodname");
  builder.setInsertionPointToStart(&gpumod.getBodyRegion().front());

  std::string legalName = op.getName().str();
  std::replace(legalName.begin(), legalName.end(), '#', '_');
  auto gpufunc = builder.create<gpu::GPUFuncOp>(loc, legalName, gpuTy);
  {
    auto entry = &gpufunc.getBody().front();
    builder.setInsertionPointToEnd(entry);
    IRMapping map;
    for (auto &&[oldarg, newarg] :
         zip(op.getArguments(), gpufunc.getArguments())) {
      Value newval = newarg;

      if (auto AT = dyn_cast<LLVM::LLVMArrayType>(oldarg.getType())) {
        auto ud =
            builder.create<LLVM::UndefOp>(newarg.getLoc(), oldarg.getType());
        int64_t c0[1] = {0};
        newval = builder.create<LLVM::InsertValueOp>(
            newarg.getLoc(), oldarg.getType(), ud, newval, c0);
      }

      map.map(oldarg, newval);
    }

    op.getFunctionBody().cloneInto(&gpufunc.getBody(), map);
    gpufunc->setAttr("gpu.kernel", builder.getUnitAttr());

    auto second = entry->getNextNode();
    entry->getOperations().splice(entry->getOperations().end(),
                                  second->getOperations());

    second->erase();

    gpufunc->walk([](LLVM::ReturnOp op) {
      OpBuilder rewriter(op);
      rewriter.create<gpu::ReturnOp>(op.getLoc());
      op.erase();
    });

    gpufunc->walk([](LLVM::UnreachableOp op) {
      OpBuilder rewriter(op);
      rewriter.create<gpu::ReturnOp>(op.getLoc());
      op.erase();
    });

    gpufunc->walk([](func::ReturnOp op) {
      OpBuilder rewriter(op);
      rewriter.create<gpu::ReturnOp>(op.getLoc());
      op.erase();
    });
  }
  SmallVector<Operation *> tocopy;
  op->walk([&](CallOpInterface cop) {
    if (auto op2 = cop.resolveCallable())
      tocopy.push_back(op2);
  });
  op->walk([&](LLVM::AddressOfOp cop) {
    if (auto op2 = cop.getGlobal(symbolTable))
      tocopy.push_back(op2);
    else if (auto op2 = cop.getFunction(symbolTable))
      tocopy.push_back(op2);
  });
  SmallPtrSet<Operation *, 1> done;

  builder.setInsertionPointToStart(&gpumod.getBodyRegion().front());
  while (tocopy.size()) {
    auto cur = tocopy.pop_back_val();
    if (done.count(cur))
      continue;
    done.insert(cur);
    builder.clone(*cur);
    cur->walk([&](CallOpInterface cop) {
      if (auto op2 = cop.resolveCallable())
        tocopy.push_back(op2);
    });
    cur->walk([&](LLVM::AddressOfOp cop) {
      if (auto op2 = cop.getGlobal(symbolTable))
        tocopy.push_back(op2);
      else if (auto op2 = cop.getFunction(symbolTable))
        tocopy.push_back(op2);
    });
  }

  builder.setInsertionPointToEnd(&submod.getBodyRegion().front());

  auto func = builder.create<func::FuncOp>(loc, "entry", calleeType);

  auto &entryBlock = *func.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  mlir::Value cufunc = entryBlock.getArgument(0);
  mlir::Value stream = entryBlock.getArgument(1);
  mlir::Value buffers = entryBlock.getArgument(2);

  auto idx = builder.getIntegerType(64);
  auto i32 = builder.getIntegerType(32);
  gpu::KernelDim3 gridSize{
      builder.create<arith::ConstantIntOp>(loc, gridx, idx),
      builder.create<arith::ConstantIntOp>(loc, gridy, idx),
      builder.create<arith::ConstantIntOp>(loc, gridz, idx),
  };

  gpu::KernelDim3 blockSize{
      builder.create<arith::ConstantIntOp>(loc, blockx, idx),
      builder.create<arith::ConstantIntOp>(loc, blocky, idx),
      builder.create<arith::ConstantIntOp>(loc, blockz, idx),
  };

  SmallVector<mlir::Value> arguments;
  for (auto arg : op.getArguments()) {
    LLVM::GEPArg args[1] = {arg.getArgNumber()};
    auto gep =
        builder.create<LLVM::GEPOp>(loc, ptrty, ptrty, buffers, args, true);
    auto argTy = arg.getType();
    if (auto AT = dyn_cast<LLVM::LLVMArrayType>(argTy)) {
      argTy = AT.getElementType();
    }
    auto ld = builder.create<LLVM::LoadOp>(loc, argTy, gep);
    arguments.push_back(ld);
  }
  auto dynshmem = builder.create<arith::ConstantIntOp>(loc, shmem, i32);

  stream = builder
               .create<UnrealizedConversionCastOp>(
                   loc, gpu::AsyncTokenType::get(stream.getContext()), stream)
               ->getResult(0);

  builder.create<gpu::LaunchFuncOp>(loc, gpufunc, gridSize, blockSize, dynshmem,
                                    arguments, stream.getType(),
                                    ValueRange(stream));

  builder.create<mlir::func::ReturnOp>(loc);

  std::string modstr;
  llvm::raw_string_ostream ss(modstr);

  ss << submod;

  if (!jit)
    return {};

  CallInfo ptr;
  {
    llvm::sys::SmartScopedWriter<true> lock(kernel_mutex);

    auto found = kernels.find(ss.str());
    if (found != kernels.end()) {
      ptr = found->second;
    } else {
      // mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);

      PassManager pm(submod.getContext());
      mlir::gpu::GPUToNVVMPipelineOptions options;
      options.indexBitWidth = indexBitWidth;
      options.cubinTriple = cubinTriple;
      options.cubinChip = cubinChip;
      options.cubinFeatures = cubinFeatures;
      options.cubinFormat = cubinFormat;
      options.optLevel = cuOptLevel;
      options.kernelUseBarePtrCallConv = false;
      options.hostUseBarePtrCallConv = false;
      buildLowerToNVVMPassPipeline(pm, options, toolkitPath, linkFiles);

      auto subres = pm.run(submod);
      if (!subres.succeeded()) {
        return {};
      }

      OpBuilder builder(submod);

      builder.setInsertionPointToStart(&submod.getBodyRegion().front());
      auto ptrty = LLVM::LLVMPointerType::get(builder.getContext());
      auto i64 = builder.getIntegerType(64);
      auto i32 = builder.getIntegerType(32);
      auto idx = i64;
      auto voidty = LLVM::LLVMVoidType::get(submod.getContext());

      mlir::Type cumodtys[] = {ptrty, ptrty};
      auto modload_ty = LLVM::LLVMFunctionType::get(i32, cumodtys);
      LLVM::LLVMFuncOp modload =
          builder.create<LLVM::LLVMFuncOp>(loc, "cuModuleLoadData", modload_ty);

      mlir::Type cutys[] = {ptrty, idx, idx,   idx,   idx,  idx,
                            idx,   i32, ptrty, ptrty, ptrty};

      auto launch_ty = LLVM::LLVMFunctionType::get(i32, cutys);
      mlir::Type curesulttys[] = {i32};
      auto curesult_handler_ty =
          LLVM::LLVMFunctionType::get(voidty, curesulttys);
      LLVM::LLVMFuncOp launch =
          builder.create<LLVM::LLVMFuncOp>(loc, "cuLaunchKernel", launch_ty);
      auto cusync_ty = LLVM::LLVMFunctionType::get(i32, {ptrty});

      mlir::Type cufunctys[] = {ptrty, ptrty, ptrty};
      auto funcload_ty = LLVM::LLVMFunctionType::get(i32, cufunctys);
      LLVM::LLVMFuncOp funcload = builder.create<LLVM::LLVMFuncOp>(
          loc, "cuModuleGetFunction", funcload_ty);

      LLVM::GlobalOp kernStr;
      {
        auto type = LLVM::LLVMArrayType::get(
            mlir::IntegerType::get(builder.getContext(), 8),
            legalName.size() + 1);
        kernStr = builder.create<LLVM::GlobalOp>(
            loc, type, /*isConstant=*/true, LLVM::Linkage::Internal, "str",
            builder.getStringAttr(legalName + '\0'));
      }

      builder.setInsertionPointToStart(&submod.getBodyRegion().front());

      LLVM::LLVMFuncOp initfn = builder.create<LLVM::LLVMFuncOp>(
          loc, "nv_func_init", LLVM::LLVMFunctionType::get(ptrty, {}, false),
          LLVM::Linkage::External);

      LLVM::LLVMFuncOp printfunc = nullptr;
      LLVM::LLVMFuncOp putfunc = nullptr;

      if (debug) {
        printfunc = builder.create<LLVM::LLVMFuncOp>(
            loc, "printf",
            LLVM::LLVMFunctionType::get(ptrty, {ptrty, ptrty}, false),
            LLVM::Linkage::External);
        printfunc.setVisibility(SymbolTable::Visibility::Private);
        putfunc = builder.create<LLVM::LLVMFuncOp>(
            loc, "puts", LLVM::LLVMFunctionType::get(voidty, {ptrty}, false),
            LLVM::Linkage::External);
        putfunc.setVisibility(SymbolTable::Visibility::Private);
      }

      LLVM::GlobalOp loadModuleStr = nullptr;
      if (debug) {
        std::string value = "load Module result = %d\n";
        auto type = LLVM::LLVMArrayType::get(
            mlir::IntegerType::get(builder.getContext(), 8), value.size() + 1);
        loadModuleStr = builder.create<LLVM::GlobalOp>(
            loc, type, /*isConstant=*/true, LLVM::Linkage::Internal, "strmod",
            builder.getStringAttr(value + '\0'));
      }
      LLVM::GlobalOp loadFuncStr = nullptr;
      if (debug) {
        std::string value = "load Func result = %d\n";
        auto type = LLVM::LLVMArrayType::get(
            mlir::IntegerType::get(builder.getContext(), 8), value.size() + 1);
        loadFuncStr = builder.create<LLVM::GlobalOp>(
            loc, type, /*isConstant=*/true, LLVM::Linkage::Internal, "strfunc",
            builder.getStringAttr(value + '\0'));
      }
      LLVM::GlobalOp launchKernelStr = nullptr;
      if (debug) {
        std::string value = "launch Kernel result = %d\n";
        auto type = LLVM::LLVMArrayType::get(
            mlir::IntegerType::get(builder.getContext(), 8), value.size() + 1);
        launchKernelStr = builder.create<LLVM::GlobalOp>(
            loc, type, /*isConstant=*/true, LLVM::Linkage::Internal,
            "strlaunch", builder.getStringAttr(value + '\0'));
      }
      LLVM::GlobalOp modOpStr = nullptr;
      if (debug) {
        std::string opstr;
        llvm::raw_string_ostream ss(opstr);

        ss << kernelCallOp;
        std::string value = "modstr=" + modstr + "\n" + opstr + "\n\n";
        auto type = LLVM::LLVMArrayType::get(
            mlir::IntegerType::get(builder.getContext(), 8), value.size() + 1);
        modOpStr = builder.create<LLVM::GlobalOp>(
            loc, type, /*isConstant=*/true, LLVM::Linkage::Internal,
            "strmlirmod", builder.getStringAttr(value + '\0'));
      }

      LLVM::GlobalOp binary = nullptr;
      submod.walk([&](gpu::BinaryOp op) {
        gpu::ObjectAttr object = getSelectedObject(op);
        auto value = object.getObject().getValue();
        auto type = LLVM::LLVMArrayType::get(
            mlir::IntegerType::get(builder.getContext(), 8), value.size());
        binary = builder.create<LLVM::GlobalOp>(
            loc, type, /*isConstant=*/true, LLVM::Linkage::Internal, "binary",
            builder.getStringAttr(value));

        if (object.getProperties()) {
          if (auto section = mlir::dyn_cast_or_null<mlir::StringAttr>(
                  object.getProperties().get("section"))) {
            binary.setSectionAttr(section);
          }
        }

        binary.setAlignmentAttr(builder.getI64IntegerAttr(8));
        binary.setUnnamedAddrAttr(LLVM::UnnamedAddrAttr::get(
            builder.getContext(), mlir::LLVM::UnnamedAddr::None));
        op.erase();
      });
      if (!binary) {
        llvm::errs() << "could not find binary object in submod:\n"
                     << *submod << "\n";
        assert(binary);
      }

      {
        auto blk = new Block();
        initfn.getRegion().push_back(blk);
        builder.setInsertionPointToEnd(blk);

        auto one = builder.create<LLVM::ConstantOp>(
            loc, i64, builder.getI64IntegerAttr(1));
        auto modptr = builder.create<LLVM::AllocaOp>(loc, ptrty, ptrty, one);
        auto funcptr = builder.create<LLVM::AllocaOp>(loc, ptrty, ptrty, one);

        auto addr_modbin = builder.create<LLVM::AddressOfOp>(loc, binary);
        SmallVector<mlir::Value> modargs = {modptr->getResult(0),
                                            addr_modbin->getResult(0)};

        mlir::Value loadModRes = nullptr;
        if (cuModuleLoadDataPtr) {
          auto addr_glob_int = builder.create<LLVM::ConstantOp>(
              loc, i64, builder.getI64IntegerAttr(cuModuleLoadDataPtr));
          auto addr_glob =
              builder.create<LLVM::IntToPtrOp>(loc, ptrty, addr_glob_int);
          modargs.insert(modargs.begin(), addr_glob);
          loadModRes = builder.create<LLVM::CallOp>(loc, modload_ty, modargs)
                           ->getResult(0);
        } else {
          loadModRes =
              builder.create<LLVM::CallOp>(loc, modload, modargs)->getResult(0);
        }

        if (debug) {
          Value printargs1[] = {
              builder.create<LLVM::AddressOfOp>(loc, loadModuleStr)
                  ->getResult(0),
              builder.create<LLVM::IntToPtrOp>(loc, ptrty, loadModRes)
                  ->getResult(0)};
          builder.create<LLVM::CallOp>(loc, printfunc, printargs1);
        }
        if (cuResultHandlerPtr) {
          auto addr_glob_int = builder.create<LLVM::ConstantOp>(
              loc, i64, builder.getI64IntegerAttr(cuResultHandlerPtr));
          auto addr_glob =
              builder.create<LLVM::IntToPtrOp>(loc, ptrty, addr_glob_int)
                  ->getResult(0);
          mlir::Value args[2] = {addr_glob, loadModRes};
          builder.create<LLVM::CallOp>(loc, curesult_handler_ty, args);
        }

        auto mod = builder.create<LLVM::LoadOp>(loc, ptrty, modptr);

        auto addr_kernstr =
            builder.create<LLVM::AddressOfOp>(loc, ptrty, "str");

        SmallVector<mlir::Value> funcargs = {funcptr->getResult(0),
                                             mod->getResult(0),
                                             addr_kernstr->getResult(0)};
        mlir::Value loadFuncRes = nullptr;
        if (cuModuleGetFunctionPtr) {
          auto addr_glob_int = builder.create<LLVM::ConstantOp>(
              loc, i64, builder.getI64IntegerAttr(cuModuleGetFunctionPtr));
          auto addr_glob =
              builder.create<LLVM::IntToPtrOp>(loc, ptrty, addr_glob_int);
          funcargs.insert(funcargs.begin(), addr_glob);
          loadFuncRes =
              builder.create<LLVM::CallOp>(loc, funcload_ty, funcargs)
                  ->getResult(0);
        } else {
          loadFuncRes = builder.create<LLVM::CallOp>(loc, funcload, funcargs)
                            ->getResult(0);
        }

        if (debug) {
          Value printargs1[] = {
              builder.create<LLVM::AddressOfOp>(loc, loadFuncStr)->getResult(0),
              builder.create<LLVM::IntToPtrOp>(loc, ptrty, loadFuncRes)
                  ->getResult(0)};
          builder.create<LLVM::CallOp>(loc, printfunc, printargs1);
        }
        if (cuResultHandlerPtr) {
          auto addr_glob_int = builder.create<LLVM::ConstantOp>(
              loc, i64, builder.getI64IntegerAttr(cuResultHandlerPtr));
          auto addr_glob =
              builder.create<LLVM::IntToPtrOp>(loc, ptrty, addr_glob_int)
                  ->getResult(0);
          mlir::Value args[2] = {addr_glob, loadFuncRes};
          builder.create<LLVM::CallOp>(loc, curesult_handler_ty, args);
        }

        auto func = builder.create<LLVM::LoadOp>(loc, ptrty, funcptr);

        builder.create<LLVM::ReturnOp>(loc, ValueRange(func));
      }

      submod.walk([&](gpu::LaunchFuncOp op) {
        builder.setInsertionPoint(op);
        auto pfunc = op->getParentOfType<LLVM::LLVMFuncOp>();
        mlir::Value cufunc = pfunc.getBody().begin()->getArgument(0);

        auto ldop =
            op.getKernelOperands().front().getDefiningOp<LLVM::LoadOp>();
        assert(ldop);
        auto params = ldop.getOperand();

        llvm::SmallVector<mlir::Value> args = {
            cufunc,
            op.getGridSizeX(),
            op.getGridSizeY(),
            op.getGridSizeZ(),
            op.getBlockSizeX(),
            op.getBlockSizeY(),
            op.getBlockSizeZ(),
            op.getDynamicSharedMemorySize(),
            op.getAsyncObject(),
            params,
            builder.create<LLVM::ZeroOp>(loc, ptrty)};

        Value kernRes;
        if (cuLaunchKernelPtr) {
          auto addr_glob_int = builder.create<LLVM::ConstantOp>(
              loc, i64, builder.getI64IntegerAttr(cuLaunchKernelPtr));
          auto addr_glob =
              builder.create<LLVM::IntToPtrOp>(loc, ptrty, addr_glob_int);
          args.insert(args.begin(), addr_glob);
          kernRes =
              builder.create<LLVM::CallOp>(loc, launch_ty, args)->getResult(0);
        } else {
          kernRes =
              builder.create<LLVM::CallOp>(loc, launch, args)->getResult(0);
        }
        if (debug) {
          Value printargs1[] = {
              builder.create<LLVM::AddressOfOp>(loc, launchKernelStr)
                  ->getResult(0),
              builder.create<LLVM::IntToPtrOp>(loc, ptrty, kernRes)
                  ->getResult(0)};
          builder.create<LLVM::CallOp>(loc, printfunc, printargs1);
        }
        if (debug) {
          Value printargs1[] = {
              builder.create<LLVM::AddressOfOp>(loc, modOpStr)->getResult(0)};
          builder.create<LLVM::CallOp>(loc, putfunc, printargs1);
        }
        if (cuResultHandlerPtr) {
          auto addr_glob_int = builder.create<LLVM::ConstantOp>(
              loc, i64, builder.getI64IntegerAttr(cuResultHandlerPtr));
          auto addr_glob =
              builder.create<LLVM::IntToPtrOp>(loc, ptrty, addr_glob_int)
                  ->getResult(0);
          mlir::Value args[2] = {addr_glob, kernRes};
          builder.create<LLVM::CallOp>(loc, curesult_handler_ty, args);
        }

        if (cuStreamSynchronizePtr) {
          auto addr_glob_int = builder.create<LLVM::ConstantOp>(
              loc, i64, builder.getI64IntegerAttr(cuStreamSynchronizePtr));
          auto addr_glob =
              builder.create<LLVM::IntToPtrOp>(loc, ptrty, addr_glob_int);
          mlir::Value args[2] = {addr_glob, op.getAsyncObject()};
          auto syncRes =
              builder.create<LLVM::CallOp>(loc, cusync_ty, args)->getResult(0);

          if (debug) {
            Value printargs1[] = {
                builder.create<LLVM::AddressOfOp>(loc, modOpStr)->getResult(0)};
            builder.create<LLVM::CallOp>(loc, putfunc, printargs1);
          }
          if (cuResultHandlerPtr) {
            auto addr_glob_int = builder.create<LLVM::ConstantOp>(
                loc, i64, builder.getI64IntegerAttr(cuResultHandlerPtr));
            auto addr_glob =
                builder.create<LLVM::IntToPtrOp>(loc, ptrty, addr_glob_int)
                    ->getResult(0);
            mlir::Value args[2] = {addr_glob, syncRes};
            builder.create<LLVM::CallOp>(loc, curesult_handler_ty, args);
          }
        }

        op.erase();
        ldop.erase();
      });

      if (!compileLaunch)
        return {};

      ptr = CompileHostModule(ss.str(), submod, run_init, &cuLaunchKernelPtr);
      kernels[ss.str()] = ptr;

      submod.erase();
    }
  }

  return ptr;
};

namespace {

struct LowerKernelPass : public LowerKernelPassBase<LowerKernelPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    OpPassManager pm;
    mlir::gpu::GPUToNVVMPipelineOptions options;
    options.indexBitWidth = 64;
    options.cubinTriple = "nvptx64-nvidia-cuda";
    options.cubinChip = "sm_50";
    options.cubinFeatures = "+ptx60";
    options.cubinFormat = "fatbin";
    options.optLevel = 2;
    options.kernelUseBarePtrCallConv = false;
    options.hostUseBarePtrCallConv = false;
    std::string toolkitPath = "";
    SmallVector<std::string> linkFiles;
    buildLowerToNVVMPassPipeline(pm, options, toolkitPath, linkFiles);
    pm.getDependentDialects(registry);

    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::math::MathDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect, mlir::vector::VectorDialect,
                    mlir::gpu::GPUDialect, mlir::nvgpu::NVGPUDialect,
                    mlir::NVVM::NVVMDialect, mlir::LLVM::LLVMDialect>();
  }

  SmallVector<std::string> parseLinkFilesString(StringRef inp) {
    if (inp.size() == 0)
      return {};
    SmallVector<StringRef, 1> split;
    SmallVector<std::string> out;
    StringRef(inp.data(), inp.size()).split(split, ';');
    for (auto &str : split) {
      out.push_back(str.str());
    }
    return out;
  }

  void runOnOperation() override {
    auto context = getOperation()->getContext();

    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(getOperation());

    llvm::SmallVector<std::string> linkFilesArray =
        parseLinkFilesString(linkFiles.getValue());
    getOperation()->walk([&](KernelCallOp op) {
      mlir::ArrayAttr operand_layouts =
          op.getOperandLayouts()
              ? cast<mlir::ArrayAttr>(*op.getOperandLayouts())
              : nullptr;
      mlir::ArrayAttr result_layouts =
          op.getResultLayouts() ? cast<mlir::ArrayAttr>(*op.getResultLayouts())
                                : nullptr;
      mlir::ArrayAttr output_operand_aliases = op.getOutputOperandAliases();

      size_t data[8];

      auto *symbolOp = symbolTable.lookupNearestSymbolFrom(op, op.getFnAttr());
      auto fn = cast<FunctionOpInterface>(symbolOp);
      if (fn.getArguments().size() != op.getInputs().size()) {
        op->emitError() << "Kernel_call had " << op.getInputs().size()
                        << " whereas called kernel requires "
                        << fn.getArguments().size() << "\n";
        return;
      }

      Value vals[] = {op.getGridx(),  op.getGridy(),  op.getGridz(),
                      op.getBlockx(), op.getBlocky(), op.getBlockz(),
                      op.getShmem()};
      for (auto en : llvm::enumerate(vals)) {
        DenseIntElementsAttr stepAttr;
        if (!matchPattern(en.value(), m_Constant(&stepAttr))) {
          op->emitError() << "Cannot lower kernel with a grid/block size which "
                             "is not a constant integer tensor";
          return;
        }
        if (stepAttr.size() != 1) {
          op->emitError() << "Cannot lower kernel with a grid/block size which "
                             "is not a constant integer tensor of size 1";
          return;
        }
        auto val = (*stepAttr.begin()).getZExtValue();
        data[1 + en.index()] = val;
      }

      // Compiled kernel goes here once ready
      CallInfo cdata = CompileKernel(
          symbolTable, op.getLoc(), fn, jit, data[1], data[2], data[3], data[4],
          data[5], data[6], data[7], toolkitPath.getValue(), linkFilesArray,
          indexBitWidth.getValue(), cubinChip.getValue(),
          cubinFeatures.getValue(), cuLaunchKernelPtr, cuModuleLoadDataPtr,
          cuModuleGetFunctionPtr, compileLaunch, run_init, op, debug,
          cuResultHandlerPtr, cuStreamSynchronizePtr, cubinFormat, cuOptLevel,
          cubinTriple);

      std::string backendinfo((char *)&cdata, sizeof(CallInfo));

      OpBuilder rewriter(op);

      SmallVector<NamedAttribute> names;
      names.push_back(NamedAttribute(rewriter.getStringAttr("attr"),
                                     rewriter.getStringAttr(backendinfo)));
      auto dattr = DictionaryAttr::get(op.getContext(), names);

      auto replacement = rewriter.create<stablehlo::CustomCallOp>(
          op.getLoc(), op.getResultTypes(), op.getInputs(),
          rewriter.getStringAttr("enzymexla_compile_gpu"),
          /* has_side_effect*/ rewriter.getBoolAttr(false),
          /*backend_config*/ dattr,
          /* api_version*/
          CustomCallApiVersionAttr::get(
              rewriter.getContext(),
              mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
          /*calledcomputations*/ nullptr, operand_layouts, result_layouts,
          output_operand_aliases);

      op.replaceAllUsesWith(replacement);
      op.erase();
    });
  }
};

} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createLowerKernelPass() {
  return std::make_unique<LowerKernelPass>();
}
} // namespace enzyme
} // namespace mlir
