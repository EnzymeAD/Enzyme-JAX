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

typedef void XlaCustomCallStatus;

llvm::StringMap<void *> kernels;
llvm::sys::SmartRWMutex<true> kernel_mutex;
std::unique_ptr<llvm::orc::LLJIT> JIT = nullptr;

void *CompileHostModule(std::string &key, mlir::ModuleOp modOp) {
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
      return nullptr;
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
      return nullptr;
    }

    JIT->getMainJITDylib().addGenerator(std::move(ProcessSymsGenerator.get()));
  }

  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext);
  auto llvmModule = translateModuleToLLVMIR(modOp, *ctx);
  if (!llvmModule) {
    llvm::errs() << "could not convert to LLVM IR\n";
    return nullptr;
  }
  llvmModule->setDataLayout(JIT->getDataLayout());
  llvmModule->setTargetTriple(JIT->getTargetTriple().getTriple());

  auto LibA =
      JIT->createJITDylib("enzymecudadl_" + std::to_string(kernels.size()));
  if (auto Err = JIT->addIRModule(
          LibA.get(),
          llvm::orc::ThreadSafeModule(std::move(llvmModule), std::move(ctx)))) {
    llvm::errs() << " addIRModuleError " << Err << "\n";
    return nullptr;
  }

  // Look up the JIT'd code entry point.
  auto EntrySym = JIT->lookup(LibA.get(), "entry");
  if (!EntrySym) {
    llvm::errs() << " lookupError " << EntrySym.takeError() << "\n";
    return nullptr;
  }

  auto ptr = (void *)EntrySym->getValue();

  kernels[key] = ptr;
  return ptr;
}

// See API details at
// https://github.com/openxla/xla/blob/37fb0612d36ac3d08ff984b1d61e4bc4dedf4809/xla/service/hlo.proto#L73
extern "C" void EnzymeGPUCustomCall(void *__restrict__ stream,
                                    void **__restrict__ buffers,
                                    void **__restrict__ opaqueptr,
                                    size_t opaque_len,
                                    XlaCustomCallStatus *__restrict__ status) {
  auto ptr = (void (*)(void *, void **))(opaqueptr[0]);
  // auto ptr = (void(*)(void*, void**, size_t, size_t, size_t, size_t, size_t,
  // size_t)) (opaqueptr[0][0]);

  // size_t gridx = opaqueptr[0][1];
  // size_t gridy = opaqueptr[0][2];
  // size_t gridz = opaqueptr[0][3];

  // size_t blockx = opaqueptr[0][4];
  // size_t blocky = opaqueptr[0][5];
  // size_t blockz = opaqueptr[0][6];

  ptr(stream, buffers); //, gridx, gridy, gridz, blockx, blocky, blockz);
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

void *CompileKernel(SymbolTableCollection &symbolTable, mlir::Location loc,
                    FunctionOpInterface op, bool jit, size_t gridx,
                    size_t gridy, size_t gridz, size_t blockx, size_t blocky,
                    size_t blockz, size_t shmem, std::string toolkitPath,
                    llvm::SmallVectorImpl<std::string> &linkFiles,
                    int indexBitWidth, std::string cubinChip,
                    std::string cubinFeatures, size_t cuLaunchKernelPtr,
                    size_t cuModuleLoadDataPtr, size_t cuModuleGetFunctionPtr) {

  OpBuilder builder(op);

  auto ptrty = LLVM::LLVMPointerType::get(builder.getContext());
  mlir::Type intys[] = {ptrty, ptrty};
  FunctionType calleeType = builder.getFunctionType(intys, {});

  FunctionType gpuTy = dyn_cast<FunctionType>(op.getFunctionType());
  if (!gpuTy) {
    if (auto lty = dyn_cast<LLVM::LLVMFunctionType>(op.getFunctionType())) {
      gpuTy = builder.getFunctionType(lty.getParams(), {});
    } else {
      op.emitError(
          "Require target operand to have functiontype or llvmfunctiontype");
      return nullptr;
    }
  }

  auto submod = builder.create<ModuleOp>(loc, "offload");
  submod->setAttr("gpu.container_module", builder.getUnitAttr());
  builder.setInsertionPointToStart(&submod.getBodyRegion().front());

  auto gpumod = builder.create<gpu::GPUModuleOp>(loc, "gpumodname");
  builder.setInsertionPointToStart(&gpumod.getBodyRegion().front());

  auto gpufunc = builder.create<gpu::GPUFuncOp>(loc, "kernel", gpuTy);
  {
    IRMapping map;
    map.map(op.getArguments(), gpufunc.getArguments());
    op.getFunctionBody().cloneInto(&gpufunc.getBody(), map);
    gpufunc->setAttr("gpu.kernel", builder.getUnitAttr());

    auto entry = &gpufunc.getBody().front();
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
  }

  builder.setInsertionPointToEnd(&submod.getBodyRegion().front());

  auto func = builder.create<func::FuncOp>(loc, "entry", calleeType);

  auto &entryBlock = *func.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  mlir::Value stream = entryBlock.getArgument(0);
  auto buffers = entryBlock.getArgument(1);

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
    auto ld = builder.create<LLVM::LoadOp>(loc, arg.getType(), gep);
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
    return nullptr;

  void *ptr = nullptr;
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
      options.cubinTriple = "nvptx64-nvidia-cuda";
      options.cubinChip = cubinChip;
      options.cubinFeatures = cubinFeatures;
      options.cubinFormat = "fatbin";
      options.optLevel = 2;
      options.kernelUseBarePtrCallConv = false;
      options.hostUseBarePtrCallConv = false;
      buildLowerToNVVMPassPipeline(pm, options, toolkitPath, linkFiles);

      pm.run(submod);

      OpBuilder builder(submod);
      builder.setInsertionPointToStart(&submod.getBodyRegion().front());
      auto ptrty = LLVM::LLVMPointerType::get(builder.getContext());
      auto i64 = builder.getIntegerType(64);
      auto i32 = builder.getIntegerType(32);
      auto idx = i64;
      auto voidty = LLVM::LLVMVoidType::get(submod.getContext());

      auto glob = builder.create<LLVM::GlobalOp>(loc, ptrty, /*constant*/ false,
                                                 LLVM::Linkage::Private,
                                                 "nv_func", mlir::Attribute());

      mlir::Type cumodtys[] = {ptrty, ptrty};
      LLVM::LLVMFuncOp modload = builder.create<LLVM::LLVMFuncOp>(
          loc, "cuModuleLoadData", LLVM::LLVMFunctionType::get(i32, cumodtys));

      mlir::Type cutys[] = {ptrty, idx, idx,   idx,   idx,  idx,
                            idx,   i32, ptrty, ptrty, ptrty};

      LLVM::LLVMFuncOp launch = builder.create<LLVM::LLVMFuncOp>(
          loc, "cuLaunchKernel", LLVM::LLVMFunctionType::get(voidty, cutys));

      mlir::Type cufunctys[] = {ptrty, ptrty, ptrty};
      LLVM::LLVMFuncOp funcload = builder.create<LLVM::LLVMFuncOp>(
          loc, "cuModuleGetFunction",
          LLVM::LLVMFunctionType::get(i32, cufunctys));

      LLVM::GlobalOp kernStr;
      {
        std::string value = "kernel";
        auto type = LLVM::LLVMArrayType::get(
            mlir::IntegerType::get(builder.getContext(), 8), value.size() + 1);
        kernStr = builder.create<LLVM::GlobalOp>(
            loc, type, /*isConstant=*/true, LLVM::Linkage::Internal, "str",
            builder.getStringAttr(value + '\0'));
      }

      builder.setInsertionPointToStart(&submod.getBodyRegion().front());

      LLVM::LLVMFuncOp initfn = builder.create<LLVM::LLVMFuncOp>(
          loc, "nv_func_init", LLVM::LLVMFunctionType::get(voidty, {}, false),
          LLVM::Linkage::Private);

      mlir::Attribute funcs[] = {FlatSymbolRefAttr::get(initfn)};
      mlir::Attribute idxs[] = {builder.getI32IntegerAttr(0)};
      builder.create<LLVM::GlobalCtorsOp>(loc, builder.getArrayAttr(funcs),
                                          builder.getArrayAttr(idxs));

      LLVM::GlobalOp binary;
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

      {
        auto blk = new Block();
        initfn.getRegion().push_back(blk);
        builder.setInsertionPointToEnd(blk);

        auto one = builder.create<LLVM::ConstantOp>(
            loc, i64, builder.getI64IntegerAttr(1));
        auto modptr = builder.create<LLVM::AllocaOp>(loc, ptrty, ptrty, one);
        auto funcptr = builder.create<LLVM::AllocaOp>(loc, ptrty, ptrty, one);

        auto addr_modbin = builder.create<LLVM::AddressOfOp>(loc, binary);
        mlir::Value modargs[] = {modptr->getResult(0),
                                 addr_modbin->getResult(0)};
        if (cuModuleLoadDataPtr) {
          auto addr_glob_int = builder.create<LLVM::ConstantOp>(
              loc, i64, builder.getI64IntegerAttr(cuModuleLoadDataPtr));
          auto addr_glob =
              builder.create<LLVM::IntToPtrOp>(loc, ptrty, addr_glob_int);
          modargs.insert(modargs.begin(), addr_glob);
          builder.create<LLVM::CallOp>(loc, TypeRange(i32), modargs);
        } else {
          builder.create<LLVM::CallOp>(loc, modload, modargs);
        }
        auto mod = builder.create<LLVM::LoadOp>(loc, ptrty, modptr);

        auto addr_kernstr =
            builder.create<LLVM::AddressOfOp>(loc, ptrty, "str");

        SmallVector<mlir::Value> funcargs = {funcptr->getResult(0),
                                             mod->getResult(0),
                                             addr_kernstr->getResult(0)};
        if (cuModuleGetFunctionPtr) {
          auto addr_glob_int = builder.create<LLVM::ConstantOp>(
              loc, i64, builder.getI64IntegerAttr(cuModuleGetFunctionPtr));
          auto addr_glob =
              builder.create<LLVM::IntToPtrOp>(loc, ptrty, addr_glob_int);
          funcargs.insert(funcargs.begin(), addr_glob);
          builder.create<LLVM::CallOp>(loc, TypeRange(i32), funcargs);
        } else {
          builder.create<LLVM::CallOp>(loc, funcload, funcargs);
        }
        auto func = builder.create<LLVM::LoadOp>(loc, ptrty, funcptr);

        auto addr_glob = builder.create<LLVM::AddressOfOp>(loc, glob);
        builder.create<LLVM::StoreOp>(loc, func, addr_glob);
        builder.create<LLVM::ReturnOp>(loc, ValueRange());
      }

      submod.walk([&](gpu::LaunchFuncOp op) {
        builder.setInsertionPoint(op);
        auto ldop =
            op.getKernelOperands().front().getDefiningOp<LLVM::LoadOp>();
        assert(ldop);
        auto params = ldop.getOperand();
        auto addr_glob = builder.create<LLVM::AddressOfOp>(loc, glob);
        auto cufunc = builder.create<LLVM::LoadOp>(loc, ptrty, addr_glob);
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

        if (cuLaunchKernelPtr) {
          auto addr_glob_int = builder.create<LLVM::ConstantOp>(
              loc, i64, builder.getI64IntegerAttr(cuLaunchKernelPtr));
          auto addr_glob =
              builder.create<LLVM::IntToPtrOp>(loc, ptrty, addr_glob_int);
          args.insert(args.begin(), addr_glob);
          builder.create<LLVM::CallOp>(loc, TypeRange(voidty), args);
        } else {
          builder.create<LLVM::CallOp>(loc, launch, args);
        }

        op.erase();
        ldop.erase();
      });

      ptr = CompileHostModule(ss.str(), submod);

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
      data[0] = (size_t)CompileKernel(
          symbolTable, op.getLoc(), fn, jit, data[1], data[2], data[3], data[4],
          data[5], data[6], data[7], toolkitPath.getValue(), linkFilesArray,
          indexBitWidth.getValue(), cubinChip.getValue(),
          cubinFeatures.getValue(), cuLaunchKernelPtr, cuModuleLoadDataPtr,
          cuModuleGetFunctionPtr);

      std::string backendinfo((char *)&data, sizeof(void *));

      OpBuilder rewriter(op);
      auto replacement = rewriter.create<stablehlo::CustomCallOp>(
          op.getLoc(), op.getResultTypes(), op.getInputs(),
          rewriter.getStringAttr("enzymexla_gpu"),
          /* has_side_effect*/ rewriter.getBoolAttr(false),
          /*backend_config*/ rewriter.getStringAttr(backendinfo),
          /* api_version*/
          CustomCallApiVersionAttr::get(rewriter.getContext(),
                                        mlir::stablehlo::CustomCallApiVersion::
                                            API_VERSION_STATUS_RETURNING),
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
