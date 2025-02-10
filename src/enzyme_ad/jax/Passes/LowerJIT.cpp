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
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Passes/EnzymeHLOPatterns.h"
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

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"

#include "mlir/Target/LLVMIR/Export.h"

#define DEBUG_TYPE "lower-jit"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERJITPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

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
void buildHostPostPipeline(
    OpPassManager &pm, const mlir::gpu::GPUToNVVMPipelineOptions &options,
    std::string toolkitPath,
    const llvm::SmallVectorImpl<std::string> &linkFiles) {
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
    std::string toolkitPath,
    const llvm::SmallVectorImpl<std::string> &linkFiles) {
  // Common pipelines
  buildCommonPassPipeline(pm, options);

  // GPUModule-specific stuff
  buildGpuPassPipeline(pm, options);

  // Host post-GPUModule-specific stuff
  buildHostPostPipeline(pm, options, toolkitPath, linkFiles);
}

void buildLowerToCPUPassPipeline(OpPassManager &pm) {
  pm.addPass(createConvertPolygeistToLLVM());
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

} // namespace

struct CallInfo {
  void (*run)(void *, void *, void **);
  void *(*init)();
};

llvm::StringMap<CallInfo> jitkernels;
llvm::sys::SmartRWMutex<true> jit_kernel_mutex;
std::unique_ptr<llvm::orc::LLJIT> JIT = nullptr;
llvm::orc::SymbolMap MappedSymbols;

bool initJIT() {
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
      return false;
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
      return false;
    }

    JIT->getMainJITDylib().addGenerator(std::move(ProcessSymsGenerator.get()));
  }
  return true;
}

extern "C" void EnzymeJaXMapSymbol(const char *name, void *symbol) {
  initJIT();
  MappedSymbols[JIT->mangleAndIntern(name)] = llvm::orc::ExecutorSymbolDef(
      llvm::orc::ExecutorAddr::fromPtr(symbol), llvm::JITSymbolFlags());
}

CallInfo CompileHostModule(std::string &key, mlir::ModuleOp modOp,
                           bool compileInit) {
  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext);
  auto llvmModule = translateModuleToLLVMIR(modOp, *ctx);
  if (!llvmModule) {
    llvm::errs() << "modOp: " << *modOp << "\n";
    llvm::errs() << "could not convert to LLVM IR\n";
    return {};
  }
  if (!initJIT())
    return {};

  llvmModule->setDataLayout(JIT->getDataLayout());
  llvmModule->setTargetTriple(JIT->getTargetTriple().getTriple());

  auto LibA =
      JIT->createJITDylib("enzymejitdl_" + std::to_string(jitkernels.size()));
  if (auto Err = JIT->addIRModule(
          LibA.get(),
          llvm::orc::ThreadSafeModule(std::move(llvmModule), std::move(ctx)))) {
    llvm::errs() << " addIRModuleError " << Err << "\n";
    return {};
  }
  if (auto Err = LibA->define(llvm::orc::absoluteSymbols(MappedSymbols))) {
    llvm::errs() << " Symbol define Error " << Err << "\n";
    return {};
  }

  llvm::Expected<llvm::orc::ExecutorAddr> NVSym(llvm::orc::ExecutorAddr{});
  if (compileInit) {
    NVSym = JIT->lookup(LibA.get(), "nv_func_init");
    if (!NVSym) {
      llvm::errs() << " lookupError " << NVSym.takeError() << "\n";
      return {};
    }
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

void rewriteKernelCallABI(
    mlir::ModuleOp &submod, mlir::Location loc, const std::string &legalName,
    bool debug, enzymexla::JITCallOp jitCallOp, const std::string &modstr,
    size_t cuResultHandlerPtr, size_t cuStreamSynchronizePtr, int indexBitWidth,
    const std::string &cubinTriple, const std::string &cubinChip,
    const std::string &cubinFeatures, const std::string &cubinFormat,
    int cuOptLevel, const std::string &toolkitPath,
    const llvm::SmallVectorImpl<std::string> &linkFiles) {
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
  auto curesult_handler_ty = LLVM::LLVMFunctionType::get(voidty, curesulttys);
  LLVM::LLVMFuncOp launch =
      builder.create<LLVM::LLVMFuncOp>(loc, "cuLaunchKernel", launch_ty);
  auto cusync_ty = LLVM::LLVMFunctionType::get(i32, {ptrty});

  mlir::Type cufunctys[] = {ptrty, ptrty, ptrty};
  auto funcload_ty = LLVM::LLVMFunctionType::get(i32, cufunctys);
  LLVM::LLVMFuncOp funcload =
      builder.create<LLVM::LLVMFuncOp>(loc, "cuModuleGetFunction", funcload_ty);

  LLVM::GlobalOp kernStr;
  {
    auto type = LLVM::LLVMArrayType::get(
        mlir::IntegerType::get(builder.getContext(), 8), legalName.size() + 1);
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
        loc, type, /*isConstant=*/true, LLVM::Linkage::Internal, "strlaunch",
        builder.getStringAttr(value + '\0'));
  }
  LLVM::GlobalOp modOpStr = nullptr;
  if (debug) {
    std::string opstr;
    llvm::raw_string_ostream ss(opstr);

    ss << jitCallOp;
    std::string value = "modstr=" + modstr + "\n" + opstr + "\n\n";
    auto type = LLVM::LLVMArrayType::get(
        mlir::IntegerType::get(builder.getContext(), 8), value.size() + 1);
    modOpStr = builder.create<LLVM::GlobalOp>(
        loc, type, /*isConstant=*/true, LLVM::Linkage::Internal, "strmlirmod",
        builder.getStringAttr(value + '\0'));
  }

  LLVM::GlobalOp binary = nullptr;
  submod.walk([&](gpu::BinaryOp op) {
    gpu::ObjectAttr object = getSelectedObject(op);
    auto value = object.getObject().getValue();
    auto type = LLVM::LLVMArrayType::get(
        mlir::IntegerType::get(builder.getContext(), 8), value.size());
    binary = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                            LLVM::Linkage::Internal, "binary",
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

    auto one = builder.create<LLVM::ConstantOp>(loc, i64,
                                                builder.getI64IntegerAttr(1));
    auto modptr = builder.create<LLVM::AllocaOp>(loc, ptrty, ptrty, one);
    auto funcptr = builder.create<LLVM::AllocaOp>(loc, ptrty, ptrty, one);

    auto addr_modbin = builder.create<LLVM::AddressOfOp>(loc, binary);
    SmallVector<mlir::Value> modargs = {modptr->getResult(0),
                                        addr_modbin->getResult(0)};

    mlir::Value loadModRes =
        builder.create<LLVM::CallOp>(loc, modload, modargs)->getResult(0);

    if (debug) {
      Value printargs1[] = {
          builder.create<LLVM::AddressOfOp>(loc, loadModuleStr)->getResult(0),
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

    auto addr_kernstr = builder.create<LLVM::AddressOfOp>(loc, ptrty, "str");

    SmallVector<mlir::Value> funcargs = {
        funcptr->getResult(0), mod->getResult(0), addr_kernstr->getResult(0)};
    mlir::Value loadFuncRes =
        builder.create<LLVM::CallOp>(loc, funcload, funcargs)->getResult(0);

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

  SmallVector<enzymexla::GetStreamOp> streams;
  submod.walk([&](enzymexla::GetStreamOp op) { streams.push_back(op); });
  for (auto op : streams) {
    OpBuilder builder(op);
    auto pfunc = op->getParentOfType<LLVM::LLVMFuncOp>();
    mlir::Value stream = pfunc.getBody().begin()->getArgument(1);
    for (auto u : llvm::make_early_inc_range(op->getResult(0).getUsers())) {
      auto ur = cast<UnrealizedConversionCastOp>(u);
      assert(ur->getResult(0).getType() == stream.getType());
      ur->getResult(0).replaceAllUsesWith(stream);
      ur.erase();
    }
    op.erase();
  }

  submod.walk([&](gpu::LaunchFuncOp op) {
    builder.setInsertionPoint(op);
    auto pfunc = op->getParentOfType<LLVM::LLVMFuncOp>();
    mlir::Value cufunc = pfunc.getBody().begin()->getArgument(2);

    auto ldop = op.getKernelOperands().front().getDefiningOp<LLVM::LoadOp>();
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

    Value kernRes =
        builder.create<LLVM::CallOp>(loc, launch, args)->getResult(0);
    if (debug) {
      Value printargs1[] = {
          builder.create<LLVM::AddressOfOp>(loc, launchKernelStr)->getResult(0),
          builder.create<LLVM::IntToPtrOp>(loc, ptrty, kernRes)->getResult(0)};
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
}

CallInfo
CompileCall(SymbolTableCollection &symbolTable, mlir::Location loc,
            FunctionOpInterface op, bool jit, enzymexla::JITCallOp jcall,
            bool openmp, size_t cuResultHandlerPtr,
            size_t cuStreamSynchronizePtr, int indexBitWidth,
            const std::string &cubinTriple, const std::string &cubinChip,
            const std::string &cubinFeatures, const std::string &cubinFormat,
            int cuOptLevel, const std::string &toolkitPath,
            const llvm::SmallVectorImpl<std::string> &linkFiles, bool debug) {

  OpBuilder builder(op);

  auto ptrty = LLVM::LLVMPointerType::get(builder.getContext());

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

  auto submod = builder.create<ModuleOp>(loc);

  int numGPUModule = 0;

  SmallVector<Operation *> tocopy;
  op->walk([&](gpu::LaunchFuncOp cop) {
    tocopy.push_back(SymbolTable::lookupNearestSymbolFrom<gpu::GPUModuleOp>(
        cop, cop.getKernelModuleName()));
  });
  op->walk([&](CallOpInterface cop) {
    if (auto op2 = cop.resolveCallable()) {
      if (auto gpu = dyn_cast<gpu::GPUFuncOp>(op2)) {
        tocopy.push_back(gpu.getParentOp());
      } else
        tocopy.push_back(op2);
    }
  });
  op->walk([&](LLVM::AddressOfOp cop) {
    if (auto op2 = cop.getGlobal(symbolTable))
      tocopy.push_back(op2);
    else if (auto op2 = cop.getFunction(symbolTable))
      tocopy.push_back(op2);
  });
  SmallPtrSet<Operation *, 1> done;

  builder.setInsertionPointToStart(&submod.getBodyRegion().front());
  while (tocopy.size()) {
    auto cur = tocopy.pop_back_val();
    if (done.count(cur))
      continue;
    done.insert(cur);
    if (isa<gpu::GPUModuleOp>(cur))
      numGPUModule++;
    builder.clone(*cur);
    if (isa<gpu::GPUModuleOp>(cur))
      continue;
    cur->walk([&](gpu::LaunchFuncOp cop) {
      tocopy.push_back(SymbolTable::lookupNearestSymbolFrom<gpu::GPUModuleOp>(
          cop, cop.getKernelModuleName()));
    });
    cur->walk([&](CallOpInterface cop) {
      if (auto op2 = cop.resolveCallable()) {
        if (auto gpu = dyn_cast<gpu::GPUFuncOp>(op2)) {
          tocopy.push_back(gpu.getParentOp());
        } else {
          tocopy.push_back(op2);
        }
      }
    });
    cur->walk([&](LLVM::AddressOfOp cop) {
      if (auto op2 = cop.getGlobal(symbolTable))
        tocopy.push_back(op2);
      else if (auto op2 = cop.getFunction(symbolTable))
        tocopy.push_back(op2);
    });
  }

  builder.setInsertionPointToEnd(&submod.getBodyRegion().front());

  SmallVector<mlir::Type, 1> intys = {ptrty};
  if (numGPUModule != 0) {
    intys.push_back(ptrty);
    intys.push_back(ptrty);
  }
  FunctionType calleeType = builder.getFunctionType(intys, {});
  auto func = builder.create<func::FuncOp>(loc, "entry", calleeType);

  auto &entryBlock = *func.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  mlir::Value buffers = entryBlock.getArgument(0);

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

  IRMapping map;
  for (auto &&[oldarg, newarg] : zip(op.getArguments(), arguments)) {
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

  op.getFunctionBody().cloneInto(&func.getBody(), map);

  auto second = entryBlock.getNextNode();
  entryBlock.getOperations().splice(entryBlock.getOperations().end(),
                                    second->getOperations());

  second->erase();

  func.getBody().walk([](LLVM::ReturnOp op) {
    OpBuilder rewriter(op);
    rewriter.create<mlir::func::ReturnOp>(op.getLoc());
    op.erase();
  });

  func.getBody().walk([](LLVM::UnreachableOp op) {
    OpBuilder rewriter(op);
    rewriter.create<mlir::func::ReturnOp>(op.getLoc());
    op.erase();
  });

  std::string modstr;
  llvm::raw_string_ostream ss(modstr);

  ss << submod;

  if (!jit) {
    submod.erase();
    return {};
  }

  llvm::sys::SmartScopedWriter<true> jit_lock(jit_kernel_mutex);

  auto found = jitkernels.find(ss.str());
  if (found != jitkernels.end()) {
    submod.erase();
    return found->second;
  } else {
    if (numGPUModule != 0)
      submod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                      UnitAttr::get(jcall.getContext()));
    static size_t id = 0;
    submod.setName("jitoffload" + std::to_string(id));
    id++;
    PassManager pm(submod.getContext());
    if (numGPUModule == 0) {
      pm.addPass(createLowerAffinePass());
      if (openmp)
        pm.addPass(createConvertSCFToOpenMPPass());
      else
        pm.addPass(createConvertSCFToCFPass());

      buildLowerToCPUPassPipeline(pm);
      auto subres = pm.run(submod);
      if (!subres.succeeded()) {
        submod.erase();
        return {};
      }
    } else {
      submod->walk([](gpu::GPUModuleOp gmod) {
        auto str = gmod.getName();
        if (str.size() > 200)
          gmod.setName(str.substr(0, 200));
      });

      std::string legalName;
      submod->walk([&](gpu::LaunchFuncOp gmod) {
        if (legalName.size())
          assert(legalName == gmod.getKernelName());
        else
          legalName = gmod.getKernelName().str();
        auto str = gmod.getKernelModuleName().getValue();
        if (str.size() > 200)
          gmod.setKernelAttr(SymbolRefAttr::get(
              StringAttr::get(gmod.getContext(), str.substr(0, 200)),
              gmod.getKernel().getNestedReferences()));
      });
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
      if (numGPUModule != 1) {
        llvm::errs() << " only single gpu module calls supported atm\n";
        submod.erase();
        return {};
      }
      auto subres = pm.run(submod);
      if (!subres.succeeded()) {
        submod.erase();
        return {};
      }
      rewriteKernelCallABI(submod, loc, legalName, debug, jcall, modstr,
                           cuResultHandlerPtr, cuStreamSynchronizePtr,
                           indexBitWidth, cubinTriple, cubinChip, cubinFeatures,
                           cubinFormat, cuOptLevel, toolkitPath, linkFiles);
    }

    auto ptr = CompileHostModule(ss.str(), submod, numGPUModule != 0);
    jitkernels[ss.str()] = ptr;
    submod.erase();
    return ptr;
  }
};

namespace {

struct LowerJITPass
    : public mlir::enzyme::impl::LowerJITPassBase<LowerJITPass> {
  using LowerJITPassBase::LowerJITPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    OpPassManager pm;
    buildLowerToCPUPassPipeline(pm);
    // if (backend == "cuda")
    {
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
      /*
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
      */
    }
    pm.getDependentDialects(registry);
    if (openmp)
      registry.insert<mlir::omp::OpenMPDialect>();
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
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(getOperation());
    llvm::SmallVector<std::string> linkFilesArray =
        parseLinkFilesString(linkFiles.getValue());

    SetVector<FunctionOpInterface> callees;
    getOperation()->walk([&](JITCallOp op) {
      mlir::ArrayAttr operand_layouts =
          op.getOperandLayouts()
              ? cast<mlir::ArrayAttr>(*op.getOperandLayouts())
              : nullptr;
      mlir::ArrayAttr result_layouts =
          op.getResultLayouts() ? cast<mlir::ArrayAttr>(*op.getResultLayouts())
                                : nullptr;
      mlir::ArrayAttr output_operand_aliases = op.getOutputOperandAliases();

      auto *symbolOp = symbolTable.lookupNearestSymbolFrom(op, op.getFnAttr());
      auto fn = cast<FunctionOpInterface>(symbolOp);
      if (fn.getArguments().size() != op.getInputs().size()) {
        op->emitError() << "JITCallOp had " << op.getInputs().size()
                        << " whereas called fn requires "
                        << fn.getArguments().size() << "\n";
        return;
      }

      CallInfo cdata =
          CompileCall(symbolTable, op.getLoc(), fn, jit, op, openmp,
                      cuResultHandlerPtr, cuStreamSynchronizePtr, indexBitWidth,
                      cubinTriple, cubinChip, cubinFeatures, cubinFormat,
                      cuOptLevel, toolkitPath, linkFilesArray, debug);

      std::string backendinfo((char *)&cdata, sizeof(CallInfo));

      OpBuilder rewriter(op);

      auto backendstr = rewriter.getStringAttr(backendinfo);
      SmallVector<NamedAttribute> names;
      names.push_back(
          NamedAttribute(rewriter.getStringAttr("attr"), backendstr));
      auto dattr = DictionaryAttr::get(op.getContext(), names);

      Operation *replacement;
      if (backend == "cuda")
        replacement = rewriter.create<stablehlo::CustomCallOp>(
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
      else if (backend == "cpu")
        replacement = rewriter.create<stablehlo::CustomCallOp>(
            op.getLoc(), op.getResultTypes(), op.getInputs(),
            rewriter.getStringAttr("enzymexla_compile_cpu"),
            /* has_side_effect*/ rewriter.getBoolAttr(false),
            /*backend_config*/ backendstr,
            /* api_version*/
            CustomCallApiVersionAttr::get(
                rewriter.getContext(),
                mlir::stablehlo::CustomCallApiVersion::
                    API_VERSION_STATUS_RETURNING_UNIFIED),
            /*calledcomputations*/ nullptr, operand_layouts, result_layouts,
            output_operand_aliases);

      op.replaceAllUsesWith(replacement);
      op.erase();
      callees.insert(fn);
    });
    for (auto callee : callees)
      callee.erase();
    getOperation()->walk([&](gpu::GPUModuleOp op) { op.erase(); });
  }
};

} // end anonymous namespace
