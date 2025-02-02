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

#define DEBUG_TYPE "lower-call"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERCALLPASS
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

void buildLowerToCPUPassPipeline(OpPassManager &pm) {
  pm.addPass(createConvertPolygeistToLLVM());
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
                           bool run_init,
                           bool compileInit = true) {
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
      JIT->createJITDylib("enzymecudadl_" + std::to_string(kernels.size()));
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

  if (cuLaunchPtr && cuLaunchPtr[0] == 0) {
    // Look up the JIT'd code entry point.
    auto LaunchSym = JIT->lookup(LibA.get(), "cuLaunchKernel");
    if (!LaunchSym) {
      llvm::errs() << " lookupError[cuLaunchKernel] " << LaunchSym.takeError()
                   << "\n";
      return {};
    }
    *cuLaunchPtr = (size_t)LaunchSym->getValue();
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

CallInfo CompileCall(SymbolTableCollection &symbolTable,
                          mlir::Location loc, FunctionOpInterface op, bool jit,
                          enzymexla::JITCallOp) {

  OpBuilder builder(op);

  auto ptrty = LLVM::LLVMPointerType::get(builder.getContext());
  mlir::Type intys[] = {ptrty, ptrty, ptrty};
  FunctionType calleeType = builder.getFunctionType(intys, {});

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
  auto submod =
      builder.create<ModuleOp>(loc, "jitoffload" + std::to_string(id));

  std::string legalName = op.getName().str();
  std::replace(legalName.begin(), legalName.end(), '#', '_');

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

  builder.setInsertionPointToStart(&submod.getBodyRegion().front());
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

  mlir::Value buffers = entryBlock.getArgument(1);

  auto idx = builder.getIntegerType(64);
  auto i32 = builder.getIntegerType(32);

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

  auto second = entryBlock->getNextNode();
  entryBlock->getOperations().splice(entryBlock->getOperations().end(),
                                second->getOperations());

  second->erase();

  func.getBody()->walk([](LLVM::ReturnOp op) {
    OpBuilder rewriter(op);
    rewriter.create<mlir::func::ReturnOp>(op.getLoc());
    op.erase();
  });

  func.getBody()->walk([](LLVM::UnreachableOp op) {
    OpBuilder rewriter(op);
    rewriter.create<mlir::func::ReturnOp>(op.getLoc());
    op.erase();
  });

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
      PassManager pm(submod.getContext());
      buildLowerToCPUPassPipeline(pm);

      auto subres = pm.run(submod);
      if (!subres.succeeded()) {
        return {};
      }
      ptr = CompileHostModule(ss.str(), submod, false, 0, false);
      kernels[ss.str()] = ptr;
      submod.erase();
    }
  }

  return ptr;
};

namespace {

struct LowerJITPass
    : public mlir::enzyme::impl::LowerJITPassBase<LowerJITPass> {
  using LowerKernelPassBase::LowerKernelPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    OpPassManager pm;
    buildLowerToCPUPassPipeline(pm);
    pm.getDependentDialects(registry);
    registry.insert<mlir::omp::OpenMPDialect>();
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::math::MathDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect, mlir::vector::VectorDialect,
                    mlir::gpu::GPUDialect, mlir::nvgpu::NVGPUDialect,
                    mlir::NVVM::NVVMDialect, mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(getOperation());

    llvm::SmallVector<std::string> linkFilesArray =
        parseLinkFilesString(linkFiles.getValue());
    getOperation()->walk([&](JITCallOp op) {
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
        op->emitError() << "JITCallOp had " << op.getInputs().size()
                        << " whereas called fn requires "
                        << fn.getArguments().size() << "\n";
        return;
      }

      CallInfo cdata = CompileCall(symbolTable, op.getLoc(), fn, jit, op);

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
    });
  }
};

} // end anonymous namespace
