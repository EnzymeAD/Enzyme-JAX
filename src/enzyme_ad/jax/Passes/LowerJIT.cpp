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

#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Passes/EnzymeHLOPatterns.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"

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

// Entry point of a jitted call: `run(buffers)`, or `run(buffers, stream)` when
// the call needs the stream it is executed on (see `enzymexla.get_stream`).
struct CallInfo {
  void (*run)(void *, void *);
};

llvm::StringMap<CallInfo> jitkernels;
llvm::sys::SmartRWMutex<true> jit_kernel_mutex;
std::unique_ptr<llvm::orc::LLJIT> JIT = nullptr;
llvm::orc::SymbolMap MappedSymbols;

bool initJIT();

llvm::Expected<void *> mlir::enzyme::lookupSymbol(const char *name) {
  if (!JIT)
    return llvm::make_error<llvm::StringError>("JIT not initialized",
                                               llvm::inconvertibleErrorCode());

  auto mangled_name = JIT->mangleAndIntern(name);
  if (!MappedSymbols.contains(mangled_name))
    return llvm::make_error<llvm::StringError>("Symbol not found: " +
                                                   std::string(name),
                                               llvm::inconvertibleErrorCode());

  auto addr = MappedSymbols[mangled_name];
  return addr.toPtr<void *>();
}

extern "C" void EnzymeJaXMapSymbol(const char *name, void *symbol) {
  initJIT();
  MappedSymbols[JIT->mangleAndIntern(name)] = llvm::orc::ExecutorSymbolDef(
      llvm::orc::ExecutorAddr::fromPtr(symbol), llvm::JITSymbolFlags());
}

#if defined(_WIN32)
#ifdef __MINGW32__
#if defined(__i386__)
#undef _alloca
extern "C" void _alloca(void);
#elif defined(__x86_64__)
extern "C" void ___chkstk_ms(void);
#else
extern "C" void __chkstk(void);
#endif
#else
extern "C" void __chkstk(void);
#endif
#endif

bool initJIT() {
  if (!JIT) {
    auto tJTMB = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!tJTMB) {
      llvm::errs() << " jit host detection error: " << tJTMB.takeError()
                   << "\n";
      return false;
    }

    // On Windows, compile as if we were mingw rather than MSVC. With an MSVC
    // environment LLVM emits every mergeable floating point constant into its
    // own COMDAT section carrying a global `__real@<hex>` (or `__xmm@<hex>`)
    // symbol -- see TargetLoweringObjectFileCOFF::getSectionForConstant
    // together with AsmPrinter::GetCPISymbol, which is gated on
    // isWindowsMSVCEnvironment(). LLJIT links x86-64 COFF objects with
    // RuntimeDyld, whose COMDAT support cannot resolve those symbols
    // (llvm.org/PR40074, which RTDyldObjectLinkingLayer::onObjLoad only
    // partially works around), so a kernel containing a literal such as `0.5`
    // intermittently fails to materialize:
    //
    //   Failed to materialize symbols:
    //     { (enzymejitdl_12, { __real@3fe0000000000000 }) }
    //
    // The GNU environment selects MCAsmInfoGNUCOFF, which sets
    // HasCOFFComdatConstants = false, so constants are emitted as ordinary
    // constant-pool entries with local labels that RuntimeDyld handles fine.
    // Both environments share the same architecture, object format, data
    // layout and calling convention, so this only changes how constants are
    // emitted. The one externally visible difference is the name of the stack
    // probe helper, which is mapped below. See EnzymeAD/Reactant.jl#1673.
    if (tJTMB->getTargetTriple().isWindowsMSVCEnvironment())
      tJTMB->getTargetTriple().setEnvironment(llvm::Triple::GNU);

    auto tJIT =
        llvm::orc::LLJITBuilder()
            .setJITTargetMachineBuilder(std::move(*tJTMB))
            .setLinkProcessSymbolsByDefault(true)
            .setObjectLinkingLayerCreator(
                [](llvm::orc::ExecutionSession &ES,
                   llvm::jitlink::JITLinkMemoryManager &)
                    -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
                  auto obj = std::make_unique<
                      llvm::orc::RTDyldObjectLinkingLayer>(
                      ES, [](const llvm::MemoryBuffer &) {
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

#if defined(_WIN32)
#ifdef __MINGW32__
#if defined(__i386__)
    void *StackProbe = (void *)&_alloca;
#elif defined(__x86_64__)
    void *StackProbe = (void *)&___chkstk_ms;
#else
    void *StackProbe = (void *)&__chkstk;
#endif
#else
    void *StackProbe = (void *)&__chkstk;
#endif
    EnzymeJaXMapSymbol("__chkstk", StackProbe);
#if defined(_M_X64) || defined(__x86_64__)
    // We select the GNU environment above, and x86-64 mingw names the stack
    // probe ___chkstk_ms rather than __chkstk (see RuntimeLibcalls.td, where
    // ___chkstk_ms is isCygwinMinGW64 and __chkstk is isWin64NotCygMing). The
    // two are interchangeable there: both take the allocation size in %rax,
    // only probe, and leave %rsp and %rax alone (see the comment in
    // X86FrameLowering::emitStackProbeCall), so whichever one this process was
    // built with can serve both names.
    EnzymeJaXMapSymbol("___chkstk_ms", StackProbe);
#endif
#endif
  }
  return true;
}

CallInfo CompileHostModule(std::string &key, mlir::ModuleOp modOp,
                           bool dump_final_module) {
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
  llvmModule->setTargetTriple(JIT->getTargetTriple());

  if (dump_final_module) {
    llvm::errs() << " final_llvm_module before jit: " << *llvmModule << "\n";
  }
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

  auto Entry = JIT->lookup(LibA.get(), "entry");
  if (!Entry) {
    llvm::errs() << " lookupError " << Entry.takeError() << "\n";
    return {};
  }

  // Run the static constructors of the module.
  if (auto Err = JIT->initialize(*LibA)) {
    llvm::errs() << " initializeError " << Err << "\n";
    return {};
  }

  auto ptr = (void *)Entry->getValue();

  return CallInfo{(void (*)(void *, void *))ptr};
}

// Replaces `enzymexla.get_stream` (and the casts convert-polygeist-to-llvm left
// on it) with the stream argument of the jitted entry point.
static LogicalResult replaceGetStreamOpsWithStreamArg(mlir::ModuleOp submod) {
  SmallVector<enzymexla::GetStreamOp> streams;
  submod.walk([&](enzymexla::GetStreamOp op) { streams.push_back(op); });
  for (auto op : streams) {
    auto pfunc = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!pfunc || pfunc.getName() != "entry")
      return op.emitError("enzymexla.get_stream must be used directly in the "
                          "body of the jitted function");
    mlir::Value stream = pfunc.getBody().front().getArgument(1);
    for (auto u : llvm::make_early_inc_range(op.getResult().getUsers())) {
      if (auto ur = dyn_cast<UnrealizedConversionCastOp>(u)) {
        if (ur->getResult(0).getType() != stream.getType())
          return ur.emitError("unexpected type for stream cast");
        ur->getResult(0).replaceAllUsesWith(stream);
        ur.erase();
        continue;
      }
      if (op.getResult().getType() != stream.getType())
        return op.emitError("unexpected stream type");
      u->replaceUsesOfWith(op.getResult(), stream);
    }
    op.erase();
  }
  return success();
}

CallInfo CompileCall(SymbolTableCollection &symbolTable, mlir::Location loc,
                     FunctionOpInterface op, bool jit, bool returnPtr,
                     bool dump_final_module, bool &usesStream) {
  if (!isa<LLVM::LLVMFuncOp>(op)) {
    op.emitError("jitted function must be lowered to llvm.func (e.g. by "
                 "convert-polygeist-to-llvm) before lower-jit");
    return {};
  }

  OpBuilder builder(op);

  auto ptrty = LLVM::LLVMPointerType::get(builder.getContext());

  auto parentModule = op->getParentOfType<ModuleOp>();
  auto submod = ModuleOp::create(builder, loc);

  SmallVector<Operation *> tocopy;
  auto collect = [&](Operation *cur) {
    cur->walk([&](CallOpInterface cop) {
      if (auto callee = cop.resolveCallable())
        tocopy.push_back(callee);
    });
    cur->walk([&](LLVM::AddressOfOp cop) {
      if (auto global = cop.getGlobal(symbolTable))
        tocopy.push_back(global);
      else if (auto fn = cop.getFunction(symbolTable))
        tocopy.push_back(fn);
    });
  };
  SmallPtrSet<Operation *, 8> done;
  builder.setInsertionPointToStart(&submod.getBodyRegion().front());
  auto copyClosure = [&]() {
    while (tocopy.size()) {
      auto cur = tocopy.pop_back_val();
      if (!done.insert(cur).second)
        continue;
      builder.clone(*cur);
      collect(cur);
    }
  };
  collect(op);
  copyClosure();

  // Static constructors (e.g. the fat binary registration emitted by
  // convert-polygeist-to-llvm) are only reachable through
  // llvm.mlir.global_ctors: keep those referring to a copied symbol, they run
  // when the module is jitted.
  auto refersToCopied = [&](Operation *fn) {
    auto uses = SymbolTable::getSymbolUses(fn);
    if (!uses)
      return false;
    for (auto use : *uses)
      if (auto sym = symbolTable.lookupNearestSymbolFrom(parentModule,
                                                         use.getSymbolRef()))
        if (done.contains(sym))
          return true;
    return false;
  };
  SmallVector<Attribute> ctors, priorities, data;
  for (bool changed = true; changed;) {
    changed = false;
    for (auto ctorsOp : parentModule.getOps<LLVM::GlobalCtorsOp>())
      for (auto [sym, priority, datum] :
           llvm::zip(ctorsOp.getCtors(), ctorsOp.getPriorities(),
                     ctorsOp.getData())) {
        auto fn = symbolTable.lookupNearestSymbolFrom(
            parentModule, cast<FlatSymbolRefAttr>(sym));
        if (!fn || done.contains(fn) || !refersToCopied(fn))
          continue;
        ctors.push_back(sym);
        priorities.push_back(priority);
        data.push_back(datum);
        tocopy.push_back(fn);
        changed = true;
      }
    copyClosure();
  }
  if (!ctors.empty())
    LLVM::GlobalCtorsOp::create(builder, loc, builder.getArrayAttr(ctors),
                                builder.getArrayAttr(priorities),
                                builder.getArrayAttr(data));

  // Calls which need the stream they are executed on run on the device, and
  // take it as a second argument of the entry point.
  auto hasGetStream = [](Operation *root) {
    return root
        ->walk([](enzymexla::GetStreamOp) { return WalkResult::interrupt(); })
        .wasInterrupted();
  };
  usesStream = hasGetStream(op) || hasGetStream(submod);

  builder.setInsertionPointToEnd(&submod.getBodyRegion().front());

  SmallVector<mlir::Type, 2> intys = {ptrty};
  if (usesStream)
    intys.push_back(ptrty);
  Type retty =
      returnPtr ? Type(ptrty) : LLVM::LLVMVoidType::get(op.getContext());
  auto func = LLVM::LLVMFuncOp::create(
      builder, loc, "entry", LLVM::LLVMFunctionType::get(retty, intys));

  auto &entryBlock = *func.addEntryBlock(builder);
  builder.setInsertionPointToStart(&entryBlock);

  mlir::Value buffers = entryBlock.getArgument(0);

  SmallVector<mlir::Value> arguments;
  for (auto arg : op.getArguments()) {
    LLVM::GEPArg args[1] = {arg.getArgNumber()};
    auto gep = LLVM::GEPOp::create(
        builder, loc, ptrty, ptrty, buffers, args,
        mlir::LLVM::GEPNoWrapFlags(mlir::LLVM::GEPNoWrapFlags::inbounds));
    auto argTy = arg.getType();
    if (auto AT = dyn_cast<LLVM::LLVMArrayType>(argTy)) {
      argTy = AT.getElementType();
    }
    auto ld = LLVM::LoadOp::create(builder, loc, argTy, gep);
    arguments.push_back(ld);
  }

  IRMapping map;
  for (auto &&[oldarg, newarg] : zip(op.getArguments(), arguments)) {
    Value newval = newarg;

    if (auto AT = dyn_cast<LLVM::LLVMArrayType>(oldarg.getType())) {
      auto ud =
          LLVM::UndefOp::create(builder, newarg.getLoc(), oldarg.getType());
      int64_t c0[1] = {0};
      newval = LLVM::InsertValueOp::create(builder, newarg.getLoc(),
                                           oldarg.getType(), ud, newval, c0);
    }

    map.map(oldarg, newval);
  }

  op.getFunctionBody().cloneInto(&func.getBody(), map);

  auto second = entryBlock.getNextNode();
  entryBlock.getOperations().splice(entryBlock.getOperations().end(),
                                    second->getOperations());

  second->erase();

  // Reaching an unreachable (e.g. after a device `exit`) ends the call.
  func.getBody().walk([&](LLVM::UnreachableOp op) {
    OpBuilder rewriter(op);
    if (returnPtr)
      LLVM::ReturnOp::create(
          rewriter, op.getLoc(),
          ValueRange(LLVM::ZeroOp::create(rewriter, op.getLoc(), ptrty)));
    else
      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange());
    op.erase();
  });

  std::string modstr;
  llvm::raw_string_ostream ss(modstr);

  ss << submod;

  if (!jit) {
    submod.erase();
    return {};
  }

  if (submod->walk([](gpu::LaunchFuncOp) { return WalkResult::interrupt(); })
          .wasInterrupted()) {
    op.emitError("kernel launches must be lowered by convert-polygeist-to-llvm "
                 "before lower-jit");
    submod.erase();
    return {};
  }

  llvm::sys::SmartScopedWriter<true> jit_lock(jit_kernel_mutex);

  auto found = jitkernels.find(ss.str());
  if (found != jitkernels.end()) {
    submod.erase();
    return found->second;
  }

  static size_t id = 0;
  submod.setName("jitoffload" + std::to_string(id));
  id++;

  SmallVector<Operation *> toErase;
  submod.walk([&](LLVM::InlineAsmOp asmop) {
    if (asmop.getAsmString() == "exit;") {
      toErase.push_back(asmop);
    }
  });
  for (auto op : toErase) {
    op->erase();
  }

  if (usesStream && failed(replaceGetStreamOpsWithStreamArg(submod))) {
    submod.erase();
    return {};
  }

  auto ptr = CompileHostModule(ss.str(), submod, dump_final_module);
  jitkernels[ss.str()] = ptr;
  submod.erase();
  return ptr;
}

namespace {

struct LowerJITPass
    : public mlir::enzyme::impl::LowerJITPassBase<LowerJITPass> {
  using LowerJITPassBase::LowerJITPassBase;

  // convert-polygeist-to-llvm registers each GPU module with the CUDA runtime
  // from a placeholder fat binary global `<module>_gpubin_cst`. Once the module
  // is serialized (e.g. by enzymexla-gpu-module-to-binary), move the binary
  // into that global.
  LogicalResult embedGPUBinaries(Operation *root) {
    SmallVector<gpu::BinaryOp> binaries;
    root->walk([&](gpu::BinaryOp binary) { binaries.push_back(binary); });
    for (auto binary : binaries) {
      auto storage =
          dyn_cast_or_null<LLVM::GlobalOp>(SymbolTable::lookupNearestSymbolFrom(
              binary->getParentOp(),
              StringAttr::get(&getContext(),
                              binary.getName() + "_gpubin_cst")));
      if (!storage)
        return binary.emitError(
            "no fat binary global registering this binary, the gpu module "
            "must be lowered by convert-polygeist-to-llvm");
      gpu::ObjectAttr object = getSelectedObject(binary);
      if (!object)
        return failure();
      StringRef value = object.getObject().getValue();
      storage.setGlobalType(LLVM::LLVMArrayType::get(
          IntegerType::get(&getContext(), 8), value.size()));
      storage.setValueAttr(StringAttr::get(&getContext(), value));
      binary.erase();
    }
    return success();
  }

  void runOnOperation() override {
    if (failed(embedGPUBinaries(getOperation())))
      return signalPassFailure();

    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(getOperation());

    SetVector<FunctionOpInterface> callees;
    bool failed = false;
    getOperation()->walk([&](JITCallOp op) {
      mlir::ArrayAttr operand_layouts =
          op.getOperandLayouts()
              ? cast<mlir::ArrayAttr>(*op.getOperandLayouts())
              : nullptr;
      mlir::ArrayAttr output_operand_aliases = op.getOutputOperandAliases();
      mlir::ArrayAttr result_layouts = nullptr;
      if (operand_layouts) {
        result_layouts = getAliasedResultLayouts(op, operand_layouts,
                                                 output_operand_aliases);
        if (!result_layouts) {
          signalPassFailure();
          return;
        }
      }

      auto *symbolOp = symbolTable.lookupNearestSymbolFrom(op, op.getFnAttr());
      auto fn = cast<FunctionOpInterface>(symbolOp);
      if (fn.getArguments().size() != op.getInputs().size()) {
        op->emitError() << "JITCallOp had " << op.getInputs().size()
                        << " whereas called fn requires "
                        << fn.getArguments().size() << "\n";
        return;
      }

      bool hasReturn = false;
      if (auto llvmfnty =
              dyn_cast<LLVM::LLVMFunctionType>(fn.getFunctionType())) {
        hasReturn = llvmfnty.getReturnType() !=
                    LLVM::LLVMVoidType::get(op.getContext());
      } else if (auto fnty = dyn_cast<FunctionType>(fn.getFunctionType())) {
        hasReturn = !fnty.getResults().empty();
      }

      bool usesStream = false;
      CallInfo cdata = CompileCall(symbolTable, op.getLoc(), fn, jit, hasReturn,
                                   dump_final_module, usesStream);

      std::string backendinfo((char *)&cdata, sizeof(CallInfo));
      if (jit) {
        if (!cdata.run) {
          failed = true;
          return;
        }
      }
      OpBuilder rewriter(op);

      auto backendstr = rewriter.getStringAttr(backendinfo);
      SmallVector<NamedAttribute> names;
      names.push_back(
          NamedAttribute(rewriter.getStringAttr("attr"), backendstr));
      auto dattr = DictionaryAttr::get(op.getContext(), names);

      BoolAttr hasSideEffectAttr;
      if (op.getXlaSideEffectFreeAttr()) {
        hasSideEffectAttr = rewriter.getBoolAttr(false);
      } else { // use our analysis if no attribute is provided
        hasSideEffectAttr = rewriter.getBoolAttr(!isMemoryEffectFree(op));
      }

      Operation *replacement;
      if (usesStream)
        replacement = stablehlo::CustomCallOp::create(
            rewriter, op.getLoc(), op.getResultTypes(), op.getInputs(),
            hasReturn
                ? rewriter.getStringAttr("enzymexla_compile_gpu_with_error")
                : rewriter.getStringAttr("enzymexla_compile_gpu"),
            /* has_side_effect*/ hasSideEffectAttr,
            /*backend_config*/ dattr,
            /* api_version*/
            CustomCallApiVersionAttr::get(
                rewriter.getContext(),
                mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
            /*calledcomputations*/ nullptr, operand_layouts, result_layouts,
            output_operand_aliases, /*result_tilings*/ nullptr);
      else
        replacement = stablehlo::CustomCallOp::create(
            rewriter, op.getLoc(), op.getResultTypes(), op.getInputs(),
            hasReturn
                ? rewriter.getStringAttr("enzymexla_compile_cpu_with_error")
                : rewriter.getStringAttr("enzymexla_compile_cpu"),
            /* has_side_effect*/ hasSideEffectAttr,
            /*backend_config*/ backendstr,
            /* api_version*/
            CustomCallApiVersionAttr::get(
                rewriter.getContext(),
                mlir::stablehlo::CustomCallApiVersion::
                    API_VERSION_STATUS_RETURNING_UNIFIED),
            /*calledcomputations*/ nullptr, operand_layouts, result_layouts,
            output_operand_aliases, /*result_tilings*/ nullptr);

      op.replaceAllUsesWith(replacement);
      op.erase();
      callees.insert(fn);
    });
    for (auto callee : callees)
      callee.erase();
    getOperation()->walk([&](gpu::GPUModuleOp op) { op.erase(); });
    // Static constructors and destructors belong to the jitted host code.
    getOperation()->walk([&](LLVM::GlobalCtorsOp op) { op.erase(); });
    getOperation()->walk([&](LLVM::GlobalDtorsOp op) { op.erase(); });
    if (failed)
      return signalPassFailure();
  }
};

} // end anonymous namespace
