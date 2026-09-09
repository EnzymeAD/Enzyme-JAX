#include "src/enzyme_ad/jax/Runtime/jit/jit.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"

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

std::unique_ptr<llvm::orc::LLJIT> JIT = nullptr;
llvm::orc::SymbolMap MappedSymbols;

using namespace enzymexla;

bool enzymexla::init_jit() {
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
    enzymexla::map_symbol("__chkstk", StackProbe);
#if defined(_M_X64) || defined(__x86_64__)
    // We select the GNU environment above, and x86-64 mingw names the stack
    // probe ___chkstk_ms rather than __chkstk (see RuntimeLibcalls.td, where
    // ___chkstk_ms is isCygwinMinGW64 and __chkstk is isWin64NotCygMing). The
    // two are interchangeable there: both take the allocation size in %rax,
    // only probe, and leave %rsp and %rax alone (see the comment in
    // X86FrameLowering::emitStackProbeCall), so whichever one this process was
    // built with can serve both names.
    enzymexla::map_symbol("___chkstk_ms", StackProbe);
#endif
#endif
  }
  return true;
}

enzymexla::CallInfo enzymexla::CompileHostModule(std::string &key,
                                                 mlir::ModuleOp modOp,
                                                 bool compileInit,
                                                 bool dump_final_module,
                                                 int lib_counter) {
  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext);
  auto llvmModule = translateModuleToLLVMIR(modOp, *ctx);
  if (!llvmModule) {
    llvm::errs() << "modOp: " << *modOp << "\n";
    llvm::errs() << "could not convert to LLVM IR\n";
    return {};
  }
  if (!::enzymexla::init_jit())
    return {};

  llvmModule->setDataLayout(JIT->getDataLayout());
  llvmModule->setTargetTriple(JIT->getTargetTriple());

  if (dump_final_module) {
    llvm::errs() << " final_llvm_module before jit: " << *llvmModule << "\n";
  }
  auto LibA = JIT->createJITDylib("enzymejitdl_" + std::to_string(lib_counter));
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

llvm::Error enzymexla::map_symbol(const char *name, void *symbol) {
  if (!init_jit())
    return llvm::make_error<llvm::StringError>("Failed to initialize JIT",
                                               llvm::inconvertibleErrorCode());

  MappedSymbols[JIT->mangleAndIntern(name)] = llvm::orc::ExecutorSymbolDef(
      llvm::orc::ExecutorAddr::fromPtr(symbol), llvm::JITSymbolFlags());

  return llvm::Error::success();
}

llvm::Expected<void *> enzymexla::lookup_symbol(const char *name) {
  if (!JIT)
    return llvm::make_error<llvm::StringError>("JIT not initialized",
                                               llvm::inconvertibleErrorCode());

  auto mangled_name = JIT->mangleAndIntern(name);
  if (!MappedSymbols.contains(mangled_name))
    return llvm::make_error<llvm::StringError>("Symbol not found",
                                               llvm::inconvertibleErrorCode());

  return MappedSymbols[mangled_name].toPtr<void *>();
}
