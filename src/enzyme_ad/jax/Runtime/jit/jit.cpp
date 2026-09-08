#include "src/enzyme_ad/jax/Runtime/jit/jit.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"

bool enzymexla::init_jit() {
  if (JIT)
    return true;

  auto tJTMB = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tJTMB) {
    llvm::errs() << " jit host detection error: " << tJTMB.takeError() << "\n";
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
                auto obj =
                    std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
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

llvm::Error enzymexla::map_symbol(const char *name, void *symbol) {
  if (!JIT)
    return llvm::make_error<llvm::StringError>("JIT not initialized",
                                               llvm::inconvertibleErrorCode());

  MappedSymbols[JIT->mangleAndIntern(name)] = llvm::orc::ExecutorSymbolDef(
      llvm::orc::ExecutorAddr::fromPtr(symbol), llvm::JITSymbolFlags());

  return llvm::Error::success();
}

llvm::Expected<void *> enzymexla::lookup_symbol(const char *name) {
  if (!JIT)
    return llvm::make_error<llvm::StringError>("JIT not initialized",
                                               llvm::inconvertibleErrorCode());

  auto addr = JIT->lookup(name);
  if (!addr)
    return llvm::make_error<llvm::StringError>("Symbol not found",
                                               llvm::inconvertibleErrorCode());

  return addr->toPtr<void *>();
}
