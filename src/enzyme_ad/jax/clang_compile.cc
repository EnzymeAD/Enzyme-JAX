//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang_compile.h"
#include "llvm/IRReader/IRReader.h"

#include <cstring>
#include <iostream>
#include <memory>
#include <setjmp.h>
#include <signal.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <unistd.h>

#include "clang/CodeGen/CodeGenAction.h"
#include "llvm-c/Core.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/LLLexer.h"
#include "llvm/AsmParser/LLParser.h"
#include "llvm/AsmParser/LLToken.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/AsmParser/SlotMapping.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "clang/AST/Decl.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LangStandard.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/Version.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/FrontendTool/Utils.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"

#include <Python.h>
#include <pybind11/pybind11.h>

#include "Enzyme/Enzyme.h"
#include "Enzyme/Utils.h"

namespace clang {
namespace driver {
namespace tools {
/// \p EnvVar is split by system delimiter for environment variables.
/// If \p ArgName is "-I", "-L", or an empty string, each entry from \p EnvVar
/// is prefixed by \p ArgName then added to \p Args. Otherwise, for each
/// entry of \p EnvVar, \p ArgName is added to \p Args first, then the entry
/// itself is added.
void addDirectoryList(const llvm::opt::ArgList &Args,
                      llvm::opt::ArgStringList &CmdArgs, const char *ArgName,
                      const char *EnvVar);
} // namespace tools
} // namespace driver
} // namespace clang

using namespace clang;
using namespace llvm;

class ArgumentList {
private:
  /// Helper storage.
  llvm::SmallVector<llvm::SmallString<0>> Storage;
  /// List of arguments
  llvm::opt::ArgStringList Args;

public:
  /// Add argument.
  ///
  /// The element stored will not be owned by this.
  void push_back(const char *Arg) { Args.push_back(Arg); }

  /// Add argument and ensure it will be valid before this passer's destruction.
  ///
  /// The element stored will be owned by this.
  /*
  template <typename... ArgTy> void emplace_back(ArgTy &&...Args) {
    // Store as a string
    std::string Buffer;
    llvm::raw_string_ostream Stream(Buffer);
    (Stream << ... << Args);
    emplace_back(llvm::StringRef(Stream.str()));
  }
  */

  void emplace_back(llvm::StringRef &&Arg) {
    push_back(Storage.emplace_back(Arg).c_str());
  }

  /// Return the underling argument list.
  ///
  /// The return value of this operation could be invalidated by subsequent
  /// calls to push_back() or emplace_back().
  llvm::opt::ArgStringList &getArguments() { return Args; }
};

/*
template <class T> class ptr_wrapper
{
    public:
        ptr_wrapper() : ptr(nullptr) {}
        ptr_wrapper(T* ptr) : ptr(ptr) {}
        ptr_wrapper(const ptr_wrapper& other) : ptr(other.ptr) {}
        T& operator* () const { return *ptr; }
        T* operator->() const { return  ptr; }
        T* get() const { return ptr; }
        T& operator[](std::size_t idx) const { return ptr[idx]; }
    private:
        T* ptr;
};
PYBIND11_DECLARE_HOLDER_TYPE(T, ptr_wrapper<T>, true);
*/

// Returns the TargetMachine instance or zero if no triple is provided.
static TargetMachine *GetTargetMachine(llvm::Triple TheTriple, StringRef CPUStr,
                                       StringRef FeaturesStr,
                                       const llvm::TargetOptions &Options,
                                       CodeGenOptLevel level) {
  std::string Error;
  const Target *TheTarget =
      TargetRegistry::lookupTarget(codegen::getMArch(), TheTriple, Error);
  // Some modules don't specify a triple, and this is okay.
  if (!TheTarget) {
    return nullptr;
  }

  return TheTarget->createTargetMachine(
      TheTriple.getTriple(), codegen::getCPUStr(), codegen::getFeaturesStr(),
      Options, codegen::getExplicitRelocModel(),
      codegen::getExplicitCodeModel(), level);
}

std::unique_ptr<llvm::Module>
GetLLVMFromJob(std::string filename, std::string filecontents, bool cpp,
               ArrayRef<std::string> pyargv, LLVMContext *Context,
               std::unique_ptr<llvm::Module> linkMod) {
  const llvm::opt::InputArgList Args;
  const char *binary = cpp ? "clang++" : "clang";
  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  auto *DiagsBuffer0 = new IgnoringDiagConsumer;

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagsBuffer);
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts0 = new DiagnosticOptions();
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID0(new DiagnosticIDs());
  DiagnosticsEngine Diags0(DiagID0, &*DiagOpts0, DiagsBuffer0);
  const std::unique_ptr<clang::driver::Driver> driver(new clang::driver::Driver(
      binary, llvm::sys::getDefaultTargetTriple(), Diags0));
  ArgumentList Argv;

  Argv.emplace_back(StringRef(filename));
  for (auto v : pyargv)
    Argv.emplace_back(v);

  SmallVector<const char *> PreArgs;
  PreArgs.push_back(binary);
  PreArgs.append(Argv.getArguments());
  PreArgs[1] = "-";
  const std::unique_ptr<clang::driver::Compilation> compilation(
      driver->BuildCompilation(PreArgs));

  Argv.push_back("-emit-llvm");
  Argv.push_back("-I/enzyme");
  Argv.push_back("-O1");
  Argv.push_back("-disable-llvm-passes");
  // Parse additional include paths from environment variables.
  // FIXME: We should probably sink the logic for handling these from the
  // frontend into the driver. It will allow deleting 4 otherwise unused flags.
  // CPATH - included following the user specified includes (but prior to
  // builtin and standard includes).
  clang::driver::tools::addDirectoryList(Args, Argv.getArguments(), "-I",
                                         "CPATH");
  // C_INCLUDE_PATH - system includes enabled when compiling C.
  clang::driver::tools::addDirectoryList(Args, Argv.getArguments(),
                                         "-c-isystem", "C_INCLUDE_PATH");
  // CPLUS_INCLUDE_PATH - system includes enabled when compiling C++.
  clang::driver::tools::addDirectoryList(Args, Argv.getArguments(),
                                         "-cxx-isystem", "CPLUS_INCLUDE_PATH");
  // OBJC_INCLUDE_PATH - system includes enabled when compiling ObjC.
  clang::driver::tools::addDirectoryList(Args, Argv.getArguments(),
                                         "-objc-isystem", "OBJC_INCLUDE_PATH");
  // OBJCPLUS_INCLUDE_PATH - system includes enabled when compiling ObjC++.
  clang::driver::tools::addDirectoryList(
      Args, Argv.getArguments(), "-objcxx-isystem", "OBJCPLUS_INCLUDE_PATH");

  auto &TC = compilation->getDefaultToolChain();
  if (cpp) {
    bool HasStdlibxxIsystem =
        false; // Args.hasArg(options::OPT_stdlibxx_isystem);
    HasStdlibxxIsystem
        ? TC.AddClangCXXStdlibIsystemArgs(Args, Argv.getArguments())
        : TC.AddClangCXXStdlibIncludeArgs(Args, Argv.getArguments());
  }

  TC.AddClangSystemIncludeArgs(Args, Argv.getArguments());

  SmallVector<char, 1> outputvec;

  std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());

  // Register the support for object-file-wrapped Clang modules.
  // auto PCHOps = Clang->getPCHContainerOperations();
  // PCHOps->registerWriter(std::make_unique<ObjectFilePCHContainerWriter>());
  // PCHOps->registerReader(std::make_unique<ObjectFilePCHContainerReader>());

  auto baseFS = createVFSFromCompilerInvocation(Clang->getInvocation(), Diags);

  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs(
      new llvm::vfs::InMemoryFileSystem());

  struct tm y2k = {};

  y2k.tm_hour = 0;
  y2k.tm_min = 0;
  y2k.tm_sec = 0;
  y2k.tm_year = 100;
  y2k.tm_mon = 0;
  y2k.tm_mday = 1;
  time_t timer = mktime(&y2k);

  fs->addFile(filename, timer,
              llvm::MemoryBuffer::getMemBuffer(
                  filecontents, filename, /*RequiresNullTerminator*/ false));
  fs->addFile("/enzyme/enzyme/utils", timer,
              llvm::MemoryBuffer::getMemBuffer(
                  R"(
namespace enzyme {
  template<typename RT=void, typename... Args>
  RT __enzyme_fwddiff(Args...);
  template<typename RT=void, typename... Args>
  RT __enzyme_autodiff(Args...);
  template<typename RT, typename... Args>
  RT __enzyme_augmentfwd(Args...);
  template<typename RT, typename... Args>
  RT __enzyme_reverse(Args...);
  template<typename... Args>
  std::size_t __enzyme_augmentsize(Args...);
}
extern "C" void prevent_stores(void*, ...);
extern "C" int enzyme_dup;
extern "C" int enzyme_const;
extern "C" int enzyme_dupnoneed;
extern "C" int enzyme_nooverwrite;
extern "C" int enzyme_tape;
extern "C" int enzyme_allocated;
  )",
                  "/enzyme/enzyme/utils", /*RequiresNullTerminator*/ false));
  fs->addFile("/enzyme/enzyme/tensor", timer,
              llvm::MemoryBuffer::getMemBuffer(
                  R"(
#include <stdint.h>
#include <tuple>
namespace enzyme {
using size_t=std::size_t;
template <typename T, size_t... n>
struct tensor;

template <typename T>
struct tensor<T>
{
   using dtype = T;
   auto static constexpr shape = std::make_tuple();

   T values;

   __attribute__((always_inline))
   T& operator[](size_t) {
     return values;
   }
   __attribute__((always_inline))
   const T& operator[](size_t) const {
     return values;
   }
   __attribute__((always_inline))
   T& operator()() {
     return values;
   }
   __attribute__((always_inline))
   const T& operator()() const {
     return values;
   }
   __attribute__((always_inline))
   operator T() const {
     return values;
   }

    __attribute__((always_inline))
    T operator=(T rhs)
    {
      return values = rhs;
    }
    __attribute__((always_inline))
    T operator+=(T rhs)
    {
      return values += rhs;
    }
    __attribute__((always_inline))
    T operator-=(T rhs)
    {
      return values -= rhs;
    }
    __attribute__((always_inline))
    T operator*=(T rhs)
    {
      return values *= rhs;
    }
    __attribute__((always_inline))
    T operator/=(T rhs)
    {
      return values /= rhs;
    }
};

template <typename T, size_t n0>
struct tensor<T, n0>
{
   using dtype = T;
   auto static constexpr shape = std::make_tuple(n0);

   T values[n0];

   __attribute__((always_inline))
   T& operator[](size_t i) {
     return values[i];
   }
   __attribute__((always_inline))
   const T& operator[](size_t i) const {
     return values[i];
   }
   __attribute__((always_inline))
   T& operator()(size_t i) {
     return values[i];
   }
   __attribute__((always_inline))
   const T& operator()(size_t i) const {
     return values[i];
   }

    __attribute__((always_inline))
    void operator=(T rhs)
    {
      for (size_t i=0; i<n0; i++)
        values[i] = rhs;
    }
    __attribute__((always_inline))
    void operator+=(T rhs)
    {
      for (size_t i=0; i<n0; i++)
        values[i] += rhs;
    }
    __attribute__((always_inline))
    void operator-=(T rhs)
    {
      for (size_t i=0; i<n0; i++)
        values[i] -= rhs;
    }
    __attribute__((always_inline))
    void operator*=(T rhs)
    {
      for (size_t i=0; i<n0; i++)
        values[i] *= rhs;
    }
    __attribute__((always_inline))
    void operator/=(T rhs)
    {
      for (size_t i=0; i<n0; i++)
        values[i] /= rhs;
    }
};

template<typename T, size_t n0, size_t... N>
struct tensor<T, n0, N...>
{
   using dtype = T;
   auto static constexpr shape = std::make_tuple(n0, N...);
   using ST = tensor<T, N...>;

   ST values[n0];

   __attribute__((always_inline))
   ST& operator[](size_t i) {
     return values[i];
   }
   __attribute__((always_inline))
   const ST& operator[](size_t i) const {
     return values[i];
   }
   __attribute__((always_inline))
   ST& operator()(size_t i) {
     return values[i];
   }
   __attribute__((always_inline))
   const ST& operator()(size_t i) const {
     return values[i];
   }

    __attribute__((always_inline))
    void operator=(T rhs)
    {
      for (size_t i=0; i<n0; i++)
        values[i] = rhs;
    }
    __attribute__((always_inline))
    void operator+=(T rhs)
    {
      for (size_t i=0; i<n0; i++)
        values[i] += rhs;
    }
    __attribute__((always_inline))
    void operator-=(T rhs)
    {
      for (size_t i=0; i<n0; i++)
        values[i] -= rhs;
    }
    __attribute__((always_inline))
    void operator*=(T rhs)
    {
      for (size_t i=0; i<n0; i++)
        values[i] *= rhs;
    }
    __attribute__((always_inline))
    void operator/=(T rhs)
    {
      for (size_t i=0; i<n0; i++)
        values[i] /= rhs;
    }
};

}
  )",
                  "/enzyme/enzyme/tensor", /*RequiresNullTerminator*/ false));

  std::unique_ptr<llvm::raw_pwrite_stream> outputStream(
      new llvm::raw_svector_ostream(outputvec));
  Clang->setOutputStream(std::move(outputStream));

  IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> fuseFS(
      new llvm::vfs::OverlayFileSystem(baseFS));
  fuseFS->pushOverlay(fs);
  fuseFS->pushOverlay(baseFS);

  Clang->createFileManager(fuseFS);

  bool Success = CompilerInvocation::CreateFromArgs(
      Clang->getInvocation(), Argv.getArguments(), Diags, binary);

  // Infer the builtin include path if unspecified.
  if (Clang->getHeaderSearchOpts().UseBuiltinIncludes &&
      Clang->getHeaderSearchOpts().ResourceDir.empty())
    Clang->getHeaderSearchOpts().ResourceDir =
        CompilerInvocation::GetResourcesPath(binary, /*MainAddr*/ 0x0);

  // Create the actual diagnostics engine.
  Clang->createDiagnostics(*fuseFS);
  if (!Clang->hasDiagnostics()) {
    llvm::errs() << " failed create diag\n";
    return {};
  }

  // Set an error handler, so that any LLVM backend diagnostics go through our
  // error handler.
  // llvm::install_fatal_error_handler(LLVMErrorHandler,
  //                                 static_cast<void*>(&Clang->getDiagnostics()));

  DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());
  if (!Success) {
    Clang->getDiagnosticClient().finish();
    llvm::errs() << " failed diag\n";
    return {};
  }

  assert(Context);
  auto Act = std::make_unique<EmitLLVMOnlyAction>(Context);
  Success = Clang->ExecuteAction(*Act);
  if (!Success) {
    llvm::errs() << " failed execute\n";
    return {};
  }

  auto mod = Act->takeModule();

  if (linkMod) {
    Linker::linkModules(*mod, std::move(linkMod));
  }

  for (auto &f : *mod) {
    if (f.empty())
      continue;
    if (f.getName() == "entry")
      continue;
    f.setLinkage(Function::LinkageTypes::InternalLinkage);
  }

  PipelineTuningOptions PTO;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  // Register the target library analysis directly and give it a customized
  // preset TLI.
  llvm::Triple triple(mod->getTargetTriple());
  std::unique_ptr<TargetLibraryInfoImpl> TLII(
      llvm::driver::createTLII(triple, Clang->getCodeGenOpts().getVecLib()));
  FAM.registerPass([&] { return TargetLibraryAnalysis(*TLII); });

  auto level = CodeGenOptLevel::Aggressive; // OptimizationLevel::O3;

  Triple ModuleTriple(mod->getTargetTriple());
  std::string CPUStr, FeaturesStr;

  auto ETM =
      llvm::orc::JITTargetMachineBuilder(llvm::Triple(mod->getTargetTriple()))
          .createTargetMachine();
  if (!ETM) {
    throw pybind11::value_error("failed to create targetmachine");
  }
  auto TM = std::move(ETM.get());

  std::optional<PGOOptions> PGOOpt;
  PassInstrumentationCallbacks PIC;
  PassBuilder PB(TM.get(), PTO, PGOOpt, &PIC);

  augmentPassBuilder(PB);

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM;
  if (Error Err = PB.parsePassPipeline(MPM, "default<O3>")) {
    throw pybind11::value_error(
        (Twine("failed to parse pass pipeline: ") + toString(std::move(Err)))
            .str());
  }
  MPM.run(*mod, MAM);

  auto F = mod->getFunction("prevent_stores");
  if (F) {
    for (const auto user : llvm::make_early_inc_range(F->users())) {
      auto CI = dyn_cast<CallInst>(user);
      if (!CI)
        continue;
      std::deque<std::pair<llvm::Value *, llvm::Value *>> todo;
      SmallVector<Value *, 1> cargs;
      for (auto &arg : CI->args())
        cargs.push_back(arg);
      CI->eraseFromParent();
      for (auto &arg : cargs) {
        Value *cur = getBaseObject(arg);
        assert(isa<LoadInst>(cur));
        for (auto U : cur->users())
          todo.emplace_back(U, cur);
      }
      std::set<std::pair<Value *, Value *>> seen;
      SmallPtrSet<Instruction *, 32> toErase;
      while (todo.size()) {
        auto pair = todo.back();
        todo.pop_back();
        auto [cur, prev] = pair;
        if (seen.count(pair))
          continue;
        seen.insert(pair);
        if (isPointerArithmeticInst(cur)) {
          for (auto u : cur->users())
            todo.emplace_back(u, cur);
          continue;
        }
        if (isa<LoadInst>(cur))
          continue;
        if (auto MTI = dyn_cast<MemTransferInst>(cur)) {
          if (MTI->getSource() == prev)
            continue;
        }
        if (auto CI = dyn_cast<CallInst>(cur))
          if (auto F = CI->getCalledFunction())
            if (F->getName() == "memset_pattern16")
              continue;
        if (auto MS = dyn_cast<MemSetInst>(cur)) {
          toErase.insert(MS);
          continue;
        }
        if (auto II = dyn_cast<IntrinsicInst>(cur)) {
          if (II->getIntrinsicID() == llvm::Intrinsic::dbg_value)
            continue;
        }
        if (isa<ICmpInst>(cur))
          continue;
        if (auto SI = dyn_cast<StoreInst>(cur)) {
          assert(SI->getPointerOperand() == prev);
          auto C = dyn_cast<Constant>(SI->getValueOperand());
          if (C && C->isNullValue()) {
          } else if (auto CF = dyn_cast_or_null<ConstantFP>(C)) {
            assert(CF->isZero());
          } else {
            llvm::errs() << "SI: " << *SI << " C: " << *SI->getValueOperand()
                         << "\n";
            assert(0);
          }
          toErase.insert(SI);
          continue;
        }
        std::string err_str;
        llvm::raw_string_ostream ss(err_str);
        ss << *mod << "\n";
        ss << " unsupported value to erase:\n";
        ss << " cur: " << *cur << " prev: " << *prev << "\n";
        throw pybind11::value_error(ss.str());
      }
      for (auto I : toErase) {
        I->eraseFromParent();
      }
    }
  }

  return mod;
}
