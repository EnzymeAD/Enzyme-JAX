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
#include <regex>
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

#include "nanobind/nanobind.h"
// #include <Python.h>

#include "Enzyme/Enzyme.h"
#include "Enzyme/Utils.h"

#include "compile_with_xla.h"

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/cpu/cpu_executable.h"

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
      TheTriple, codegen::getCPUStr(), codegen::getFeaturesStr(), Options,
      codegen::getExplicitRelocModel(), codegen::getExplicitCodeModel(), level);
}

std::unique_ptr<llvm::Module>
GetLLVMFromJob(std::string filename, std::string filecontents, bool cpp,
               ArrayRef<std::string> pyargv, LLVMContext *Context,
               std::unique_ptr<llvm::Module> linkMod) {
  const llvm::opt::InputArgList Args;
  const char *binary = cpp ? "clang++" : "clang";
  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  DiagnosticOptions DiagOpts;
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  auto *DiagsBuffer0 = new IgnoringDiagConsumer;

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  DiagnosticsEngine Diags(DiagID, DiagOpts, DiagsBuffer);

  DiagnosticOptions DiagOpts0;
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID0(new DiagnosticIDs());
  DiagnosticsEngine Diags0(DiagID0, DiagOpts0, DiagsBuffer0);

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

  Clang->createVirtualFileSystem(fuseFS);
  Clang->createFileManager();

  bool Success = CompilerInvocation::CreateFromArgs(
      Clang->getInvocation(), Argv.getArguments(), Diags, binary);

  // Infer the builtin include path if unspecified.
  if (Clang->getHeaderSearchOpts().UseBuiltinIncludes &&
      Clang->getHeaderSearchOpts().ResourceDir.empty())
    Clang->getHeaderSearchOpts().ResourceDir = clang::GetResourcesPath(binary);

  // Create the actual diagnostics engine.
  Clang->setDiagnostics(Clang->createDiagnostics(*fuseFS, DiagOpts));
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
    throw nanobind::value_error("failed to create targetmachine");
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
    throw nanobind::value_error(
        (Twine("failed to parse pass pipeline: ") + toString(std::move(Err)))
            .str()
            .c_str());
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
        throw nanobind::value_error(ss.str().c_str());
      }
      for (auto I : toErase) {
        I->eraseFromParent();
      }
    }
  }

  return mod;
}

std::string make_type(std::string typenam, llvm::ArrayRef<int64_t> shape,
                      bool constv, ::Language lang) {
  std::string s =
      std::string(constv ? "const " : "") + "enzyme::tensor<" + typenam;
  for (auto v : shape) {
    s += ", " + std::to_string(v);
  }
  return s + ">";
}

std::tuple<std::unique_ptr<llvm::Module>, std::unique_ptr<llvm::LLVMContext>,
           size_t, size_t>
createLLVMMod(std::string fn, llvm::StringRef source,
              llvm::ArrayRef<llvm::SmallVector<int64_t>> out_shapes,
              llvm::ArrayRef<std::string> out_names,
              llvm::ArrayRef<llvm::SmallVector<int64_t>> in_shapes,
              llvm::ArrayRef<std::string> in_names, PyObject *pyargv, ABI mode,
              ::Language lang, bool xla_runtime,
              const std::string &pass_pipeline) {
  auto llvm_ctx = std::make_unique<llvm::LLVMContext>();

  std::string input;
  llvm::raw_string_ostream ss(input);
  ss << "#include <cstdint>\n";
  ss << "#include <enzyme/tensor>\n";
  ss << "#include <enzyme/utils>\n";

  std::unique_ptr<llvm::Module> linkMod;
  std::unique_ptr<xla::LocalExecutable> local_executable;
  std::string stringbuf;

  size_t tmpBuf = 0;
  llvm::StringRef origSource = source;
  switch (lang) {
  case ::Language::CPP:
    ss << source << "\n";
    break;

  case ::Language::MHLO: {
    local_executable = compile_mhlo_to_llvm_with_xla(
        source, stringbuf, xla_runtime, pass_pipeline);
    auto *cpu_executable =
        static_cast<xla::cpu::CpuExecutable *>(local_executable->executable());
    auto &assignment = cpu_executable->buffer_assignment();
    if (!xla_runtime) {
      size_t num_in = 0;
      for (auto &buf2 : assignment.Allocations()) {
        if (buf2.is_entry_computation_parameter()) {
          num_in++;
        }
      }
      if (num_in != in_shapes.size()) {
        std::string err_str;
        llvm::raw_string_ostream ss(err_str);
        ss << assignment.ToString() << "\n";
        ss << source << "\n";
        ss << " Number of mhlo inputs (" << num_in
           << ") != number of jax inputs (" << in_shapes.size() << "):\n";
        ss << source << "\n";
        throw nanobind::value_error(ss.str().c_str());
      }
      for (size_t i = 0; i < in_shapes.size(); i++) {
        ssize_t idx = -1;
        for (auto &buf2 : assignment.Allocations()) {
          if (!buf2.is_entry_computation_parameter())
            continue;
          if (buf2.parameter_number() != i)
            continue;
          assert(idx == -1);
          idx = buf2.index();
        }
        if (idx == -1) {
          std::string err_str;
          llvm::raw_string_ostream ss(err_str);
          ss << " Could not find input parameter (" << i
             << ") as hlo parameter:\n";
          ss << source << "\n";
          throw nanobind::value_error(ss.str().c_str());
        }
      }
    }
    source = stringbuf;
    if (xla_runtime)
      tmpBuf = 0;
    else
      tmpBuf = assignment.temp_allocation_total_size();
    // explicitly fall through
  }
  case ::Language::LLVM:
    llvm::SMDiagnostic Err;
    linkMod =
        llvm::parseIR(llvm::MemoryBufferRef(source, "<input>"), Err, *llvm_ctx);
    if (!linkMod) {
      std::string err_str;
      llvm::raw_string_ostream ss(err_str);
      Err.print("llvmsource", ss, false);
      throw nanobind::value_error(
          ("failed to compile LLVM: " + ss.str()).c_str());
    }
    assert(linkMod);
    if (lang == ::Language::MHLO) {
      auto *cpu_executable = static_cast<xla::cpu::CpuExecutable *>(
          local_executable->executable());
      llvm::StringRef fname = cpu_executable->module_name();
      if (fname.size() && fname[0] == '_')
        fname = fname.substr(1);
      auto F = linkMod->getFunction(fname);
      if (!F) {
        llvm::errs() << *linkMod << "\n";
        llvm::errs() << "fname: " << fname << "\n";
      }
      assert(F);
      fn = "mhlo_main";
      F->setName(fn);
      assert(!F->empty());
      for (auto &F2 : *linkMod)
        if (!F2.empty()) {
          F2.addFnAttr(llvm::Attribute::AlwaysInline);
          // Remove invariant_load if we expect enzyme to cache explicitly all
          // data. Otherwise invariant_load allows Enzyme to assume it need
          // not cache, and it is illegal for us to pass in nullptr as the
          // primal (since it may be needed).
          if (mode == ABI::Augmented || mode == ABI::Reverse ||
              mode == ABI::Tape) {
            for (auto &BB : F2)
              for (auto &I : BB)
                if (auto LI = llvm::dyn_cast<llvm::LoadInst>(&I))
                  if (LI->hasMetadata(llvm::LLVMContext::MD_invariant_load))
                    LI->setMetadata(llvm::LLVMContext::MD_invariant_load,
                                    nullptr);
          }
        }
    }
    if (xla_runtime) {
      ss << " extern \"C\" void " << fn << "(void* exec";

      for (size_t i = 0; i < in_shapes.size(); i++) {
        ss << ", void*";
        ss << ", void*";
        ss << ", int64_t";
        for (auto idx : in_shapes[i])
          ss << ", int64_t";
        for (auto idx : in_shapes[i])
          ss << ", int64_t";
      }

      for (size_t i = 0; i < out_shapes.size(); i++) {
        ss << ", void*";
        ss << ", void*";
        ss << ", int64_t";
        for (auto idx : out_shapes[i])
          ss << ", int64_t";
        for (auto idx : out_shapes[i])
          ss << ", int64_t";
      }
      ss << ");\n\n";
    } else {
      ss << " extern \"C\" void " << fn
         << "(void* retval, void* run_options, void* params, void* "
            "buffer_table, void* status, void* prof_counters);\n\n";
    }

    if (local_executable && !xla_runtime) {
      auto *cpu_executable = static_cast<xla::cpu::CpuExecutable *>(
          local_executable->executable());
      auto &assignment = cpu_executable->buffer_assignment();
      for (auto &buf : assignment.Allocations()) {
        if (!buf.is_constant())
          continue;
        assert(buf.assigned_buffers().size() == 1);
        auto hlo = buf.assigned_buffers().begin()->first;
        auto tyenum = hlo->shape().element_type();
        std::string ty;
        switch (tyenum) {
        case xla::PrimitiveType::S8:
          ty = "int8_t";
          break;
        case xla::PrimitiveType::S16:
          ty = "int16_t";
          break;
        case xla::PrimitiveType::S32:
          ty = "int32_t";
          break;
        case xla::PrimitiveType::S64:
          ty = "int64_t";
          break;
        case xla::PrimitiveType::U8:
          ty = "uint8_t";
          break;
        case xla::PrimitiveType::U16:
          ty = "uint16_t";
          break;
        case xla::PrimitiveType::U32:
          ty = "uint32_t";
          break;
        case xla::PrimitiveType::U64:
          ty = "uint64_t";
          break;
        case xla::PrimitiveType::F16:
          ty = "half";
          break;
        case xla::PrimitiveType::F32:
          ty = "float";
          break;
        case xla::PrimitiveType::F64:
          ty = "double";
          break;
        case xla::PrimitiveType::PRED:
          ty = "bool";
          break;
        default: {
          std::string err;
          llvm::raw_string_ostream ess(err);
          ess << " Failed to compile mhlo, unknown constant element type: "
              << hlo->shape().ToString() << "\n";
          throw std::runtime_error(ess.str());
        }
        }
        auto val = xla::Cast<xla::HloConstantInstruction>(hlo->instruction());

        llvm::ArrayRef<int64_t> shape(hlo->shape().dimensions().begin(),
                                      hlo->shape().dimensions().end());
        ss << "  static constexpr "
           << make_type(ty, shape, /*const*/ false, lang) << " const_"
           << buf.index() << " = ";

        xla::StringPrinter printer;
        val->literal().PrintWithoutShape(&printer);
        auto str = std::move(printer).ToString();
        if (shape.size() == 0)
          ss << "{";
        str = std::regex_replace(str, std::regex("\\{"), "{{");
        str = std::regex_replace(str, std::regex("\\}"), "}}");
        ss << str;
        if (shape.size() == 0)
          ss << "}";
        ss << ";\n";
      }
    }

    llvm::StringRef abiName = "abi_wrap";
    if (mode == ABI::Augmented)
      abiName = "aug_abi_wrap";
    else if (mode == ABI::Reverse)
      abiName = "rev_abi_wrap";
    ss << " __attribute__((always_inline)) static inline void " << abiName
       << "(";
    bool comma = false;
    for (size_t i = 0; i < out_shapes.size(); i++) {
      if (comma)
        ss << ", ";
      ss << " " << make_type(out_names[i], out_shapes[i], false, lang)
         << "& __restrict__ out_" << i;
      comma = true;
    }
    if (tmpBuf != 0) {
      if (comma)
        ss << ", ";
      ss << " enzyme::tensor<char, " << tmpBuf << "> & __restrict__ tmpBuf";
      comma = true;
    }
    for (size_t i = 0; i < in_shapes.size(); i++) {
      if (comma)
        ss << ", ";
      ss << " " << make_type(in_names[i], in_shapes[i], true, lang) << "& in_"
         << i;
      comma = true;
    }
    ss << ") {\n";
    if (xla_runtime) {
      ss << "  " << fn << "(/*exec*/nullptr";
      for (size_t i = 0; i < in_shapes.size(); i++) {
        ss << ", (void*)&in_" << i;
        ss << ", (void*)&in_" << i;
        ss << ", 0";
        for (auto idx : in_shapes[i])
          ss << ", " << idx;
        for (auto idx : in_shapes[i])
          ss << ", 0";
      }

      for (size_t i = 0; i < out_shapes.size(); i++) {
        ss << ", (void*)&out_" << i;
        ss << ", (void*)&out_" << i;
        ss << ", 0";
        for (auto idx : out_shapes[i])
          ss << ", " << idx;
        for (auto idx : out_shapes[i])
          ss << ", 0";
      }
      ss << ");\n";
    } else {
      size_t numBuffers = out_shapes.size() + in_shapes.size();
      std::vector<int> out_idxs;
      if (local_executable) {
        auto *cpu_executable = static_cast<xla::cpu::CpuExecutable *>(
            local_executable->executable());
        auto &assignment = cpu_executable->buffer_assignment();
        numBuffers = assignment.Allocations().size();
        if (out_shapes.size() == 1) {
          ssize_t idx = -1;
          for (auto &buf2 : assignment.Allocations()) {
            if (!buf2.maybe_live_out())
              continue;
            assert(!buf2.is_tuple());
            assert(idx == -1);
            idx = buf2.index();
          }
          assert(idx != -1);
          out_idxs.push_back(idx);
        } else {
          // If a tuple, find the tuple buf, then use that to index the
          // outputs.
          ssize_t tupidx = -1;
          for (auto &buf2 : assignment.Allocations()) {
            if (!buf2.maybe_live_out())
              continue;
            if (!buf2.is_tuple())
              continue;
            assert(tupidx == -1);
            tupidx = buf2.index();
          }
          assert(tupidx != -1);
          auto &tup_buf = assignment.Allocations()[tupidx];
          assert(tup_buf.assigned_buffers().size() == 1);
          auto hlo = tup_buf.assigned_buffers().begin()->first;
          auto val = hlo->instruction();
          assert(val->operand_count() == out_shapes.size());
          for (size_t i = 0; i < out_shapes.size(); i++) {
            ssize_t found = -1;
            auto operand = val->operand(i);
            while (found == -1) {
              for (auto &buf : assignment.Allocations()) {
                if (!buf.maybe_live_out())
                  continue;
                if (buf.is_tuple())
                  continue;
                bool contains_output = false;
                for (auto &pair : buf.assigned_buffers()) {
                  if (pair.first->instruction() != operand)
                    continue;
                  assert(!contains_output);
                  contains_output = true;
                  assert(pair.second.offset == 0);
                }
                if (!contains_output)
                  continue;
                assert(found == -1);
                found = buf.index();
              }
              if (operand->opcode() == xla::HloOpcode::kBitcast) {
                operand = operand->operand(0);
                continue;
              }
              break;
            }
            if (found == -1) {
              llvm::errs() << "assignment: " << assignment.ToString() << "\n";
              llvm::errs() << "val: " << val->ToString() << "\n";
              llvm::errs() << "vop: " << val->operand(i)->ToString() << "\n";
              llvm::errs() << "i: " << i << "\n";
            }
            assert(found != -1);
            out_idxs.push_back((int)found);
          }
        }
        for (auto &buf : assignment.Allocations()) {
          if (buf.is_thread_local()) {
            ss << "  char local_" << buf.index() << "[" << buf.size() << "];\n";
            continue;
          }
          if (!buf.maybe_live_out())
            continue;
          if (!buf.is_tuple())
            continue;
          ss << "  void* tup_" << buf.index() << "[" << out_idxs.size()
             << "] = {";

          for (size_t i = 0; i < out_idxs.size(); i++) {
            if (i != 0)
              ss << ", ";
            ss << " "
               << "(void*)&out_" << i;
          }
          ss << "};\n";
        }
      }
      ss << "  void* buffers[" << numBuffers << "] = {";

      if (local_executable) {
        auto *cpu_executable = static_cast<xla::cpu::CpuExecutable *>(
            local_executable->executable());
        auto &assignment = cpu_executable->buffer_assignment();
        for (auto &buf : assignment.Allocations()) {
          if (buf.index() != 0)
            ss << ", ";
          if (buf.is_entry_computation_parameter()) {
            ss << " "
               << "(void*)&in_" << buf.parameter_number();
          } else if (buf.IsPreallocatedTempBuffer()) {
            ss << " "
               << "(void*)&tmpBuf";
          } else if (buf.maybe_live_out()) {
            if (buf.is_tuple()) {
              assert(out_shapes.size() != 1);
              ss << " "
                 << "(void*)&tup_" << buf.index();
              continue;
            }
            auto it = std::find(out_idxs.begin(), out_idxs.end(), buf.index());
            assert(it != out_idxs.end());
            int index = it - out_idxs.begin();
            ss << " "
               << "(void*)&out_" << index;
          } else if (buf.is_constant()) {
            ss << " "
               << "(void*)&const_" << buf.index();
          } else if (buf.is_thread_local()) {
            ss << " "
               << "(void*)&local_" << buf.index();
          } else {
            std::string err;
            llvm::raw_string_ostream ess(err);
            ess << " Failed to compile mhlo, unknown buffer type\n";
            ess << origSource << "\n";
            ess << source << "\n";
            ess << local_executable->executable()->module().ToString() << "\n";
            ess << " unknown buffer type: " << buf.ToString() << "\n";
            throw std::runtime_error(ess.str());
          }
        }
      } else {
        comma = false;
        for (size_t i = 0; i < out_shapes.size(); i++) {
          if (comma)
            ss << ", ";
          ss << " "
             << "(void*)&out_" << i;
          comma = true;
        }
        for (size_t i = 0; i < in_shapes.size(); i++) {
          if (comma)
            ss << ", ";
          ss << " "
             << "(void*)&in_" << i;
          comma = true;
        }
        if (tmpBuf != 0) {
          if (comma)
            ss << ", ";
          ss << " "
             << "(void*)&tmpBuf";
          comma = true;
        }
      }
      ss << "};\n";
      ss << "  " << fn
         << "(nullptr, nullptr, nullptr, buffers, nullptr, nullptr);\n";
    }
    ss << "};\n";
    fn = abiName;
  }
  if (mode != ABI::Primal) {
    ss << " void entry_wrap(";
    bool comma = false;
    for (size_t i = 0; i < out_shapes.size(); i++) {
      if (comma)
        ss << ", ";
      ss << " " << make_type(out_names[i], out_shapes[i], false, lang)
         << "& __restrict__ out_" << i;
      comma = true;
    }
    if (tmpBuf != 0) {
      if (comma)
        ss << ", ";
      ss << " enzyme::tensor<char, " << tmpBuf << "> & __restrict__ tmpBuf";
      comma = true;
    }
    for (size_t i = 0; i < in_shapes.size(); i++) {
      if (comma)
        ss << ", ";
      ss << " " << make_type(in_names[i], in_shapes[i], true, lang) << "& in_"
         << i;
      comma = true;
    }
    ss << ") {\n";
    ss << "  " << fn << "(";
    comma = false;
    for (size_t i = 0; i < out_shapes.size(); i++) {
      if (comma)
        ss << ", ";
      ss << " "
         << "out_" << i;
      comma = true;
    }
    if (tmpBuf != 0) {
      if (comma)
        ss << ", ";
      ss << " "
         << "tmpBuf";
      comma = true;
    }
    for (size_t i = 0; i < in_shapes.size(); i++) {
      if (comma)
        ss << ", ";
      ss << " "
         << "in_" << i;
      comma = true;
    }
    ss << ");\n";
    ss << "}\n";
    fn = "entry_wrap";
  }
  if (mode == ABI::Tape)
    ss << "extern \"C\" std::size_t entry() {\n";
  else
    ss << "extern \"C\" void entry(void** __restrict__ outs, void** "
          "__restrict__ ins) {\n";
  size_t out_off = 0;
  size_t in_off = 0;

  if (mode == ABI::Reverse) {
    ss << " void*& tape = "
       << "*(void**)ins[" << in_off << "];\n";
    in_off++;
  }

  for (size_t i = 0; i < out_shapes.size(); i++) {
    if (mode != ABI::Reverse && mode != ABI::Tape) {
      ss << " " << make_type(out_names[i], out_shapes[i], false, lang)
         << "& out_" << i << " = "
         << "*(" << make_type(out_names[i], out_shapes[i], false, lang)
         << "*)outs[" << out_off << "];\n";
      out_off++;
    }
    if (mode == ABI::Forward) {
      ss << " " << make_type(out_names[i], out_shapes[i], false, lang)
         << "& dout_" << i << " = "
         << "*(" << make_type(out_names[i], out_shapes[i], false, lang)
         << "*)outs[" << out_off << "];\n";
      out_off++;
    }
    if (mode == ABI::Reverse) {
      ss << " " << make_type(out_names[i], out_shapes[i], true, lang)
         << "& dout_" << i << " = "
         << "*(" << make_type(out_names[i], out_shapes[i], true, lang)
         << "*)ins[" << in_off << "];\n";
      in_off++;
    }
  }

  for (size_t i = 0; i < in_shapes.size(); i++) {
    if (mode != ABI::Reverse && mode != ABI::Tape) {
      ss << " " << make_type(in_names[i], in_shapes[i], true, lang) << "& in_"
         << i << " = "
         << "*(" << make_type(in_names[i], in_shapes[i], true, lang) << "*)ins["
         << in_off << "];\n";
      in_off++;
    }
    if (mode == ABI::Forward) {
      ss << " " << make_type(in_names[i], in_shapes[i], true, lang) << "& din_"
         << i << " = "
         << "*(" << make_type(in_names[i], in_shapes[i], true, lang) << "*)ins["
         << in_off << "];\n";
      in_off++;
    }
    if (mode == ABI::Reverse) {
      ss << " " << make_type(in_names[i], in_shapes[i], false, lang) << "& din_"
         << i << " = "
         << "*(" << make_type(in_names[i], in_shapes[i], false, lang)
         << "*)outs[" << out_off << "];\n";
      out_off++;
    }
  }
  if (mode == ABI::Augmented) {
    ss << " void*& tape = "
       << "*(void**)outs[" << out_off << "];\n";
    out_off++;
  }
  if (mode != ABI::Tape && mode != ABI::Reverse && tmpBuf != 0) {
    ss << " enzyme::tensor<char, " << tmpBuf << ">& tmpBuf = "
       << "*(enzyme::tensor<char, " << tmpBuf << ">*)outs[" << out_off
       << "];\n";
    out_off++;
  }
  // forward mode, we have undef dtmpbuf
  if (mode == ABI::Forward && tmpBuf != 0) {
    ss << " enzyme::tensor<char, " << tmpBuf << ">& dtmpBuf = "
       << "*(enzyme::tensor<char, " << tmpBuf << ">*)outs[" << out_off
       << "];\n";
    out_off++;
  }
  // augmented forward mode, we have nullptr dtmpBuf
  if (mode == ABI::Augmented && tmpBuf != 0) {
    ss << "#pragma clang diagnostic push\n";
    ss << "#pragma clang diagnostic ignored \"-Wnull-dereference\"\n";
    ss << " enzyme::tensor<char, " << tmpBuf << ">& dtmpBuf = "
       << "*(enzyme::tensor<char, " << tmpBuf << ">*)(nullptr);\n";
    ss << "#pragma clang diagnostic pop\n";
  }
  // reverse mode, we have zero'd
  if (mode == ABI::Reverse && tmpBuf != 0) {
    ss << "#pragma clang diagnostic push\n";
    ss << "#pragma clang diagnostic ignored \"-Wnull-dereference\"\n";
    ss << " enzyme::tensor<char, " << tmpBuf << ">& tmpBuf = "
       << "*(enzyme::tensor<char, " << tmpBuf << ">*)(nullptr);\n";
    ss << "#pragma clang diagnostic pop\n";
    ss << " __builtin_memset(outs[" << out_off << "], 0, " << tmpBuf << ");\n";
    ss << " enzyme::tensor<char, " << tmpBuf << ">& dtmpBuf = "
       << "*(enzyme::tensor<char, " << tmpBuf << ">*)outs[" << out_off
       << "];\n";
    out_off++;
  }

  if (mode == ABI::Primal) {
    ss << "  " << fn << "(";
    bool comma = false;
    for (size_t i = 0; i < out_shapes.size(); i++) {
      if (comma)
        ss << ", ";
      ss << "out_" << i;
      comma = true;
    }
    if (tmpBuf != 0) {
      if (comma)
        ss << ", ";
      ss << "tmpBuf";
      comma = true;
    }
    for (size_t i = 0; i < in_shapes.size(); i++) {
      if (comma)
        ss << ", ";
      ss << "in_" << i;
      comma = true;
    }
    ss << ");\n";
  } else if (mode == ABI::Forward) {
    ss << "  enzyme::__enzyme_fwddiff(" << fn;
    for (size_t i = 0; i < out_shapes.size(); i++) {
      ss << ", enzyme_dup, ";
      ss << "&out_" << i << ", ";
      ss << "&dout_" << i;
    }
    if (tmpBuf != 0) {
      ss << ", enzyme_dup, &tmpBuf, &dtmpBuf";
    }
    for (size_t i = 0; i < in_shapes.size(); i++) {
      ss << ", enzyme_dup, ";
      ss << "&in_" << i << ", ";
      ss << "&din_" << i;
    }
    ss << ");\n";
  } else if (mode == ABI::Augmented) {
    // outs, tapeout
    // ins
    ss << "  std::size_t tapesize = enzyme::__enzyme_augmentsize(" << fn;
    for (size_t i = 0; i < out_shapes.size(); i++) {
      ss << ", enzyme_dup";
    }
    if (tmpBuf != 0) {
      ss << ", enzyme_dup";
    }
    for (size_t i = 0; i < in_shapes.size(); i++) {
      ss << ", enzyme_dup";
    }
    ss << ");\n";
    ss << "  enzyme::__enzyme_augmentfwd<void*>(" << fn
       << ", enzyme_allocated, tapesize, enzyme_tape, &tape";
    for (size_t i = 0; i < out_shapes.size(); i++) {
      ss << ", enzyme_dup, &out_" << i << ", nullptr";
    }
    if (tmpBuf != 0) {
      ss << ", enzyme_dup, &tmpBuf, &dtmpBuf";
    }
    for (size_t i = 0; i < in_shapes.size(); i++) {
      ss << ", enzyme_dup, &in_" << i << ", nullptr";
    }
    ss << ");\n";
  } else if (mode == ABI::Reverse) {

    // d_ins
    // tape, d_out

    // og outputs, og inputs
    //     doutputs (in), dinputs (out)
    ss << "  std::size_t tapesize = enzyme::__enzyme_augmentsize(" << fn;
    for (size_t i = 0; i < out_shapes.size(); i++) {
      ss << ", enzyme_dup";
    }
    if (tmpBuf != 0) {
      ss << ", enzyme_dup";
    }
    for (size_t i = 0; i < in_shapes.size(); i++) {
      ss << ", enzyme_dup";
    }
    ss << ");\n";
    for (size_t i = 0; i < in_shapes.size(); i++) {
      ss << "  din_" << i << " = (" << in_names[i] << ")0;\n";
    }
    ss << "  enzyme::__enzyme_reverse<void>(" << fn
       << ", enzyme_allocated, tapesize, enzyme_tape, &tape";
    for (size_t i = 0; i < out_shapes.size(); i++) {
      ss << ", enzyme_dup, nullptr, &dout_" << i;
    }
    if (tmpBuf != 0) {
      ss << ", enzyme_dup, &tmpBuf, &dtmpBuf";
    }
    for (size_t i = 0; i < in_shapes.size(); i++) {
      ss << ", enzyme_dup, nullptr, &din_" << i;
    }
    ss << ");\n";
    ss << "prevent_stores(";
    for (size_t i = 0; i < out_shapes.size(); i++) {
      if (i != 0)
        ss << ", ";
      ss << "(void*)&dout_" << i;
    }
    ss << ");\n";
  } else if (mode == ABI::Tape) {
    // outs, tapeout
    // ins
    ss << "  std::size_t tapesize = enzyme::__enzyme_augmentsize(" << fn;
    for (size_t i = 0; i < out_shapes.size(); i++) {
      ss << ", enzyme_dup";
    }
    if (tmpBuf != 0) {
      ss << ", enzyme_dup";
    }
    for (size_t i = 0; i < in_shapes.size(); i++) {
      ss << ", enzyme_dup";
    }
    ss << ");\n";
    ss << "  return tapesize;\n";
  } else {
    assert(0 && "unhandled mode");
  }
  ss << "}\n";

  llvm::SmallVector<std::string> pyargv_strs;
  assert(PySequence_Check(pyargv));
  auto sz = PySequence_Size(pyargv);
  for (Py_ssize_t i = 0; i < sz; ++i) {
    PyObject *item = PySequence_GetItem(pyargv, i);
#if PY_VERSION_HEX < 0x03000000
    auto argv = PyString_AsString(item);
#else
    auto argv = PyUnicode_AsUTF8(item);
#endif
    Py_DECREF(item);
    assert(argv);
    pyargv_strs.emplace_back(argv);
#if PY_VERSION_HEX < 0x03000000
    free(argv);
#else
    // should not free py3+
#endif
  }

  auto mod = GetLLVMFromJob("/enzyme_call/source.cpp", ss.str(), /*cpp*/ true,
                            pyargv_strs, llvm_ctx.get(), std::move(linkMod));
  if (!mod) {
    llvm::errs() << "Source:\n" << ss.str() << "\n";
    throw nanobind::value_error("failed to compile C++");
  }
  return std::make_tuple(std::move(mod), std::move(llvm_ctx), out_off, tmpBuf);
}