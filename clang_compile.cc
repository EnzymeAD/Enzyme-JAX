//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang_compile.h"
#include "llvm/IRReader/IRReader.h"

#include <iostream>
#include <memory>
#include <string>
#include <cstring>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <setjmp.h>
#include <sys/time.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/LLLexer.h"
#include "llvm/AsmParser/LLParser.h"
#include "llvm/AsmParser/LLToken.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/AsmParser/SlotMapping.h"
#include "llvm-c/Core.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "clang/CodeGen/CodeGenAction.h"

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
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "llvm/Support/Host.h"
#include "clang/FrontendTool/Utils.h"

#include "llvm/Support/MemoryBufferRef.h"

#include <Python.h>
#include <pybind11/pybind11.h>

#include "Enzyme/Enzyme.h"

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
}
}
}

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
  llvm::opt::ArgStringList& getArguments() { return Args; }
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

static TargetLibraryInfoImpl *createTLII(llvm::Triple &&TargetTriple,
                                         const CodeGenOptions &CodeGenOpts) {
  TargetLibraryInfoImpl *TLII = new TargetLibraryInfoImpl(TargetTriple);
 
  switch (CodeGenOpts.getVecLib()) {
  case CodeGenOptions::Accelerate:
    TLII->addVectorizableFunctionsFromVecLib(TargetLibraryInfoImpl::Accelerate,
                                             TargetTriple);
    break;
  case CodeGenOptions::LIBMVEC:
    TLII->addVectorizableFunctionsFromVecLib(TargetLibraryInfoImpl::LIBMVEC_X86,
                                             TargetTriple);
    break;
  case CodeGenOptions::MASSV:
    TLII->addVectorizableFunctionsFromVecLib(TargetLibraryInfoImpl::MASSV,
                                             TargetTriple);
    break;
  case CodeGenOptions::SVML:
    TLII->addVectorizableFunctionsFromVecLib(TargetLibraryInfoImpl::SVML,
                                             TargetTriple);
    break;
  case CodeGenOptions::SLEEF:
    TLII->addVectorizableFunctionsFromVecLib(TargetLibraryInfoImpl::SLEEFGNUABI,
                                             TargetTriple);
    break;
  case CodeGenOptions::Darwin_libsystem_m:
    TLII->addVectorizableFunctionsFromVecLib(
        TargetLibraryInfoImpl::DarwinLibSystemM, TargetTriple);
    break;
  default:
    break;
  }
  return TLII;
}

static LLVMContext GlobalContext;

std::unique_ptr<llvm::Module> GetLLVMFromJob(std::string filename, std::string filecontents, bool cpp, PyObject* pyargv, LLVMContext* Context) {
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
  const std::unique_ptr<clang::driver::Driver> driver(
      new clang::driver::Driver(binary, llvm::sys::getDefaultTargetTriple(), Diags0));
  ArgumentList Argv;
  
  Argv.emplace_back(StringRef(filename));

  assert (PySequence_Check(pyargv));
	auto sz = PySequence_Size(pyargv);
    for (Py_ssize_t i = 0; i < sz; ++i) {
        PyObject* item = PySequence_GetItem(pyargv, i);
#if PY_VERSION_HEX < 0x03000000
        auto argv = PyString_AsString(item);
#else
        auto argv = PyUnicode_AsUTF8(item);
#endif
        Py_DECREF(item);
		assert(argv);
		Argv.emplace_back(StringRef(argv));
#if PY_VERSION_HEX < 0x03000000
        free(argv);
#else
        // should not free py3+
#endif
    }

  SmallVector<const char*> PreArgs;
  PreArgs.push_back(binary);
  PreArgs.append(Argv.getArguments());
  PreArgs[1] = "-";
  const std::unique_ptr<clang::driver::Compilation> compilation(
      driver->BuildCompilation(PreArgs));

  Argv.push_back("-S");
  Argv.push_back("-emit-llvm");
  Argv.push_back("-I/enzyme");
  // Parse additional include paths from environment variables.
  // FIXME: We should probably sink the logic for handling these from the
  // frontend into the driver. It will allow deleting 4 otherwise unused flags.
  // CPATH - included following the user specified includes (but prior to
  // builtin and standard includes).
  clang::driver::tools::addDirectoryList(Args, Argv.getArguments(), "-I", "CPATH");
  // C_INCLUDE_PATH - system includes enabled when compiling C.
  clang::driver::tools::addDirectoryList(Args, Argv.getArguments(), "-c-isystem", "C_INCLUDE_PATH");
  // CPLUS_INCLUDE_PATH - system includes enabled when compiling C++.
  clang::driver::tools::addDirectoryList(Args, Argv.getArguments(), "-cxx-isystem", "CPLUS_INCLUDE_PATH");
  // OBJC_INCLUDE_PATH - system includes enabled when compiling ObjC.
  clang::driver::tools::addDirectoryList(Args, Argv.getArguments(), "-objc-isystem", "OBJC_INCLUDE_PATH");
  // OBJCPLUS_INCLUDE_PATH - system includes enabled when compiling ObjC++.
  clang::driver::tools::addDirectoryList(Args, Argv.getArguments(), "-objcxx-isystem", "OBJCPLUS_INCLUDE_PATH");

  auto &TC = compilation->getDefaultToolChain();
  if (cpp) {
    bool HasStdlibxxIsystem = false; // Args.hasArg(options::OPT_stdlibxx_isystem);
          HasStdlibxxIsystem ? TC.AddClangCXXStdlibIsystemArgs(Args, Argv.getArguments())
                             : TC.AddClangCXXStdlibIncludeArgs(Args, Argv.getArguments());
  }

                                 TC.AddClangSystemIncludeArgs(Args, Argv.getArguments());
  
  SmallVector<char, 1> outputvec;
  
  std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());

  // Register the support for object-file-wrapped Clang modules.
  // auto PCHOps = Clang->getPCHContainerOperations();
  // PCHOps->registerWriter(std::make_unique<ObjectFilePCHContainerWriter>());
  // PCHOps->registerReader(std::make_unique<ObjectFilePCHContainerReader>());


  auto baseFS = createVFSFromCompilerInvocation(Clang->getInvocation(),
                                                 Diags);

  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs(new llvm::vfs::InMemoryFileSystem());

  struct tm y2k = {};

  y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
  y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;
  time_t timer = mktime(&y2k);

  fs->addFile(filename, timer, llvm::MemoryBuffer::getMemBuffer(filecontents, filename, /*RequiresNullTerminator*/false));
  fs->addFile("/enzyme/enzyme/utils", timer, llvm::MemoryBuffer::getMemBuffer(R"(
namespace enzyme {
  template<typename RT=void, typename... Args>
  RT __enzyme_fwddiff(Args...);
  template<typename RT=void, typename... Args>
  RT __enzyme_autodiff(Args...);
}
extern "C" int enzyme_dup;
extern "C" int enzyme_const;
extern "C" int enzyme_dupnoneed;
  )", "/enzyme/enzyme/utils", /*RequiresNullTerminator*/false));
  fs->addFile("/enzyme/enzyme/tensor", timer, llvm::MemoryBuffer::getMemBuffer(R"(
// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#define MFEM_HOST_DEVICE 
#include <stdint.h>
namespace enzyme {
using size_t=std::size_t;
template <typename T, size_t... n>
struct tensor;

/// The implementation can be drastically generalized by using concepts of the
/// c++17 standard.

template < typename T >
struct tensor<T>
{
   using type = T;
   static constexpr size_t ndim      = 1;
   static constexpr size_t first_dim = 0;
   T& operator[](size_t /*unused*/) { return values; }
   MFEM_HOST_DEVICE const T& operator[](size_t /*unused*/) const { return values; }
   MFEM_HOST_DEVICE T& operator()(size_t /*unused*/) { return values; }
   MFEM_HOST_DEVICE const T& operator()(size_t /*unused*/) const { return values; }
   MFEM_HOST_DEVICE operator T() const { return values; }
   T values;

  __attribute__((always_inline))
  MFEM_HOST_DEVICE T operator=(T rhs)
  {
    values = rhs;
    return rhs;
  }
};

template < typename T, size_t n0 >
struct tensor<T, n0>
{
   using type = T;
   static constexpr size_t ndim      = 1;
   static constexpr size_t first_dim = n0;
   MFEM_HOST_DEVICE T& operator[](size_t i) { return values[i]; }
   MFEM_HOST_DEVICE const T& operator[](size_t i) const { return values[i]; }
   MFEM_HOST_DEVICE T& operator()(size_t i) { return values[i]; }
   MFEM_HOST_DEVICE const T& operator()(size_t i) const { return values[i]; }
   T values[n0];

  __attribute__((always_inline))
  MFEM_HOST_DEVICE T operator=(T rhs)
  {
    for (size_t i=0; i<n0; i++)
      values[i] = rhs;
    return rhs;
  }
};

template < typename T, size_t n0, size_t n1 >
struct tensor<T, n0, n1>
{
   using type = T;
   static constexpr size_t ndim      = 2;
   static constexpr size_t first_dim = n0;
   MFEM_HOST_DEVICE tensor< T, n1 >& operator[](size_t i) { return values[i]; }
   MFEM_HOST_DEVICE const tensor< T, n1 >& operator[](size_t i) const { return values[i]; }
   MFEM_HOST_DEVICE tensor< T, n1 >& operator()(size_t i) { return values[i]; }
   MFEM_HOST_DEVICE const tensor< T, n1 >& operator()(size_t i) const { return values[i]; }
   MFEM_HOST_DEVICE T& operator()(size_t i, size_t j) { return values[i][j]; }
   MFEM_HOST_DEVICE const T& operator()(size_t i, size_t j) const { return values[i][j]; }
   tensor < T, n1 > values[n0];

  __attribute__((always_inline))
  MFEM_HOST_DEVICE T operator=(T rhs)
  {
    for (size_t i=0; i<n0; i++)
      values[i] = rhs;
    return rhs;
  }
};

template < typename T, size_t n0, size_t n1, size_t n2 >
struct tensor<T, n0, n1, n2>
{
   using type = T;
   static constexpr size_t ndim      = 3;
   static constexpr size_t first_dim = n0;
   MFEM_HOST_DEVICE tensor< T, n1, n2 >& operator[](size_t i) { return values[i]; }
   MFEM_HOST_DEVICE const tensor< T, n1, n2 >& operator[](size_t i) const { return values[i]; }
   MFEM_HOST_DEVICE tensor< T, n1, n2 >& operator()(size_t i) { return values[i]; }
   MFEM_HOST_DEVICE const tensor< T, n1, n2 >& operator()(size_t i) const { return values[i]; }
   MFEM_HOST_DEVICE tensor< T, n2 >& operator()(size_t i, size_t j) { return values[i][j]; }
   MFEM_HOST_DEVICE const tensor< T, n2 >& operator()(size_t i, size_t j) const { return values[i][j]; }
   MFEM_HOST_DEVICE T& operator()(size_t i, size_t j, size_t k) { return values[i][j][k]; }
   MFEM_HOST_DEVICE const T& operator()(size_t i, size_t j, size_t k) const { return values[i][j][k]; }
   tensor < T, n1, n2 > values[n0];

  __attribute__((always_inline))
  MFEM_HOST_DEVICE T operator=(T rhs)
  {
    for (size_t i=0; i<n0; i++)
      values[i] = rhs;
    return rhs;
  }
};

template < typename T, size_t n0, size_t n1, size_t n2, size_t n3 >
struct tensor<T, n0, n1, n2, n3>
{
   using type = T;
   static constexpr size_t ndim      = 4;
   static constexpr size_t first_dim = n0;
   MFEM_HOST_DEVICE tensor< T, n1, n2, n3 >& operator[](size_t i) { return values[i]; }
   MFEM_HOST_DEVICE const tensor< T, n1, n2, n3 >& operator[](size_t i) const { return values[i]; }
   MFEM_HOST_DEVICE tensor< T, n1, n2, n3 >& operator()(size_t i) { return values[i]; }
   MFEM_HOST_DEVICE const tensor< T, n1, n2, n3 >& operator()(size_t i) const { return values[i]; }
   MFEM_HOST_DEVICE tensor< T, n2, n3 >& operator()(size_t i, size_t j) { return values[i][j]; }
   MFEM_HOST_DEVICE const tensor< T, n2, n3 >& operator()(size_t i, size_t j) const { return values[i][j]; }
   MFEM_HOST_DEVICE tensor< T, n3 >& operator()(size_t i, size_t j, size_t k) { return values[i][j][k]; }
   MFEM_HOST_DEVICE const tensor< T, n3 >& operator()(size_t i, size_t j, size_t k) const { return values[i][j][k]; }
   MFEM_HOST_DEVICE T& operator()(size_t i, size_t j, size_t k, size_t l) { return values[i][j][k][l]; }
   MFEM_HOST_DEVICE const T&  operator()(size_t i, size_t j, size_t k, size_t l) const { return values[i][j][k][l]; }
   tensor < T, n1, n2, n3 > values[n0];

  __attribute__((always_inline))
  MFEM_HOST_DEVICE T operator=(T rhs)
  {
    for (size_t i=0; i<n0; i++)
      values[i] = rhs;
    return rhs;
  }
};

template < typename T, size_t n0, size_t n1, size_t n2, size_t n3, size_t n4 >
struct tensor<T, n0, n1, n2, n3, n4>
{
   using type = T;
   static constexpr size_t ndim      = 5;
   static constexpr size_t first_dim = n0;
   MFEM_HOST_DEVICE tensor< T, n1, n2, n3, n4 >& operator[](size_t i) { return values[i]; }
   MFEM_HOST_DEVICE const tensor< T, n1, n2, n3, n4 >& operator[](size_t i) const { return values[i]; }
   MFEM_HOST_DEVICE tensor< T, n1, n2, n3, n4 >& operator()(size_t i) { return values[i]; }
   MFEM_HOST_DEVICE const tensor< T, n1, n2, n3, n4 >& operator()(size_t i) const { return values[i]; }
   MFEM_HOST_DEVICE tensor< T, n2, n3, n4 >& operator()(size_t i, size_t j) { return values[i][j]; }
   MFEM_HOST_DEVICE const tensor< T, n2, n3, n4 >& operator()(size_t i,
                                                              size_t j) const { return values[i][j]; }
   MFEM_HOST_DEVICE tensor< T, n3, n4>& operator()(size_t i, size_t j, size_t k) { return values[i][j][k]; }
   MFEM_HOST_DEVICE const tensor< T, n3, n4>& operator()(size_t i, size_t j,
                                                         size_t k) const { return values[i][j][k]; }
   MFEM_HOST_DEVICE tensor< T, n4 >& operator()(size_t i, size_t j, size_t k, size_t l) { return values[i][j][k][l]; }
   MFEM_HOST_DEVICE const tensor< T, n4 >& operator()(size_t i, size_t j, size_t k,
                                                      size_t l) const { return values[i][j][k][l]; }
   MFEM_HOST_DEVICE T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m) { return values[i][j][k][l][m]; }
   MFEM_HOST_DEVICE const T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m) const { return values[i][j][k][l][m]; }
   tensor < T, n1, n2, n3, n4 > values[n0];

  __attribute__((always_inline))
  MFEM_HOST_DEVICE T operator+=(T rhs)
  {
    for (size_t i=0; i<n0; i++)
      values[i] = rhs;
    return rhs;
  }
};



template<typename T, size_t... N>
void operator+=(tensor<T, N...> & lhs, T rhs);

template<typename T, size_t N0>
__attribute__((always_inline))
void operator+=(tensor<T, N0> & lhs, T rhs)
{
  for (size_t i=0; i<N0; i++)
    lhs.values[i] += rhs;
}

template<typename T, size_t N0, size_t... N>
__attribute__((always_inline))
void operator+=(tensor<T, N0, N...> & lhs, T rhs)
{
  for (size_t i=0; i<N0; i++)
    lhs.values[i] += rhs;
}

}
  )", "/enzyme/enzyme/tensor", /*RequiresNullTerminator*/false));

  std::unique_ptr<llvm::raw_pwrite_stream> outputStream(new llvm::raw_svector_ostream(outputvec));
  Clang->setOutputStream(std::move(outputStream));

  IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> fuseFS(new llvm::vfs::OverlayFileSystem(baseFS));
  fuseFS->pushOverlay(fs);
  fuseFS->pushOverlay(baseFS);

  Clang->createFileManager(fuseFS);


  bool Success = CompilerInvocation::CreateFromArgs(Clang->getInvocation(),
                                                    Argv.getArguments(), Diags, binary);

  // Infer the builtin include path if unspecified.
  if (Clang->getHeaderSearchOpts().UseBuiltinIncludes &&
      Clang->getHeaderSearchOpts().ResourceDir.empty())
    Clang->getHeaderSearchOpts().ResourceDir =
      CompilerInvocation::GetResourcesPath(binary, /*MainAddr*/0x0);

  // Create the actual diagnostics engine.
  Clang->createDiagnostics();
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

  if (!Context) Context=&GlobalContext;
  auto Act = std::make_unique<EmitLLVMOnlyAction>(Context);
  Success = Clang->ExecuteAction(*Act);
  if (!Success) {
    llvm::errs() << " failed execute\n";
    return {};
  }

  auto mod = Act->takeModule();
  
  PipelineTuningOptions PTO;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  // Register the target library analysis directly and give it a customized
  // preset TLI.
  std::unique_ptr<TargetLibraryInfoImpl> TLII(
      createTLII(llvm::Triple(mod->getTargetTriple()), Clang->getCodeGenOpts()));
  FAM.registerPass([&] { return TargetLibraryAnalysis(*TLII); });

  std::optional<PGOOptions> PGOOpt;
  PassInstrumentationCallbacks PIC;
  PassBuilder PB(nullptr, PTO, PGOOpt, &PIC);

  augmentPassBuilder(PB);

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM;

    // Map our optimization levels into one of the distinct levels used to
    // configure the pipeline.
    OptimizationLevel Level = OptimizationLevel::O3;
    
    MPM = PB.buildPerModuleDefaultPipeline(Level);
 
    MPM.run(*mod, MAM);
  return mod;
}

