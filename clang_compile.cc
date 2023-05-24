//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang_compile.h"

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

#include <Python.h>
#include <pybind11/pybind11.h>
#if PY_VERSION_HEX < 0x03000000
#define MyPyText_AsString PyString_AsString
#else
#define MyPyText_AsString PyUnicode_AsUTF8
#endif


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

int GetLLVMFromJob(std::string filename, std::string filecontents, std::string &output, bool cpp, PyObject* pyargv) {
    const llvm::opt::InputArgList Args;
      const char *binary = cpp ? "clang++" : "clang"; 
  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagsBuffer);
  const std::unique_ptr<clang::driver::Driver> driver(
      new clang::driver::Driver(binary, llvm::sys::getDefaultTargetTriple(), Diags));
  ArgumentList Argv;
  // Argv.push_back(binary);
  Argv.emplace_back(StringRef(filename));
    Argv.push_back("-v");


  assert (PySequence_Check(pyargv));
	auto sz = PySequence_Size(pyargv);
    for (Py_ssize_t i = 0; i < sz; ++i) {
        PyObject* item = PySequence_GetItem(pyargv, i);
        auto argv = (char*)MyPyText_AsString(item);
        Py_DECREF(item);
		assert(argv);
		Argv.emplace_back(StringRef(argv));
        free(argv);
    }



  const std::unique_ptr<clang::driver::Compilation> compilation(
      driver->BuildCompilation(Argv.getArguments()));

  /*
  for (const auto &filename : filenames) {
    Argv.emplace_back(filename);
  }
  */
  /*
  if (FOpenMP)
    Argv.push_back("-fopenmp");
  if (TargetTripleOpt != "") {
    Argv.push_back("-target");
    Argv.emplace_back(TargetTripleOpt);
  }
  if (McpuOpt != "") {
    Argv.emplace_back("-mcpu=", McpuOpt);
  }
  if (Standard != "") {
    Argv.emplace_back("-std=", Standard);
  }
  if (ResourceDir != "") {
    Argv.push_back("-resource-dir");
    Argv.emplace_back(ResourceDir);
  }
  if (SysRoot != "") {
    Argv.push_back("--sysroot");
    Argv.emplace_back(SysRoot);
  }
  if (Verbose) {
    Argv.push_back("-v");
  }
  if (NoCUDAInc) {
    Argv.push_back("-nocudainc");
  }
  if (NoCUDALib) {
    Argv.push_back("-nocudalib");
  }
  if (CUDAGPUArch != "") {
    Argv.emplace_back("--cuda-gpu-arch=", CUDAGPUArch);
  }
  if (CUDAPath != "") {
    Argv.emplace_back("--cuda-path=", CUDAPath);
  }
  if (MArch != "") {
    Argv.emplace_back("-march=", MArch);
  }
  for (const auto &dir : includeDirs) {
    Argv.push_back("-I");
    Argv.emplace_back(dir);
  }
  for (const auto &define : defines) {
    Argv.emplace_back("-D", define);
  }
  for (const auto &Include : Includes) {
    Argv.push_back("-include");
    Argv.emplace_back(Include);
  }
  */

  Argv.push_back("-emit-llvm");
  Argv.push_back("-S");
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
  
  {
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
  if (!Clang->hasDiagnostics())
    return 1;

  // Set an error handler, so that any LLVM backend diagnostics go through our
  // error handler.
  // llvm::install_fatal_error_handler(LLVMErrorHandler,
  //                                 static_cast<void*>(&Clang->getDiagnostics()));

  DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());
  if (!Success) {
    Clang->getDiagnosticClient().finish();
    return 1;
  }

  // Execute the frontend actions.
  {
    Success = ExecuteCompilerInvocation(Clang.get());
  }
  }

  output.assign(outputvec.data(), outputvec.size());

  return 0;
}

