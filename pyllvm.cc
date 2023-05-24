//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <memory>
#include <string>
#include <cstring>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <setjmp.h>
#include <sys/time.h>

#include <llvm/AsmParser/LLLexer.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/AsmParser/LLParser.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm-c/Core.h>
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"


#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/Support/raw_ostream.h>

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

using namespace clang;
using namespace llvm;

class ArgumentList {
private:
  /// Helper storage.
  llvm::SmallVector<llvm::SmallString<0>> Storage;
  /// List of arguments
  llvm::SmallVector<const char *> Args;

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
  llvm::ArrayRef<const char *> getArguments() const { return Args; }
};



namespace py = pybind11;
using namespace llvm;

#include "llvm/Support/TargetSelect.h"
#include "llvm/AsmParser/LLLexer.h"
#include "llvm/AsmParser/LLToken.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/SlotMapping.h"

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

int GetLLVMFromJob(std::string filename, std::string filecontents, std::string &output) {
      const char *binary = "clang"; //Argv0; // CudaLower ? "clang++" : "clang";
  //const std::unique_ptr<clang::driver::Driver> driver(
  //    new clang::driver::Driver(binary, llvm::sys::getDefaultTargetTriple(), Diags));
  ArgumentList Argv;
  // Argv.push_back(binary);
  Argv.emplace_back(StringRef(filename));
    Argv.push_back("-v");
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
  SmallVector<char, 1> outputvec;
  
  {
  std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

  // Register the support for object-file-wrapped Clang modules.
  // auto PCHOps = Clang->getPCHContainerOperations();
  // PCHOps->registerWriter(std::make_unique<ObjectFilePCHContainerWriter>());
  // PCHOps->registerReader(std::make_unique<ObjectFilePCHContainerReader>());

  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagsBuffer);

  auto baseFS = createVFSFromCompilerInvocation(Clang->getInvocation(),
                                                 Diags);

  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs(new llvm::vfs::InMemoryFileSystem());

  struct tm y2k = {};

  y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
  y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;
  time_t timer = mktime(&y2k);

  fs->addFile(filename, timer, llvm::MemoryBuffer::getMemBuffer(filecontents, filename, /*RequiresNullTerminator*/false));

  IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> fuseFS(new llvm::vfs::OverlayFileSystem(baseFS));
  fuseFS->pushOverlay(fuseFS);

  std::unique_ptr<llvm::raw_pwrite_stream> outputStream(new llvm::raw_svector_ostream(outputvec));
  Clang->setOutputStream(std::move(outputStream));
  Clang->createFileManager(fs);


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

}


PYBIND11_MODULE(pyllvm, m) {
  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  m.def("compile", [&](std::string &s) {
    std::string output;
    GetLLVMFromJob("/pyenzyme/input.c", s, output);
      //obvious memory leak, but need to do something with memory since not taken
      return output; //std::shared_ptr<llvm::LLLexer>(new llvm::LLLexer(*(new std::string(s)),  *sm, *sd, Ctx));
  });
  m.def("compilepp", [&](std::string &s) {
    std::string output;
    GetLLVMFromJob("/pyenzyme/input.cpp", s, output);
      //obvious memory leak, but need to do something with memory since not taken
      return output; //std::shared_ptr<llvm::LLLexer>(new llvm::LLLexer(*(new std::string(s)),  *sm, *sd, Ctx));
  });
}
