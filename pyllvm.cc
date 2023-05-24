//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "clang_compile.h"

#include "llvm/Support/TargetSelect.h"

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
