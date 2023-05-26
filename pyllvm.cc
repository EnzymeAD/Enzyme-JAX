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
#include <iostream>

PYBIND11_MODULE(pyllvm, m) {
  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  m.def("compile", [&](std::string &s, pybind11::object args) {
    return GetLLVMFromJob("/pyenzyme/input.c", s, /*cpp*/false, args.ptr());
  });
  m.def("compilepp", [&](std::string &s, pybind11::object args) {
    return GetLLVMFromJob("/pyenzyme/input.cpp", s, /*cpp*/true, args.ptr());
  });

  pybind11::class_<llvm::Module>(m, "Module")
    .def("__str__", [&](llvm::Module &mod) {
      assert(&mod);
      llvm::SmallVector<char, 1> outputvec;
      llvm::raw_svector_ostream outputStream(outputvec);
      mod.print(outputStream, nullptr);
      std::string st(outputvec.data(), outputvec.size());
      return st;
  });
}
