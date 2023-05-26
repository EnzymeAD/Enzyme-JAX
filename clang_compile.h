//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_JAX_CLANG_COMPILE_H
#define ENZYME_JAX_CLANG_COMPILE_H

#include <Python.h>
#include <string>
#include "llvm/IR/Module.h"

std::unique_ptr<llvm::Module> GetLLVMFromJob(std::string filename, std::string filecontents, bool cpp, PyObject* pyargv, llvm::LLVMContext*ctx=nullptr);

#endif  // ENZYME_JAX_CLANG_COMPILE_H
