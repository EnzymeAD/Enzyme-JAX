//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_JAX_CLANG_COMPILE_H
#define ENZYME_JAX_CLANG_COMPILE_H

#include "absl/status/statusor.h"
#include "llvm/IR/Module.h"
#include <string>

enum class ABI { Primal, Forward, Augmented, Reverse, Tape };

enum class Language : int { CPP = 0, LLVM = 1, MHLO = 2 };

enum class CallABI : int { Tensor = 0, RawEntry = 1 };

absl::StatusOr<std::unique_ptr<llvm::Module>>
GetLLVMFromJob(std::string filename, std::string filecontents, bool cpp,
               llvm::ArrayRef<std::string> pyargv,
               llvm::LLVMContext *ctx = nullptr,
               std::unique_ptr<llvm::Module> linkMod = nullptr);

absl::StatusOr<std::tuple<std::unique_ptr<llvm::Module>,
                          std::unique_ptr<llvm::LLVMContext>, size_t, size_t>>
createLLVMMod(std::string fn, llvm::StringRef source,
              llvm::ArrayRef<llvm::SmallVector<int64_t>> out_shapes,
              llvm::ArrayRef<std::string> out_names,
              llvm::ArrayRef<llvm::SmallVector<int64_t>> in_shapes,
              llvm::ArrayRef<std::string> in_names,
              const std::vector<std::string> &pyargv_strs, ABI mode,
              Language lang, CallABI call_abi, bool xla_runtime,
              const std::string &pass_pipeline);
#endif // ENZYME_JAX_CLANG_COMPILE_H
