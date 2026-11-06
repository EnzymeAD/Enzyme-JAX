//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_JAX_FLANG_COMPILE_H
#define ENZYME_JAX_FLANG_COMPILE_H

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace enzyme_jax {

/// Calling convention used by Flang for a dummy argument in lowered LLVM IR.
enum class ParameterPassingKind {
  RawPointer,
  BoxDescriptor,
  ByValue,
};

/// ABI metadata extracted from a Flang FIR procedure parameter or result.
struct FortranParameter {
  ParameterPassingKind PassingKind = ParameterPassingKind::RawPointer;
  int Rank = 0;
  int CfiTypeCode = -1;
  int CfiAttributeCode = 0;
  size_t ElementSizeInBytes = 0;
};

/// Compiled LLVM IR module and extracted Fortran procedure ABI metadata.
struct CompiledFortranModule {
  std::unique_ptr<llvm::LLVMContext> Context;
  std::unique_ptr<llvm::Module> Module;
  std::string ResolvedSymbolName;
  std::vector<FortranParameter> Parameters;
  bool ReturnsScalarByValue = false;
};

/// Compiles `FortranSource` with Flang and resolves `ProcedureName` against the
/// expected JAX output and input shapes and C element type names.
absl::StatusOr<CompiledFortranModule> compileFortranSourceToLLVM(
    llvm::StringRef ProcedureName, llvm::StringRef FortranSource,
    llvm::ArrayRef<llvm::SmallVector<int64_t>> OutputShapes,
    llvm::ArrayRef<std::string> OutputTypeNames,
    llvm::ArrayRef<llvm::SmallVector<int64_t>> InputShapes,
    llvm::ArrayRef<std::string> InputTypeNames,
    llvm::ArrayRef<std::string> CommandLineArgs);

} // namespace enzyme_jax

#endif // ENZYME_JAX_FLANG_COMPILE_H
