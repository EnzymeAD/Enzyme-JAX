//===- ReactantPipelineDriver.cpp - Exercise the Reactant entry point -------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/raise.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <string>

// Use the production importer, complete default pipeline, and LLVM exporter.
// No database is contacted: the exported module still calls external libpq.
int main(int argc, char **argv) {
  if ((argc != 3 && argc != 4) ||
      (std::string(argv[1]) != "cpu" && std::string(argv[1]) != "xla") ||
      (argc == 4 && std::string(argv[3]) != "--mlir")) {
    llvm::errs() << "usage: reactant-pipeline-driver <cpu|xla> <input.ll> "
                    "[--mlir]\n";
    return 1;
  }
  // These tests must use the real pipeline, regardless of the caller's shell.
  unsetenv("OVERRIDE_PASS_PIPELINE");
  unsetenv("REACTANT_OMP");
  unsetenv("EXPORT_REACTANT");
  auto input = llvm::MemoryBuffer::getFile(argv[2]);
  if (!input) {
    llvm::errs() << "cannot read " << argv[2] << ": "
                 << input.getError().message() << "\n";
    return 1;
  }
  MLIRRoundTripOptions options{};
  options.verifyEach = true;
  options.unrollMaxTripCount = 8;
  options.unrollMaxNestedOps = 256;
  options.exportMLIROnly = argc == 4;
  std::string result = runLLVMToMLIRRoundTrip(
      (*input)->getBuffer().str(), "", argv[1], "", &options);
  if (result.empty())
    return 1;
  if (options.exportMLIROnly) {
    llvm::outs() << result;
    return 0;
  }
  // The entry point returns bitcode. Reparse it and print readable LLVM IR so
  // FileCheck tests the ABI after the exporter as well as after MLIR passes.
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  auto module = llvm::parseIR(llvm::MemoryBufferRef(result, "pipeline-output"),
                             diagnostic, context);
  if (!module) {
    diagnostic.print(argv[0], llvm::errs());
    return 1;
  }
  if (llvm::verifyModule(*module, &llvm::errs()))
    return 1;
  module->print(llvm::outs(), nullptr);
  return 0;
}
