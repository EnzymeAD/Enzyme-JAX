//===- enzymemlir-opt.cpp - The enzymemlir-opt driver ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'enzymemlir-opt' tool, which is the enzyme analog
// of mlir-opt, used to drive compiler passes, e.g. for testing.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "src/enzyme_ad/jax/RegistryUtils.h"
#include "stablehlo/reference/InterpreterOps.h"
#include "stablehlo/tests/CheckOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LibraryOptionsContext.h"
#include "llvm/Support/TargetSelect.h"

using namespace mlir;

int main(int argc, char **argv) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::DialectRegistry registry;

  mlir::enzyme::prepareRegistry(registry);
  mlir::enzyme::registerDialects(registry);

  registry.insert<mlir::stablehlo::check::CheckDialect>();
  registry.insert<mlir::stablehlo::interpreter::InterpreterDialect>();

  mlir::enzyme::registerInterfaces(registry);
  mlir::enzyme::initializePasses();
  mlir::registerTransformsPasses();

  std::string helpHeader =
      mlir::registerCLIOptions("Enzyme modular optimizer driver", registry);
  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::init("-"));
  static llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));
  llvm::cl::parseCommandLineOptions(llvm::cl::getCAPILibraryOptions(),
                                    llvm::ArrayRef(argv, argc), helpHeader);
  llvm::ArrayRef<const char *> libArgs =
      llvm::cl::getCAPILibraryOptions().args();
  if (!libArgs.empty()) {
    llvm::SmallVector<const char *, 8> globalArgv;
    globalArgv.reserve(libArgs.size() + 1);
    globalArgv.push_back(argv[0]);
    globalArgv.append(libArgs.begin(), libArgs.end());
    (void)llvm::cl::ParseCommandLineOptions(static_cast<int>(globalArgv.size()),
                                            globalArgv.data(), helpHeader);
  }
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, inputFilename.getValue(),
                        outputFilename.getValue(), registry));
}
