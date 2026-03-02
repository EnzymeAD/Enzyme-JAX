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
#include "llvm/Support/TargetSelect.h"

using namespace mlir;

int main(int argc, char **argv) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::DialectRegistry registry;

  mlir::enzyme::prepareRegistry(registry);
  mlir::enzyme::registerDialects(registry);
  mlir::enzyme::registerInterfaces(registry);
  mlir::enzyme::initializePasses();
  mlir::registerTransformsPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Enzyme modular optimizer driver", registry));
}
