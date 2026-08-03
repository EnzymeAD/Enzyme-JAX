//===- EnzymeXLALspServerMain.cpp - Enzyme-JAX MLIR language server -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "src/enzyme_ad/jax/RegistryUtils.h"
#include "stablehlo/reference/InterpreterOps.h"
#include "stablehlo/tests/CheckOps.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  mlir::enzyme::prepareRegistry(registry);
  mlir::enzyme::registerDialects(registry);

  registry.insert<mlir::stablehlo::check::CheckDialect>();
  registry.insert<mlir::stablehlo::interpreter::InterpreterDialect>();

  mlir::enzyme::registerInterfaces(registry);

  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
