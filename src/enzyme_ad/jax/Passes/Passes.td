//===- Passes.td - EnzymeXLA pass tablegen macros  ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYMEXLA_PASSES
#define ENZYMEXLA_PASSES

include "mlir/Pass/PassBase.td"

def ArithRaisingPass : Pass<"arith-raise"> {
  let summary = "Raise Arith to mhlo";
  let dependentDialects = [
    "arith::ArithDialect",
    "mhlo::MhloDialect",
    "stablehlo::StablehloDialect"
  ];
  let constructor = "mlir::enzyme::createArithRaisingPass()";
  let options = [
    Option<
      /*C++ variable name=*/"use_stablehlo",
      /*CLI argument=*/"stablehlo",
      /*type=*/"bool",
      /*default=*/"true",
      /*description=*/"Whether to raise to stablehlo vs mhlo"
    >
    ];
}

def EnzymeHLOOptPass : Pass<"enzyme-hlo-opt"> {
  let summary = "Optimize stablehlo";
  let dependentDialects = [
    "stablehlo::StablehloDialect"
  ];
  let constructor = "mlir::enzyme::createEnzymeHLOOptPass()";
}

def PrintPass : Pass<"print"> {
  let summary = "Print the module";
  let dependentDialects = [
  ];
  let constructor = "mlir::enzyme::createPrintPass()";
}

#endif
