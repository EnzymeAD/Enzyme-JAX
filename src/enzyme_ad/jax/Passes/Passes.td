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
    "stablehlo::StablehloDialect",
    "tensor::TensorDialect"
  ];
  let constructor = "mlir::enzyme::createEnzymeHLOOptPass()";
  let options = [
    Option<
      /*C++ variable name=*/"all_finite",
      /*CLI argument=*/"all_finite",
      /*type=*/"bool",
      /*default=*/"false",
      /*description=*/"Whether to raise to assume all variables are finite"
    >,
    Option<
      /*C++ variable name=*/"no_nan",
      /*CLI argument=*/"no_nan",
      /*type=*/"bool",
      /*default=*/"false",
      /*description=*/"Whether to raise to assume no variables are nan"
    >,
    Option<
      /*C++ variable name=*/"max_constant_expansion",
      /*CLI argument=*/"max_constant_expansion",
      /*type=*/"size_t",
      /*default=*/"1024",
      /*description=*/"Maximum size to expand constants into"
    >,
    Option<
      /*C++ variable name=*/"max_iterations",
      /*CLI argument=*/"max_iterations",
      /*type=*/"int64_t",
      /*default=*/"100",
      /*description=*/"Maximum number of pattern iterations"
    >,
    Option<
      /*C++ variable name=*/"top_down",
      /*CLI argument=*/"top_down",
      /*type=*/"bool",
      /*default=*/"false",
      /*description=*/"Use top down traversal"
    >
    ];
}

def EnzymeHLOUnrollPass : Pass<"enzyme-hlo-unroll"> {
  let summary = "Unroll stablehlo";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
    "tensor::TensorDialect"
  ];
  let constructor = "mlir::enzyme::createEnzymeHLOUnrollPass()";
}

def PrintPass : Pass<"print"> {
  let summary = "Print the module";
  let dependentDialects = [
  ];
  let constructor = "mlir::enzyme::createPrintPass()";
}

#endif
