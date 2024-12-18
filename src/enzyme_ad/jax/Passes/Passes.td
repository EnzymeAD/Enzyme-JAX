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
    "stablehlo::StablehloDialect",
    "chlo::ChloDialect"
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
    >,
    Option<
      /*C++ variable name=*/"cse",
      /*CLI argument=*/"cse",
      /*type=*/"bool",
      /*default=*/"true",
      /*description=*/"Run CSE alongside"
    >,
    Option<
      /*C++ variable name=*/"passses",
      /*CLI argument=*/"passses",
      /*type=*/"uint64_t",
      /*default=*/"24575",
      /*description=*/"Additional optimization passes"
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
  let options = [
    Option<
      /*C++ variable name=*/"use_stdout",
      /*CLI argument=*/"stdout",
      /*type=*/"bool",
      /*default=*/"true",
      /*description=*/"Whether to print to stdout (vs stderr)"
    >
    ];
}

def LowerKernelPass : Pass<"lower-kernel"> {
  let summary = "Lower kernel to custom call";
  let dependentDialects = [
  ];
  let constructor = "mlir::enzyme::createLowerKernelPass()";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
    "gpu::GPUDialect",
    "func::FuncDialect",
    "math::MathDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "vector::VectorDialect",
    "nvgpu::NVGPUDialect",
    "NVVM::NVVMDialect",
    "LLVM::LLVMDialect",
    "arith::ArithDialect",
    "tensor::TensorDialect",
  ];

  let options = [
    Option<
      /*C++ variable name=*/"jit",
      /*CLI argument=*/"jit",
      /*type=*/"bool",
      /*default=*/"true",
      /*description=*/"Whether to jit the kernel"
    >,
    Option<
      /*C++ variable name=*/"toolkitPath",
      /*CLI argument=*/"toolkitPath",
      /*type=*/"std::string",
      /*default=*/"",
      /*description=*/"The location of the cuda toolkit"
    >,
    Option<
      /*C++ variable name=*/"linkFiles",
      /*CLI argument=*/"linkFiles",
      /*type=*/"std::string",
      /*default=*/"",
      /*description=*/"Semicolon separated list of files to link"
    >,
    Option<
      /*C++ variable name=*/"cubinChip",
      /*CLI argument=*/"cubinChip",
      /*type=*/"std::string",
      /*default=*/"\"sm_50\"",
      /*description=*/"cubinChip"
    >,
    Option<
      /*C++ variable name=*/"cubinFeatures",
      /*CLI argument=*/"cubinFeatures",
      /*type=*/"std::string",
      /*default=*/"\"+ptx60\"",
      /*description=*/"cubinChip"
    >,
    Option<
      /*C++ variable name=*/"indexBitWidth",
      /*CLI argument=*/"indexBitWidth",
      /*type=*/"int",
      /*default=*/"64",
      /*description=*/"indexBitWidth"
    >,
    Option<
      /*C++ variable name=*/"cuLaunchKernelPtr",
      /*CLI argument=*/"cuLaunchKernelPtr",
      /*type=*/"size_t",
      /*default=*/"0",
      /*description=*/"cuLaunchKernelPtr"
    >,
    Option<
      /*C++ variable name=*/"cuModuleLoadDataPtr",
      /*CLI argument=*/"cuModuleLoadDataPtr",
      /*type=*/"size_t",
      /*default=*/"0",
      /*description=*/"cuModuleLoadDataPtr"
    >,
    Option<
      /*C++ variable name=*/"cuModuleGetFunctionPtr",
      /*CLI argument=*/"cuModuleGetFunctionPtr",
      /*type=*/"size_t",
      /*default=*/"0",
      /*description=*/"cuModuleGetFunctionPtr"
    >,
    ];
}

#endif
