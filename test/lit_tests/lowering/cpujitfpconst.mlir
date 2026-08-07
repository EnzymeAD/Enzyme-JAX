// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-jit{backend=cpu})" | FileCheck %s

// A CPU jit_call whose body contains a floating point literal. Note that `jit`
// defaults to true, so unlike cpujit.mlir this actually compiles @scale through
// the in-process ORC JIT, and lower-jit signals pass failure if the module
// cannot be materialized.
//
// On Windows with an MSVC target triple, LLVM emits the 0.5 below into its own
// COMDAT section with a global `__real@3fe0000000000000` symbol, which
// RuntimeDyld cannot resolve (llvm.org/PR40074):
//
//   JIT session error: Failed to materialize symbols:
//     { (enzymejitdl_12, { __real@3fe0000000000000 }) }
//
// initJIT() therefore selects the mingw environment on Windows, which does not
// use COMDAT constants. See EnzymeAD/Reactant.jl#1673.

module {
  func.func private @scale(%arg0: !llvm.ptr<1>) {
    %cst = llvm.mlir.constant(5.000000e-01 : f64) : f64
    %0 = llvm.load %arg0 {alignment = 8 : i64} : !llvm.ptr<1> -> f64
    %1 = llvm.fmul %cst, %0 : f64
    llvm.store %1, %arg0 {alignment = 8 : i64} : f64, !llvm.ptr<1>
    return
  }
  func.func @main(%arg0: tensor<64xf64>) -> tensor<64xf64> {
    %0 = enzymexla.jit_call @scale (%arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<64xf64>) -> tensor<64xf64>
    return %0 : tensor<64xf64>
  }
}

// CHECK-LABEL: @main
// CHECK: stablehlo.custom_call @enzymexla_compile_cpu
