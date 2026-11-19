// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-kernel,canonicalize)" | FileCheck %s

// PTX identifiers may only contain [A-Za-z0-9_$], so the kernel inherits a
// legalized name, not only one without the `#` of Julia's symbols.

module {
  llvm.func internal ptx_kernelcc @"##kern{Float32}!#12"(%arg0: !llvm.ptr<1>) {
    %0 = llvm.load %arg0 : !llvm.ptr<1> -> i64
    %1 = llvm.mul %0, %0 : i64
    llvm.store %1, %arg0 : i64, !llvm.ptr<1>
    llvm.return
  }
  func.func @main(%arg0: tensor<64xi64>) -> tensor<64xi64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %0 = enzymexla.kernel_call @"##kern{Float32}!#12" blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) shmem=%c0 (%arg0) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> : (tensor<64xi64>) -> tensor<64xi64>
    return %0 : tensor<64xi64>
  }
}

// CHECK: gpu.module @gpumod___kern_Float32___12 {
// CHECK:   gpu.func @__kern_Float32___12(%arg0: !llvm.ptr<1>) kernel {
// CHECK: gpu.launch_func async [%{{.+}}] @gpumod___kern_Float32___12::@__kern_Float32___12
