// RUN: enzymexlamlir-opt %s --lower-kernel-to-polygeist | FileCheck %s

// Kernel symbols may contain characters PTX does not accept, e.g. Julia's
// `##kernel#123`; the wrapper the device kernel is outlined from must not.

module {
  llvm.func private @"##kernel#123"(%arg0: !llvm.ptr<1>) {
    %0 = llvm.load %arg0 : !llvm.ptr<1> -> f32
    %1 = llvm.fmul %0, %0 : f32
    llvm.store %1, %arg0 : f32, !llvm.ptr<1>
    llvm.return
  }

  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %0 = enzymexla.kernel_call @"##kernel#123" blocks in(%c1, %c1, %c1) threads in(%c1, %c1, %c1) shmem = %c0 (%arg0) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK: enzymexla.jit_call @__kernel_123_gpu_wrapper
// CHECK: llvm.func private @__kernel_123_gpu_wrapper(
