// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(kernel-call-to-gpu-wrapper,fuse-jit,symbol-dce)" | FileCheck %s

// Dependent kernel calls fuse into one jit_call of their gpu_wrappers, and
// symbol-dce erases the wrappers fusion leaves uncalled.

module {
  llvm.func private @square(%arg0: !llvm.ptr<1>) {
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    %1 = llvm.zext %0 : i32 to i64
    %2 = llvm.getelementptr inbounds %arg0[%1] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %3 = llvm.load %2 : !llvm.ptr<1> -> f32
    %4 = llvm.fmul %3, %3 : f32
    llvm.store %4, %2 : f32, !llvm.ptr<1>
    llvm.return
  }

  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c4 = stablehlo.constant dense<4> : tensor<i64>
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %0 = enzymexla.kernel_call @square blocks in(%c1, %c1, %c1) threads in(%c4, %c1, %c1) shmem = %c0 (%arg0) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> : (tensor<4xf32>) -> tensor<4xf32>
    %1 = enzymexla.kernel_call @square blocks in(%c1, %c1, %c1) threads in(%c4, %c1, %c1) shmem = %c0 (%0) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> : (tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}

// CHECK-NOT:   _gpu_wrapper(
// CHECK-LABEL: func.func @main(
// CHECK-SAME:      %[[A:.+]]: tensor<4xf32>)
// CHECK-NEXT:    %[[R:.+]] = enzymexla.jit_call @fused__square_gpu_wrapper_square_gpu_wrapper_0 (%[[A]])
// CHECK-NEXT:    return %[[R]]
// CHECK-NOT:   _gpu_wrapper(
// CHECK:       llvm.func private @fused__square_gpu_wrapper_square_gpu_wrapper_0(%{{[^:]+}}: !llvm.ptr<1>) attributes {sym_visibility = "private"} {
// CHECK-COUNT-2: "enzymexla.gpu_wrapper"
// CHECK:         llvm.return
// CHECK-NOT:   _gpu_wrapper(
