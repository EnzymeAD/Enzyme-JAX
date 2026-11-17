// RUN: enzymexlamlir-opt %s --lower-kernel-to-polygeist | FileCheck %s
// RUN: enzymexlamlir-opt %s --lower-kernel-to-polygeist --jit-call-to-jit-region --canonicalize | FileCheck %s --check-prefix=REGION

// The function running the kernel body keeps the attributes of the pointers of
// the kernel.
module {
  llvm.func private @kernel(%arg0: !llvm.ptr<1> {llvm.noalias}, %arg1: !llvm.ptr<1> {llvm.noalias, llvm.readonly}) {
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    %1 = llvm.zext %0 : i32 to i64
    %2 = llvm.getelementptr inbounds %arg0[%1] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %3 = llvm.getelementptr inbounds %arg1[%1] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %4 = llvm.load %2 : !llvm.ptr<1> -> f32
    %5 = llvm.load %3 : !llvm.ptr<1> -> f32
    %6 = llvm.fmul %4, %5 : f32
    llvm.store %6, %2 : f32, !llvm.ptr<1>
    llvm.return
  }

  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c4 = stablehlo.constant dense<4> : tensor<i64>
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = enzymexla.kernel_call @kernel blocks in(%c1, %c1, %c1) threads in(%c4, %c1, %c1) shmem = %c0 (%arg0, %arg1) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>]}> : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
    return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
  }
}

// CHECK-LABEL: llvm.func private @kernel_gpu_wrapper(
// CHECK-SAME:      %{{.+}}: !llvm.ptr<1> {llvm.noalias}, %{{.+}}: !llvm.ptr<1> {llvm.noalias, llvm.readonly}) {

// Once inlined into a jit_region, the readonly buffer of %arg1 is no longer a
// result of the region, which reads it but does not update it.
// REGION-LABEL: func.func @main(
// REGION-SAME:      %[[A:.+]]: tensor<4xf32>, %[[B:.+]]: tensor<4xf32>)
// REGION:         %[[R:.+]] = "enzymexla.jit_region"(%[[A]], %[[B]]) <{arg_attrs = [{llvm.noalias}, {llvm.noalias, llvm.readonly}], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> ({
// REGION:         }) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// REGION-NEXT:    return %[[R]], %[[B]] : tensor<4xf32>, tensor<4xf32>
