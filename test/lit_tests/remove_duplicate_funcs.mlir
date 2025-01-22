// RUN: enzymexlamlir-opt %s -remove-duplicate-func-def | FileCheck %s

// CHECK-LABEL: module
module {
  llvm.func local_unnamed_addr @malloc(i64) -> !llvm.ptr

  // CHECK: gpu_malloc(
  llvm.func local_unnamed_addr @gpu_malloc(%arg0: i64 {llvm.zeroext}) -> i64 {
    %0 = llvm.call @malloc(%arg0) : (i64) -> !llvm.ptr
    %1 = llvm.ptrtoint %0 : !llvm.ptr to i64
    llvm.return %1 : i64
  }

  // CHECK-NOT: gpu_malloc_10
  llvm.func local_unnamed_addr @gpu_malloc_10(%arg0: i64 {llvm.zeroext}) -> i64 {
    %0 = llvm.call @malloc(%arg0) : (i64) -> !llvm.ptr
    %1 = llvm.ptrtoint %0 : !llvm.ptr to i64
    llvm.return %1 : i64
  }

  // CHECK-NOT: gpu_malloc_11
  llvm.func local_unnamed_addr @gpu_malloc_11(%arg0: i64 {llvm.zeroext}) -> i64 {
    %0 = llvm.call @malloc(%arg0) : (i64) -> !llvm.ptr
    %1 = llvm.ptrtoint %0 : !llvm.ptr to i64
    llvm.return %1 : i64
  }

  // CHECK: "##foo#3874"(
  llvm.func ptx_kernelcc @"##foo#3874"() { 
    llvm.return
  }

  // CHECK-NOT: "##foo#3876"
  llvm.func ptx_kernelcc @"##foo#3876"() { 
    llvm.return
  }
  
  // CHECK-LABEL: @main
  func.func @main(%arg0: tensor<16x34x34xf64>, %arg1: tensor<16x34x34xf64>, %arg2: tensor<16x34x34xf64>, %arg3: i64) {
    %c_4 = stablehlo.constant dense<256> : tensor<i64>
    %c_6 = stablehlo.constant dense<0> : tensor<i64>
    %c_8 = stablehlo.constant dense<1> : tensor<i64>
    %c_9 = stablehlo.constant dense<8> : tensor<i64>
    // CHECK: enzymexla.kernel_call @"##foo#3874"
    %0 = enzymexla.kernel_call @"##foo#3874" blocks in(%c_9, %c_8, %c_8) threads in(%c_4, %c_8, %c_8) shmem = %c_6 (%arg0, %arg1, %arg2) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<16x34x34xf64>, tensor<16x34x34xf64>, tensor<16x34x34xf64>) -> tensor<16x34x34xf64>
    // CHECK-NEXT: enzymexla.kernel_call @"##foo#3874"
    %1 = enzymexla.kernel_call @"##foo#3876" blocks in(%c_9, %c_8, %c_8) threads in(%c_4, %c_8, %c_8) shmem = %c_6 (%arg0, %arg1, %arg2) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<16x34x34xf64>, tensor<16x34x34xf64>, tensor<16x34x34xf64>) -> tensor<16x34x34xf64> 
    // CHECK: @gpu_malloc(
    %2 = llvm.call @gpu_malloc_11(%arg3) : (i64) -> i64
    return 
  }

}