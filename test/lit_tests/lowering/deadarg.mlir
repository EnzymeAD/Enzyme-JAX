// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(canonicalize)" | FileCheck %s

module {
  func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> (tensor<i64>, tensor<i64>, tensor<i64>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %0:3 = enzymexla.kernel_call @k1 blocks in(%c, %c, %c) threads in(%c, %c, %c) shmem = %c_0 (%arg0, %arg1, %arg2) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>] } : (tensor<i64>, tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>, tensor<i64>)
    return %0#0, %0#1, %0#2 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  llvm.func ptx_kernelcc @k1(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) {
    %a0 = llvm.load %arg0 {alignment = 1 : i64} : !llvm.ptr<1> -> i64
    llvm.store %a0, %arg2 {alignment = 1 : i64} : i64, !llvm.ptr<1>
    llvm.return
  }
  func.func @main2(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> (tensor<i64>, tensor<i64>, tensor<i64>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %0:3 = enzymexla.kernel_call @k2 blocks in(%c, %c, %c) threads in(%c, %c, %c) shmem = %c_0 (%arg0, %arg1, %arg2) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>] } : (tensor<i64>, tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>, tensor<i64>)
    return %0#0, %0#1, %0#2 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  func.func @main3(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> (tensor<i64>, tensor<i64>, tensor<i64>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %0:3 = enzymexla.kernel_call @k2 blocks in(%c, %c, %c) threads in(%c, %c, %c) shmem = %c_0 (%arg0, %arg1, %arg2) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>] } : (tensor<i64>, tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>, tensor<i64>)
    return %0#0, %0#1, %0#2 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  llvm.func ptx_kernelcc @k2(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) {
    %a0 = llvm.load %arg2 {alignment = 1 : i64} : !llvm.ptr<1> -> i64
    %t = llvm.mul %a0, %a0 : i64
    llvm.store %t, %arg2 {alignment = 1 : i64} : i64, !llvm.ptr<1>
    llvm.return
  }
}

// CHECK:  func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> (tensor<i64>, tensor<i64>, tensor<i64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0:2 = enzymexla.kernel_call @k1 blocks in(%c, %c, %c) threads in(%c, %c, %c) shmem = %c_0 (%arg0, %arg2) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>]} : (tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>)
// CHECK-NEXT:    return %0#0, %arg1, %0#1 : tensor<i64>, tensor<i64>, tensor<i64>
// CHECK-NEXT:  }
// CHECK:  llvm.func ptx_kernelcc @k1(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) {
// CHECK-NEXT:    %0 = llvm.load %arg0 {alignment = 1 : i64} : !llvm.ptr<1> -> i64
// CHECK-NEXT:    llvm.store %0, %arg1 {alignment = 1 : i64} : i64, !llvm.ptr<1>
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
// CHECK:  func.func @main2(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> (tensor<i64>, tensor<i64>, tensor<i64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = enzymexla.kernel_call @k2 blocks in(%c, %c, %c) threads in(%c, %c, %c) shmem = %c_0 (%arg2) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<i64>) -> tensor<i64>
// CHECK-NEXT:    return %arg0, %arg1, %0 : tensor<i64>, tensor<i64>, tensor<i64>
// CHECK-NEXT:  }
// CHECK:  func.func @main3(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> (tensor<i64>, tensor<i64>, tensor<i64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = enzymexla.kernel_call @k2 blocks in(%c, %c, %c) threads in(%c, %c, %c) shmem = %c_0 (%arg2) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<i64>) -> tensor<i64>
// CHECK-NEXT:    return %arg0, %arg1, %0 : tensor<i64>, tensor<i64>, tensor<i64>
// CHECK-NEXT:  }
// CHECK:  llvm.func ptx_kernelcc @k2(%arg0: !llvm.ptr<1>) {
// CHECK-NEXT:    %0 = llvm.load %arg0 {alignment = 1 : i64} : !llvm.ptr<1> -> i64
// CHECK-NEXT:    %1 = llvm.mul %0, %0 : i64
// CHECK-NEXT:    llvm.store %1, %arg0 {alignment = 1 : i64} : i64, !llvm.ptr<1>
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
