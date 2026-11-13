// RUN: enzymexlamlir-opt %s --enzyme-batch | FileCheck %s

// The batched call must keep `enzymexla.require_jit_ordering`.

llvm.func private @inc(%arg0: !llvm.ptr) {
  llvm.return
}

func.func @single(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = enzymexla.jit_call @inc (%arg0) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]> {enzymexla.require_jit_ordering} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @single_unordered(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = enzymexla.jit_call @inc (%arg0) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]> : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @main(%arg0: tensor<3x4xf32>) -> (tensor<3x4xf32>, tensor<3x4xf32>) {
  %0 = enzyme.batch @single(%arg0) <{batch_shape = array<i64: 3>}> : (tensor<3x4xf32>) -> tensor<3x4xf32>
  %1 = enzyme.batch @single_unordered(%arg0) <{batch_shape = array<i64: 3>}> : (tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0, %1 : tensor<3x4xf32>, tensor<3x4xf32>
}

// CHECK-LABEL: func.func @main
// CHECK: enzymexla.jit_call @inc_batched
// CHECK-SAME: {enzymexla.require_jit_ordering}
// CHECK: enzymexla.jit_call @inc_batched
// CHECK-NOT: enzymexla.require_jit_ordering
