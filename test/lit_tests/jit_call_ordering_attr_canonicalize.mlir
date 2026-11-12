// RUN: enzymexlamlir-opt --canonicalize %s | FileCheck %s

// Rebuilding a call must keep `enzymexla.require_jit_ordering`.

llvm.func @wrap(%ro: !llvm.ptr {llvm.readonly}, %out: !llvm.ptr) {
  %v = llvm.load %ro : !llvm.ptr -> f32
  llvm.store %v, %out : f32, !llvm.ptr
  llvm.return
}

// The result aliasing the read only operand is dropped, which rebuilds the call.
// CHECK-LABEL: func.func @ordered
// CHECK: enzymexla.jit_call @wrap
// CHECK-SAME: {enzymexla.require_jit_ordering}
// CHECK-SAME: -> tensor<f32>
func.func @ordered(%a: tensor<f32>, %b: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %r:2 = enzymexla.jit_call @wrap (%a, %b) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>]> {enzymexla.require_jit_ordering} : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
  return %r#0, %r#1 : tensor<f32>, tensor<f32>
}

// CHECK-LABEL: func.func @unordered
// CHECK: enzymexla.jit_call @wrap
// CHECK-NOT: enzymexla.require_jit_ordering
func.func @unordered(%a: tensor<f32>, %b: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %r:2 = enzymexla.jit_call @wrap (%a, %b) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>]> : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
  return %r#0, %r#1 : tensor<f32>, tensor<f32>
}
