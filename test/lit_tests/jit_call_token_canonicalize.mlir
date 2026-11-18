// RUN: enzymexlamlir-opt --canonicalize %s | FileCheck %s

// The token operand has no callee argument, so the canonicalizers must neither
// read past the callee arguments nor drop the token.

llvm.func @wrap(%buf: !llvm.ptr {llvm.readonly}, %dead: !llvm.ptr, %out: !llvm.ptr) {
  %v = llvm.load %buf : !llvm.ptr -> f32
  llvm.store %v, %out : f32, !llvm.ptr
  llvm.return
}

// CHECK-LABEL: func.func @main
// CHECK-SAME: (%[[A:.+]]: tensor<f32>, %[[B:.+]]: tensor<f32>, %[[C:.+]]: tensor<f32>)
// CHECK: %[[TOK:.+]] = stablehlo.after_all : !stablehlo.token
// CHECK: %[[R:.+]]:2 = enzymexla.jit_call @wrap (%[[A]], %[[C]], %[[TOK]])
// CHECK-SAME: output_tuple_indices = [0], operand_index = 1
// CHECK-SAME: output_tuple_indices = [1], operand_index = 2
// CHECK-SAME: : (tensor<f32>, tensor<f32>, !stablehlo.token) -> (tensor<f32>, !stablehlo.token)
// CHECK: return %[[R]]#0, %[[R]]#1
func.func @main(%a: tensor<f32>, %b: tensor<f32>, %c: tensor<f32>) -> (tensor<f32>, !stablehlo.token) {
  %tok = stablehlo.after_all : !stablehlo.token
  %r:2 = enzymexla.jit_call @wrap (%a, %b, %c, %tok) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 3, operand_tuple_indices = []>]> : (tensor<f32>, tensor<f32>, tensor<f32>, !stablehlo.token) -> (tensor<f32>, !stablehlo.token)
  return %r#0, %r#1 : tensor<f32>, !stablehlo.token
}
