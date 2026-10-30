// RUN: enzymexlamlir-opt --canonicalize %s | FileCheck %s

// The result read from the buffer of %b is unused, so it is dropped; the
// buffer itself stays a block argument.
func.func @drop_unused_result(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %r:2 = "enzymexla.jit_region"(%a, %b) <{
    result_attrs = [{enzymexla.tag = 0 : i64}, {enzymexla.tag = 1 : i64}],
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 1, operand_tuple_indices = []>,
      #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 0, operand_tuple_indices = []>
    ]
  }> ({
  ^bb0(%ma: memref<4xf32>, %mb: memref<4xf32>):
    %c0 = arith.constant 0 : index
    %v = memref.load %mb[%c0] : memref<4xf32>
    memref.store %v, %ma[%c0] : memref<4xf32>
    enzymexla.jit_region_yield
  }) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  return %r#1 : tensor<4xf32>
}

// CHECK-LABEL: func.func @drop_unused_result
// CHECK-SAME:      (%[[A:.+]]: tensor<4xf32>, %[[B:.+]]: tensor<4xf32>)
// CHECK:         %[[R:.+]] = "enzymexla.jit_region"(%[[A]], %[[B]]) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_attrs = [{enzymexla.tag = 1 : i64}]}> ({
// CHECK-NEXT:    ^bb0(%[[MA:.+]]: memref<4xf32>, %[[MB:.+]]: memref<4xf32>):
// CHECK:           memref.load %[[MB]]
// CHECK:           memref.store {{.+}}, %[[MA]]
// CHECK:           enzymexla.jit_region_yield
// CHECK-NEXT:    }) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    return %[[R]] : tensor<4xf32>

// The buffer of %b is never accessed and no result is read from it, so %b is
// erased along with its operand_attrs entry.
func.func @erase_unused_input(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %r = "enzymexla.jit_region"(%a, %b) <{
    operand_attrs = [{enzymexla.tag = 0 : i64}, {enzymexla.tag = 1 : i64}],
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>
    ]
  }> ({
  ^bb0(%ma: memref<4xf32>, %mb: memref<4xf32>):
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.0 : f32
    memref.store %cst, %ma[%c0] : memref<4xf32>
    enzymexla.jit_region_yield
  }) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// CHECK-LABEL: func.func @erase_unused_input
// CHECK-SAME:      (%[[A:.+]]: tensor<4xf32>, %{{.+}}: tensor<4xf32>)
// CHECK:         %[[R:.+]] = "enzymexla.jit_region"(%[[A]]) <{operand_attrs = [{enzymexla.tag = 0 : i64}], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> ({
// CHECK-NEXT:    ^bb0(%[[MA:.+]]: memref<4xf32>):
// CHECK:           memref.store {{.+}}, %[[MA]]
// CHECK:         }) : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    return %[[R]] : tensor<4xf32>
