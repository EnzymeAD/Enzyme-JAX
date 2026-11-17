// RUN: enzymexlamlir-opt --canonicalize %s | FileCheck %s

// The buffer of %b is readonly, so the result read from it is dropped; the
// buffer itself stays a block argument.
func.func @drop_unused_result(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %r:2 = "enzymexla.jit_region"(%a, %b) <{
    arg_attrs = [{}, {llvm.readonly}],
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
// CHECK:         %[[R:.+]] = "enzymexla.jit_region"(%[[A]], %[[B]]) <{arg_attrs = [{}, {llvm.readonly}], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_attrs = [{enzymexla.tag = 1 : i64}]}> ({
// CHECK-NEXT:    ^bb0(%[[MA:.+]]: memref<4xf32>, %[[MB:.+]]: memref<4xf32>):
// CHECK:           memref.load %[[MB]]
// CHECK:           memref.store {{.+}}, %[[MA]]
// CHECK:           enzymexla.jit_region_yield
// CHECK-NEXT:    }) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    return %[[R]] : tensor<4xf32>

// The buffer of %b is never accessed and no result is read from it, so %b is
// erased along with its operand_attrs and arg_attrs entries.
func.func @erase_unused_input(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %r = "enzymexla.jit_region"(%a, %b) <{
    operand_attrs = [{enzymexla.tag = 0 : i64}, {enzymexla.tag = 1 : i64}],
    arg_attrs = [{llvm.noalias}, {llvm.readnone}],
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
// CHECK:         %[[R:.+]] = "enzymexla.jit_region"(%[[A]]) <{arg_attrs = [{llvm.noalias}], operand_attrs = [{enzymexla.tag = 0 : i64}], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> ({
// CHECK-NEXT:    ^bb0(%[[MA:.+]]: memref<4xf32>):
// CHECK:           memref.store {{.+}}, %[[MA]]
// CHECK:         }) : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    return %[[R]] : tensor<4xf32>

// The result read from the buffer of %a is unused, but the buffer is not
// readonly, so the result and its alias are kept: without the alias, XLA would
// assume that %a is not updated and could hand the region a buffer whose
// value is still needed.
func.func @keep_unused_written_result(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %r:2 = "enzymexla.jit_region"(%a, %b) <{
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>,
      #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>
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

// CHECK-LABEL: func.func @keep_unused_written_result
// CHECK:         %{{.+}}:2 = "enzymexla.jit_region"
// CHECK-SAME:      output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>]

// The buffer of %b is readonly, so the result read from it is %b itself, even
// though it is used.
func.func @forward_read_only_result(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %r:2 = "enzymexla.jit_region"(%a, %b) <{
    arg_attrs = [{}, {llvm.readonly}],
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>,
      #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>
    ]
  }> ({
  ^bb0(%ma: memref<4xf32>, %mb: memref<4xf32>):
    %c0 = arith.constant 0 : index
    %v = memref.load %mb[%c0] : memref<4xf32>
    memref.store %v, %ma[%c0] : memref<4xf32>
    enzymexla.jit_region_yield
  }) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  return %r#0, %r#1 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: func.func @forward_read_only_result
// CHECK-SAME:      (%[[A:.+]]: tensor<4xf32>, %[[B:.+]]: tensor<4xf32>)
// CHECK:         %[[R:.+]] = "enzymexla.jit_region"(%[[A]], %[[B]]) <{arg_attrs = [{}, {llvm.readonly}], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> ({
// CHECK:         }) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    return %[[R]], %[[B]] : tensor<4xf32>, tensor<4xf32>
