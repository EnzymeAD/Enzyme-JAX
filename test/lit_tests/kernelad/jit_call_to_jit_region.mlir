// RUN: enzymexlamlir-opt %s --jit-call-to-jit-region | FileCheck %s

module {
  llvm.func private @wrapper(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr) {
    %0 = llvm.load %arg1 : !llvm.ptr -> f32
    llvm.store %0, %arg0 : f32, !llvm.ptr<1>
    llvm.return
  }

  func.func private @memref_wrapper(%arg0: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.0 : f32
    memref.store %cst, %arg0[%c0] : memref<4xf32>
    return
  }

  llvm.func @external(!llvm.ptr)

  func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %0 = enzymexla.jit_call @wrapper(%a, %b) <
      arg_attrs = [{}, {enzymexla.reverse_operand_index = 0 : i64}],
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ]
    > : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %1 = enzymexla.jit_call @memref_wrapper(%0) <
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ]
    > : (tensor<4xf32>) -> tensor<4xf32>
    %2 = enzymexla.jit_call @external(%1) <
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ]
    > : (tensor<4xf32>) -> tensor<4xf32>
    return %0, %1, %2 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }
}

// CHECK-NOT:   @wrapper
// CHECK-NOT:   @memref_wrapper
// CHECK:       llvm.func @external(!llvm.ptr)

// CHECK-LABEL: func.func @main(
// CHECK-SAME:      %[[A:.+]]: tensor<4xf32>, %[[B:.+]]: tensor<4xf32>)
// CHECK:         %[[R0:.+]] = "enzymexla.jit_region"(%[[A]], %[[B]]) <{operand_attrs = [{}, {enzymexla.reverse_operand_index = 0 : i64}], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> ({
// CHECK-NEXT:    ^bb0(%[[MA:.+]]: memref<4xf32, 1>, %[[MB:.+]]: memref<4xf32>):
// CHECK-NEXT:      %[[PA:.+]] = "enzymexla.memref2pointer"(%[[MA]]) : (memref<4xf32, 1>) -> !llvm.ptr<1>
// CHECK-NEXT:      %[[PB:.+]] = "enzymexla.memref2pointer"(%[[MB]]) : (memref<4xf32>) -> !llvm.ptr
// CHECK-NEXT:      %[[V:.+]] = llvm.load %[[PB]] : !llvm.ptr -> f32
// CHECK-NEXT:      llvm.store %[[V]], %[[PA]] : f32, !llvm.ptr<1>
// CHECK-NEXT:      enzymexla.jit_region_yield
// CHECK-NEXT:    }) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK:         %[[R1:.+]] = "enzymexla.jit_region"(%[[R0]])
// CHECK-NEXT:    ^bb0(%[[M:.+]]: memref<4xf32>):
// CHECK-DAG:       %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:       %[[CST:.+]] = arith.constant 1.000000e+00 : f32
// CHECK:           memref.store %[[CST]], %[[M]][%[[C0]]] : memref<4xf32>
// CHECK-NEXT:      enzymexla.jit_region_yield
// CHECK:         %[[R2:.+]] = enzymexla.jit_call @external (%[[R1]])
// CHECK:         return %[[R0]], %[[R1]], %[[R2]]
