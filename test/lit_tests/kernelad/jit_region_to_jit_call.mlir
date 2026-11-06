// RUN: enzymexlamlir-opt %s --jit-region-to-jit-call | FileCheck %s

module {
  func.func @main(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> tensor<100xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %cst = arith.constant 2.0 : f32

    %0 = "enzymexla.jit_region"(%arg0, %arg1) <{
      operand_attrs = [{}, {enzymexla.reverse_operand_index = 0 : i64}],
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 1,
                                        operand_tuple_indices = []>
      ]
    }> ({
    ^bb0(%a: memref<100xf32>, %b: memref<100xf32, 1>):
      scf.parallel (%i) = (%c0) to (%c100) step (%c1) {
        %v = memref.load %a[%i] : memref<100xf32>
        %w = arith.mulf %v, %cst : f32
        memref.store %w, %b[%i] : memref<100xf32, 1>
        scf.reduce
      }
      "enzymexla.jit_region_yield"() : () -> ()
    }) : (tensor<100xf32>, tensor<100xf32>) -> tensor<100xf32>

    return %0 : tensor<100xf32>
  }
}

// CHECK-LABEL: func.func @main(
// CHECK-SAME:      %[[A:.+]]: tensor<100xf32>, %[[B:.+]]: tensor<100xf32>)
// CHECK-NOT:     enzymexla.jit_region
// CHECK:         %[[R:.+]] = enzymexla.jit_call @main_jit_region (%[[A]], %[[B]]) <arg_attrs = [{}, {enzymexla.reverse_operand_index = 0 : i64}], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]> : (tensor<100xf32>, tensor<100xf32>) -> tensor<100xf32>
// CHECK-NEXT:    return %[[R]]

// CHECK-LABEL: func.func private @main_jit_region(
// CHECK-SAME:      %[[MA:.+]]: memref<100xf32>, %[[MB:.+]]: memref<100xf32, 1>) {
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
// CHECK-DAG:     %[[CST:.+]] = arith.constant 2.000000e+00 : f32
// CHECK:         scf.parallel (%[[I:.+]]) = (%[[C0]]) to (%[[C100]]) step (%[[C1]]) {
// CHECK-NEXT:      %[[V:.+]] = memref.load %[[MA]][%[[I]]] : memref<100xf32>
// CHECK-NEXT:      %[[W:.+]] = arith.mulf %[[V]], %[[CST]] : f32
// CHECK-NEXT:      memref.store %[[W]], %[[MB]][%[[I]]] : memref<100xf32, 1>
// CHECK:         return
