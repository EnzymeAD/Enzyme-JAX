// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

module {
  func.func @test_symbolic_load(%memref: memref<?xf64>, %symbol_memref: memref<1xi64>, %res_memref: memref<1xf64>) {
    %cst = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index
    %symbol_i64 = affine.load %symbol_memref[0] : memref<1xi64>
    %symbol = arith.index_cast %symbol_i64 : i64 to index
    %res = affine.for %i = 0 to 10 iter_args(%acc = %cst) -> (f64) {
      %val = "affine.load"(%memref, %symbol) <{map = affine_map<()[s0] -> (s0 + 1)>}> : (memref<?xf64>, index) -> f64
      %add = arith.addf %acc, %val : f64
      affine.yield %add : f64
    }
    affine.store %res, %res_memref[0] : memref<1xf64>
    return
  }
}

// CHECK-LABEL: func.func private @test_symbolic_load_raised(
// CHECK-SAME:                                               %[[ARG0:.*]]: tensor<?xf64>,
// CHECK-SAME:                                               %[[ARG1:.*]]: tensor<1xi64>,
// CHECK-SAME:                                               %[[ARG2:.*]]: tensor<1xf64>) -> (tensor<?xf64>, tensor<1xi64>, tensor<1xf64>) {
// CHECK-DAG:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG:     %[[V0:.*]] = stablehlo.reshape %[[ARG1]] : (tensor<1xi64>) -> tensor<i64>
// CHECK-DAG:     %[[V1:.*]] = stablehlo.iota dim = 0 : tensor<10xi64>
// CHECK-DAG:     %[[C0:.*]] = stablehlo.constant dense<0> : tensor<10xi64>
// CHECK-DAG:     %[[V2:.*]] = stablehlo.add %[[V1]], %[[C0]] : tensor<10xi64>
// CHECK-DAG:     %[[C1:.*]] = stablehlo.constant dense<1> : tensor<10xi64>
// CHECK-DAG:     %[[V3:.*]] = stablehlo.multiply %[[V2]], %[[C1]] : tensor<10xi64>
// CHECK-DAG:     %[[C_1:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-DAG:     %[[V4:.*]] = stablehlo.add %[[V0]], %[[C_1]] : tensor<i64>
// CHECK-DAG:     %[[V5:.*]] = stablehlo.reshape %[[V4]] : (tensor<i64>) -> tensor<1xi64>
// CHECK-DAG:     %[[V6:.*]] = "stablehlo.gather"(%[[ARG0]], %[[V5]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<?xf64>, tensor<1xi64>) -> tensor<f64>
// CHECK-DAG:     %[[V9:.*]] = stablehlo.broadcast_in_dim %[[V6]], dims = [] : (tensor<f64>) -> tensor<10xf64>
// CHECK-DAG:     %[[V10:.*]] = stablehlo.broadcast_in_dim %[[CST]], dims = [] : (tensor<f64>) -> tensor<10xf64>
// CHECK-DAG:     %[[CST_2:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG:     %[[V11:.*]] = "stablehlo.reduce_window"(%[[V9]], %[[CST_2]]) <{base_dilations = array<i64: 1>, padding = dense<{{\[\[}}9, 0{{\]\]}}> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 10>, window_strides = array<i64: 1>}> ({
// CHECK-NEXT:    ^bb0(%[[ARG3:.*]]: tensor<f64>, %[[ARG4:.*]]: tensor<f64>):
// CHECK-NEXT:      %[[V17:.*]] = stablehlo.add %[[ARG3]], %[[ARG4]] : tensor<f64>
// CHECK-NEXT:      stablehlo.return %[[V17]] : tensor<f64>
// CHECK-NEXT:    }) : (tensor<10xf64>, tensor<f64>) -> tensor<10xf64>
// CHECK-DAG:     %[[V12:.*]] = stablehlo.add %[[V11]], %[[V10]] : tensor<10xf64>
// CHECK-DAG:     %[[C0_IDX:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[V13:.*]] = stablehlo.slice %[[V12]] [9:10] : (tensor<10xf64>) -> tensor<1xf64>
// CHECK-DAG:     %[[V14:.*]] = stablehlo.reshape %[[V13]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-DAG:     %[[C_8:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-DAG:     %[[V15:.*]] = stablehlo.broadcast_in_dim %[[V14]], dims = [] : (tensor<f64>) -> tensor<1xf64>
// CHECK-DAG:     %[[V16:.*]] = stablehlo.dynamic_update_slice %[[ARG2]], %[[V15]], %[[C_8]] : (tensor<1xf64>, tensor<1xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-DAG:     return %[[ARG0]], %[[ARG1]], %[[V16]] : tensor<?xf64>, tensor<1xi64>, tensor<1xf64>
// CHECK: }
