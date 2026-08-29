// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// A dead-lane scatter sends its index out of bounds (a negative select
// arm) and relies on the update being dropped; a dynamic-update-slice
// clamps instead, so the single-index scatter-to-DUS conversion must not
// fire on masked indices.

// CHECK-LABEL: @masked_single_scatter
// CHECK: "stablehlo.scatter"
// CHECK-NOT: stablehlo.dynamic_update_slice

module {
  func.func @masked_single_scatter(%buf: tensor<12xf64>, %idx: tensor<i64>, %pred: tensor<i1>, %val: tensor<f64>) -> tensor<12xf64> {
    %cm1 = stablehlo.constant dense<-1> : tensor<i64>
    %sel = stablehlo.select %pred, %idx, %cm1 : tensor<i1>, tensor<i64>
    %indices = stablehlo.reshape %sel : (tensor<i64>) -> tensor<1xi64>
    %r = "stablehlo.scatter"(%buf, %indices, %val) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = false}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      stablehlo.return %arg1 : tensor<f64>
    }) : (tensor<12xf64>, tensor<1xi64>, tensor<f64>) -> tensor<12xf64>
    return %r : tensor<12xf64>
  }
}
