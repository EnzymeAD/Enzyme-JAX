// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=parallel_while_to_batched_scatter" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// Iteration k writes row k of the carried vector viewed as [6, 16]: the
// reshapes to and from the view are links of the buffer's chain of writes,
// and the write batches into one scatter of the view.
func.func @reshape_link(%buf: tensor<96xf64>, %vals: tensor<6x16xf64>) -> tensor<96xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c6 = stablehlo.constant dense<6> : tensor<i64>
  %r:2 = stablehlo.while(%k = %c0, %acc = %buf) : tensor<i64>, tensor<96xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %k, %c6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %vals, %k, %c0, sizes = [1, 16] : (tensor<6x16xf64>, tensor<i64>, tensor<i64>) -> tensor<1x16xf64>
    %view = stablehlo.reshape %acc : (tensor<96xf64>) -> tensor<6x16xf64>
    %w = stablehlo.dynamic_update_slice %view, %row, %k, %c0 : (tensor<6x16xf64>, tensor<1x16xf64>, tensor<i64>, tensor<i64>) -> tensor<6x16xf64>
    %back = stablehlo.reshape %w : (tensor<6x16xf64>) -> tensor<96xf64>
    %k2 = stablehlo.add %k, %c1 : tensor<i64>
    stablehlo.return %k2, %back : tensor<i64>, tensor<96xf64>
  }
  return %r#1 : tensor<96xf64>
}

// CHECK:  func.func @reshape_link(%arg0: tensor<96xf64>, %arg1: tensor<6x16xf64>) -> tensor<96xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<6xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg1 : (tensor<6x16xf64>) -> tensor<6x1x16xf64>
// CHECK-NEXT:   %2 = stablehlo.reshape %arg0 : (tensor<96xf64>) -> tensor<6x16xf64>
// CHECK-NEXT:   %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<6xi64>
// CHECK-NEXT:   %4 = stablehlo.clamp %c_0, %0, %c : (tensor<i64>, tensor<6xi64>, tensor<i64>) -> tensor<6xi64>
// CHECK-NEXT:   %5 = stablehlo.reshape %4 : (tensor<6xi64>) -> tensor<6x1xi64>
// CHECK-NEXT:   %6 = stablehlo.clamp %c_0, %3, %c_0 : (tensor<i64>, tensor<6xi64>, tensor<i64>) -> tensor<6xi64>
// CHECK-NEXT:   %7 = stablehlo.reshape %6 : (tensor<6xi64>) -> tensor<6x1xi64>
// CHECK-NEXT:   %8 = stablehlo.concatenate %5, %7, dim = 1 : (tensor<6x1xi64>, tensor<6x1xi64>) -> tensor<6x2xi64>
// CHECK-NEXT:   %9 = "stablehlo.scatter"(%2, %8, %1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<6x16xf64>, tensor<6x2xi64>, tensor<6x1x16xf64>) -> tensor<6x16xf64>
// CHECK-NEXT:   %10 = stablehlo.reshape %9 : (tensor<6x16xf64>) -> tensor<96xf64>
// CHECK-NEXT:   return %10 : tensor<96xf64>
// CHECK-NEXT:  }
