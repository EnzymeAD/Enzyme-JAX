// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=parallel_while_to_batched_scatter" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// Each iteration reads an element of the buffer it writes, through a view of
// the buffer with another shape, and writes it back doubled: a read of the
// buffer through a reshape is a read of it.
func.func @double_in_place(%idx: tensor<8xi64>, %y: tensor<32xf64>) -> tensor<32xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %two = stablehlo.constant dense<2.0> : tensor<f64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<32xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %j = stablehlo.dynamic_slice %idx, %i, sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
    %js = stablehlo.reshape %j : (tensor<1xi64>) -> tensor<i64>
    %view = stablehlo.reshape %acc : (tensor<32xf64>) -> tensor<1x32xf64>
    %e = stablehlo.dynamic_slice %view, %c0, %js, sizes = [1, 1] : (tensor<1x32xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
    %es = stablehlo.reshape %e : (tensor<1x1xf64>) -> tensor<f64>
    %d = stablehlo.multiply %es, %two : tensor<f64>
    %u = "stablehlo.scatter"(%acc, %j, %d) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) : (tensor<32xf64>, tensor<1xi64>, tensor<f64>) -> tensor<32xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %u : tensor<i64>, tensor<32xf64>
  }
  return %0#1 : tensor<32xf64>
}

// CHECK-LABEL: func.func @double_in_place(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8xi64>, %[[V1:[^ ,:)]+]]: tensor<32xf64>) -> tensor<32xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<2.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V4]] : (tensor<8x1xi64>) -> tensor<8xi64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<32xf64>) -> tensor<1x32xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V2]], dims = [] : (tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.reshape %[[V7]] : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.reshape %[[V5]] : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.concatenate %[[V8]], %[[V9]], dim = 1 : (tensor<8x1xi64>, tensor<8x1xi64>) -> tensor<8x2xi64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = "stablehlo.gather"(%[[V6]], %[[V10]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<1x32xf64>, tensor<8x2xi64>) -> tensor<8x1x1xf64>
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = stablehlo.reshape %[[V11]] : (tensor<8x1x1xf64>) -> tensor<8xf64>
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V3]], dims = [] : (tensor<f64>) -> tensor<8xf64>
// CHECK-NEXT: %[[V14:[^ ,:)]+]] = stablehlo.multiply %[[V12]], %[[V13]] : tensor<8xf64>
// CHECK-NEXT: %[[V15:[^ ,:)]+]] = "stablehlo.scatter"(%[[V1]], %[[V4]], %[[V14]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT: ^bb0(%[[V16:[^ ,:)]+]]: tensor<f64>, %[[V17:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V17]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<32xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<32xf64>
// CHECK-NEXT: return %[[V15]] : tensor<32xf64>
// CHECK-NEXT: }
