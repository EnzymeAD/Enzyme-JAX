// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=parallel_while_to_batched_scatter" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// Every iteration writes its own scratch, a tensor of 32 zeros, at a place
// it reads, and sums it: batched whole, the scratch would be broadcast to
// 2^25 elements, past the budget, so the loop is batched in the largest
// chunks that take it: two of 2^19.
func.func @scratch_per_iteration(%idx: tensor<1048576xi64>, %x: tensor<1048576xf64>, %y: tensor<1048576xf64>) -> tensor<1048576xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %cn = stablehlo.constant dense<1048576> : tensor<i64>
  %zero = stablehlo.constant dense<0.0> : tensor<32xf64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<1048576xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %cn : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %j = stablehlo.dynamic_slice %idx, %i, sizes = [1] : (tensor<1048576xi64>, tensor<i64>) -> tensor<1xi64>
    %xi = stablehlo.dynamic_slice %x, %i, sizes = [1] : (tensor<1048576xf64>, tensor<i64>) -> tensor<1xf64>
    %xs = stablehlo.reshape %xi : (tensor<1xf64>) -> tensor<f64>
    %s = "stablehlo.scatter"(%zero, %j, %xs) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) : (tensor<32xf64>, tensor<1xi64>, tensor<f64>) -> tensor<32xf64>
    %c0f = stablehlo.constant dense<0.0> : tensor<f64>
    %r = stablehlo.reduce(%s init: %c0f) applies stablehlo.add across dimensions = [0] : (tensor<32xf64>, tensor<f64>) -> tensor<f64>
    %ir = stablehlo.reshape %i : (tensor<i64>) -> tensor<1xi64>
    %u = "stablehlo.scatter"(%acc, %ir, %r) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) : (tensor<1048576xf64>, tensor<1xi64>, tensor<f64>) -> tensor<1048576xf64>
    %n1 = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n1, %u : tensor<i64>, tensor<1048576xf64>
  }
  return %0#1 : tensor<1048576xf64>
}

// CHECK-LABEL: func.func @scratch_per_iteration(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<1048576xi64>, %[[V1:[^ ,:)]+]]: tensor<1048576xf64>, %[[V2:[^ ,:)]+]]: tensor<1048576xf64>) -> tensor<1048576xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<524288> : tensor<i64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<32xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]]:2 = stablehlo.while(%[[V10:[^ ,:)]+]] = %[[V8]], %[[V11:[^ ,:)]+]] = %[[V2]]) : tensor<i64>, tensor<1048576xf64> attributes {enzymexla.chunks}
// CHECK-NEXT: cond {
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = stablehlo.compare LT, %[[V10]], %[[V5]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT: stablehlo.return %[[V12]] : tensor<i1>
// CHECK-NEXT: } do {
// CHECK-NEXT: %[[V12]] = stablehlo.multiply %[[V10]], %[[V4]] : tensor<i64>
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.iota dim = 0 : tensor<524288xi64>
// CHECK-NEXT: %[[V14:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V12]], dims = [] : (tensor<i64>) -> tensor<524288xi64>
// CHECK-NEXT: %[[V15:[^ ,:)]+]] = stablehlo.add %[[V13]], %[[V14]] : tensor<524288xi64>
// CHECK-NEXT: %[[V16:[^ ,:)]+]] = stablehlo.reshape %[[V15]] : (tensor<524288xi64>) -> tensor<524288x1xi64>
// CHECK-NEXT: %[[V17:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V16]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<1048576xi64>, tensor<524288x1xi64>) -> tensor<524288x1xi64>
// CHECK-NEXT: %[[V18:[^ ,:)]+]] = stablehlo.reshape %[[V15]] : (tensor<524288xi64>) -> tensor<524288x1xi64>
// CHECK-NEXT: %[[V19:[^ ,:)]+]] = "stablehlo.gather"(%[[V1]], %[[V18]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<1048576xf64>, tensor<524288x1xi64>) -> tensor<524288x1xf64>
// CHECK-NEXT: %[[V20:[^ ,:)]+]] = stablehlo.reshape %[[V19]] : (tensor<524288x1xf64>) -> tensor<524288xf64>
// CHECK-NEXT: %[[V21:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V7]], dims = [1] : (tensor<32xf64>) -> tensor<524288x32xf64>
// CHECK-NEXT: %[[V22:[^ ,:)]+]] = "stablehlo.scatter"(%[[V21]], %[[V17]], %[[V20]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT: ^bb0(%[[V23:[^ ,:)]+]]: tensor<f64>, %[[V24:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V24]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<524288x32xf64>, tensor<524288x1xi64>, tensor<524288xf64>) -> tensor<524288x32xf64>
// CHECK-NEXT: %[[V25:[^ ,:)]+]] = stablehlo.reduce(%[[V22]] init: %[[V6]]) applies stablehlo.add across dimensions = [1] : (tensor<524288x32xf64>, tensor<f64>) -> tensor<524288xf64>
// CHECK-NEXT: %[[V26:[^ ,:)]+]] = stablehlo.reshape %[[V15]] : (tensor<524288xi64>) -> tensor<524288x1xi64>
// CHECK-NEXT: %[[V27:[^ ,:)]+]] = "stablehlo.scatter"(%[[V11]], %[[V26]], %[[V25]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT: ^bb0(%[[V23]]: tensor<f64>, %[[V24]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V24]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<1048576xf64>, tensor<524288x1xi64>, tensor<524288xf64>) -> tensor<1048576xf64>
// CHECK-NEXT: %[[V28:[^ ,:)]+]] = stablehlo.add %[[V10]], %[[V3]] : tensor<i64>
// CHECK-NEXT: stablehlo.return %[[V28]], %[[V27]] : tensor<i64>, tensor<1048576xf64>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[V9]]#1 : tensor<1048576xf64>
// CHECK-NEXT: }
