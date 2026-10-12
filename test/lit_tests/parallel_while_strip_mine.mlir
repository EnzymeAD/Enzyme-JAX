// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=parallel_while_to_batched_scatter" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// Every iteration scales a row of 1024 and sums it into its place of the
// result: batched, the values an iteration computes (its row, its products,
// its indices) are some 32 GiB, past the 8 GiB a batch may compute, so the
// iterations are batched in four chunks of 2^18, one after the other.
func.func @rows(%m: tensor<1048576x1024xf64>, %y: tensor<1048576xf64>) -> tensor<1048576xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %cn = stablehlo.constant dense<1048576> : tensor<i64>
  %two = stablehlo.constant dense<2.0> : tensor<1x1024xf64>
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<1048576xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %cn : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %m, %i, %c0, sizes = [1, 1024] : (tensor<1048576x1024xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1024xf64>
    %p = stablehlo.multiply %row, %two : tensor<1x1024xf64>
    %s = stablehlo.reduce(%p init: %z) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1024xf64>, tensor<f64>) -> tensor<f64>
    %ir = stablehlo.reshape %i : (tensor<i64>) -> tensor<1xi64>
    %u = "stablehlo.scatter"(%acc, %ir, %s) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) : (tensor<1048576xf64>, tensor<1xi64>, tensor<f64>) -> tensor<1048576xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %u : tensor<i64>, tensor<1048576xf64>
  }
  return %0#1 : tensor<1048576xf64>
}

// The same over 2^16 iterations (1 GiB): batched at once.
func.func @rows_small(%m: tensor<65536x1024xf64>, %y: tensor<65536xf64>) -> tensor<65536xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %cn = stablehlo.constant dense<65536> : tensor<i64>
  %two = stablehlo.constant dense<2.0> : tensor<1x1024xf64>
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<65536xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %cn : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %m, %i, %c0, sizes = [1, 1024] : (tensor<65536x1024xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1024xf64>
    %p = stablehlo.multiply %row, %two : tensor<1x1024xf64>
    %s = stablehlo.reduce(%p init: %z) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1024xf64>, tensor<f64>) -> tensor<f64>
    %ir = stablehlo.reshape %i : (tensor<i64>) -> tensor<1xi64>
    %u = "stablehlo.scatter"(%acc, %ir, %s) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) : (tensor<65536xf64>, tensor<1xi64>, tensor<f64>) -> tensor<65536xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %u : tensor<i64>, tensor<65536xf64>
  }
  return %0#1 : tensor<65536xf64>
}

// CHECK-LABEL: func.func @rows(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<1048576x1024xf64>, %[[V1:[^ ,:)]+]]: tensor<1048576xf64>) -> tensor<1048576xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<262144> : tensor<i64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.constant dense<2.000000e+00> : tensor<1x1024xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]]:2 = stablehlo.while(%[[V9:[^ ,:)]+]] = %[[V5]], %[[V10:[^ ,:)]+]] = %[[V1]]) : tensor<i64>, tensor<1048576xf64>
// CHECK-NEXT: cond {
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.compare LT, %[[V9]], %[[V4]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT: stablehlo.return %[[V11]] : tensor<i1>
// CHECK-NEXT: } do {
// CHECK-NEXT: %[[V11]] = stablehlo.multiply %[[V9]], %[[V3]] : tensor<i64>
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = stablehlo.iota dim = 0 : tensor<262144xi64>
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V11]], dims = [] : (tensor<i64>) -> tensor<262144xi64>
// CHECK-NEXT: %[[V14:[^ ,:)]+]] = stablehlo.add %[[V12]], %[[V13]] : tensor<262144xi64>
// CHECK-NEXT: %[[V15:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V5]], dims = [] : (tensor<i64>) -> tensor<262144xi64>
// CHECK-NEXT: %[[V16:[^ ,:)]+]] = stablehlo.reshape %[[V14]] : (tensor<262144xi64>) -> tensor<262144x1xi64>
// CHECK-NEXT: %[[V17:[^ ,:)]+]] = stablehlo.reshape %[[V15]] : (tensor<262144xi64>) -> tensor<262144x1xi64>
// CHECK-NEXT: %[[V18:[^ ,:)]+]] = stablehlo.concatenate %[[V16]], %[[V17]], dim = 1 : (tensor<262144x1xi64>, tensor<262144x1xi64>) -> tensor<262144x2xi64>
// CHECK-NEXT: %[[V19:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V18]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1024>}> : (tensor<1048576x1024xf64>, tensor<262144x2xi64>) -> tensor<262144x1x1024xf64>
// CHECK-NEXT: %[[V20:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V6]], dims = [1, 2] : (tensor<1x1024xf64>) -> tensor<262144x1x1024xf64>
// CHECK-NEXT: %[[V21:[^ ,:)]+]] = stablehlo.multiply %[[V19]], %[[V20]] : tensor<262144x1x1024xf64>
// CHECK-NEXT: %[[V22:[^ ,:)]+]] = stablehlo.reduce(%[[V21]] init: %[[V7]]) applies stablehlo.add across dimensions = [1, 2] : (tensor<262144x1x1024xf64>, tensor<f64>) -> tensor<262144xf64>
// CHECK-NEXT: %[[V23:[^ ,:)]+]] = stablehlo.reshape %[[V14]] : (tensor<262144xi64>) -> tensor<262144x1xi64>
// CHECK-NEXT: %[[V24:[^ ,:)]+]] = "stablehlo.scatter"(%[[V10]], %[[V23]], %[[V22]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT: ^bb0(%[[V25:[^ ,:)]+]]: tensor<f64>, %[[V26:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V26]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<1048576xf64>, tensor<262144x1xi64>, tensor<262144xf64>) -> tensor<1048576xf64>
// CHECK-NEXT: %[[V27:[^ ,:)]+]] = stablehlo.add %[[V9]], %[[V2]] : tensor<i64>
// CHECK-NEXT: stablehlo.return %[[V27]], %[[V24]] : tensor<i64>, tensor<1048576xf64>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[V8]]#1 : tensor<1048576xf64>
// CHECK-NEXT: }
// CHECK-LABEL: func.func @rows_small(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<65536x1024xf64>, %[[V1:[^ ,:)]+]]: tensor<65536xf64>) -> tensor<65536xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<2.000000e+00> : tensor<1x1024xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.iota dim = 0 : tensor<65536xi64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<65536x1024xf64>) -> tensor<65536x1x1024xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V2]], dims = [1, 2] : (tensor<1x1024xf64>) -> tensor<65536x1x1024xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.multiply %[[V5]], %[[V6]] : tensor<65536x1x1024xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.reduce(%[[V7]] init: %[[V3]]) applies stablehlo.add across dimensions = [1, 2] : (tensor<65536x1x1024xf64>, tensor<f64>) -> tensor<65536xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.reshape %[[V4]] : (tensor<65536xi64>) -> tensor<65536x1xi64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = "stablehlo.scatter"(%[[V1]], %[[V9]], %[[V8]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT: ^bb0(%[[V11:[^ ,:)]+]]: tensor<f64>, %[[V12:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V12]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<65536xf64>, tensor<65536x1xi64>, tensor<65536xf64>) -> tensor<65536xf64>
// CHECK-NEXT: return %[[V10]] : tensor<65536xf64>
// CHECK-NEXT: }
