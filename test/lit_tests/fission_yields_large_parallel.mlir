// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=greedy_while_loop_batch_fission" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// A parallel loop whose values, batched, are past the 8 GiB a batch may hold:
// fission lifts nothing out of it (each op it lifts would hold every
// iteration's value at once), leaving it whole to
// parallel_while_to_batched_scatter, which batches it a chunk at a time.
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

// The same over 2^16 iterations (1 GiB): fission lifts the row work out.
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
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<1048576> : tensor<i64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.constant dense<2.000000e+00> : tensor<1x1024xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]]:2 = stablehlo.while(%[[V8:[^ ,:)]+]] = %[[V2]], %[[V9:[^ ,:)]+]] = %[[V1]]) : tensor<i64>, tensor<1048576xf64> attributes {enzymexla.parallel}
// CHECK-NEXT: cond {
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.compare LT, %[[V8]], %[[V4]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT: stablehlo.return %[[V10]] : tensor<i1>
// CHECK-NEXT: } do {
// CHECK-NEXT: %[[V10]] = stablehlo.dynamic_slice %[[V0]], %[[V8]], %[[V2]], sizes = [1, 1024] : (tensor<1048576x1024xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1024xf64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.multiply %[[V10]], %[[V5]] : tensor<1x1024xf64>
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = stablehlo.reduce(%[[V11]] init: %[[V6]]) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1024xf64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.reshape %[[V8]] : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT: %[[V14:[^ ,:)]+]] = "stablehlo.scatter"(%[[V9]], %[[V13]], %[[V12]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
// CHECK-NEXT: ^bb0(%[[V15:[^ ,:)]+]]: tensor<f64>, %[[V16:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V16]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<1048576xf64>, tensor<1xi64>, tensor<f64>) -> tensor<1048576xf64>
// CHECK-NEXT: %[[V17:[^ ,:)]+]] = stablehlo.add %[[V8]], %[[V3]] : tensor<i64>
// CHECK-NEXT: stablehlo.return %[[V17]], %[[V14]] : tensor<i64>, tensor<1048576xf64>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[V7]]#1 : tensor<1048576xf64>
// CHECK-NEXT: }
// CHECK-LABEL: func.func @rows_small(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<65536x1024xf64>, %[[V1:[^ ,:)]+]]: tensor<65536xf64>) -> tensor<65536xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<2.000000e+00> : tensor<65536x1x1024xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.constant dense<65536> : tensor<i64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<65536x1024xf64>) -> tensor<65536x1x1024xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.multiply %[[V7]], %[[V3]] : tensor<65536x1x1024xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.reduce(%[[V8]] init: %[[V2]]) applies stablehlo.add across dimensions = [1, 2] : (tensor<65536x1x1024xf64>, tensor<f64>) -> tensor<65536xf64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]]:2 = stablehlo.while(%[[V11:[^ ,:)]+]] = %[[V4]], %[[V12:[^ ,:)]+]] = %[[V1]]) : tensor<i64>, tensor<65536xf64> attributes {enzymexla.parallel}
// CHECK-NEXT: cond {
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.compare LT, %[[V11]], %[[V6]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT: stablehlo.return %[[V13]] : tensor<i1>
// CHECK-NEXT: } do {
// CHECK-NEXT: %[[V13]] = stablehlo.dynamic_slice %[[V9]], %[[V11]], sizes = [1] : (tensor<65536xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT: %[[V14:[^ ,:)]+]] = stablehlo.reshape %[[V13]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[V15:[^ ,:)]+]] = stablehlo.reshape %[[V11]] : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT: %[[V16:[^ ,:)]+]] = "stablehlo.scatter"(%[[V12]], %[[V15]], %[[V14]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
// CHECK-NEXT: ^bb0(%[[V17:[^ ,:)]+]]: tensor<f64>, %[[V18:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V18]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<65536xf64>, tensor<1xi64>, tensor<f64>) -> tensor<65536xf64>
// CHECK-NEXT: %[[V19:[^ ,:)]+]] = stablehlo.add %[[V11]], %[[V5]] : tensor<i64>
// CHECK-NEXT: stablehlo.return %[[V19]], %[[V16]] : tensor<i64>, tensor<65536xf64>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[V10]]#1 : tensor<65536xf64>
// CHECK-NEXT: }
