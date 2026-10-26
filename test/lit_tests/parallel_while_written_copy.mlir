// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=parallel_while_to_batched_scatter" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// Every iteration writes its own scratch, a tensor of 32 zeros, at a place
// it reads, and sums it. Batched, the scratch is a copy per iteration: 2^25
// elements, more than an invariant operand may be broadcast to, but the
// batched scatter's result is that big anyway: the loop batches.
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
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<32xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.iota dim = 0 : tensor<1048576xi64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<1048576xi64>) -> tensor<1048576x1xi64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<1048576xf64>) -> tensor<1048576x1xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.reshape %[[V7]] : (tensor<1048576x1xf64>) -> tensor<1048576xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V4]], dims = [1] : (tensor<32xf64>) -> tensor<1048576x32xf64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = "stablehlo.scatter"(%[[V9]], %[[V6]], %[[V8]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT: ^bb0(%[[V11:[^ ,:)]+]]: tensor<f64>, %[[V12:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V12]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<1048576x32xf64>, tensor<1048576x1xi64>, tensor<1048576xf64>) -> tensor<1048576x32xf64>
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.reduce(%[[V10]] init: %[[V3]]) applies stablehlo.add across dimensions = [1] : (tensor<1048576x32xf64>, tensor<f64>) -> tensor<1048576xf64>
// CHECK-NEXT: %[[V14:[^ ,:)]+]] = stablehlo.reshape %[[V5]] : (tensor<1048576xi64>) -> tensor<1048576x1xi64>
// CHECK-NEXT: %[[V15:[^ ,:)]+]] = "stablehlo.scatter"(%[[V2]], %[[V14]], %[[V13]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT: ^bb0(%[[V11]]: tensor<f64>, %[[V12]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V12]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<1048576xf64>, tensor<1048576x1xi64>, tensor<1048576xf64>) -> tensor<1048576xf64>
// CHECK-NEXT: return %[[V15]] : tensor<1048576xf64>
// CHECK-NEXT: }

// The same with a scratch of 1024: 8 GiB of copies, which the batched
// write needs all the same -- no budget, the loop batches.
func.func @scratch_too_large(%idx: tensor<1048576xi64>, %x: tensor<1048576xf64>, %y: tensor<1048576xf64>) -> tensor<1048576xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %cn = stablehlo.constant dense<1048576> : tensor<i64>
  %zero = stablehlo.constant dense<0.0> : tensor<1024xf64>
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
    }) : (tensor<1024xf64>, tensor<1xi64>, tensor<f64>) -> tensor<1024xf64>
    %c0f = stablehlo.constant dense<0.0> : tensor<f64>
    %r = stablehlo.reduce(%s init: %c0f) applies stablehlo.add across dimensions = [0] : (tensor<1024xf64>, tensor<f64>) -> tensor<f64>
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

// CHECK-LABEL: func.func @scratch_too_large(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<1048576xi64>, %[[V1:[^ ,:)]+]]: tensor<1048576xf64>, %[[V2:[^ ,:)]+]]: tensor<1048576xf64>) -> tensor<1048576xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<1024xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.iota dim = 0 : tensor<1048576xi64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<1048576xi64>) -> tensor<1048576x1xi64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<1048576xf64>) -> tensor<1048576x1xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.reshape %[[V7]] : (tensor<1048576x1xf64>) -> tensor<1048576xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V4]], dims = [1] : (tensor<1024xf64>) -> tensor<1048576x1024xf64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = "stablehlo.scatter"(%[[V9]], %[[V6]], %[[V8]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT: ^bb0(%[[V11:[^ ,:)]+]]: tensor<f64>, %[[V12:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V12]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<1048576x1024xf64>, tensor<1048576x1xi64>, tensor<1048576xf64>) -> tensor<1048576x1024xf64>
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.reduce(%[[V10]] init: %[[V3]]) applies stablehlo.add across dimensions = [1] : (tensor<1048576x1024xf64>, tensor<f64>) -> tensor<1048576xf64>
// CHECK-NEXT: %[[V14:[^ ,:)]+]] = stablehlo.reshape %[[V5]] : (tensor<1048576xi64>) -> tensor<1048576x1xi64>
// CHECK-NEXT: %[[V15:[^ ,:)]+]] = "stablehlo.scatter"(%[[V2]], %[[V14]], %[[V13]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT: ^bb0(%[[V11]]: tensor<f64>, %[[V12]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V12]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<1048576xf64>, tensor<1048576x1xi64>, tensor<1048576xf64>) -> tensor<1048576xf64>
// CHECK-NEXT: return %[[V15]] : tensor<1048576xf64>
// CHECK-NEXT: }
