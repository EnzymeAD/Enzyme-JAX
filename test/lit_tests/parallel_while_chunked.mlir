// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=parallel_while_to_batched_scatter_chunked(3);enzyme_hlo_unroll(8)" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// Asked for at least 3 chunks, a loop of 8 iterations (from 2 to 10) is
// batched in 4 chunks of 2, one after the other: the loop over the chunks
// carries the buffer the iterations write and the sum they accumulate, and
// each chunk reads its iterations' rows of the matrix. The loop over the
// chunks is kept a loop, though it runs fewer times than the unroll bound.
func.func @chunks(%m: tensor<8x4xf64>, %perm: tensor<8xi64>, %y: tensor<12xf64>, %s0: tensor<f64>) -> (tensor<12xf64>, tensor<f64>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %c10 = stablehlo.constant dense<10> : tensor<i64>
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %0:3 = stablehlo.while(%i = %c2, %acc = %y, %s = %s0) : tensor<i64>, tensor<12xf64>, tensor<f64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %j = stablehlo.subtract %i, %c2 : tensor<i64>
    %row = stablehlo.dynamic_slice %m, %j, %c0, sizes = [1, 4] : (tensor<8x4xf64>, tensor<i64>, tensor<i64>) -> tensor<1x4xf64>
    %r = stablehlo.reduce(%row init: %z) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x4xf64>, tensor<f64>) -> tensor<f64>
    %p = stablehlo.dynamic_slice %perm, %j, sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
    %u = "stablehlo.scatter"(%acc, %p, %r) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) : (tensor<12xf64>, tensor<1xi64>, tensor<f64>) -> tensor<12xf64>
    %s2 = stablehlo.add %s, %r : tensor<f64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %u, %s2 : tensor<i64>, tensor<12xf64>, tensor<f64>
  }
  return %0#1, %0#2 : tensor<12xf64>, tensor<f64>
}

// CHECK-LABEL: func.func @chunks(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x4xf64>, %[[V1:[^ ,:)]+]]: tensor<8xi64>, %[[V2:[^ ,:)]+]]: tensor<12xf64>, %[[V3:[^ ,:)]+]]: tensor<f64>) -> (tensor<12xf64>, tensor<f64>) {
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]]:3 = stablehlo.while(%[[V10:[^ ,:)]+]] = %[[V6]], %[[V11:[^ ,:)]+]] = %[[V2]], %[[V12:[^ ,:)]+]] = %[[V3]]) : tensor<i64>, tensor<12xf64>, tensor<f64> attributes {enzymexla.chunks}
// CHECK-NEXT: cond {
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.compare LT, %[[V10]], %[[V5]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT: stablehlo.return %[[V13]] : tensor<i1>
// CHECK-NEXT: } do {
// CHECK-NEXT: %[[V13]] = stablehlo.multiply %[[V10]], %[[V7]] : tensor<i64>
// CHECK-NEXT: %[[V14:[^ ,:)]+]] = stablehlo.add %[[V13]], %[[V7]] : tensor<i64>
// CHECK-NEXT: %[[V15:[^ ,:)]+]] = stablehlo.iota dim = 0 : tensor<2xi64>
// CHECK-NEXT: %[[V16:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V14]], dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT: %[[V17:[^ ,:)]+]] = stablehlo.add %[[V15]], %[[V16]] : tensor<2xi64>
// CHECK-NEXT: %[[V18:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V7]], dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT: %[[V19:[^ ,:)]+]] = stablehlo.subtract %[[V17]], %[[V18]] : tensor<2xi64>
// CHECK-NEXT: %[[V20:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V6]], dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT: %[[V21:[^ ,:)]+]] = stablehlo.reshape %[[V19]] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT: %[[V22:[^ ,:)]+]] = stablehlo.reshape %[[V20]] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT: %[[V23:[^ ,:)]+]] = stablehlo.concatenate %[[V21]], %[[V22]], dim = 1 : (tensor<2x1xi64>, tensor<2x1xi64>) -> tensor<2x2xi64>
// CHECK-NEXT: %[[V24:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V23]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4>}> : (tensor<8x4xf64>, tensor<2x2xi64>) -> tensor<2x1x4xf64>
// CHECK-NEXT: %[[V25:[^ ,:)]+]] = stablehlo.reduce(%[[V24]] init: %[[V8]]) applies stablehlo.add across dimensions = [1, 2] : (tensor<2x1x4xf64>, tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT: %[[V26:[^ ,:)]+]] = stablehlo.reshape %[[V19]] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT: %[[V27:[^ ,:)]+]] = "stablehlo.gather"(%[[V1]], %[[V26]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<8xi64>, tensor<2x1xi64>) -> tensor<2x1xi64>
// CHECK-NEXT: %[[V28:[^ ,:)]+]] = "stablehlo.scatter"(%[[V11]], %[[V27]], %[[V25]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT: ^bb0(%[[V29:[^ ,:)]+]]: tensor<f64>, %[[V30:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V30]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<12xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<12xf64>
// CHECK-NEXT: %[[V31:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V8]], dims = [] : (tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT: %[[V32:[^ ,:)]+]] = stablehlo.add %[[V31]], %[[V25]] : tensor<2xf64>
// CHECK-NEXT: %[[V33:[^ ,:)]+]] = stablehlo.reduce(%[[V32]] init: %[[V8]]) applies stablehlo.add across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[V34:[^ ,:)]+]] = stablehlo.add %[[V12]], %[[V33]] : tensor<f64>
// CHECK-NEXT: %[[V35:[^ ,:)]+]] = stablehlo.add %[[V10]], %[[V4]] : tensor<i64>
// CHECK-NEXT: stablehlo.return %[[V35]], %[[V28]], %[[V34]] : tensor<i64>, tensor<12xf64>, tensor<f64>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[V9]]#1, %[[V9]]#2 : tensor<12xf64>, tensor<f64>
// CHECK-NEXT: }
