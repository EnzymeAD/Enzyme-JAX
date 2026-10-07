// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=scatter_indices_are_unique" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Batched indices broadcast from a constant: each batch writes its own copy
// of the same distinct slots, so no two scatter points write one element.
func.func @batched_broadcast(%x: tensor<8x375xf64>, %u: tensor<8x5xf64>) -> tensor<8x375xf64> {
  %c = stablehlo.constant dense<[[125], [150], [175], [200], [225]]> : tensor<5x1xi64>
  %idx = stablehlo.broadcast_in_dim %c, dims = [1, 2] : (tensor<5x1xi64>) -> tensor<8x5x1xi64>
  %0 = "stablehlo.scatter"(%x, %idx, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<8x375xf64>, tensor<8x5x1xi64>, tensor<8x5xf64>) -> tensor<8x375xf64>
  return %0 : tensor<8x375xf64>
}

// CHECK:  func.func @batched_broadcast(%arg0: tensor<8x375xf64>, %arg1: tensor<8x5xf64>) -> tensor<8x375xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<{{\[\[}}125], [150], [175], [200], [225]]> : tensor<5x1xi64>
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %c, dims = [1, 2] : (tensor<5x1xi64>) -> tensor<8x5x1xi64>
// CHECK-NEXT:   %1 = "stablehlo.scatter"(%arg0, %0, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<8x375xf64>, tensor<8x5x1xi64>, tensor<8x5xf64>) -> tensor<8x375xf64>
// CHECK-NEXT:   return %1 : tensor<8x375xf64>
// CHECK-NEXT: }

// Indices computed from constants by a slice, an add and a clamp, left
// unfolded (the constant has another user).
func.func @tree(%x: tensor<3000xf64>, %u: tensor<8x1xf64>) -> (tensor<3000xf64>, tensor<2x8xi64>) {
  %c = stablehlo.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]> : tensor<2x8xi64>
  %s = stablehlo.slice %c [0:1, 0:8] : (tensor<2x8xi64>) -> tensor<1x8xi64>
  %r = stablehlo.reshape %s : (tensor<1x8xi64>) -> tensor<8xi64>
  %k = stablehlo.constant dense<375> : tensor<8xi64>
  %m = stablehlo.multiply %r, %k : tensor<8xi64>
  %o = stablehlo.constant dense<301> : tensor<8xi64>
  %a = stablehlo.add %m, %o : tensor<8xi64>
  %lo = stablehlo.constant dense<0> : tensor<i64>
  %hi = stablehlo.constant dense<2999> : tensor<i64>
  %cl = stablehlo.clamp %lo, %a, %hi : (tensor<i64>, tensor<8xi64>, tensor<i64>) -> tensor<8xi64>
  %idx = stablehlo.reshape %cl : (tensor<8xi64>) -> tensor<8x1xi64>
  %0 = "stablehlo.scatter"(%x, %idx, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a0: tensor<f64>, %b0: tensor<f64>):
    stablehlo.return %b0 : tensor<f64>
  }) : (tensor<3000xf64>, tensor<8x1xi64>, tensor<8x1xf64>) -> tensor<3000xf64>
  return %0, %c : tensor<3000xf64>, tensor<2x8xi64>
}

// CHECK:  func.func @tree(%arg0: tensor<3000xf64>, %arg1: tensor<8x1xf64>) -> (tensor<3000xf64>, tensor<2x8xi64>) {
// CHECK-NEXT:   %c = stablehlo.constant dense<2999> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<301> : tensor<8xi64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<375> : tensor<8xi64>
// CHECK-NEXT:   %c_3 = stablehlo.constant dense<{{\[\[}}0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]> : tensor<2x8xi64>
// CHECK-NEXT:   %0 = stablehlo.slice %c_3 [0:1, 0:8] : (tensor<2x8xi64>) -> tensor<1x8xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %0 : (tensor<1x8xi64>) -> tensor<8xi64>
// CHECK-NEXT:   %2 = stablehlo.multiply %1, %c_2 : tensor<8xi64>
// CHECK-NEXT:   %3 = stablehlo.add %2, %c_1 : tensor<8xi64>
// CHECK-NEXT:   %4 = stablehlo.clamp %c_0, %3, %c : (tensor<i64>, tensor<8xi64>, tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %5 = stablehlo.reshape %4 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:   %6 = "stablehlo.scatter"(%arg0, %5, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<3000xf64>, tensor<8x1xi64>, tensor<8x1xf64>) -> tensor<3000xf64>
// CHECK-NEXT:   return %6, %c_3 : tensor<3000xf64>, tensor<2x8xi64>
// CHECK-NEXT: }

// Two scatter points of one batch at the same slot: not unique.
func.func @batched_duplicate(%x: tensor<8x375xf64>, %u: tensor<8x2xf64>) -> tensor<8x375xf64> {
  %c = stablehlo.constant dense<[[7], [7]]> : tensor<2x1xi64>
  %idx = stablehlo.broadcast_in_dim %c, dims = [1, 2] : (tensor<2x1xi64>) -> tensor<8x2x1xi64>
  %0 = "stablehlo.scatter"(%x, %idx, %u) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<8x375xf64>, tensor<8x2x1xi64>, tensor<8x2xf64>) -> tensor<8x375xf64>
  return %0 : tensor<8x375xf64>
}

// CHECK:  func.func @batched_duplicate(%arg0: tensor<8x375xf64>, %arg1: tensor<8x2xf64>) -> tensor<8x375xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<7> : tensor<2x1xi64>
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %c, dims = [1, 2] : (tensor<2x1xi64>) -> tensor<8x2x1xi64>
// CHECK-NEXT:   %1 = "stablehlo.scatter"(%arg0, %0, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<8x375xf64>, tensor<8x2x1xi64>, tensor<8x2xf64>) -> tensor<8x375xf64>
// CHECK-NEXT:   return %1 : tensor<8x375xf64>
// CHECK-NEXT: }
