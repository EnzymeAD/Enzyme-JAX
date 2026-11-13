// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// Main diagonal via gather(iota) (#1617)
// CHECK-LABEL: func.func @gather_iota_main_diag
// CHECK-SAME: (%[[A:.+]]: tensor<7x4xf64>, %[[B:.+]]: tensor<4x32xf64>)
// CHECK-NOT: tensor<7x32xf64>
// CHECK-NOT: stablehlo.gather
// CHECK: %[[BS:.+]] = stablehlo.slice %[[B]] [0:4, 0:7] : (tensor<4x32xf64>) -> tensor<4x7xf64>
// CHECK: %[[DOT:.+]] = stablehlo.dot_general %[[A]], %[[BS]], batching_dims = [0] x [1], contracting_dims = [1] x [0] : (tensor<7x4xf64>, tensor<4x7xf64>) -> tensor<7xf64>
// CHECK: return %[[DOT]] : tensor<7xf64>
func.func @gather_iota_main_diag(%a: tensor<7x4xf64>, %b: tensor<4x32xf64>) -> tensor<7xf64> {
  %dot = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<7x4xf64>, tensor<4x32xf64>) -> tensor<7x32xf64>
  %idx = stablehlo.iota dim = 0 : tensor<7x2xi64>
  %g = "stablehlo.gather"(%dot, %idx) <{
    dimension_numbers = #stablehlo.gather<
      offset_dims = [],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    slice_sizes = array<i64: 1, 1>
  }> : (tensor<7x32xf64>, tensor<7x2xi64>) -> tensor<7xf64>
  return %g : tensor<7xf64>
}

// Super-diagonal (k = +2) via gather(concat(iota, iota + 2)) (#1617)
// CHECK-LABEL: func.func @gather_super_diag_concat
// CHECK-SAME: (%[[A:.+]]: tensor<6x4xf64>, %[[B:.+]]: tensor<4x10xf64>)
// CHECK-NOT: tensor<6x10xf64>
// CHECK-NOT: stablehlo.gather
// CHECK: %[[BS:.+]] = stablehlo.slice %[[B]] [0:4, 2:8] : (tensor<4x10xf64>) -> tensor<4x6xf64>
// CHECK: %[[DOT:.+]] = stablehlo.dot_general %[[A]], %[[BS]], batching_dims = [0] x [1], contracting_dims = [1] x [0] : (tensor<6x4xf64>, tensor<4x6xf64>) -> tensor<6xf64>
// CHECK: return %[[DOT]] : tensor<6xf64>
func.func @gather_super_diag_concat(%a: tensor<6x4xf64>, %b: tensor<4x10xf64>) -> tensor<6xf64> {
  %dot = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<6x4xf64>, tensor<4x10xf64>) -> tensor<6x10xf64>
  %r = stablehlo.iota dim = 0 : tensor<6x1xi64>
  %c2 = stablehlo.constant dense<2> : tensor<6x1xi64>
  %c = stablehlo.add %r, %c2 : tensor<6x1xi64>
  %idx = stablehlo.concatenate %r, %c, dim = 1 : (tensor<6x1xi64>, tensor<6x1xi64>) -> tensor<6x2xi64>
  %g = "stablehlo.gather"(%dot, %idx) <{
    dimension_numbers = #stablehlo.gather<
      offset_dims = [],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    slice_sizes = array<i64: 1, 1>
  }> : (tensor<6x10xf64>, tensor<6x2xi64>) -> tensor<6xf64>
  return %g : tensor<6xf64>
}

// Sub-diagonal (k = -2) via reshape + slice (#1617)
// On 8x6, sub-diagonal k = -2 starts at (2, 0) = 2 * 6 + 0 = 12 with stride 7, 5 elements: (2,0)..(6,4)
// CHECK-LABEL: func.func @reshape_slice_sub_diag
// CHECK-SAME: (%[[A:.+]]: tensor<8x4xf64>, %[[B:.+]]: tensor<4x6xf64>)
// CHECK-NOT: tensor<8x6xf64>
// CHECK: %[[AS:.+]] = stablehlo.slice %[[A]] [2:7, 0:4] : (tensor<8x4xf64>) -> tensor<5x4xf64>
// CHECK: %[[BS:.+]] = stablehlo.slice %[[B]] [0:4, 0:5] : (tensor<4x6xf64>) -> tensor<4x5xf64>
// CHECK: %[[DOT:.+]] = stablehlo.dot_general %[[AS]], %[[BS]], batching_dims = [0] x [1], contracting_dims = [1] x [0] : (tensor<5x4xf64>, tensor<4x5xf64>) -> tensor<5xf64>
// CHECK: return %[[DOT]] : tensor<5xf64>
func.func @reshape_slice_sub_diag(%a: tensor<8x4xf64>, %b: tensor<4x6xf64>) -> tensor<5xf64> {
  %dot = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<8x4xf64>, tensor<4x6xf64>) -> tensor<8x6xf64>
  %flat = stablehlo.reshape %dot : (tensor<8x6xf64>) -> tensor<48xf64>
  %s = stablehlo.slice %flat [12:41:7] : (tensor<48xf64>) -> tensor<5xf64>
  return %s : tensor<5xf64>
}

// Dynamic diagonal indices concat(idx, idx) on square matrix
// CHECK-LABEL: func.func @gather_dynamic_concat_diag
// CHECK-SAME: (%[[A:.+]]: tensor<16x4xf32>, %[[B:.+]]: tensor<4x16xf32>, %[[IDX:.+]]: tensor<5x1xi32>)
// CHECK-NOT: tensor<16x16xf32>
// CHECK: %[[DOT:.+]] = stablehlo.dot_general %[[A]], %[[B]], batching_dims = [0] x [1], contracting_dims = [1] x [0] : (tensor<16x4xf32>, tensor<4x16xf32>) -> tensor<16xf32>
// CHECK: %[[G:.+]] = "stablehlo.gather"(%[[DOT]], %[[IDX]])
// CHECK: return %[[G]] : tensor<5xf32>
func.func @gather_dynamic_concat_diag(%a: tensor<16x4xf32>, %b: tensor<4x16xf32>, %idx: tensor<5x1xi32>) -> tensor<5xf32> {
  %dot = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<16x4xf32>, tensor<4x16xf32>) -> tensor<16x16xf32>
  %pair = stablehlo.concatenate %idx, %idx, dim = 1 : (tensor<5x1xi32>, tensor<5x1xi32>) -> tensor<5x2xi32>
  %g = "stablehlo.gather"(%dot, %pair) <{
    dimension_numbers = #stablehlo.gather<
      offset_dims = [],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 1>
  }> : (tensor<16x16xf32>, tensor<5x2xi32>) -> tensor<5xf32>
  return %g : tensor<5xf32>
}

// Negative test: two users reading different diagonals (k = 0 and k = 1) must not rewrite to a single diagonal
// CHECK-LABEL: func.func @mixed_diagonals_not_rewritten
// CHECK: %[[DOT:.+]] = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<6x4xf64>, tensor<4x6xf64>) -> tensor<6x6xf64>
func.func @mixed_diagonals_not_rewritten(%a: tensor<6x4xf64>, %b: tensor<4x6xf64>) -> (tensor<5xf64>, tensor<5xf64>) {
  %dot = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<6x4xf64>, tensor<4x6xf64>) -> tensor<6x6xf64>
  %flat = stablehlo.reshape %dot : (tensor<6x6xf64>) -> tensor<36xf64>
  %d0 = stablehlo.slice %flat [0:35:7] : (tensor<36xf64>) -> tensor<5xf64>
  %d1 = stablehlo.slice %flat [1:36:7] : (tensor<36xf64>) -> tensor<5xf64>
  return %d0, %d1 : tensor<5xf64>, tensor<5xf64>
}
