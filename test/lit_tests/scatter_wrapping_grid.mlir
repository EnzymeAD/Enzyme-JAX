// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt="max_constant_expansion=0" | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt="max_constant_expansion=0" --inline --canonicalize --symbol-dce | stablehlo-translate --interpret

// A flat affine grid can cross boundaries of its reshaped operand. Split the
// update at those boundaries, preserving both the destination addresses and
// their corresponding update values. The runtime checks at the end distinguish
// correct wrapping from DUS clamping and cover out-of-bounds scatter semantics.

// indices[i,j] = 7 + i + 8*j. Split into [0:3,7:8] and [1:4,0:1].
// CHECK-LABEL: func.func @wrap_row
// CHECK-SAME: (%[[DEST:.*]]: tensor<32xf32>, %[[UPDATE:.*]]: tensor<2x3xf32>)
// CHECK-NOT: stablehlo.scatter
// CHECK-DAG: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-DAG: %[[C1:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-DAG: %[[C7:.*]] = stablehlo.constant dense<7> : tensor<i32>
// CHECK: %[[VIEW:.*]] = stablehlo.reshape %[[DEST]] : (tensor<32xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[SLICE0:.*]] = stablehlo.slice %[[UPDATE]] [0:1, 0:3] : (tensor<2x3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[TILE0:.*]] = stablehlo.reshape %[[SLICE0]] : (tensor<1x3xf32>) -> tensor<3x1xf32>
// CHECK-NEXT: %[[DUS0:.*]] = stablehlo.dynamic_update_slice %[[VIEW]], %[[TILE0]], %[[C0]], %[[C7]] : (tensor<4x8xf32>, tensor<3x1xf32>, tensor<i32>, tensor<i32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[SLICE1:.*]] = stablehlo.slice %[[UPDATE]] [1:2, 0:3] : (tensor<2x3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[TILE1:.*]] = stablehlo.reshape %[[SLICE1]] : (tensor<1x3xf32>) -> tensor<3x1xf32>
// CHECK-NEXT: %[[DUS1:.*]] = stablehlo.dynamic_update_slice %[[DUS0]], %[[TILE1]], %[[C1]], %[[C0]] : (tensor<4x8xf32>, tensor<3x1xf32>, tensor<i32>, tensor<i32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[RESULT:.*]] = stablehlo.reshape %[[DUS1]] : (tensor<4x8xf32>) -> tensor<32xf32>
// CHECK-NEXT: return %[[RESULT]] : tensor<32xf32>
func.func @wrap_row(%dest: tensor<32xf32>, %update: tensor<2x3xf32>) -> tensor<32xf32> {
  %i0 = stablehlo.iota dim = 0 : tensor<2x3x1xi64>
  %i1 = stablehlo.iota dim = 1 : tensor<2x3x1xi64>
  %scale1 = stablehlo.constant dense<8> : tensor<2x3x1xi64>
  %term1 = stablehlo.multiply %i1, %scale1 : tensor<2x3x1xi64>
  %sum0 = stablehlo.add %i0, %term1 : tensor<2x3x1xi64>
  %offset = stablehlo.constant dense<7> : tensor<2x3x1xi64>
  %grid = stablehlo.add %sum0, %offset : tensor<2x3x1xi64>
  %result = "stablehlo.scatter"(%dest, %grid, %update) <{
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>,
    unique_indices = true
  }> ({
  ^bb0(%old: tensor<f32>, %value: tensor<f32>):
    stablehlo.return %value : tensor<f32>
  }) : (tensor<32xf32>, tensor<2x3x1xi64>, tensor<2x3xf32>) -> tensor<32xf32>
  return %result : tensor<32xf32>
}

// indices[i,j] = 24 - i - 8*j. Reverse the update axes before splitting.
// CHECK-LABEL: func.func @reverse_both_axes
// CHECK-SAME: (%[[DEST:.*]]: tensor<32xf32>, %[[UPDATE:.*]]: tensor<2x3xf32>)
// CHECK-NOT: stablehlo.scatter
// CHECK-DAG: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-DAG: %[[C1:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-DAG: %[[C7:.*]] = stablehlo.constant dense<7> : tensor<i32>
// CHECK: %[[VIEW:.*]] = stablehlo.reshape %[[DEST]] : (tensor<32xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[TRANSPOSE:.*]] = stablehlo.transpose %[[UPDATE]], dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT: %[[REVERSE:.*]] = stablehlo.reverse %[[TRANSPOSE]], dims = [0, 1] : tensor<3x2xf32>
// CHECK-NEXT: %[[SLICE0:.*]] = stablehlo.slice %[[REVERSE]] [0:3, 0:1] : (tensor<3x2xf32>) -> tensor<3x1xf32>
// CHECK-NEXT: %[[DUS0:.*]] = stablehlo.dynamic_update_slice %[[VIEW]], %[[SLICE0]], %[[C0]], %[[C7]] : (tensor<4x8xf32>, tensor<3x1xf32>, tensor<i32>, tensor<i32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[SLICE1:.*]] = stablehlo.slice %[[REVERSE]] [0:3, 1:2] : (tensor<3x2xf32>) -> tensor<3x1xf32>
// CHECK-NEXT: %[[DUS1:.*]] = stablehlo.dynamic_update_slice %[[DUS0]], %[[SLICE1]], %[[C1]], %[[C0]] : (tensor<4x8xf32>, tensor<3x1xf32>, tensor<i32>, tensor<i32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[RESULT:.*]] = stablehlo.reshape %[[DUS1]] : (tensor<4x8xf32>) -> tensor<32xf32>
// CHECK-NEXT: return %[[RESULT]] : tensor<32xf32>
func.func @reverse_both_axes(%dest: tensor<32xf32>, %update: tensor<2x3xf32>) -> tensor<32xf32> {
  %i0 = stablehlo.iota dim = 0 : tensor<2x3x1xi64>
  %scale0 = stablehlo.constant dense<-1> : tensor<2x3x1xi64>
  %term0 = stablehlo.multiply %i0, %scale0 : tensor<2x3x1xi64>
  %i1 = stablehlo.iota dim = 1 : tensor<2x3x1xi64>
  %scale1 = stablehlo.constant dense<-8> : tensor<2x3x1xi64>
  %term1 = stablehlo.multiply %i1, %scale1 : tensor<2x3x1xi64>
  %sum0 = stablehlo.add %term0, %term1 : tensor<2x3x1xi64>
  %offset = stablehlo.constant dense<24> : tensor<2x3x1xi64>
  %grid = stablehlo.add %sum0, %offset : tensor<2x3x1xi64>
  %result = "stablehlo.scatter"(%dest, %grid, %update) <{
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>,
    unique_indices = true
  }> ({
  ^bb0(%old: tensor<f32>, %value: tensor<f32>):
    stablehlo.return %value : tensor<f32>
  }) : (tensor<32xf32>, tensor<2x3x1xi64>, tensor<2x3xf32>) -> tensor<32xf32>
  return %result : tensor<32xf32>
}

// The 4x4x4 destination view crosses both its inner and middle dimensions.
// CHECK-LABEL: func.func @wrap_multiple_axes
// CHECK-SAME: (%[[DEST:.*]]: tensor<64xf32>, %[[UPDATE:.*]]: tensor<2x2x2xf32>)
// CHECK-NOT: stablehlo.scatter
// CHECK-DAG: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-DAG: %[[C1:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-DAG: %[[C3:.*]] = stablehlo.constant dense<3> : tensor<i32>
// CHECK: %[[VIEW:.*]] = stablehlo.reshape %[[DEST]] : (tensor<64xf32>) -> tensor<4x4x4xf32>
// CHECK-NEXT: %[[SLICE0:.*]] = stablehlo.slice %[[UPDATE]] [0:1, 0:1, 0:2] : (tensor<2x2x2xf32>) -> tensor<1x1x2xf32>
// CHECK-NEXT: %[[TILE0:.*]] = stablehlo.reshape %[[SLICE0]] : (tensor<1x1x2xf32>) -> tensor<2x1x1xf32>
// CHECK-NEXT: %[[DUS0:.*]] = stablehlo.dynamic_update_slice %[[VIEW]], %[[TILE0]], %[[C0]], %[[C3]], %[[C3]] : (tensor<4x4x4xf32>, tensor<2x1x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x4x4xf32>
// CHECK-NEXT: %[[SLICE1:.*]] = stablehlo.slice %[[UPDATE]] [0:1, 1:2, 0:2] : (tensor<2x2x2xf32>) -> tensor<1x1x2xf32>
// CHECK-NEXT: %[[TILE1:.*]] = stablehlo.reshape %[[SLICE1]] : (tensor<1x1x2xf32>) -> tensor<2x1x1xf32>
// CHECK-NEXT: %[[DUS1:.*]] = stablehlo.dynamic_update_slice %[[DUS0]], %[[TILE1]], %[[C1]], %[[C0]], %[[C3]] : (tensor<4x4x4xf32>, tensor<2x1x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x4x4xf32>
// CHECK-NEXT: %[[SLICE2:.*]] = stablehlo.slice %[[UPDATE]] [1:2, 0:2, 0:2] : (tensor<2x2x2xf32>) -> tensor<1x2x2xf32>
// CHECK-NEXT: %[[TILE2:.*]] = stablehlo.transpose %[[SLICE2]], dims = [2, 1, 0] : (tensor<1x2x2xf32>) -> tensor<2x2x1xf32>
// CHECK-NEXT: %[[DUS2:.*]] = stablehlo.dynamic_update_slice %[[DUS1]], %[[TILE2]], %[[C1]], %[[C0]], %[[C0]] : (tensor<4x4x4xf32>, tensor<2x2x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x4x4xf32>
// CHECK-NEXT: %[[RESULT:.*]] = stablehlo.reshape %[[DUS2]] : (tensor<4x4x4xf32>) -> tensor<64xf32>
// CHECK-NEXT: return %[[RESULT]] : tensor<64xf32>
func.func @wrap_multiple_axes(%dest: tensor<64xf32>, %update: tensor<2x2x2xf32>) -> tensor<64xf32> {
  %i0 = stablehlo.iota dim = 0 : tensor<2x2x2x1xi64>
  %i1 = stablehlo.iota dim = 1 : tensor<2x2x2x1xi64>
  %scale1 = stablehlo.constant dense<4> : tensor<2x2x2x1xi64>
  %term1 = stablehlo.multiply %i1, %scale1 : tensor<2x2x2x1xi64>
  %i2 = stablehlo.iota dim = 2 : tensor<2x2x2x1xi64>
  %scale2 = stablehlo.constant dense<16> : tensor<2x2x2x1xi64>
  %term2 = stablehlo.multiply %i2, %scale2 : tensor<2x2x2x1xi64>
  %sum0 = stablehlo.add %i0, %term1 : tensor<2x2x2x1xi64>
  %sum1 = stablehlo.add %sum0, %term2 : tensor<2x2x2x1xi64>
  %offset = stablehlo.constant dense<15> : tensor<2x2x2x1xi64>
  %grid = stablehlo.add %sum1, %offset : tensor<2x2x2x1xi64>
  %result = "stablehlo.scatter"(%dest, %grid, %update) <{
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 3>,
    unique_indices = true
  }> ({
  ^bb0(%old: tensor<f32>, %value: tensor<f32>):
    stablehlo.return %value : tensor<f32>
  }) : (tensor<64xf32>, tensor<2x2x2x1xi64>, tensor<2x2x2xf32>) -> tensor<64xf32>
  return %result : tensor<64xf32>
}

// Some indices exceed the flat buffer. A clamped DUS would change the writes.
// CHECK-LABEL: func.func @out_of_bounds
// CHECK: "stablehlo.scatter"
// CHECK: return
func.func @out_of_bounds(%dest: tensor<32xf32>, %update: tensor<2x3xf32>) -> tensor<32xf32> {
  %i0 = stablehlo.iota dim = 0 : tensor<2x3x1xi64>
  %i1 = stablehlo.iota dim = 1 : tensor<2x3x1xi64>
  %scale1 = stablehlo.constant dense<8> : tensor<2x3x1xi64>
  %term1 = stablehlo.multiply %i1, %scale1 : tensor<2x3x1xi64>
  %sum0 = stablehlo.add %i0, %term1 : tensor<2x3x1xi64>
  %offset = stablehlo.constant dense<24> : tensor<2x3x1xi64>
  %grid = stablehlo.add %sum0, %offset : tensor<2x3x1xi64>
  %result = "stablehlo.scatter"(%dest, %grid, %update) <{
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>,
    unique_indices = true
  }> ({
  ^bb0(%old: tensor<f32>, %value: tensor<f32>):
    stablehlo.return %value : tensor<f32>
  }) : (tensor<32xf32>, tensor<2x3x1xi64>, tensor<2x3xf32>) -> tensor<32xf32>
  return %result : tensor<32xf32>
}

// i8 arithmetic wraps past 127 even though the mathematical grid fits in 256.
// CHECK-LABEL: func.func @overflowing_indices
// CHECK: "stablehlo.scatter"
// CHECK: return
func.func @overflowing_indices(%dest: tensor<256xf32>, %update: tensor<2x3xf32>) -> tensor<256xf32> {
  %i0 = stablehlo.iota dim = 0 : tensor<2x3x1xi8>
  %i1 = stablehlo.iota dim = 1 : tensor<2x3x1xi8>
  %scale1 = stablehlo.constant dense<8> : tensor<2x3x1xi8>
  %term1 = stablehlo.multiply %i1, %scale1 : tensor<2x3x1xi8>
  %sum0 = stablehlo.add %i0, %term1 : tensor<2x3x1xi8>
  %offset = stablehlo.constant dense<120> : tensor<2x3x1xi8>
  %grid = stablehlo.add %sum0, %offset : tensor<2x3x1xi8>
  %result = "stablehlo.scatter"(%dest, %grid, %update) <{
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>,
    unique_indices = true
  }> ({
  ^bb0(%old: tensor<f32>, %value: tensor<f32>):
    stablehlo.return %value : tensor<f32>
  }) : (tensor<256xf32>, tensor<2x3x1xi8>, tensor<2x3xf32>) -> tensor<256xf32>
  return %result : tensor<256xf32>
}

// The j coordinate repeats the same addresses; it is not a rectangular update.
// CHECK-LABEL: func.func @repeated_indices
// CHECK: "stablehlo.scatter"
// CHECK: return
func.func @repeated_indices(%dest: tensor<32xf32>, %update: tensor<2x3xf32>) -> tensor<32xf32> {
  %i0 = stablehlo.iota dim = 0 : tensor<2x3x1xi64>
  %i1 = stablehlo.iota dim = 1 : tensor<2x3x1xi64>
  %scale1 = stablehlo.constant dense<0> : tensor<2x3x1xi64>
  %term1 = stablehlo.multiply %i1, %scale1 : tensor<2x3x1xi64>
  %sum0 = stablehlo.add %i0, %term1 : tensor<2x3x1xi64>
  %offset = stablehlo.constant dense<7> : tensor<2x3x1xi64>
  %grid = stablehlo.add %sum0, %offset : tensor<2x3x1xi64>
  %result = "stablehlo.scatter"(%dest, %grid, %update) <{
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>,
    unique_indices = false
  }> ({
  ^bb0(%old: tensor<f32>, %value: tensor<f32>):
    stablehlo.return %value : tensor<f32>
  }) : (tensor<32xf32>, tensor<2x3x1xi64>, tensor<2x3xf32>) -> tensor<32xf32>
  return %result : tensor<32xf32>
}

// Reshaping 2x3 indices to 3x2 preserves their linear order, not their axes.
// CHECK-LABEL: func.func @reshaped_index_grid
// CHECK-SAME: (%[[DEST:.*]]: tensor<32xf32>, %[[UPDATE:.*]]: tensor<3x2xf32>)
// CHECK-NOT: stablehlo.scatter
// CHECK-DAG: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-DAG: %[[C1:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-DAG: %[[C7:.*]] = stablehlo.constant dense<7> : tensor<i32>
// CHECK: %[[VIEW:.*]] = stablehlo.reshape %[[DEST]] : (tensor<32xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[GRID_UPDATE:.*]] = stablehlo.reshape %[[UPDATE]] : (tensor<3x2xf32>) -> tensor<2x3xf32>
// CHECK-NEXT: %[[SLICE0:.*]] = stablehlo.slice %[[GRID_UPDATE]] [0:1, 0:3] : (tensor<2x3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[TILE0:.*]] = stablehlo.reshape %[[SLICE0]] : (tensor<1x3xf32>) -> tensor<3x1xf32>
// CHECK-NEXT: %[[DUS0:.*]] = stablehlo.dynamic_update_slice %[[VIEW]], %[[TILE0]], %[[C0]], %[[C7]] : (tensor<4x8xf32>, tensor<3x1xf32>, tensor<i32>, tensor<i32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[SLICE1:.*]] = stablehlo.slice %[[GRID_UPDATE]] [1:2, 0:3] : (tensor<2x3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[TILE1:.*]] = stablehlo.reshape %[[SLICE1]] : (tensor<1x3xf32>) -> tensor<3x1xf32>
// CHECK-NEXT: %[[DUS1:.*]] = stablehlo.dynamic_update_slice %[[DUS0]], %[[TILE1]], %[[C1]], %[[C0]] : (tensor<4x8xf32>, tensor<3x1xf32>, tensor<i32>, tensor<i32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[RESULT:.*]] = stablehlo.reshape %[[DUS1]] : (tensor<4x8xf32>) -> tensor<32xf32>
// CHECK-NEXT: return %[[RESULT]] : tensor<32xf32>
func.func @reshaped_index_grid(%dest: tensor<32xf32>, %update: tensor<3x2xf32>) -> tensor<32xf32> {
  %i0 = stablehlo.iota dim = 0 : tensor<2x3xi64>
  %i1 = stablehlo.iota dim = 1 : tensor<2x3xi64>
  %scale1 = stablehlo.constant dense<8> : tensor<2x3xi64>
  %term1 = stablehlo.multiply %i1, %scale1 : tensor<2x3xi64>
  %sum0 = stablehlo.add %i0, %term1 : tensor<2x3xi64>
  %offset = stablehlo.constant dense<7> : tensor<2x3xi64>
  %grid = stablehlo.add %sum0, %offset : tensor<2x3xi64>
  %indices = stablehlo.reshape %grid : (tensor<2x3xi64>) -> tensor<3x2xi64>
  %result = "stablehlo.scatter"(%dest, %indices, %update) <{
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>,
    unique_indices = true
  }> ({
  ^bb0(%old: tensor<f32>, %value: tensor<f32>):
    stablehlo.return %value : tensor<f32>
  }) : (tensor<32xf32>, tensor<3x2xi64>, tensor<3x2xf32>) -> tensor<32xf32>
  return %result : tensor<32xf32>
}

// Interpret nonconstant function inputs after rewriting their scatter bodies.
func.func @main() {
  %d0 = stablehlo.constant dense<0.0> : tensor<32xf32>
  %u0 = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %e0 = stablehlo.constant dense<[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]> : tensor<32xf32>
  %r0 = func.call @wrap_row(%d0, %u0) : (tensor<32xf32>, tensor<2x3xf32>) -> tensor<32xf32>
  check.expect_eq %r0, %e0 : tensor<32xf32>
  %d1 = stablehlo.constant dense<0.0> : tensor<32xf32>
  %u1 = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %e1 = stablehlo.constant dense<[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]> : tensor<32xf32>
  %r1 = func.call @reverse_both_axes(%d1, %u1) : (tensor<32xf32>, tensor<2x3xf32>) -> tensor<32xf32>
  check.expect_eq %r1, %e1 : tensor<32xf32>
  %d2 = stablehlo.constant dense<0.0> : tensor<64xf32>
  %u2 = stablehlo.constant dense<[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]> : tensor<2x2x2xf32>
  %e2 = stablehlo.constant dense<[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 3.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 6.0, 0.0, 0.0, 4.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]> : tensor<64xf32>
  %r2 = func.call @wrap_multiple_axes(%d2, %u2) : (tensor<64xf32>, tensor<2x2x2xf32>) -> tensor<64xf32>
  check.expect_eq %r2, %e2 : tensor<64xf32>
  %d3 = stablehlo.constant dense<0.0> : tensor<32xf32>
  %u3 = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %e3 = stablehlo.constant dense<[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]> : tensor<32xf32>
  %r3 = func.call @out_of_bounds(%d3, %u3) : (tensor<32xf32>, tensor<2x3xf32>) -> tensor<32xf32>
  check.expect_eq %r3, %e3 : tensor<32xf32>
  %d6 = stablehlo.constant dense<0.0> : tensor<32xf32>
  %u6 = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32>
  %e6 = stablehlo.constant dense<[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]> : tensor<32xf32>
  %r6 = func.call @reshaped_index_grid(%d6, %u6) : (tensor<32xf32>, tensor<3x2xf32>) -> tensor<32xf32>
  check.expect_eq %r6, %e6 : tensor<32xf32>
  return
}
