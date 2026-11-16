// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=gather_op_canon;scatter_op_canon" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// The BK6 form: the third component is the piece's position along the
// concatenated grid dimension.
func.func @pieces_step(%x: tensor<5x5x8xf64>) -> tensor<3x5x5xf64> {
  %c0 = stablehlo.constant dense<[[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], [[0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0], [4, 1, 0]], [[0, 2, 0], [1, 2, 0], [2, 2, 0], [3, 2, 0], [4, 2, 0]], [[0, 3, 0], [1, 3, 0], [2, 3, 0], [3, 3, 0], [4, 3, 0]], [[0, 4, 0], [1, 4, 0], [2, 4, 0], [3, 4, 0], [4, 4, 0]]]> : tensor<5x5x3xi64>
  %c1 = stablehlo.constant dense<[[[0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, 1], [4, 0, 1]], [[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 1, 1]], [[0, 2, 1], [1, 2, 1], [2, 2, 1], [3, 2, 1], [4, 2, 1]], [[0, 3, 1], [1, 3, 1], [2, 3, 1], [3, 3, 1], [4, 3, 1]], [[0, 4, 1], [1, 4, 1], [2, 4, 1], [3, 4, 1], [4, 4, 1]]]> : tensor<5x5x3xi64>
  %c2 = stablehlo.constant dense<[[[0, 0, 2], [1, 0, 2], [2, 0, 2], [3, 0, 2], [4, 0, 2]], [[0, 1, 2], [1, 1, 2], [2, 1, 2], [3, 1, 2], [4, 1, 2]], [[0, 2, 2], [1, 2, 2], [2, 2, 2], [3, 2, 2], [4, 2, 2]], [[0, 3, 2], [1, 3, 2], [2, 3, 2], [3, 3, 2], [4, 3, 2]], [[0, 4, 2], [1, 4, 2], [2, 4, 2], [3, 4, 2], [4, 4, 2]]]> : tensor<5x5x3xi64>
  %r0 = stablehlo.reshape %c0 : (tensor<5x5x3xi64>) -> tensor<1x5x5x3xi64>
  %r1 = stablehlo.reshape %c1 : (tensor<5x5x3xi64>) -> tensor<1x5x5x3xi64>
  %r2 = stablehlo.reshape %c2 : (tensor<5x5x3xi64>) -> tensor<1x5x5x3xi64>
  %i = stablehlo.concatenate %r0, %r1, %r2, dim = 0 : (tensor<1x5x5x3xi64>, tensor<1x5x5x3xi64>, tensor<1x5x5x3xi64>) -> tensor<3x5x5x3xi64>
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [2, 1, 0], index_vector_dim = 3>, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<5x5x8xf64>, tensor<3x5x5x3xi64>) -> tensor<3x5x5xf64>
  return %g : tensor<3x5x5xf64>
}

// CHECK-LABEL: func.func @pieces_step(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<5x5x8xf64>) -> tensor<3x5x5xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:3, 0:5, 0:5] : (tensor<5x5x8xf64>) -> tensor<3x5x5xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V1]], dims = [0, 1, 2] : (tensor<3x5x5xf64>) -> tensor<3x5x5xf64>
// CHECK-NEXT: return %[[V2]] : tensor<3x5x5xf64>
// CHECK-NEXT: }

// Pieces of different widths along the concatenated dimension.
func.func @uneven(%x: tensor<16xf64>) -> tensor<6xf64> {
  %a = stablehlo.iota dim = 0 : tensor<2x1xi64>
  %c2 = stablehlo.constant dense<[[2], [3], [4], [5]]> : tensor<4x1xi64>
  %i = stablehlo.concatenate %a, %c2, dim = 0 : (tensor<2x1xi64>, tensor<4x1xi64>) -> tensor<6x1xi64>
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<16xf64>, tensor<6x1xi64>) -> tensor<6xf64>
  return %g : tensor<6xf64>
}

// CHECK-LABEL: func.func @uneven(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<16xf64>) -> tensor<6xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:6] : (tensor<16xf64>) -> tensor<6xf64>
// CHECK-NEXT: return %[[V1]] : tensor<6xf64>
// CHECK-NEXT: }

// Stays: the second piece does not continue the first.
func.func @broken_step(%x: tensor<16xf64>) -> tensor<4xf64> {
  %a = stablehlo.iota dim = 0 : tensor<2x1xi64>
  %c = stablehlo.constant dense<[[5], [6]]> : tensor<2x1xi64>
  %i = stablehlo.concatenate %a, %c, dim = 0 : (tensor<2x1xi64>, tensor<2x1xi64>) -> tensor<4x1xi64>
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<16xf64>, tensor<4x1xi64>) -> tensor<4xf64>
  return %g : tensor<4xf64>
}

// CHECK-LABEL: func.func @broken_step(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<16xf64>) -> tensor<4xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<{{\[\[}}5], [6]]> : tensor<2x1xi64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.iota dim = 0 : tensor<2x1xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.concatenate %[[V2]], %[[V1]], dim = 0 : (tensor<2x1xi64>, tensor<2x1xi64>) -> tensor<4x1xi64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V3]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<16xf64>, tensor<4x1xi64>) -> tensor<4xf64>
// CHECK-NEXT: return %[[V4]] : tensor<4xf64>
// CHECK-NEXT: }

// The index grid transposed: column-major reads of a 4x6 box.
func.func @transposed_indices(%x: tensor<64xf64>) -> tensor<6x4xf64> {
  %r = stablehlo.iota dim = 0 : tensor<4x6xi64>
  %c = stablehlo.iota dim = 1 : tensor<4x6xi64>
  %c8 = stablehlo.constant dense<8> : tensor<4x6xi64>
  %r8 = stablehlo.multiply %r, %c8 : tensor<4x6xi64>
  %s = stablehlo.add %r8, %c : tensor<4x6xi64>
  %t = stablehlo.transpose %s, dims = [1, 0] : (tensor<4x6xi64>) -> tensor<6x4xi64>
  %i = stablehlo.reshape %t : (tensor<6x4xi64>) -> tensor<6x4x1xi64>
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1>}> : (tensor<64xf64>, tensor<6x4x1xi64>) -> tensor<6x4xf64>
  return %g : tensor<6x4xf64>
}

// CHECK-LABEL: func.func @transposed_indices(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<64xf64>) -> tensor<6x4xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<64xf64>) -> tensor<8x8xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:4, 0:6] : (tensor<8x8xf64>) -> tensor<4x6xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V2]], dims = [1, 0] : (tensor<4x6xf64>) -> tensor<6x4xf64>
// CHECK-NEXT: return %[[V3]] : tensor<6x4xf64>
// CHECK-NEXT: }

// A strided slice of a wider index grid.
func.func @sliced_indices(%x: tensor<5x24xf64>, %u: tensor<5x3xf64>) -> tensor<5x24xf64> {
  %r = stablehlo.iota dim = 0 : tensor<5x24x1xi64>
  %c = stablehlo.iota dim = 1 : tensor<5x24x1xi64>
  %i0 = stablehlo.concatenate %r, %c, dim = 2 : (tensor<5x24x1xi64>, tensor<5x24x1xi64>) -> tensor<5x24x2xi64>
  %i = stablehlo.slice %i0 [0:5, 2:8:2, 0:2] : (tensor<5x24x2xi64>) -> tensor<5x3x2xi64>
  %s = "stablehlo.scatter"(%x, %i, %u) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<5x24xf64>, tensor<5x3x2xi64>, tensor<5x3xf64>) -> tensor<5x24xf64>
  return %s : tensor<5x24xf64>
}

// CHECK-LABEL: func.func @sliced_indices(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<5x24xf64>, %[[V1:[^ ,:)]+]]: tensor<5x3xf64>) -> tensor<5x24xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<5x24xf64>) -> tensor<5x12x2xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.transpose %[[V1]], dims = [0, 1] : (tensor<5x3xf64>) -> tensor<5x3xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.reshape %[[V5]] : (tensor<5x3xf64>) -> tensor<5x3x1xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V4]], %[[V6]], %[[V3]], %[[V2]], %[[V3]] : (tensor<5x12x2xf64>, tensor<5x3x1xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x12x2xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.reshape %[[V7]] : (tensor<5x12x2xf64>) -> tensor<5x24xf64>
// CHECK-NEXT: return %[[V8]] : tensor<5x24xf64>
// CHECK-NEXT: }
