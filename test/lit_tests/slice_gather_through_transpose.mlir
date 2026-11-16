// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=slice_gather" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// The BK6 form: a gather over padded lanes, read only in its live box, once
// through a transpose.
func.func @through_transpose(%x: tensor<64xf64>, %i: tensor<8x8x1xi64>) -> (tensor<3x3xf64>, tensor<3x2xf64>, tensor<2x3xf64>) {
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1>}> : (tensor<64xf64>, tensor<8x8x1xi64>) -> tensor<8x8xf64>
  %a = stablehlo.slice %g [0:3, 0:3] : (tensor<8x8xf64>) -> tensor<3x3xf64>
  %b = stablehlo.slice %g [0:3, 0:2] : (tensor<8x8xf64>) -> tensor<3x2xf64>
  %t = stablehlo.transpose %g, dims = [1, 0] : (tensor<8x8xf64>) -> tensor<8x8xf64>
  %c = stablehlo.slice %t [1:3, 0:3] : (tensor<8x8xf64>) -> tensor<2x3xf64>
  return %a, %b, %c : tensor<3x3xf64>, tensor<3x2xf64>, tensor<2x3xf64>
}

// CHECK-LABEL: func.func @through_transpose(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<64xf64>, %[[V1:[^ ,:)]+]]: tensor<8x8x1xi64>) -> (tensor<3x3xf64>, tensor<3x2xf64>, tensor<2x3xf64>) {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:3, 0:3, 0:1] : (tensor<8x8x1xi64>) -> tensor<3x3x1xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V2]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1>}> : (tensor<64xf64>, tensor<3x3x1xi64>) -> tensor<3x3xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.slice %[[V3]] [0:3, 0:2] : (tensor<3x3xf64>) -> tensor<3x2xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.transpose %[[V3]], dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.slice %[[V5]] [1:3, 0:3] : (tensor<3x3xf64>) -> tensor<2x3xf64>
// CHECK-NEXT: return %[[V3]], %[[V4]], %[[V6]] : tensor<3x3xf64>, tensor<3x2xf64>, tensor<2x3xf64>
// CHECK-NEXT: }

// Stays: the transpose has a user that is not a slice.
func.func @transpose_used_whole(%x: tensor<64xf64>, %i: tensor<8x8x1xi64>) -> (tensor<3x3xf64>, tensor<8x8xf64>) {
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1>}> : (tensor<64xf64>, tensor<8x8x1xi64>) -> tensor<8x8xf64>
  %a = stablehlo.slice %g [0:3, 0:3] : (tensor<8x8xf64>) -> tensor<3x3xf64>
  %t = stablehlo.transpose %g, dims = [1, 0] : (tensor<8x8xf64>) -> tensor<8x8xf64>
  return %a, %t : tensor<3x3xf64>, tensor<8x8xf64>
}

// CHECK-LABEL: func.func @transpose_used_whole(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<64xf64>, %[[V1:[^ ,:)]+]]: tensor<8x8x1xi64>) -> (tensor<3x3xf64>, tensor<8x8xf64>) {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V1]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1>}> : (tensor<64xf64>, tensor<8x8x1xi64>) -> tensor<8x8xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.slice %[[V2]] [0:3, 0:3] : (tensor<8x8xf64>) -> tensor<3x3xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.transpose %[[V2]], dims = [1, 0] : (tensor<8x8xf64>) -> tensor<8x8xf64>
// CHECK-NEXT: return %[[V3]], %[[V4]] : tensor<3x3xf64>, tensor<8x8xf64>
// CHECK-NEXT: }

// Slices and a batching pair: the box slices the operand along the pair.
func.func @batching(%x: tensor<6x16xf64>, %i: tensor<6x8x1xi64>) -> (tensor<2x3xf64>, tensor<2x2xf64>) {
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<6x16xf64>, tensor<6x8x1xi64>) -> tensor<6x8xf64>
  %a = stablehlo.slice %g [1:3, 0:6:2] : (tensor<6x8xf64>) -> tensor<2x3xf64>
  %b = stablehlo.slice %g [2:4, 1:3] : (tensor<6x8xf64>) -> tensor<2x2xf64>
  return %a, %b : tensor<2x3xf64>, tensor<2x2xf64>
}

// CHECK-LABEL: func.func @batching(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<6x16xf64>, %[[V1:[^ ,:)]+]]: tensor<6x8x1xi64>) -> (tensor<2x3xf64>, tensor<2x2xf64>) {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V1]] [1:4, 0:5, 0:1] : (tensor<6x8x1xi64>) -> tensor<3x5x1xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.slice %[[V0]] [1:4, 0:16] : (tensor<6x16xf64>) -> tensor<3x16xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = "stablehlo.gather"(%[[V3]], %[[V2]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<3x16xf64>, tensor<3x5x1xi64>) -> tensor<3x5xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.slice %[[V4]] [0:2, 0:5:2] : (tensor<3x5xf64>) -> tensor<2x3xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.slice %[[V4]] [1:3, 1:3] : (tensor<3x5xf64>) -> tensor<2x2xf64>
// CHECK-NEXT: return %[[V5]], %[[V6]] : tensor<2x3xf64>, tensor<2x2xf64>
// CHECK-NEXT: }

// Stays: the slices together cover the whole gather.
func.func @covers_whole(%x: tensor<64xf64>, %i: tensor<8x1xi64>) -> (tensor<4xf64>, tensor<4xf64>) {
  %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<64xf64>, tensor<8x1xi64>) -> tensor<8xf64>
  %a = stablehlo.slice %g [0:4] : (tensor<8xf64>) -> tensor<4xf64>
  %b = stablehlo.slice %g [4:8] : (tensor<8xf64>) -> tensor<4xf64>
  return %a, %b : tensor<4xf64>, tensor<4xf64>
}

// CHECK-LABEL: func.func @covers_whole(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<64xf64>, %[[V1:[^ ,:)]+]]: tensor<8x1xi64>) -> (tensor<4xf64>, tensor<4xf64>) {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V1]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<64xf64>, tensor<8x1xi64>) -> tensor<8xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.slice %[[V2]] [0:4] : (tensor<8xf64>) -> tensor<4xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.slice %[[V2]] [4:8] : (tensor<8xf64>) -> tensor<4xf64>
// CHECK-NEXT: return %[[V3]], %[[V4]] : tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT: }
