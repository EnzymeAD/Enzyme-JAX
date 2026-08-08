// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// ------------------------------------------------------------------------
// Case 1: float gather whose second affine iota is hidden under
//         broadcast_in_dim(multiply(iota, stride)).
//
// indices[i, j] = 8 + i + 8*j
//
// This is a regular 2-D tile.  It can be implemented as:
//   reshape 64 -> 8x8
//   slice [1:4, 0:2] -> 3x2
//   transpose -> 2x3
//
// ------------------------------------------------------------------------
module {
  func.func @gather_broadcast_scaled_iota(%input: tensor<64xf32>) -> tensor<2x3xf32> {
      %stride = stablehlo.constant dense<8> : tensor<3xi64>
      %offset = stablehlo.constant dense<8> : tensor<2x3x1xi64>

      %i = stablehlo.iota dim = 0 : tensor<2x3x1xi64>
      %j = stablehlo.iota dim = 0 : tensor<3xi64>
      %scaled_j = stablehlo.multiply %j, %stride : tensor<3xi64>
      %broadcast_j = stablehlo.broadcast_in_dim %scaled_j, dims = [1]
          : (tensor<3xi64>) -> tensor<2x3x1xi64>
      %grid = stablehlo.add %i, %broadcast_j : tensor<2x3x1xi64>
      %indices = stablehlo.add %grid, %offset : tensor<2x3x1xi64>

      %result = "stablehlo.gather"(%input, %indices) <{
          dimension_numbers = #stablehlo.gather<
          collapsed_slice_dims = [0],
          start_index_map = [0],
          index_vector_dim = 2>,
          indices_are_sorted = false,
          slice_sizes = array<i64: 1>
      }> : (tensor<64xf32>, tensor<2x3x1xi64>) -> tensor<2x3xf32>
      return %result : tensor<2x3xf32>
  }

// CHECK-LABEL: @gather_broadcast_scaled_iota
// CHECK-SAME:     %[[VAL_0:.*]]: tensor<64xf32>
// CHECK-NEXT:     %[[VAL_1:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<64xf32>) -> tensor<8x8xf32>
// CHECK-NEXT:     %[[VAL_2:.*]] = stablehlo.slice %[[VAL_1]] [1:4, 0:2] : (tensor<8x8xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:     %[[VAL_3:.*]] = stablehlo.transpose %[[VAL_2]], dims = [1, 0] : (tensor<3x2xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:     return %[[VAL_3]] : tensor<2x3xf32>

  // ------------------------------------------------------------------------
  // Case 2: byte/flag gather whose first affine iota is hidden under a
  //         reshape(multiply(iota, byte_stride)).
  //
  // indices[i, j] = 16 + 4*i + 32*j
  //
  // This models the LBM flag load after the f32 buffer is bitcast to bytes.
  // Currently the gather remains because detectIotaLikeTensor does not look
  // through the reshape.
  // ------------------------------------------------------------------------
  func.func @gather_reshaped_scaled_iota(
      %bytes: tensor<256xi8>) -> tensor<2x3xi8> {
    %inner_stride = stablehlo.constant dense<4> : tensor<2x3xi64>
    %row_stride = stablehlo.constant dense<32> : tensor<2x3x1xi64>
    %offset = stablehlo.constant dense<16> : tensor<2x3x1xi64>

    %i = stablehlo.iota dim = 0 : tensor<2x3xi64>
    %scaled_i = stablehlo.multiply %i, %inner_stride : tensor<2x3xi64>
    %reshaped_i = stablehlo.reshape %scaled_i
      : (tensor<2x3xi64>) -> tensor<2x3x1xi64>

    %j = stablehlo.iota dim = 1 : tensor<2x3x1xi64>
    %scaled_j = stablehlo.multiply %j, %row_stride : tensor<2x3x1xi64>
    %grid = stablehlo.add %reshaped_i, %scaled_j : tensor<2x3x1xi64>
    %indices = stablehlo.add %grid, %offset : tensor<2x3x1xi64>

    %result = "stablehlo.gather"(%bytes, %indices) <{
      dimension_numbers = #stablehlo.gather<
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1>
    }> : (tensor<256xi8>, tensor<2x3x1xi64>) -> tensor<2x3xi8>
    return %result : tensor<2x3xi8>
  }

// CHECK-LABEL: @gather_reshaped_scaled_iota
// CHECK-SAME:     %[[VAL_0:.*]]: tensor<256xi8>
// CHECK-NEXT:     %[[VAL_1:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<256xi8>) -> tensor<8x8x4xi8>
// CHECK-NEXT:     %[[VAL_2:.*]] = stablehlo.slice %[[VAL_1]] [0:3, 4:6, 0:1] : (tensor<8x8x4xi8>) -> tensor<3x2x1xi8>
// CHECK-NEXT:     %[[VAL_3:.*]] = stablehlo.reshape %[[VAL_2]] : (tensor<3x2x1xi8>) -> tensor<3x2xi8>
// CHECK-NEXT:     %[[VAL_4:.*]] = stablehlo.transpose %[[VAL_3]], dims = [1, 0] : (tensor<3x2xi8>) -> tensor<2x3xi8>
// CHECK-NEXT:     return %[[VAL_4]] : tensor<2x3xi8>
}