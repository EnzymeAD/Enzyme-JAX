// RUN: enzymexlamlir-opt --pass-pipeline="any(enzyme-hlo-generate-td{patterns=convolution_lhs_dilation_swap},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

// The filter gradient of a 2x2 stride 2 transposed convolution, as the
// convolution derivative emits it: the input of the transposed convolution,
// base dilated, against the output gradient as the window.
func.func @filter_grad(%x: tensor<4x4x3x2xf32>, %dy: tensor<8x8x5x2xf32>) -> tensor<3x5x2x2xf32> {
  %0 = stablehlo.convolution(%x, %dy) dim_numbers = [0, 1, b, f]x[0, 1, o, i]->[b, f, 1, 0], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<4x4x3x2xf32>, tensor<8x8x5x2xf32>) -> tensor<3x5x2x2xf32>
  return %0 : tensor<3x5x2x2xf32>
}

// CHECK-LABEL: func.func @filter_grad
// CHECK-SAME: (%[[X:.+]]: tensor<4x4x3x2xf32>, %[[DY:.+]]: tensor<8x8x5x2xf32>)
// CHECK-NEXT{LITERAL}: %0 = stablehlo.convolution(%arg1, %arg0) dim_numbers = [0, 1, b, f]x[0, 1, o, i]->[f, b, 1, 0], window = {pad = [[0, 0], [0, 0]], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<8x8x5x2xf32>, tensor<4x4x3x2xf32>) -> tensor<3x5x2x2xf32>
// CHECK-NEXT: %[[R:.+]] = stablehlo.reverse %0, dims = [3, 2] : tensor<3x5x2x2xf32>
// CHECK-NEXT: return %[[R]]

// The transposed convolution itself has a short window: unchanged.
func.func @conv_transpose(%x: tensor<4x4x3x2xf32>, %w: tensor<2x2x3x5xf32>) -> tensor<8x8x5x2xf32> {
  %0 = stablehlo.convolution(%x, %w) dim_numbers = [0, 1, f, b]x[0, 1, i, o]->[0, 1, f, b], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<4x4x3x2xf32>, tensor<2x2x3x5xf32>) -> tensor<8x8x5x2xf32>
  return %0 : tensor<8x8x5x2xf32>
}

// CHECK-LABEL: func.func @conv_transpose
// CHECK-NEXT: stablehlo.convolution(%arg0, %arg1)
// CHECK-SAME: lhs_dilate = [2, 2]
// CHECK-NOT: stablehlo.reverse
