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

// Anisotropic / 1D-in-2D transposed convolution filter gradient with a unit
// spatial dimension: swaps operands and only reverses axes with size > 1.
func.func @filter_grad_anisotropic(%x: tensor<1x4x3x2xf32>, %dy: tensor<1x8x5x2xf32>) -> tensor<3x5x2x1xf32> {
  %0 = stablehlo.convolution(%x, %dy) dim_numbers = [0, 1, b, f]x[0, 1, o, i]->[b, f, 1, 0], window = {stride = [1, 1], pad = [[0, 0], [1, 1]], lhs_dilate = [1, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x4x3x2xf32>, tensor<1x8x5x2xf32>) -> tensor<3x5x2x1xf32>
  return %0 : tensor<3x5x2x1xf32>
}

// CHECK-LABEL: func.func @filter_grad_anisotropic
// CHECK-NEXT{LITERAL}: %0 = stablehlo.convolution(%arg1, %arg0) dim_numbers = [0, 1, b, f]x[0, 1, o, i]->[f, b, 1, 0], window = {pad = [[0, 0], [0, 0]], rhs_dilate = [1, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x8x5x2xf32>, tensor<1x4x3x2xf32>) -> tensor<3x5x2x1xf32>
// CHECK-NEXT: %[[R:.+]] = stablehlo.reverse %0, dims = [2] : tensor<3x5x2x1xf32>
// CHECK-NEXT: return %[[R]]

// Cropped transposed convolution filter gradient (negative input padding turns
// into non-negative padding after swapping).
func.func @filter_grad_cropped(%x: tensor<4x4x3x2xf32>, %dy: tensor<4x4x5x2xf32>) -> tensor<3x5x2x2xf32> {
  %0 = stablehlo.convolution(%x, %dy) dim_numbers = [0, 1, b, f]x[0, 1, o, i]->[b, f, 1, 0], window = {stride = [1, 1], pad = [[-1, -1], [-1, -1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<4x4x3x2xf32>, tensor<4x4x5x2xf32>) -> tensor<3x5x2x2xf32>
  return %0 : tensor<3x5x2x2xf32>
}

// CHECK-LABEL: func.func @filter_grad_cropped
// CHECK-NEXT{LITERAL}: %0 = stablehlo.convolution(%arg1, %arg0) dim_numbers = [0, 1, b, f]x[0, 1, o, i]->[f, b, 1, 0], window = {pad = [[2, 2], [2, 2]], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<4x4x5x2xf32>, tensor<4x4x3x2xf32>) -> tensor<3x5x2x2xf32>
// CHECK-NEXT: %[[R:.+]] = stablehlo.reverse %0, dims = [3, 2] : tensor<3x5x2x2xf32>
// CHECK-NEXT: return %[[R]]

// 1x1 filter gradient omits the no-op reverse.
func.func @filter_grad_1x1(%x: tensor<4x4x3x2xf32>, %dy: tensor<7x7x5x2xf32>) -> tensor<3x5x1x1xf32> {
  %0 = stablehlo.convolution(%x, %dy) dim_numbers = [0, 1, b, f]x[0, 1, o, i]->[b, f, 1, 0], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<4x4x3x2xf32>, tensor<7x7x5x2xf32>) -> tensor<3x5x1x1xf32>
  return %0 : tensor<3x5x1x1xf32>
}

// CHECK-LABEL: func.func @filter_grad_1x1
// CHECK-NEXT{LITERAL}: %0 = stablehlo.convolution(%arg1, %arg0) dim_numbers = [0, 1, b, f]x[0, 1, o, i]->[f, b, 1, 0], window = {pad = [[0, 0], [0, 0]], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<7x7x5x2xf32>, tensor<4x4x3x2xf32>) -> tensor<3x5x1x1xf32>
// CHECK-NEXT: return %0

// The transposed convolution itself has a short window: unchanged.
func.func @conv_transpose(%x: tensor<4x4x3x2xf32>, %w: tensor<2x2x3x5xf32>) -> tensor<8x8x5x2xf32> {
  %0 = stablehlo.convolution(%x, %w) dim_numbers = [0, 1, f, b]x[0, 1, i, o]->[0, 1, f, b], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<4x4x3x2xf32>, tensor<2x2x3x5xf32>) -> tensor<8x8x5x2xf32>
  return %0 : tensor<8x8x5x2xf32>
}

// CHECK-LABEL: func.func @conv_transpose
// CHECK-NEXT: stablehlo.convolution(%arg0, %arg1)
// CHECK-SAME: lhs_dilate = [2, 2]
// CHECK-NOT: stablehlo.reverse

// Backward input of a strided convolution (rhs is spatially reversed): unchanged.
func.func @backward_input_reversed_rhs(%dy: tensor<2x2x5x2xf32>, %w: tensor<5x5x5x3xf32>) -> tensor<3x3x3x2xf32> {
  %rev = stablehlo.reverse %w, dims = [0, 1] : tensor<5x5x5x3xf32>
  %0 = stablehlo.convolution(%dy, %rev) dim_numbers = [0, 1, f, b]x[0, 1, i, o]->[0, 1, f, b], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2x2x5x2xf32>, tensor<5x5x5x3xf32>) -> tensor<3x3x3x2xf32>
  return %0 : tensor<3x3x3x2xf32>
}

// CHECK-LABEL: func.func @backward_input_reversed_rhs
// CHECK-NEXT: %[[REV:.+]] = stablehlo.reverse %arg1, dims = [0, 1]
// CHECK-NEXT: %[[CONV:.+]] = stablehlo.convolution(%arg0, %[[REV]])
// CHECK-SAME: lhs_dilate = [2, 2]
// CHECK-NEXT: return %[[CONV]]
