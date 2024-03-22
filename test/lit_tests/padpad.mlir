// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// CHECK-LABEL: @pad_pad
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x1x1xf32>, %[[ARG1:.+]]: tensor<f32>
// CHECK: stablehlo.pad %[[ARG0]], %[[ARG1]], low = [6, 0, 1], high = [1, 0, 8], interior = [0, 0, 0]
// CHECK-NOT: pad
func.func @pad_pad(%arg0: tensor<1x1x1xf32>, %arg1: tensor<f32>) -> tensor<8x1x10xf32> {
  %0 = stablehlo.pad %arg0, %arg1, low = [2, 0, 0], high = [0, 0, 3], interior = [0, 0, 0] : (tensor<1x1x1xf32>, tensor<f32>) -> tensor<3x1x4xf32>
  %1 = stablehlo.pad %0, %arg1, low = [4, 0, 1], high = [1, 0, 5], interior = [0, 0, 0] : (tensor<3x1x4xf32>, tensor<f32>) -> tensor<8x1x10xf32>
  return %1 : tensor<8x1x10xf32>
}

// CHECK-LABEL: @pad_pad_interior2
// CHECK-COUNT-2: pad
func.func @pad_pad_interior2(%arg0: tensor<1x1x1xf32>, %arg1: tensor<f32>) -> tensor<10x1x10xf32> {
  %0 = stablehlo.pad %arg0, %arg1, low = [2, 0, 0], high = [0, 0, 3], interior = [0, 1, 0] : (tensor<1x1x1xf32>, tensor<f32>) -> tensor<3x1x4xf32>
  %1 = stablehlo.pad %0, %arg1, low = [4, 0, 1], high = [1, 0, 5], interior = [1, 0, 0] : (tensor<3x1x4xf32>, tensor<f32>) -> tensor<10x1x10xf32>
  return %1 : tensor<10x1x10xf32>
}
