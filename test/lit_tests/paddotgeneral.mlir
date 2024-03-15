// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// CHECK-LABEL: @pad_dot_general
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x3x1024x4xbf16>, %[[ARG1:.+]]: tensor<1x8x3x1024x2048xbf16>)
// CHECK:      %[[SLICE:.+]] = stablehlo.slice %[[ARG1]] [0:1, 0:8, 0:3, 0:1024, 1024:2048]
// CHECK-NOT:  pad
// CHECK:      stablehlo.dot_general %[[ARG0]], %[[SLICE]], batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [4], precision = [DEFAULT, DEFAULT]
func.func @pad_dot_general(%4 : tensor<1x3x1024x4xbf16>, %6: tensor<1x8x3x1024x2048xbf16>) -> tensor<1x3x4x8x1024xbf16> {
  %3 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
  %5 = stablehlo.pad %4, %3, low = [0, 0, 1024, 0], high = [0, 0, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x3x1024x4xbf16>, tensor<bf16>) -> tensor<1x3x2048x4xbf16>
  %7 = stablehlo.dot_general %5, %6, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x3x2048x4xbf16>, tensor<1x8x3x1024x2048xbf16>) -> tensor<1x3x4x8x1024xbf16>
  return %7 : tensor<1x3x4x8x1024xbf16>
}

