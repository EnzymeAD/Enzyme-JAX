// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{passses=65535})" %s | FileCheck %s

// CHECK-LABEL: @pad_dot_general_lhs
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x3x1024x4xbf16>, %[[ARG1:.+]]: tensor<1x8x3x1024x2048xbf16>)
// CHECK:      %[[SLICE:.+]] = stablehlo.slice %[[ARG1]] [0:1, 0:8, 0:3, 0:1024, 1024:2048]
// CHECK-NOT:  pad
// CHECK:      stablehlo.dot_general %[[ARG0]], %[[SLICE]], batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [4], precision = [DEFAULT, DEFAULT]
func.func @pad_dot_general_lhs(%4 : tensor<1x3x1024x4xbf16>, %6: tensor<1x8x3x1024x2048xbf16>) -> tensor<1x3x4x8x1024xbf16> {
  %3 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
  %5 = stablehlo.pad %4, %3, low = [0, 0, 1024, 0], high = [0, 0, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x3x1024x4xbf16>, tensor<bf16>) -> tensor<1x3x2048x4xbf16>
  %7 = stablehlo.dot_general %5, %6, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x3x2048x4xbf16>, tensor<1x8x3x1024x2048xbf16>) -> tensor<1x3x4x8x1024xbf16>
  return %7 : tensor<1x3x4x8x1024xbf16>
}

  func.func @pad_dot_general_rhs(%arg0: tensor<1x8x3x1024x1024xbf16>, %arg1: tensor<1x3x2048x4xbf16>) -> tensor<1x3x4x8x1024xbf16> {
    %0 = stablehlo.slice %arg1 [0:1, 0:3, 1024:2048, 0:4] : (tensor<1x3x2048x4xbf16>) -> tensor<1x3x1024x4xbf16>
    %1 = stablehlo.dot_general %0, %arg0, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x3x1024x4xbf16>, tensor<1x8x3x1024x1024xbf16>) -> tensor<1x3x4x8x1024xbf16>
    return %1 : tensor<1x3x4x8x1024xbf16>
  }
// CHECK:  func.func @pad_dot_general_rhs(%arg0: tensor<1x8x3x1024x1024xbf16>, %arg1: tensor<1x3x2048x4xbf16>) -> tensor<1x3x4x8x1024xbf16> {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.slice %arg1 [0:1, 0:3, 1024:2048, 0:4] : (tensor<1x3x2048x4xbf16>) -> tensor<1x3x1024x4xbf16>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.dot_general %[[i0]], %arg0, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x3x1024x4xbf16>, tensor<1x8x3x1024x1024xbf16>) -> tensor<1x3x4x8x1024xbf16>
// CHECK-NEXT:    return %[[i1]] : tensor<1x3x4x8x1024xbf16>
// CHECK-NEXT:  }

  func.func @pad_dot_general3(%arg0: tensor<4x1024xbf16>, %arg1: tensor<1048x4xbf16>) -> tensor<1024x2048xbf16> {
	%143 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
	%1428 = stablehlo.pad %arg1, %143, low = [1000, 0], high = [0, 0], interior = [0, 0] : (tensor<1048x4xbf16>, tensor<bf16>) -> tensor<2048x4xbf16>
	%1523 = stablehlo.dot_general %arg0, %1428, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x1024xbf16>, tensor<2048x4xbf16>) -> tensor<1024x2048xbf16>
    return %1523 : tensor<1024x2048xbf16>
  }

// CHECK:  func.func @pad_dot_general3(%arg0: tensor<4x1024xbf16>, %arg1: tensor<1048x4xbf16>) -> tensor<1024x2048xbf16> {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x1024xbf16>, tensor<1048x4xbf16>) -> tensor<1024x1048xbf16>
// CHECK-NEXT:    %[[i2:.+]] = stablehlo.pad %[[i1]], %[[i0]], low = [0, 1000], high = [0, 0], interior = [0, 0] : (tensor<1024x1048xbf16>, tensor<bf16>) -> tensor<1024x2048xbf16>
// CHECK-NEXT:    return %[[i2]] : tensor<1024x2048xbf16>
// CHECK-NEXT:  }

  func.func @pad_dot_general4(%arg0: tensor<4x1024xbf16>, %arg1: tensor<1048x4xbf16>, %arg2: tensor<4x1024xbf16>) -> (tensor<1024x2048xbf16>, tensor<1024x2048xbf16>) {
	%143 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
	%1428 = stablehlo.pad %arg1, %143, low = [1000, 0], high = [0, 0], interior = [0, 0] : (tensor<1048x4xbf16>, tensor<bf16>) -> tensor<2048x4xbf16>
	%1523 = stablehlo.dot_general %arg0, %1428, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x1024xbf16>, tensor<2048x4xbf16>) -> tensor<1024x2048xbf16>
	%1524 = stablehlo.dot_general %arg2, %1428, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x1024xbf16>, tensor<2048x4xbf16>) -> tensor<1024x2048xbf16>
    return %1523, %1524 : tensor<1024x2048xbf16>, tensor<1024x2048xbf16>
  }

// CHECK:  func.func @pad_dot_general4(%arg0: tensor<4x1024xbf16>, %arg1: tensor<1048x4xbf16>, %arg2: tensor<4x1024xbf16>) -> (tensor<1024x2048xbf16>, tensor<1024x2048xbf16>) {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x1024xbf16>, tensor<1048x4xbf16>) -> tensor<1024x1048xbf16>
// CHECK-NEXT:    %[[i2:.+]] = stablehlo.pad %[[i1]], %[[i0]], low = [0, 1000], high = [0, 0], interior = [0, 0] : (tensor<1024x1048xbf16>, tensor<bf16>) -> tensor<1024x2048xbf16>
// CHECK-NEXT:    %[[i3:.+]] = stablehlo.dot_general %arg2, %arg1, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x1024xbf16>, tensor<1048x4xbf16>) -> tensor<1024x1048xbf16>
// CHECK-NEXT:    %[[i4:.+]] = stablehlo.pad %[[i3]], %[[i0]], low = [0, 1000], high = [0, 0], interior = [0, 0] : (tensor<1024x1048xbf16>, tensor<bf16>) -> tensor<1024x2048xbf16>
// CHECK-NEXT:    return %[[i2]], %[[i4]] : tensor<1024x2048xbf16>, tensor<1024x2048xbf16>
// CHECK-NEXT:  }
