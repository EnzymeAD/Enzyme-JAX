// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// CHECK-LABEL: @reshape_slice
// CHECK-SAME: %[[ARG0:.+]]: tensor<
// CHECK: %[[S0:.+]] = stablehlo.slice %[[ARG0]] [0:1, 1:3, 0:1024, 0:4]
// CHECK: stablehlo.reshape %[[S0]] : (tensor<1x2x1024x4xbf16>) -> tensor<1x2x1024x1x4xbf16>
func.func @reshape_slice(%7: tensor<1x3x2048x4xbf16>) -> (tensor<1x2x1024x1x4xbf16>) {
  %8 = stablehlo.reshape %7 : (tensor<1x3x2048x4xbf16>) -> tensor<1x3x2048x1x4xbf16>
  %9 = stablehlo.slice %8 [0:1, 1:3, 0:1024, 0:1, 0:4] : (tensor<1x3x2048x1x4xbf16>) -> tensor<1x2x1024x1x4xbf16>
  return %9 : tensor<1x2x1024x1x4xbf16>
}

// CHECK-LABEL: @reshape_slice
// CHECK-SAME: %[[ARG0:.+]]: tensor<
// CHECK: %[[S1:.+]] = stablehlo.slice %arg0 [0:1, 2:3, 1024:2048, 0:4]
// CHECK: stablehlo.reshape %[[S1]] : (tensor<1x1x1024x4xbf16>) -> tensor<1x1x1024x1x4xbf16>
func.func @reshape_slice2(%7: tensor<1x3x2048x4xbf16>) -> (tensor<1x1x1024x1x4xbf16>) {
  %8 = stablehlo.reshape %7 : (tensor<1x3x2048x4xbf16>) -> tensor<1x3x2048x1x4xbf16>
  %10 = stablehlo.slice %8 [0:1, 2:3, 1024:2048, 0:1, 0:4] : (tensor<1x3x2048x1x4xbf16>) -> tensor<1x1x1024x1x4xbf16>
  return %10 : tensor<1x1x1024x1x4xbf16>
}
