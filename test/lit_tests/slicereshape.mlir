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

// CHECK-LABEL: @reshape_slice2
// CHECK-SAME: %[[ARG0:.+]]: tensor<
// CHECK: %[[S1:.+]] = stablehlo.slice %arg0 [0:1, 2:3, 1024:2048, 0:4]
// CHECK: stablehlo.reshape %[[S1]] : (tensor<1x1x1024x4xbf16>) -> tensor<1x1x1024x1x4xbf16>
func.func @reshape_slice2(%7: tensor<1x3x2048x4xbf16>) -> (tensor<1x1x1024x1x4xbf16>) {
  %8 = stablehlo.reshape %7 : (tensor<1x3x2048x4xbf16>) -> tensor<1x3x2048x1x4xbf16>
  %10 = stablehlo.slice %8 [0:1, 2:3, 1024:2048, 0:1, 0:4] : (tensor<1x3x2048x1x4xbf16>) -> tensor<1x1x1024x1x4xbf16>
  return %10 : tensor<1x1x1024x1x4xbf16>
}

func.func @reshape_slice3(%1250: tensor<1x1x8192x1x256xbf16>) -> (tensor<1x3072x1x256xbf16>) {
  %1251 = stablehlo.reshape %1250 : (tensor<1x1x8192x1x256xbf16>) -> tensor<1x8192x1x256xbf16> 
  %1252 = stablehlo.slice %1251 [0:1, 0:3072, 0:1, 0:256] : (tensor<1x8192x1x256xbf16>) -> tensor<1x3072x1x256xbf16>
  return %1252 : tensor<1x3072x1x256xbf16>
}

// CHECK:  func.func @reshape_slice3(%arg0: tensor<1x1x8192x1x256xbf16>) -> tensor<1x3072x1x256xbf16> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:1, 0:3072, 0:1, 0:256] : (tensor<1x1x8192x1x256xbf16>) -> tensor<1x1x3072x1x256xbf16>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<1x1x3072x1x256xbf16>) -> tensor<1x3072x1x256xbf16>
// CHECK-NEXT:    return %1 : tensor<1x3072x1x256xbf16>
// CHECK-NEXT:  }
