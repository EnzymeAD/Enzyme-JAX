// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// CHECK-LABEL: @transpose
// CHECK-NOT: stablehlo.transpose
func.func @transpose(%1845: tensor<32x32768xf32>) -> tensor<32x32768xf32> {
  %1846 = stablehlo.transpose %1845, dims = [1, 0] : (tensor<32x32768xf32>) -> tensor<32768x32xf32>
  %1847 = stablehlo.transpose %1846, dims = [1, 0] : (tensor<32768x32xf32>) -> tensor<32x32768xf32>
  return %1847 : tensor<32x32768xf32>
}

// CHECK-LABEL: @transpose2
// CHECK:  stablehlo.transpose %{{.*}}, dims = [1, 0, 2] : (tensor<2x3x4xf32>) -> tensor<3x2x4xf32>
func.func @transpose2(%arg: tensor<2x3x4xf32>) -> tensor<3x2x4xf32> {
  %0 = stablehlo.transpose %arg, dims = [2, 0, 1] : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
  %1 = stablehlo.transpose %0, dims = [2, 1, 0] : (tensor<4x2x3xf32>) -> tensor<3x2x4xf32>
  return %1 : tensor<3x2x4xf32>
}

// CHECK:  func.func @transpose3(%arg0: tensor<2x3x4xf32>) -> (tensor<4x2x3xf32>, tensor<3x2x4xbf16>) {
// CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
// CHECK-NEXT:    %1 = stablehlo.convert %arg0 : (tensor<2x3x4xf32>) -> tensor<2x3x4xbf16>
// CHECK-NEXT:    %2 = stablehlo.transpose %1, dims = [1, 0, 2] : (tensor<2x3x4xbf16>) -> tensor<3x2x4xbf16>
// CHECK-NEXT:    return %0, %2 : tensor<4x2x3xf32>, tensor<3x2x4xbf16>
// CHECK-NEXT:  }
func.func @transpose3(%arg: tensor<2x3x4xf32>) -> (tensor<4x2x3xf32>, tensor<3x2x4xbf16>) {
  %0 = stablehlo.transpose %arg, dims = [2, 0, 1] : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
  %conv = stablehlo.convert %0 : (tensor<4x2x3xf32>) -> tensor<4x2x3xbf16>
  %1 = stablehlo.transpose %conv, dims = [2, 1, 0] : (tensor<4x2x3xbf16>) -> tensor<3x2x4xbf16>
  return %0, %1 : tensor<4x2x3xf32>, tensor<3x2x4xbf16>
}
