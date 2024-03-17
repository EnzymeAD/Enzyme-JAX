// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{max_constant_expansion=1})" %s | FileCheck %s

module {
  func.func @main(%1826 : tensor<1x8x3x1024x2048xbf16>, %326: tensor<1x3x1024x8x4xbf16>) -> tensor<1x3x2048x4xbf16> {
    %1827 = stablehlo.transpose %1826, dims = [0, 2, 4, 3, 1] : (tensor<1x8x3x1024x2048xbf16>) -> tensor<1x3x2048x1024x8xbf16>
    %1847 = stablehlo.dot_general %1827, %326, batching_dims = [0, 1] x [0, 1], contracting_dims = [3, 4] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<1x3x2048x1024x8xbf16>, tensor<1x3x1024x8x4xbf16>) -> tensor<1x3x2048x4xbf16>
    return %1847 : tensor<1x3x2048x4xbf16>
  }
  func.func @main2(%1826 : tensor<1x8x3x1024x2048xbf16>, %326: tensor<1x3x1024x8x4xbf16>) -> tensor<1x3x4x2048xbf16> {
    %1827 = stablehlo.transpose %1826, dims = [0, 2, 4, 3, 1] : (tensor<1x8x3x1024x2048xbf16>) -> tensor<1x3x2048x1024x8xbf16>
    %1847 = stablehlo.dot_general %326, %1827, batching_dims = [0, 1] x [0, 1], contracting_dims = [2, 3] x [3, 4], precision = [DEFAULT, DEFAULT] : (tensor<1x3x1024x8x4xbf16>, tensor<1x3x2048x1024x8xbf16>) -> tensor<1x3x4x2048xbf16>
    return %1847 : tensor<1x3x4x2048xbf16>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x8x3x1024x2048xbf16>, %arg1: tensor<1x3x1024x8x4xbf16>) -> tensor<1x3x2048x4xbf16> {
// CHECK-NEXT:    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 2] x [0, 1], contracting_dims = [3, 1] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3x1024x2048xbf16>, tensor<1x3x1024x8x4xbf16>) -> tensor<1x3x2048x4xbf16>
// CHECK-NEXT:    return %0 : tensor<1x3x2048x4xbf16>
// CHECK-NEXT:  }
// CHECK:  func.func @main2(%arg0: tensor<1x8x3x1024x2048xbf16>, %arg1: tensor<1x3x1024x8x4xbf16>) -> tensor<1x3x4x2048xbf16> {
// CHECK-NEXT:    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0, 1] x [0, 2], contracting_dims = [2, 3] x [3, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x3x1024x8x4xbf16>, tensor<1x8x3x1024x2048xbf16>) -> tensor<1x3x4x2048xbf16>
// CHECK-NEXT:    return %0 : tensor<1x3x4x2048xbf16>
// CHECK-NEXT:  }
