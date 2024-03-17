// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{max_constant_expansion=1})" %s | FileCheck %s

module {
  func.func @main(%1844 : tensor<1x3x4x8x1024xbf16>, %1525: tensor<1x8x3x1024x2048xbf16>) -> tensor<1x3x2048x4xbf16> {
    %1909 = stablehlo.dot_general %1844, %1525, batching_dims = [0, 1] x [0, 2], contracting_dims = [3, 4] x [1, 3], precision = [DEFAULT, DEFAULT] : (tensor<1x3x4x8x1024xbf16>, tensor<1x8x3x1024x2048xbf16>) -> tensor<1x3x4x2048xbf16>
    %1910 = stablehlo.transpose %1909, dims = [0, 1, 3, 2] : (tensor<1x3x4x2048xbf16>) -> tensor<1x3x2048x4xbf16>
    return %1910 : tensor<1x3x2048x4xbf16>
  }
  func.func @main2(%1844 : tensor<1x3x4x8x1024xbf16>, %1525: tensor<1x8x3x1024x2048xbf16>) -> tensor<1x3x2048x4xf32> {
    %1909 = stablehlo.dot_general %1844, %1525, batching_dims = [0, 1] x [0, 2], contracting_dims = [3, 4] x [1, 3], precision = [DEFAULT, DEFAULT] : (tensor<1x3x4x8x1024xbf16>, tensor<1x8x3x1024x2048xbf16>) -> tensor<1x3x4x2048xbf16>
    %c = stablehlo.convert %1909 : (tensor<1x3x4x2048xbf16>) -> tensor<1x3x4x2048xf32>
    %1910 = stablehlo.transpose %c, dims = [0, 1, 3, 2] : (tensor<1x3x4x2048xf32>) -> tensor<1x3x2048x4xf32>
    return %1910 : tensor<1x3x2048x4xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x3x4x8x1024xbf16>, %arg1: tensor<1x8x3x1024x2048xbf16>) -> tensor<1x3x2048x4xbf16> {
// CHECK-NEXT:    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0, 2] x [0, 1], contracting_dims = [1, 3] x [3, 4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3x1024x2048xbf16>, tensor<1x3x4x8x1024xbf16>) -> tensor<1x3x2048x4xbf16>
// CHECK-NEXT:    return %0 : tensor<1x3x2048x4xbf16>
// CHECK-NEXT:  }

// CHECK:  func.func @main2(%arg0: tensor<1x3x4x8x1024xbf16>, %arg1: tensor<1x8x3x1024x2048xbf16>) -> tensor<1x3x2048x4xf32> {
// CHECK-NEXT:    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0, 2] x [0, 1], contracting_dims = [1, 3] x [3, 4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3x1024x2048xbf16>, tensor<1x3x4x8x1024xbf16>) -> tensor<1x3x2048x4xbf16>
// CHECK-NEXT:    %1 = stablehlo.convert %0 : (tensor<1x3x2048x4xbf16>) -> tensor<1x3x2048x4xf32>
// CHECK-NEXT:    return %1 : tensor<1x3x2048x4xf32>
// CHECK-NEXT:  }
