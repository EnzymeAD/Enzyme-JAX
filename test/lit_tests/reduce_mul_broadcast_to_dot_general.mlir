// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%arg0: tensor<2x3xf64>, %arg1: tensor<100x2xf64>) -> tensor<100x3xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2] : (tensor<2x3xf64>) -> tensor<2x100x3xf64>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1, 0] : (tensor<100x2xf64>) -> tensor<2x100x3xf64>
    %2 = stablehlo.multiply %0, %1 : tensor<2x100x3xf64>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2x100x3xf64>, tensor<f64>) -> tensor<100x3xf64>
    return %3 : tensor<100x3xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<2x3xf64>, %arg1: tensor<100x2xf64>) -> tensor<100x3xf64> {
// CHECK-NEXT:    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [0] : (tensor<100x2xf64>, tensor<2x3xf64>) -> tensor<100x3xf64>
// CHECK-NEXT:    return %0 : tensor<100x3xf64>
// CHECK-NEXT:  }

module {
  func.func @main(%arg0: tensor<128x32xf32>, %arg1: tensor<128x32xf32>) -> tensor<256x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [2, 0] : (tensor<128x32xf32>) -> tensor<32x256x128xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2, 0] : (tensor<128x32xf32>) -> tensor<32x256x128xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<32x256x128xf32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<32x256x128xf32>, tensor<f32>) -> tensor<256x128xf32>
    return %3 : tensor<256x128xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<128x32xf32>, %arg1: tensor<128x32xf32>) -> tensor<256x128xf32> {
// CHECK-NEXT:   %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<128x32xf32>, tensor<128x32xf32>) -> tensor<128xf32>
// CHECK-NEXT:   %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<128xf32>) -> tensor<256x128xf32>
// CHECK-NEXT:   return %1 : tensor<256x128xf32>
// CHECK-NEXT: }
