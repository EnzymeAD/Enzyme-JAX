// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=reduce_mul_to_dot_general},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

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
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %arg0, dims = [2, 0] : (tensor<128x32xf32>) -> tensor<32x256x128xf32>
// CHECK-NEXT:   %1 = stablehlo.broadcast_in_dim %arg1, dims = [2, 0] : (tensor<128x32xf32>) -> tensor<32x256x128xf32>
// CHECK-NEXT:   %2 = stablehlo.dot_general %0, %1, batching_dims = [1, 2] x [1, 2], contracting_dims = [0] x [0] : (tensor<32x256x128xf32>, tensor<32x256x128xf32>) -> tensor<256x128xf32>
// CHECK-NEXT:   return %2 : tensor<256x128xf32>
// CHECK-NEXT: }
