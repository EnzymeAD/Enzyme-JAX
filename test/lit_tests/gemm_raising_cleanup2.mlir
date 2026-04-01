// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=reshape_elementwise_only_fusible(1)},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<32x64xf32>, %arg3: tensor<64x32xf32>) -> tensor<64x64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<32x64xf32>) -> tensor<64x32xf32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64x32xf32>
    %2 = stablehlo.multiply %1, %0 : tensor<64x32xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x32xf32>) -> tensor<64x32x64x1xf32>
    %4 = stablehlo.broadcast_in_dim %arg3, dims = [2, 1] : (tensor<64x32xf32>) -> tensor<64x32x64x1xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x32x64x1xf32>
    %6 = stablehlo.multiply %cst, %arg1 : tensor<f32>
    %7 = stablehlo.reshape %5 : (tensor<64x32x64x1xf32>) -> tensor<64x1x32x64x1xf32>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [2, 1] : (tensor<64x1x32x64x1xf32>, tensor<f32>) -> tensor<64x64x1xf32>
    %9 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<64x64x1xf32>
    %10 = stablehlo.add %8, %9 : tensor<64x64x1xf32>
    %11 = stablehlo.reshape %10 : (tensor<64x64x1xf32>) -> tensor<64x64xf32>
    %12 = stablehlo.transpose %11, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    return %12 : tensor<64x64xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<32x64xf32>, %arg3: tensor<64x32xf32>) -> tensor<64x64xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %0 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<32x64xf32>) -> tensor<64x32xf32>
// CHECK-NEXT:   %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64x32xf32>
// CHECK-NEXT:   %2 = stablehlo.multiply %1, %0 : tensor<64x32xf32>
// CHECK-NEXT:   %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x32xf32>) -> tensor<64x32x64x1xf32>
// CHECK-NEXT:   %4 = stablehlo.reshape %3 : (tensor<64x32x64x1xf32>) -> tensor<64x1x32x64x1xf32>
// CHECK-NEXT:   %5 = stablehlo.broadcast_in_dim %arg3, dims = [2, 1] : (tensor<64x32xf32>) -> tensor<64x32x64x1xf32>
// CHECK-NEXT:   %6 = stablehlo.reshape %5 : (tensor<64x32x64x1xf32>) -> tensor<64x1x32x64x1xf32>
// CHECK-NEXT:   %7 = stablehlo.multiply %4, %6 : tensor<64x1x32x64x1xf32>
// CHECK-NEXT:   %8 = stablehlo.multiply %cst, %arg1 : tensor<f32>
// CHECK-NEXT:   %9 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [2, 1] : (tensor<64x1x32x64x1xf32>, tensor<f32>) -> tensor<64x64x1xf32>
// CHECK-NEXT:   %10 = stablehlo.reshape %9 : (tensor<64x64x1xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %11 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<64x64x1xf32>
// CHECK-NEXT:   %12 = stablehlo.reshape %11 : (tensor<64x64x1xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %13 = stablehlo.add %10, %12 : tensor<64x64xf32>
// CHECK-NEXT:   %14 = stablehlo.transpose %13, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:   return %14 : tensor<64x64xf32>
// CHECK-NEXT: }
