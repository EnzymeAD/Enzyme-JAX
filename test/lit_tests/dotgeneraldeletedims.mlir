// RUN: enzymexlamlir-opt --enzyme-hlo-opt="passses=131072" %s | FileCheck %s

module {
  func.func @covariance(%arg0: tensor<128x2048xf32>) -> tensor<128x128xf32> {
    %cst = stablehlo.constant dense<4.88519785E-4> : tensor<128x128xf32>
    %cst_0 = stablehlo.constant dense<4.8828125E-4> : tensor<128x1xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reshape %arg0 : (tensor<128x2048xf32>) -> tensor<128x2048x1xf32>
    %1 = stablehlo.reduce(%0 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<128x2048x1xf32>, tensor<f32>) -> tensor<128x1xf32>
    %2 = stablehlo.multiply %1, %cst_0 : tensor<128x1xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [1, 0] : (tensor<128x2048xf32>) -> tensor<2048x128x1x1xf32>
    %4 = stablehlo.broadcast_in_dim %2, dims = [1, 2] : (tensor<128x1xf32>) -> tensor<2048x128x1x1xf32>
    %5 = stablehlo.subtract %3, %4 : tensor<2048x128x1x1xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [2, 0, 3, 4] : (tensor<2048x128x1x1xf32>) -> tensor<128x128x2048x1x1xf32>
    %7 = stablehlo.broadcast_in_dim %5, dims = [2, 1, 3, 4] : (tensor<2048x128x1x1xf32>) -> tensor<128x128x2048x1x1xf32>
    %8 = stablehlo.reshape %6 : (tensor<128x128x2048x1x1xf32>) -> tensor<128x128x2048x1xf32>
    %9 = stablehlo.reshape %7 : (tensor<128x128x2048x1x1xf32>) -> tensor<128x128x2048x1xf32>
    %10 = stablehlo.dot_general %8, %9, batching_dims = [0, 1, 3] x [0, 1, 3], contracting_dims = [2] x [2] : (tensor<128x128x2048x1xf32>, tensor<128x128x2048x1xf32>) -> tensor<128x128x1xf32>
    %11 = stablehlo.transpose %10, dims = [1, 0, 2] : (tensor<128x128x1xf32>) -> tensor<128x128x1xf32>
    %12 = stablehlo.reshape %11 : (tensor<128x128x1xf32>) -> tensor<128x128xf32>
    %13 = stablehlo.multiply %12, %cst : tensor<128x128xf32>
    return %13 : tensor<128x128xf32>
  }
}

// CHECK-LABEL: func.func @covariance
// CHECK:      %6 = stablehlo.dot_general %5, %5, batching_dims = [2] x [2], contracting_dims = [0] x [0] : (tensor<2048x128x1xf32>, tensor<2048x128x1xf32>) -> tensor<1x128x128xf32>

module {
  func.func @syrk(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<2048x2048xf32>, %arg3: tensor<2048x2048xf32>) -> tensor<2048x2048xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2048x2048x1x1xf32>
    %1 = stablehlo.reshape %arg2 : (tensor<2048x2048xf32>) -> tensor<2048x2048x1x1xf32>
    %2 = stablehlo.multiply %1, %0 : tensor<2048x2048x1x1xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1, 2, 3, 4] : (tensor<2048x2048x1x1xf32>) -> tensor<2048x2048x2048x1x1xf32>
    %4 = stablehlo.reshape %3 : (tensor<2048x2048x2048x1x1xf32>) -> tensor<2048x2048x2048x1xf32>
    %5 = stablehlo.dot_general %4, %arg2, batching_dims = [0] x [1], contracting_dims = [1] x [0] : (tensor<2048x2048x2048x1xf32>, tensor<2048x2048xf32>) -> tensor<2048x2048x1xf32>
    %6 = stablehlo.broadcast_in_dim %arg3, dims = [1, 0] : (tensor<2048x2048xf32>) -> tensor<2048x2048x1xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<2048x2048x1xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<2048x2048x1xf32>
    %9 = stablehlo.add %5, %8 : tensor<2048x2048x1xf32>
    %10 = stablehlo.reshape %9 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2048x2048x1xf32>) -> tensor<2048x2048xf32>
    %11 = stablehlo.transpose %10, dims = [1, 0] : (tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    return %11 : tensor<2048x2048xf32>
  }
}

// CHECK-LABEL: func.func @syrk
// CHECK:    %0 = stablehlo.dot_general %arg2, %arg2, contracting_dims = [0] x [0] : (tensor<2048x2048xf32>, tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
