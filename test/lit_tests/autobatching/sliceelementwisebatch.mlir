// RUN: enzymexlamlir-opt --auto-batching --inline --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<6x3xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<3.000000e+01> : tensor<4xf32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2] : (tensor<6x3xf32>) -> tensor<6x1x3xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<6x1x3xf32>) -> tensor<6x4x3xf32>
    %2 = stablehlo.transpose %1, dims = [0, 1, 2] : (tensor<6x4x3xf32>) -> tensor<6x4x3xf32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.multiply across dimensions = [2] : (tensor<6x4x3xf32>, tensor<f32>) -> tensor<6x4xf32>
    %4 = stablehlo.slice %3 [0:1, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
    %5 = stablehlo.reshape %4 : (tensor<1x4xf32>) -> tensor<4xf32>
    %6 = stablehlo.slice %3 [1:2, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
    %7 = stablehlo.reshape %6 : (tensor<1x4xf32>) -> tensor<4xf32>
    %8 = stablehlo.slice %3 [2:3, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
    %9 = stablehlo.reshape %8 : (tensor<1x4xf32>) -> tensor<4xf32>
    %10 = stablehlo.slice %3 [3:4, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
    %11 = stablehlo.reshape %10 : (tensor<1x4xf32>) -> tensor<4xf32>
    %12 = stablehlo.slice %3 [4:5, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
    %13 = stablehlo.reshape %12 : (tensor<1x4xf32>) -> tensor<4xf32>
    %14 = stablehlo.slice %3 [5:6, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
    %15 = stablehlo.reshape %14 : (tensor<1x4xf32>) -> tensor<4xf32>
    %16 = stablehlo.power %15, %cst_0 : tensor<4xf32>
    %17 = stablehlo.power %13, %cst_0 : tensor<4xf32>
    %18 = stablehlo.power %11, %cst_0 : tensor<4xf32>
    %19 = stablehlo.power %9, %cst_0 : tensor<4xf32>
    %20 = stablehlo.power %7, %cst_0 : tensor<4xf32>
    %21 = stablehlo.power %5, %cst_0 : tensor<4xf32>
    return %16, %17, %18, %19, %20, %21 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<6x3xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<3.000000e+01> : tensor<6x4xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2] : (tensor<6x3xf32>) -> tensor<6x4x3xf32>
// CHECK-NEXT:    %1 = stablehlo.reduce(%0 init: %cst_0) applies stablehlo.multiply across dimensions = [2] : (tensor<6x4x3xf32>, tensor<f32>) -> tensor<6x4xf32>
// CHECK-NEXT:    %2 = stablehlo.power %1, %cst : tensor<6x4xf32>
// CHECK-NEXT:    %3 = stablehlo.slice %2 [0:1, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:    %4 = stablehlo.reshape %3 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %5 = stablehlo.slice %2 [1:2, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:    %6 = stablehlo.reshape %5 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %7 = stablehlo.slice %2 [2:3, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:    %8 = stablehlo.reshape %7 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %9 = stablehlo.slice %2 [3:4, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:    %10 = stablehlo.reshape %9 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %11 = stablehlo.slice %2 [4:5, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:    %12 = stablehlo.reshape %11 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %13 = stablehlo.slice %2 [5:6, 0:4] : (tensor<6x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:    %14 = stablehlo.reshape %13 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    return %4, %6, %8, %10, %12, %14 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
// CHECK-NEXT:  }
