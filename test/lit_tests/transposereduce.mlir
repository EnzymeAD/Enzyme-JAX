// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @main(%arg0: tensor<4x3xf64>) -> tensor<4x1xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x3xf64>) -> tensor<3x4xf64>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3x4xf64>, tensor<f64>) -> tensor<4xf64>
    %2 = stablehlo.reshape %1 : (tensor<4xf64>) -> tensor<4x1xf64>
    return %2 : tensor<4x1xf64>
}

// CHECK:  func.func @main(%arg0: tensor<4x3xf64>) -> tensor<4x1xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x3xf64>, tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<4xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:    return %1 : tensor<4x1xf64>
// CHECK-NEXT:  }

func.func @main2(%arg0: tensor<5x4x3xf64>) -> tensor<5x1x3xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<5x4x3xf64>) -> tensor<3x4x5xf64>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<3x4x5xf64>, tensor<f64>) -> tensor<3x5xf64>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<3x5xf64>) -> tensor<5x3xf64>
    %3 = stablehlo.reshape %2 : (tensor<5x3xf64>) -> tensor<5x1x3xf64>
    return %3 : tensor<5x1x3xf64>
}

// CHECK:  func.func @main2(%arg0: tensor<5x4x3xf64>) -> tensor<5x1x3xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<5x4x3xf64>, tensor<f64>) -> tensor<5x3xf64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<5x3xf64>) -> tensor<5x1x3xf64>
// CHECK-NEXT:    return %1 : tensor<5x1x3xf64>
// CHECK-NEXT:  }
