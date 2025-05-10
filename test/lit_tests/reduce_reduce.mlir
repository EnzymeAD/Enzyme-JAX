// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=reduce_reduce},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<5x4x3x2xf64>) -> tensor<1x4x1x1xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0, 2] : (tensor<5x4x3x2xf64>, tensor<f64>) -> tensor<4x2xf64>
    %4 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x2xf64>, tensor<f64>) -> tensor<4xf64>
    %5 = stablehlo.reshape %4 : (tensor<4xf64>) -> tensor<1x4x1x1xf64>
    return %5 : tensor<1x4x1x1xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<5x4x3x2xf64>) -> tensor<1x4x1x1xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<5x4x3x2xf64>, tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:     %1 = stablehlo.reshape %0 : (tensor<4xf64>) -> tensor<1x4x1x1xf64>
// CHECK-NEXT:     return %1 : tensor<1x4x1x1xf64>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<5x4x3x2xf64>) -> tensor<1x1x3x1xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<5x4x3x2xf64>, tensor<f64>) -> tensor<5x3x2xf64>
    %4 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0, 2] : (tensor<5x3x2xf64>, tensor<f64>) -> tensor<3xf64>
    %5 = stablehlo.reshape %4 : (tensor<3xf64>) -> tensor<1x1x3x1xf64>
    return %5 : tensor<1x1x3x1xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<5x4x3x2xf64>) -> tensor<1x1x3x1xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1, 0, 3] : (tensor<5x4x3x2xf64>, tensor<f64>) -> tensor<3xf64>
// CHECK-NEXT:     %1 = stablehlo.reshape %0 : (tensor<3xf64>) -> tensor<1x1x3x1xf64>
// CHECK-NEXT:     return %1 : tensor<1x1x3x1xf64>
// CHECK-NEXT: }
