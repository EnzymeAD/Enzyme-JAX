// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=concat_reshape_reduce},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=FULL

func.func @main(%arg0: tensor<5x4x3xf64>) -> tensor<4x1x1xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.slice %arg0 [0:5, 0:1, 0:3] : (tensor<5x4x3xf64>) -> tensor<5x1x3xf64>
    %1 = stablehlo.reshape %0 : (tensor<5x1x3xf64>) -> tensor<5x3xf64>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [1, 0] : (tensor<5x3xf64>, tensor<f64>) -> tensor<f64>
    %3 = stablehlo.slice %arg0 [0:5, 1:2, 0:3] : (tensor<5x4x3xf64>) -> tensor<5x1x3xf64>
    %4 = stablehlo.reshape %3 : (tensor<5x1x3xf64>) -> tensor<5x3xf64>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1, 0] : (tensor<5x3xf64>, tensor<f64>) -> tensor<f64>
    %6 = stablehlo.slice %arg0 [0:5, 2:3, 0:3] : (tensor<5x4x3xf64>) -> tensor<5x1x3xf64>
    %7 = stablehlo.reshape %6 : (tensor<5x1x3xf64>) -> tensor<5x3xf64>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [1, 0] : (tensor<5x3xf64>, tensor<f64>) -> tensor<f64>
    %9 = stablehlo.slice %arg0 [0:5, 3:4, 0:3] : (tensor<5x4x3xf64>) -> tensor<5x1x3xf64>
    %10 = stablehlo.reshape %9 : (tensor<5x1x3xf64>) -> tensor<5x3xf64>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [1, 0] : (tensor<5x3xf64>, tensor<f64>) -> tensor<f64>
    %12 = stablehlo.reshape %2 : (tensor<f64>) -> tensor<1x1x1xf64>
    %13 = stablehlo.reshape %5 : (tensor<f64>) -> tensor<1x1x1xf64>
    %14 = stablehlo.reshape %8 : (tensor<f64>) -> tensor<1x1x1xf64>
    %15 = stablehlo.reshape %11 : (tensor<f64>) -> tensor<1x1x1xf64>
    %16 = stablehlo.concatenate %12, %13, %14, %15, dim = 0 : (tensor<1x1x1xf64>, tensor<1x1x1xf64>, tensor<1x1x1xf64>, tensor<1x1x1xf64>) -> tensor<4x1x1xf64>
    return %16 : tensor<4x1x1xf64>
}

// CHECK: func.func @main(%arg0: tensor<5x4x3xf64>) -> tensor<4x1x1xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [0:5, 0:1, 0:3] : (tensor<5x4x3xf64>) -> tensor<5x1x3xf64>
// CHECK-NEXT:     %1 = stablehlo.reshape %0 : (tensor<5x1x3xf64>) -> tensor<5x3xf64>
// CHECK-NEXT:     %2 = stablehlo.slice %arg0 [0:5, 1:2, 0:3] : (tensor<5x4x3xf64>) -> tensor<5x1x3xf64>
// CHECK-NEXT:     %3 = stablehlo.reshape %2 : (tensor<5x1x3xf64>) -> tensor<5x3xf64>
// CHECK-NEXT:     %4 = stablehlo.slice %arg0 [0:5, 2:3, 0:3] : (tensor<5x4x3xf64>) -> tensor<5x1x3xf64>
// CHECK-NEXT:     %5 = stablehlo.reshape %4 : (tensor<5x1x3xf64>) -> tensor<5x3xf64>
// CHECK-NEXT:     %6 = stablehlo.slice %arg0 [0:5, 3:4, 0:3] : (tensor<5x4x3xf64>) -> tensor<5x1x3xf64>
// CHECK-NEXT:     %7 = stablehlo.reshape %6 : (tensor<5x1x3xf64>) -> tensor<5x3xf64>
// CHECK-NEXT:     %8 = stablehlo.reshape %1 : (tensor<5x3xf64>) -> tensor<1x5x3xf64>
// CHECK-NEXT:     %9 = stablehlo.reshape %3 : (tensor<5x3xf64>) -> tensor<1x5x3xf64>
// CHECK-NEXT:     %10 = stablehlo.reshape %5 : (tensor<5x3xf64>) -> tensor<1x5x3xf64>
// CHECK-NEXT:     %11 = stablehlo.reshape %7 : (tensor<5x3xf64>) -> tensor<1x5x3xf64>
// CHECK-NEXT:     %12 = stablehlo.concatenate %8, %9, %10, %11, dim = 0 : (tensor<1x5x3xf64>, tensor<1x5x3xf64>, tensor<1x5x3xf64>, tensor<1x5x3xf64>) -> tensor<4x5x3xf64>
// CHECK-NEXT:     %13 = stablehlo.reduce(%12 init: %cst) applies stablehlo.add across dimensions = [2, 1] : (tensor<4x5x3xf64>, tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:     %14 = stablehlo.reshape %13 : (tensor<4xf64>) -> tensor<4x1x1xf64>
// CHECK-NEXT:     return %14 : tensor<4x1x1xf64>
// CHECK-NEXT: }

// FULL: func.func @main(%arg0: tensor<5x4x3xf64>) -> tensor<4x1x1xf64> {
// FULL-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FULL-NEXT:     %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2, 0] : (tensor<5x4x3xf64>, tensor<f64>) -> tensor<4xf64>
// FULL-NEXT:     %1 = stablehlo.reshape %0 : (tensor<4xf64>) -> tensor<4x1x1xf64>
// FULL-NEXT:     return %1 : tensor<4x1x1xf64>
// FULL-NEXT: }
