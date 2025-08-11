// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %0 = "enzymexla.rotate"(%arg0) <{amount = 235 : si32, dimension = 1 : si32}> : (tensor<1520x3056xf64>) -> tensor<1520x3056xf64>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1520x3056xf64>
    return %1 : tensor<1520x3056xf64>
}

// CHECK: func.func @main1(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = "enzymexla.wrap"(%arg0) <{dimension = 1 : i64, lhs = 235 : i64, rhs = 0 : i64}> : (tensor<1520x3056xf64>) -> tensor<1520x3291xf64>
// CHECK-NEXT:     %1 = "stablehlo.reduce_window"(%0, %cst) <{window_dilations = array<i64: 1, 235>, window_dimensions = array<i64: 1, 2>}> ({
// CHECK-NEXT:     ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:       %2 = stablehlo.multiply %arg1, %arg2 : tensor<f64>
// CHECK-NEXT:       stablehlo.return %2 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<1520x3291xf64>, tensor<f64>) -> tensor<1520x3056xf64>
// CHECK-NEXT:     return %1 : tensor<1520x3056xf64>
// CHECK-NEXT: }
