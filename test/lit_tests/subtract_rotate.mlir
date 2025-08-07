// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %0 = "enzymexla.rotate"(%arg0) <{amount = 235 : si32, dimension = 1 : si32}> : (tensor<1520x3056xf64>) -> tensor<1520x3056xf64>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1520x3056xf64>
    return %1 : tensor<1520x3056xf64>
}

// CHECK: func.func @main1(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = "enzymexla.wrap"(%arg0) <{dimension = 1 : i64, lhs = 235 : i64, rhs = 0 : i64}> : (tensor<1520x3056xf64>) -> tensor<1520x3291xf64>
// CHECK-NEXT:     %1 = "stablehlo.reduce_window"(%0, %cst) <{window_dilations = array<i64: 1, 235>, window_dimensions = array<i64: 1, 2>}> ({
// CHECK-NEXT:     ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:       %2 = stablehlo.subtract %arg1, %arg2 : tensor<f64>
// CHECK-NEXT:       stablehlo.return %2 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<1520x3291xf64>, tensor<f64>) -> tensor<1520x3056xf64>
// CHECK-NEXT:     return %1 : tensor<1520x3056xf64>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %0 = "enzymexla.rotate"(%arg0) <{amount = 235 : si32, dimension = 1 : si32}> : (tensor<1520x3056xf64>) -> tensor<1520x3056xf64>
    %1 = stablehlo.subtract %0, %arg0 : tensor<1520x3056xf64>
    return %1 : tensor<1520x3056xf64>
}

// CHECK: func.func @main2(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = "enzymexla.wrap"(%arg0) <{dimension = 1 : i64, lhs = 235 : i64, rhs = 0 : i64}> : (tensor<1520x3056xf64>) -> tensor<1520x3291xf64>
// CHECK-NEXT:     %1 = "stablehlo.reduce_window"(%0, %cst) <{window_dilations = array<i64: 1, 235>, window_dimensions = array<i64: 1, 2>}> ({
// CHECK-NEXT:     ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:       %2 = stablehlo.subtract %arg1, %arg2 : tensor<f64>
// CHECK-NEXT:       stablehlo.return %2 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<1520x3291xf64>, tensor<f64>) -> tensor<1520x3056xf64>
// CHECK-NEXT:     return %1 : tensor<1520x3056xf64>
// CHECK-NEXT: }

func.func @main3(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %cst_1 = stablehlo.constant dense<5.000000e+00> : tensor<1520x3056xf64>
    %cst_2 = stablehlo.constant dense<-2.000000e+00> : tensor<1520x3056xf64>
    %0 = "enzymexla.rotate"(%arg0) <{amount = 235 : si32, dimension = 1 : si32}> : (tensor<1520x3056xf64>) -> tensor<1520x3056xf64>
    %1 = stablehlo.multiply %0, %cst_2 : tensor<1520x3056xf64>
    %2 = stablehlo.multiply %cst_1, %arg0 : tensor<1520x3056xf64>
    %3 = stablehlo.subtract %2, %1 : tensor<1520x3056xf64>
    return %3 : tensor<1520x3056xf64>
}

// CHECK: func.func @main3(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<-2.000000e+00> : tensor<f64>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<5.000000e+00> : tensor<f64>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = "enzymexla.wrap"(%arg0) <{dimension = 1 : i64, lhs = 235 : i64, rhs = 0 : i64}> : (tensor<1520x3056xf64>) -> tensor<1520x3291xf64>
// CHECK-NEXT:     %1 = "stablehlo.reduce_window"(%0, %cst_1) <{window_dilations = array<i64: 1, 235>, window_dimensions = array<i64: 1, 2>}> ({
// CHECK-NEXT:     ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:       %2 = stablehlo.multiply %arg1, %cst_0 : tensor<f64>
// CHECK-NEXT:       %3 = stablehlo.multiply %arg2, %cst : tensor<f64>
// CHECK-NEXT:       %4 = stablehlo.subtract %2, %3 : tensor<f64>
// CHECK-NEXT:       stablehlo.return %4 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<1520x3291xf64>, tensor<f64>) -> tensor<1520x3056xf64>
// CHECK-NEXT:     return %1 : tensor<1520x3056xf64>
// CHECK-NEXT: }

func.func @main4(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %cst_1 = stablehlo.constant dense<5.000000e+00> : tensor<1520x3056xf64>
    %cst_2 = stablehlo.constant dense<-2.000000e+00> : tensor<1520x3056xf64>
    %0 = "enzymexla.rotate"(%arg0) <{amount = 235 : si32, dimension = 1 : si32}> : (tensor<1520x3056xf64>) -> tensor<1520x3056xf64>
    %1 = stablehlo.multiply %0, %cst_2 : tensor<1520x3056xf64>
    %2 = stablehlo.multiply %cst_1, %arg0 : tensor<1520x3056xf64>
    %3 = stablehlo.subtract %1, %2 : tensor<1520x3056xf64>
    return %3 : tensor<1520x3056xf64>
}

// CHECK: func.func @main4(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<5.000000e+00> : tensor<f64>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<-2.000000e+00> : tensor<f64>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = "enzymexla.wrap"(%arg0) <{dimension = 1 : i64, lhs = 235 : i64, rhs = 0 : i64}> : (tensor<1520x3056xf64>) -> tensor<1520x3291xf64>
// CHECK-NEXT:     %1 = "stablehlo.reduce_window"(%0, %cst_1) <{window_dilations = array<i64: 1, 235>, window_dimensions = array<i64: 1, 2>}> ({
// CHECK-NEXT:     ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:       %2 = stablehlo.multiply %arg1, %cst_0 : tensor<f64>
// CHECK-NEXT:       %3 = stablehlo.multiply %arg2, %cst : tensor<f64>
// CHECK-NEXT:       %4 = stablehlo.subtract %2, %3 : tensor<f64>
// CHECK-NEXT:       stablehlo.return %4 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<1520x3291xf64>, tensor<f64>) -> tensor<1520x3056xf64>
// CHECK-NEXT:     return %1 : tensor<1520x3056xf64>
// CHECK-NEXT: }

func.func @main5(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %cst_1 = stablehlo.constant dense<5.000000e+00> : tensor<1520x3056xf64>
    %0 = "enzymexla.rotate"(%arg0) <{amount = 235 : si32, dimension = 1 : si32}> : (tensor<1520x3056xf64>) -> tensor<1520x3056xf64>
    %1 = stablehlo.multiply %cst_1, %arg0 : tensor<1520x3056xf64>
    %2 = stablehlo.subtract %0, %1 : tensor<1520x3056xf64>
    return %2 : tensor<1520x3056xf64>
}

// CHECK: func.func @main5(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<5.000000e+00> : tensor<f64>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = "enzymexla.wrap"(%arg0) <{dimension = 1 : i64, lhs = 235 : i64, rhs = 0 : i64}> : (tensor<1520x3056xf64>) -> tensor<1520x3291xf64>
// CHECK-NEXT:     %1 = "stablehlo.reduce_window"(%0, %cst_0) <{window_dilations = array<i64: 1, 235>, window_dimensions = array<i64: 1, 2>}> ({
// CHECK-NEXT:     ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:       %2 = stablehlo.multiply %arg2, %cst : tensor<f64>
// CHECK-NEXT:       %3 = stablehlo.subtract %arg1, %2 : tensor<f64>
// CHECK-NEXT:       stablehlo.return %3 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<1520x3291xf64>, tensor<f64>) -> tensor<1520x3056xf64>
// CHECK-NEXT:     return %1 : tensor<1520x3056xf64>
// CHECK-NEXT: }
