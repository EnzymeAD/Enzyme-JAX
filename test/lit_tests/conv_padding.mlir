// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3080xf64> {
    %cst = stablehlo.constant dense<[[[[-1.000000e+00, 1.000000e+00]]]]> : tensor<1x1x1x2xf64>
    %cst2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.reshape %arg0 : (tensor<1520x3056xf64>) -> tensor<1x1x1520x3056xf64>
    %1 = stablehlo.pad %0, %cst2, low = [0, 0, 0, 22], high = [0, 0, 0, 26], interior = [0, 0, 0, 0] : (tensor<1x1x1520x3056xf64>, tensor<f64>) -> tensor<1x1x1520x3104xf64>
    %2 = stablehlo.convolution(%1, %cst) dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1], window = {rhs_dilate = [1, 24]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x1520x3104xf64>, tensor<1x1x1x2xf64>) -> tensor<1x1x1520x3080xf64>
    %3 = stablehlo.reshape %2 : (tensor<1x1x1520x3080xf64>) -> tensor<1520x3080xf64>
    return %3 : tensor<1520x3080xf64>
}

// CHECK: func.func @main(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3080xf64> {
// CHECK-NEXT{LITERAL}:     %cst = stablehlo.constant dense<[[[[-1.000000e+00, 1.000000e+00]]]]> : tensor<1x1x1x2xf64>
// CHECK-NEXT:     %0 = stablehlo.reshape %arg0 : (tensor<1520x3056xf64>) -> tensor<1x1x1520x3056xf64>
// CHECK-NEXT:     %1 = stablehlo.convolution(%0, %cst) dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1], window = {pad = [[0, 0], [22, 26]], rhs_dilate = [1, 24]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x1520x3056xf64>, tensor<1x1x1x2xf64>) -> tensor<1x1x1520x3080xf64>
// CHECK-NEXT:     %2 = stablehlo.reshape %1 : (tensor<1x1x1520x3080xf64>) -> tensor<1520x3080xf64>
// CHECK-NEXT:     return %2 : tensor<1520x3080xf64>
// CHECK-NEXT: }

func.func @main_no_apply(%arg0: tensor<1520x3056xf64>) -> tensor<6x1x1520x3080xf64> {
    %cst = stablehlo.constant dense<[[[[-1.000000e+00, 1.000000e+00]]]]> : tensor<1x1x1x2xf64>
    %cst2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.reshape %arg0 : (tensor<1520x3056xf64>) -> tensor<1x1x1520x3056xf64>
    // CHECK: stablehlo.pad
    %1 = stablehlo.pad %0, %cst2, low = [3, 0, 0, 24], high = [2, 0, 0, 24], interior = [0, 0, 0, 0] : (tensor<1x1x1520x3056xf64>, tensor<f64>) -> tensor<6x1x1520x3104xf64>
    %2 = stablehlo.convolution(%1, %cst) dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1], window = {rhs_dilate = [1, 24]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<6x1x1520x3104xf64>, tensor<1x1x1x2xf64>) -> tensor<6x1x1520x3080xf64>
    return %2 : tensor<6x1x1520x3080xf64>
}

func.func @main_no_apply2(%arg0: tensor<1520x3056xf64>) -> tensor<1x1x1520x3080xf64> {
    %cst = stablehlo.constant dense<[[[[-1.000000e+00, 1.000000e+00]]], [[[-1.000000e+00, 1.000000e+00]]]]> : tensor<2x1x1x2xf64>
    %cst2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.reshape %arg0 : (tensor<1520x3056xf64>) -> tensor<1x1x1520x3056xf64>
    // CHECK: stablehlo.pad
    %1 = stablehlo.pad %0, %cst2, low = [0, 0, 0, 24], high = [0, 1, 0, 24], interior = [0, 0, 0, 0] : (tensor<1x1x1520x3056xf64>, tensor<f64>) -> tensor<1x2x1520x3104xf64>
    %2 = stablehlo.convolution(%1, %cst) dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1], window = {rhs_dilate = [1, 24]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x2x1520x3104xf64>, tensor<2x1x1x2xf64>) -> tensor<1x1x1520x3080xf64>
    return %2 : tensor<1x1x1520x3080xf64>
}
