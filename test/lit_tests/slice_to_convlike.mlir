// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main_add1(%arg0: tensor<4x1534x3070xf64>) -> tensor<3x1520x3056xf64> {
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<3x1520x3056xf64>
    %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<3x1520x3056xf64>
    %0 = stablehlo.slice %arg0 [0:3, 7:1527, 6:3062] : (tensor<4x1534x3070xf64>) -> tensor<3x1520x3056xf64>
    %1 = stablehlo.slice %arg0 [0:3, 7:1527, 5:3061] : (tensor<4x1534x3070xf64>) -> tensor<3x1520x3056xf64>
    %2 = stablehlo.multiply %0, %cst_0 : tensor<3x1520x3056xf64>
    %3 = stablehlo.multiply %1, %cst_1 : tensor<3x1520x3056xf64>
    %4 = stablehlo.add %2, %3 : tensor<3x1520x3056xf64>
    return %4 : tensor<3x1520x3056xf64>
}

// CHECK: func.func @main_add1(%arg0: tensor<4x1534x3070xf64>) -> tensor<3x1520x3056xf64> {
// CHECK-NEXT{LITERAL}:     %cst = stablehlo.constant dense<[[[[[3.000000e+00, 2.000000e+00]]]]]> : tensor<1x1x1x1x2xf64>
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [0:3, 7:1527, 5:3062] : (tensor<4x1534x3070xf64>) -> tensor<3x1520x3057xf64>
// CHECK-NEXT:     %1 = stablehlo.reshape %0 : (tensor<3x1520x3057xf64>) -> tensor<1x1x3x1520x3057xf64>
// CHECK-NEXT:     %2 = stablehlo.convolution(%1, %cst) dim_numbers = [b, f, 0, 1, 2]x[i, o, 0, 1, 2]->[b, f, 0, 1, 2], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x3x1520x3057xf64>, tensor<1x1x1x1x2xf64>) -> tensor<1x1x3x1520x
// CHECK-NEXT: 3056xf64>
// CHECK-NEXT:     %3 = stablehlo.reshape %2 : (tensor<1x1x3x1520x3056xf64>) -> tensor<3x1520x3056xf64>
// CHECK-NEXT:     return %3 : tensor<3x1520x3056xf64>
// CHECK-NEXT: }

func.func @main_add2(%arg0: tensor<4x1534x3070xf64>) -> tensor<3x1520x3056xf64> {
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<3x1520x3056xf64>
    %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<3x1520x3056xf64>
    %0 = stablehlo.slice %arg0 [0:3, 7:1527, 5:3061] : (tensor<4x1534x3070xf64>) -> tensor<3x1520x3056xf64>
    %1 = stablehlo.slice %arg0 [0:3, 7:1527, 6:3062] : (tensor<4x1534x3070xf64>) -> tensor<3x1520x3056xf64>
    %2 = stablehlo.multiply %0, %cst_0 : tensor<3x1520x3056xf64>
    %3 = stablehlo.multiply %1, %cst_1 : tensor<3x1520x3056xf64>
    %4 = stablehlo.add %3, %2 : tensor<3x1520x3056xf64>
    return %4 : tensor<3x1520x3056xf64>
}

// CHECK: func.func @main_add2(%arg0: tensor<4x1534x3070xf64>) -> tensor<3x1520x3056xf64> {
// CHECK-NEXT{LITERAL}:     %cst = stablehlo.constant dense<[[[[[2.000000e+00, 3.000000e+00]]]]]> : tensor<1x1x1x1x2xf64>
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [0:3, 7:1527, 5:3062] : (tensor<4x1534x3070xf64>) -> tensor<3x1520x3057xf64>
// CHECK-NEXT:     %1 = stablehlo.reshape %0 : (tensor<3x1520x3057xf64>) -> tensor<1x1x3x1520x3057xf64>
// CHECK-NEXT:     %2 = stablehlo.convolution(%1, %cst) dim_numbers = [b, f, 0, 1, 2]x[i, o, 0, 1, 2]->[b, f, 0, 1, 2], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x3x1520x3057xf64>, tensor<1x1x1x1x2xf64>) -> tensor<1x1x3x1520x3056xf64>
// CHECK-NEXT:     %3 = stablehlo.reshape %2 : (tensor<1x1x3x1520x3056xf64>) -> tensor<3x1520x3056xf64>
// CHECK-NEXT:     return %3 : tensor<3x1520x3056xf64>
// CHECK-NEXT: }

func.func @main_sub1(%arg0: tensor<4x1534x3070xf64>) -> tensor<3x1520x3056xf64> {
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<3x1520x3056xf64>
    %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<3x1520x3056xf64>
    %0 = stablehlo.slice %arg0 [0:3, 7:1527, 6:3062] : (tensor<4x1534x3070xf64>) -> tensor<3x1520x3056xf64>
    %1 = stablehlo.slice %arg0 [0:3, 7:1527, 5:3061] : (tensor<4x1534x3070xf64>) -> tensor<3x1520x3056xf64>
    %2 = stablehlo.multiply %0, %cst_0 : tensor<3x1520x3056xf64>
    %3 = stablehlo.multiply %1, %cst_1 : tensor<3x1520x3056xf64>
    %4 = stablehlo.subtract %2, %3 : tensor<3x1520x3056xf64>
    return %4 : tensor<3x1520x3056xf64>
}

// CHECK: func.func @main_sub1(%arg0: tensor<4x1534x3070xf64>) -> tensor<3x1520x3056xf64> {
// CHECK-NEXT{LITERAL}:     %cst = stablehlo.constant dense<[[[[[-3.000000e+00, 2.000000e+00]]]]]> : tensor<1x1x1x1x2xf64>
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [0:3, 7:1527, 5:3062] : (tensor<4x1534x3070xf64>) -> tensor<3x1520x3057xf64>
// CHECK-NEXT:     %1 = stablehlo.reshape %0 : (tensor<3x1520x3057xf64>) -> tensor<1x1x3x1520x3057xf64>
// CHECK-NEXT:     %2 = stablehlo.convolution(%1, %cst) dim_numbers = [b, f, 0, 1, 2]x[i, o, 0, 1, 2]->[b, f, 0, 1, 2], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x3x1520x3057xf64>, tensor<1x1x1x1x2xf64>) -> tensor<1x1x3x1520x3056xf64>
// CHECK-NEXT:     %3 = stablehlo.reshape %2 : (tensor<1x1x3x1520x3056xf64>) -> tensor<3x1520x3056xf64>
// CHECK-NEXT:     return %3 : tensor<3x1520x3056xf64>
// CHECK-NEXT: }

func.func @main_sub2(%arg0: tensor<4x1534x3070xf64>) -> tensor<3x1520x3056xf64> {
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<3x1520x3056xf64>
    %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<3x1520x3056xf64>
    %0 = stablehlo.slice %arg0 [0:3, 7:1527, 6:3062] : (tensor<4x1534x3070xf64>) -> tensor<3x1520x3056xf64>
    %1 = stablehlo.slice %arg0 [0:3, 7:1527, 5:3061] : (tensor<4x1534x3070xf64>) -> tensor<3x1520x3056xf64>
    %2 = stablehlo.multiply %0, %cst_0 : tensor<3x1520x3056xf64>
    %3 = stablehlo.multiply %1, %cst_1 : tensor<3x1520x3056xf64>
    %4 = stablehlo.subtract %3, %2 : tensor<3x1520x3056xf64>
    return %4 : tensor<3x1520x3056xf64>
}

// CHECK: func.func @main_sub2(%arg0: tensor<4x1534x3070xf64>) -> tensor<3x1520x3056xf64> {
// CHECK-NEXT{LITERAL}:     %cst = stablehlo.constant dense<[[[[[3.000000e+00, -2.000000e+00]]]]]> : tensor<1x1x1x1x2xf64>
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [0:3, 7:1527, 5:3062] : (tensor<4x1534x3070xf64>) -> tensor<3x1520x3057xf64>
// CHECK-NEXT:     %1 = stablehlo.reshape %0 : (tensor<3x1520x3057xf64>) -> tensor<1x1x3x1520x3057xf64>
// CHECK-NEXT:     %2 = stablehlo.convolution(%1, %cst) dim_numbers = [b, f, 0, 1, 2]x[i, o, 0, 1, 2]->[b, f, 0, 1, 2], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x3x1520x3057xf64>, tensor<1x1x1x1x2xf64>) -> tensor<1x1x3x1520x3056xf64>
// CHECK-NEXT:     %3 = stablehlo.reshape %2 : (tensor<1x1x3x1520x3056xf64>) -> tensor<3x1520x3056xf64>
// CHECK-NEXT:     return %3 : tensor<3x1520x3056xf64>
// CHECK-NEXT: }
