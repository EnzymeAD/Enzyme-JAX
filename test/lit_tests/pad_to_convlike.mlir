// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<1520x3056xf64>, %arg1: tensor<f64>) -> tensor<1520x3080xf64> {
    %0 = stablehlo.pad %arg0, %arg1, low = [0, 24], high = [0, 0], interior = [0, 0] : (tensor<1520x3056xf64>, tensor<f64>) -> tensor<1520x3080xf64>
    %1 = stablehlo.pad %arg0, %arg1, low = [0, 0], high = [0, 24], interior = [0, 0] : (tensor<1520x3056xf64>, tensor<f64>) -> tensor<1520x3080xf64>
    %2 = stablehlo.add %0, %1 : tensor<1520x3080xf64>
    return %2 : tensor<1520x3080xf64>
}

// CHECK: func.func @main1(%arg0: tensor<1520x3056xf64>, %arg1: tensor<f64>) -> tensor<1520x3080xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %arg1, low = [0, 24], high = [0, 24], interior = [0, 0] : (tensor<1520x3056xf64>, tensor<f64>) -> tensor<1520x3104xf64>
// CHECK-NEXT:     %1 = "stablehlo.reduce_window"(%0, %cst) <{window_dilations = array<i64: 1, 24>, window_dimensions = array<i64: 1, 2>}> ({
// CHECK-NEXT:     ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:       %2 = stablehlo.add %arg2, %arg3 : tensor<f64>
// CHECK-NEXT:       stablehlo.return %2 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<1520x3104xf64>, tensor<f64>) -> tensor<1520x3080xf64>
// CHECK-NEXT:     return %1 : tensor<1520x3080xf64>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<1520x3056xf64>, %arg1: tensor<f64>) -> tensor<1520x3080xf64> {
    %0 = stablehlo.pad %arg0, %arg1, low = [0, 24], high = [0, 0], interior = [0, 0] : (tensor<1520x3056xf64>, tensor<f64>) -> tensor<1520x3080xf64>
    %1 = stablehlo.pad %arg0, %arg1, low = [0, 0], high = [0, 24], interior = [0, 0] : (tensor<1520x3056xf64>, tensor<f64>) -> tensor<1520x3080xf64>
    %2 = stablehlo.add %1, %0 : tensor<1520x3080xf64>
    return %2 : tensor<1520x3080xf64>
}

// CHECK: func.func @main2(%arg0: tensor<1520x3056xf64>, %arg1: tensor<f64>) -> tensor<1520x3080xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %arg1, low = [0, 24], high = [0, 24], interior = [0, 0] : (tensor<1520x3056xf64>, tensor<f64>) -> tensor<1520x3104xf64>
// CHECK-NEXT:     %1 = "stablehlo.reduce_window"(%0, %cst) <{window_dilations = array<i64: 1, 24>, window_dimensions = array<i64: 1, 2>}> ({
// CHECK-NEXT:     ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:       %2 = stablehlo.add %arg2, %arg3 : tensor<f64>
// CHECK-NEXT:       stablehlo.return %2 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<1520x3104xf64>, tensor<f64>) -> tensor<1520x3080xf64>
// CHECK-NEXT:     return %1 : tensor<1520x3080xf64>
// CHECK-NEXT: }

func.func @main3(%arg0: tensor<1520x3056xf64>, %arg1: tensor<f64>) -> tensor<1520x3080xf64> {
    %0 = stablehlo.pad %arg0, %arg1, low = [0, 24], high = [0, 0], interior = [0, 0] : (tensor<1520x3056xf64>, tensor<f64>) -> tensor<1520x3080xf64>
    %1 = stablehlo.pad %arg0, %arg1, low = [0, 0], high = [0, 24], interior = [0, 0] : (tensor<1520x3056xf64>, tensor<f64>) -> tensor<1520x3080xf64>
    %2 = stablehlo.subtract %0, %1 : tensor<1520x3080xf64>
    return %2 : tensor<1520x3080xf64>
}

// CHECK: func.func @main3(%arg0: tensor<1520x3056xf64>, %arg1: tensor<f64>) -> tensor<1520x3080xf64> {
// CHECK-NEXT{LITERAL}:     %cst = stablehlo.constant dense<[[[[1.000000e+00, -1.000000e+00]]]]> : tensor<1x1x1x2xf64>
// CHECK-NEXT:     %0 = stablehlo.reshape %arg0 : (tensor<1520x3056xf64>) -> tensor<1x1x1520x3056xf64>
// CHECK-NEXT:     %1 = stablehlo.pad %0, %arg1, low = [0, 0, 0, 24], high = [0, 0, 0, 24], interior = [0, 0, 0, 0] : (tensor<1x1x1520x3056xf64>, tensor<f64>) -> tensor<1x1x1520x3104xf64>
// CHECK-NEXT:     %2 = stablehlo.convolution(%1, %cst) dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1], window = {rhs_dilate = [1, 24]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x1520x3104xf64>, tensor<1x1x1x2xf64>) -> tensor<1x1x1520x3080xf64>
// CHECK-NEXT:     %3 = stablehlo.reshape %2 : (tensor<1x1x1520x3080xf64>) -> tensor<1520x3080xf64>
// CHECK-NEXT:     return %3 : tensor<1520x3080xf64>
// CHECK-NEXT: }

func.func @main4(%arg0: tensor<1520x3056xf64>, %arg1: tensor<f64>) -> tensor<1520x3080xf64> {
    %0 = stablehlo.pad %arg0, %arg1, low = [0, 24], high = [0, 0], interior = [0, 0] : (tensor<1520x3056xf64>, tensor<f64>) -> tensor<1520x3080xf64>
    %1 = stablehlo.pad %arg0, %arg1, low = [0, 0], high = [0, 24], interior = [0, 0] : (tensor<1520x3056xf64>, tensor<f64>) -> tensor<1520x3080xf64>
    %2 = stablehlo.subtract %1, %0 : tensor<1520x3080xf64>
    return %2 : tensor<1520x3080xf64>
}

// CHECK: func.func @main4(%arg0: tensor<1520x3056xf64>, %arg1: tensor<f64>) -> tensor<1520x3080xf64> {
// CHECK-NEXT{LITERAL}:     %cst = stablehlo.constant dense<[[[[-1.000000e+00, 1.000000e+00]]]]> : tensor<1x1x1x2xf64>
// CHECK-NEXT:     %0 = stablehlo.reshape %arg0 : (tensor<1520x3056xf64>) -> tensor<1x1x1520x3056xf64>
// CHECK-NEXT:     %1 = stablehlo.pad %0, %arg1, low = [0, 0, 0, 24], high = [0, 0, 0, 24], interior = [0, 0, 0, 0] : (tensor<1x1x1520x3056xf64>, tensor<f64>) -> tensor<1x1x1520x3104xf64>
// CHECK-NEXT:     %2 = stablehlo.convolution(%1, %cst) dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1], window = {rhs_dilate = [1, 24]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x1520x3104xf64>, tensor<1x1x1x2xf64>) -> tensor<1x1x1520x3080xf64>
// CHECK-NEXT:     %3 = stablehlo.reshape %2 : (tensor<1x1x1520x3080xf64>) -> tensor<1520x3080xf64>
// CHECK-NEXT:     return %3 : tensor<1520x3080xf64>
// CHECK-NEXT: }
