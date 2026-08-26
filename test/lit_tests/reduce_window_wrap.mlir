// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @reduce_window_wrap_pattern1(%arg0: tensor<4x3056x6128xf64>) -> tensor<4x3056x6128xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 0 : i64, rhs = 6127 : i64}> : (tensor<4x3056x6128xf64>) -> tensor<4x3056x12255xf64>
  %1 = "stablehlo.reduce_window"(%0, %cst) <{window_dilations = array<i64: 1, 1, 6127>, window_dimensions = array<i64: 1, 1, 2>}> ({
  ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
    %2 = stablehlo.add %arg1, %arg2 : tensor<f64>
    stablehlo.return %2 : tensor<f64>
  }) : (tensor<4x3056x12255xf64>, tensor<f64>) -> tensor<4x3056x6128xf64>
  return %1 : tensor<4x3056x6128xf64>
}

// CHECK: func.func @reduce_window_wrap_pattern1(%arg0: tensor<4x3056x6128xf64>) -> tensor<4x3056x6128xf64> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %0 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<4x3056x6128xf64>) -> tensor<4x3056x6129xf64>
// CHECK-NEXT:   %1 = "stablehlo.reduce_window"(%0, %cst) <{window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 2>}> ({
// CHECK-NEXT:   ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:     %2 = stablehlo.add %arg1, %arg2 : tensor<f64>
// CHECK-NEXT:     stablehlo.return %2 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<4x3056x6129xf64>, tensor<f64>) -> tensor<4x3056x6128xf64>
// CHECK-NEXT:   return %1 : tensor<4x3056x6128xf64>
// CHECK-NEXT: }

func.func @reduce_window_wrap_pattern2(%arg0: tensor<4x3056x6128xf64>) -> tensor<4x3056x6128xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 6127 : i64, rhs = 0 : i64}> : (tensor<4x3056x6128xf64>) -> tensor<4x3056x12255xf64>
  %1 = "stablehlo.reduce_window"(%0, %cst) <{window_dilations = array<i64: 1, 1, 6127>, window_dimensions = array<i64: 1, 1, 2>}> ({
  ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
    %2 = stablehlo.add %arg1, %arg2 : tensor<f64>
    stablehlo.return %2 : tensor<f64>
  }) : (tensor<4x3056x12255xf64>, tensor<f64>) -> tensor<4x3056x6128xf64>
  return %1 : tensor<4x3056x6128xf64>
}

// CHECK: func.func @reduce_window_wrap_pattern2(%arg0: tensor<4x3056x6128xf64>) -> tensor<4x3056x6128xf64> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %0 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 0 : i64, rhs = 1 : i64}> : (tensor<4x3056x6128xf64>) -> tensor<4x3056x6129xf64>
// CHECK-NEXT:   %1 = "stablehlo.reduce_window"(%0, %cst) <{window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 2>}> ({
// CHECK-NEXT:   ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:     %2 = stablehlo.add %arg1, %arg2 : tensor<f64>
// CHECK-NEXT:     stablehlo.return %2 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<4x3056x6129xf64>, tensor<f64>) -> tensor<4x3056x6128xf64>
// CHECK-NEXT:   return %1 : tensor<4x3056x6128xf64>
// CHECK-NEXT: }

func.func @reduce_window_wrap_pattern3(%arg0: tensor<4x3056x6128xf64>) -> tensor<4x3056x6128xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 0 : i64, rhs = 6128 : i64}> : (tensor<4x3056x6128xf64>) -> tensor<4x3056x12256xf64>
  %1 = "stablehlo.reduce_window"(%0, %cst) <{window_dilations = array<i64: 1, 1, 6128>, window_dimensions = array<i64: 1, 1, 2>}> ({
  ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
    %2 = stablehlo.add %arg1, %arg2 : tensor<f64>
    stablehlo.return %2 : tensor<f64>
  }) : (tensor<4x3056x12256xf64>, tensor<f64>) -> tensor<4x3056x6128xf64>
  return %1 : tensor<4x3056x6128xf64>
}

// CHECK: func.func @reduce_window_wrap_pattern3(%arg0: tensor<4x3056x6128xf64>) -> tensor<4x3056x6128xf64> {
// CHECK-NEXT:   %0 = stablehlo.add %arg0, %arg0 : tensor<4x3056x6128xf64>
// CHECK-NEXT:   return %0 : tensor<4x3056x6128xf64>
// CHECK-NEXT: }

func.func @reduce_window_wrap_pattern4(%arg0: tensor<4x3056x6128xf64>) -> tensor<4x3056x6128xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 0 : i64, rhs = 6100 : i64}> : (tensor<4x3056x6128xf64>) -> tensor<4x3056x12228xf64>
  %1 = "stablehlo.reduce_window"(%0, %cst) <{window_dilations = array<i64: 1, 1, 6100>, window_dimensions = array<i64: 1, 1, 2>}> ({
  ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
    %2 = stablehlo.add %arg1, %arg2 : tensor<f64>
    stablehlo.return %2 : tensor<f64>
  }) : (tensor<4x3056x12228xf64>, tensor<f64>) -> tensor<4x3056x6128xf64>
  return %1 : tensor<4x3056x6128xf64>
}

// CHECK: func.func @reduce_window_wrap_pattern4(%arg0: tensor<4x3056x6128xf64>) -> tensor<4x3056x6128xf64> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %0 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 28 : i64, rhs = 0 : i64}> : (tensor<4x3056x6128xf64>) -> tensor<4x3056x6156xf64>
// CHECK-NEXT:   %1 = "stablehlo.reduce_window"(%0, %cst) <{window_dilations = array<i64: 1, 1, 28>, window_dimensions = array<i64: 1, 1, 2>}> ({
// CHECK-NEXT:   ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:     %2 = stablehlo.add %arg1, %arg2 : tensor<f64>
// CHECK-NEXT:     stablehlo.return %2 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<4x3056x6156xf64>, tensor<f64>) -> tensor<4x3056x6128xf64>
// CHECK-NEXT:   return %1 : tensor<4x3056x6128xf64>
// CHECK-NEXT: }
